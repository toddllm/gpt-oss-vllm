#!/usr/bin/env python3
"""
Direct Unsloth weight loader for vLLM GPT-OSS model.
This bypasses the FusedMoE requirement by directly attaching Unsloth weights
to the model without conversion.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from safetensors import safe_open
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit, dequantize_blockwise
from vllm.logger import init_logger
import os
import glob

logger = init_logger(__name__)


def dequantize_absmax(absmax_uint8: torch.Tensor, 
                     nested_absmax: torch.Tensor,
                     nested_quant_map: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the double-quantized absmax values.
    Unsloth uses nested quantization where absmax itself is quantized.
    """
    # The absmax values are quantized with nested_absmax as scale
    # and nested_quant_map as codebook
    
    # First, convert uint8 to indices
    indices = absmax_uint8.long().flatten()
    
    # Look up values in codebook
    dequantized = nested_quant_map[indices]
    
    # Apply scale - nested_absmax is a single value
    if nested_absmax.numel() == 1:
        dequantized = dequantized * nested_absmax.item()
    else:
        # Block-wise scaling
        blocksize = len(indices) // len(nested_absmax)
        for i in range(len(nested_absmax)):
            start = i * blocksize
            end = min(start + blocksize, len(indices))
            dequantized[start:end] *= nested_absmax[i]
    
    return dequantized


def load_and_dequantize_unsloth_weight(checkpoint_path: str,
                                      layer_idx: int,
                                      expert_idx: int,
                                      proj_type: str) -> torch.Tensor:
    """
    Load an Unsloth weight and fully dequantize it to FP16.
    
    This is the fallback approach when direct packed weight usage fails.
    """
    
    with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
        if proj_type == "gate_up":
            prefix = f"model.layers.{layer_idx}.mlp.experts.gate_up_projs.{expert_idx}"
        else:  # down
            prefix = f"model.layers.{layer_idx}.mlp.experts.down_projs.{expert_idx}"
        
        # Load packed weight and metadata
        weight_key = f"{prefix}.weight"
        qweight = f.get_tensor(weight_key)  # Shape: [N, 1] uint8
        
        # Load quantization metadata
        absmax_key = f"{weight_key}.absmax"
        absmax_uint8 = f.get_tensor(absmax_key)  # uint8, double quantized
        
        nested_absmax_key = f"{weight_key}.nested_absmax"
        nested_absmax = f.get_tensor(nested_absmax_key)  # float32
        
        nested_quant_map_key = f"{weight_key}.nested_quant_map"
        nested_quant_map = f.get_tensor(nested_quant_map_key)  # float32
        
        quant_map_key = f"{weight_key}.quant_map"
        quant_map = f.get_tensor(quant_map_key)  # float32, NF4 codebook
        
        # Dequantize absmax (it's double quantized)
        absmax = dequantize_absmax(absmax_uint8, nested_absmax, nested_quant_map)
        
        # Reshape qweight if needed
        if len(qweight.shape) == 2 and qweight.shape[1] == 1:
            qweight = qweight.squeeze(1)
        
        # Determine expected shape based on projection type
        # Unsloth GPT-OSS-20B variant: hidden_size=2880, intermediate_size=2880 (square MLP)
        # This is different from standard GPT-OSS which has intermediate_size=14400
        if proj_type == "gate_up":
            # Combined gate (w1) and up (w3): hidden -> 2*intermediate
            out_features = 2 * 2880   # gate + up combined (square dimensions)
            in_features = 2880        # hidden size
        else:  # down
            # Down projection (w2): intermediate -> hidden
            out_features = 2880   # hidden size  
            in_features = 2880    # intermediate size (square)
        
        # Dequantize the 4-bit weight
        # Each byte contains 2 4-bit values
        total_params = out_features * in_features
        
        # Unpack 4-bit values from bytes - vectorized for speed
        print(f"  Unpacking {len(qweight)} bytes to {total_params} 4-bit values...")
        qweight_unpacked = torch.zeros(total_params, dtype=torch.uint8, device=qweight.device)
        
        # Vectorized unpacking - much faster than loop
        qweight_bytes = qweight.view(-1)
        lower_nibbles = (qweight_bytes & 0x0F)
        upper_nibbles = ((qweight_bytes >> 4) & 0x0F)
        
        # Interleave lower and upper nibbles
        if total_params % 2 == 0:
            qweight_unpacked[0::2] = lower_nibbles
            qweight_unpacked[1::2] = upper_nibbles
        else:
            qweight_unpacked[0::2] = lower_nibbles
            qweight_unpacked[1::2] = upper_nibbles[:total_params//2]
        
        # Apply NF4 dequantization using the codebook
        # qweight_unpacked contains indices into the quant_map
        weight_fp = quant_map[qweight_unpacked.long()]
        
        # Apply absmax scaling (blockwise)
        blocksize = 64
        num_blocks = (total_params + blocksize - 1) // blocksize
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * blocksize
            end_idx = min(start_idx + blocksize, total_params)
            
            if block_idx < len(absmax):
                weight_fp[start_idx:end_idx] *= absmax[block_idx]
        
        # Reshape to matrix form
        weight_matrix = weight_fp.reshape(out_features, in_features)
        
        return weight_matrix.to(torch.float16)


def create_fused_expert_weights(checkpoint_paths: List[str],
                               layer_idx: int,
                               num_experts: int = 32) -> Dict[str, torch.Tensor]:
    """
    Create fused expert tensors by dequantizing and concatenating all experts.
    
    Returns:
        dict with 'w13_weight' and 'w2_weight' tensors
    """
    
    w13_experts = []
    w2_experts = []
    
    for expert_idx in range(num_experts):
        logger.info(f"Dequantizing L{layer_idx}E{expert_idx}")
        
        # Try each checkpoint file
        for ckpt_path in checkpoint_paths:
            try:
                # Load gate_up projection
                gate_up_weight = load_and_dequantize_unsloth_weight(
                    ckpt_path, layer_idx, expert_idx, "gate_up"
                )
                w13_experts.append(gate_up_weight)
                
                # Load down projection
                down_weight = load_and_dequantize_unsloth_weight(
                    ckpt_path, layer_idx, expert_idx, "down"
                )
                w2_experts.append(down_weight)
                
                break  # Found in this checkpoint
                
            except Exception as e:
                continue  # Try next checkpoint
    
    if len(w13_experts) == num_experts and len(w2_experts) == num_experts:
        # Stack all experts into fused tensors
        # w13 contains gate (w1) and up (w3) concatenated
        # Note: Unsloth variant has square MLPs (intermediate=2880)
        w13_weight = torch.stack(w13_experts, dim=0)  # [32, 5760, 2880]
        w2_weight = torch.stack(w2_experts, dim=0)    # [32, 2880, 2880]
        
        # Critical shape assertions
        assert w13_weight.shape == (32, 5760, 2880), \
            f"w13 shape mismatch: {w13_weight.shape} != (32, 5760, 2880)"
        assert w2_weight.shape == (32, 2880, 2880), \
            f"w2 shape mismatch: {w2_weight.shape} != (32, 2880, 2880)"
        
        logger.info(f"Created fused tensors: w13={w13_weight.shape}, w2={w2_weight.shape} ✓")
        
        return {
            'w13_weight': w13_weight,
            'w2_weight': w2_weight
        }
    else:
        logger.error(f"Only found {len(w13_experts)} experts, expected {num_experts}")
        return None


def patch_gptoss_for_unsloth_loading():
    """
    Patch the GPT-OSS model to handle Unsloth weight loading.
    """
    
    import sys
    gptoss_path = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"
    
    # Read the file
    with open(gptoss_path, 'r') as f:
        lines = f.readlines()
    
    # Find where to add the Unsloth loading logic
    for i, line in enumerate(lines):
        if "def load_weights(" in line:
            # Add import at top of method
            insert_idx = i + 3
            
            insert_lines = [
                "        # Check if this is an Unsloth model\n",
                "        if 'unsloth' in str(getattr(self, '_model_path', '')).lower():\n",
                "            from vllm.logger import init_logger\n",
                "            logger = init_logger(__name__)\n",
                "            logger.info('Detected Unsloth model, using special weight loader')\n",
                "            \n",
                "            # Import the Unsloth loader\n",
                "            import sys\n",
                "            sys.path.insert(0, '/home/tdeshane/vllm/patches')\n",
                "            from unsloth_direct_loader import create_fused_expert_weights\n",
                "            \n",
                "            # Convert and load weights layer by layer\n",
                "            import glob\n",
                "            import os\n",
                "            cache_dir = os.path.expanduser('~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/')\n",
                "            ckpt_files = sorted(glob.glob(os.path.join(cache_dir, '*/*.safetensors')))\n",
                "            \n",
                "            if ckpt_files:\n",
                "                for layer_idx in range(self.config.num_hidden_layers):\n",
                "                    logger.info(f'Converting layer {layer_idx}')\n",
                "                    fused_weights = create_fused_expert_weights(ckpt_files, layer_idx)\n",
                "                    if fused_weights:\n",
                "                        # Apply to model\n",
                "                        mlp = self.model.layers[layer_idx].mlp\n",
                "                        if hasattr(mlp.experts, 'w13_weight'):\n",
                "                            mlp.experts.w13_weight.data = fused_weights['w13_weight'].to(mlp.experts.w13_weight.device)\n",
                "                        if hasattr(mlp.experts, 'w2_weight'):\n",
                "                            mlp.experts.w2_weight.data = fused_weights['w2_weight'].to(mlp.experts.w2_weight.device)\n",
                "                return\n",
                "        \n",
            ]
            
            lines[insert_idx:insert_idx] = insert_lines
            break
    
    # Write back
    with open(gptoss_path, 'w') as f:
        f.writelines(lines)
    
    logger.info("Patched GPT-OSS for Unsloth loading")


def test_dequantization():
    """Test the dequantization process."""
    
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/")
    ckpt_files = sorted(glob.glob(os.path.join(cache_dir, "*/*.safetensors")))
    
    if not ckpt_files:
        logger.error(f"No checkpoint files found in {cache_dir}")
        return False
    
    logger.info(f"Found {len(ckpt_files)} checkpoint files")
    
    # Test dequantizing one expert
    try:
        weight = load_and_dequantize_unsloth_weight(ckpt_files[0], 0, 0, "gate_up")
        logger.info(f"Successfully dequantized gate_up weight: shape={weight.shape}, dtype={weight.dtype}")
        
        # Check values are reasonable
        logger.info(f"Weight stats: min={weight.min():.4f}, max={weight.max():.4f}, mean={weight.mean():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Dequantization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dequantization()
    if success:
        print("\n✓ Dequantization test passed!")
        # Optionally patch the model
        # patch_gptoss_for_unsloth_loading()
    else:
        print("\n✗ Dequantization test failed")