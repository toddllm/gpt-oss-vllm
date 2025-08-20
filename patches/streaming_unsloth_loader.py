#!/usr/bin/env python3
"""
Streaming Unsloth to BitsAndBytes converter.
Processes one expert at a time to avoid OOM on RTX 3090.
"""

import torch
import bitsandbytes as bnb
from bitsandbytes import nn as bnb_nn
from typing import Dict, List, Tuple, Optional
import safetensors.torch
import logging
import gc

logger = logging.getLogger(__name__)

# Disable gradients globally
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

# NF4 quantization map from bitsandbytes
NF4_QUANT_MAP = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


def make_bnb_linear4bit(out_features: int, in_features: int, 
                        quant_type: str = "nf4", compute_dtype=torch.float16) -> bnb_nn.Linear4bit:
    """Create an empty BnB Linear4bit module."""
    return bnb_nn.Linear4bit(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        compute_dtype=compute_dtype,
        quant_type=quant_type,
    )


def fp16_to_bnb_inplace(lin4b: bnb_nn.Linear4bit, cpu_fp16_weight: torch.Tensor, 
                        device: str = "cuda") -> None:
    """Quantize FP16 weight into BnB Linear4bit module in-place."""
    # Ensure contiguous
    cpu_fp16_weight = cpu_fp16_weight.to(memory_format=torch.contiguous_format)
    
    # Upload to GPU
    w_gpu = cpu_fp16_weight.pin_memory().to(device, non_blocking=False)
    del cpu_fp16_weight
    
    # Quantize using BnB API
    if hasattr(bnb.nn, "Params4bit"):
        # Newer BnB API
        lin4b.weight = bnb.nn.Params4bit(
            w_gpu, 
            requires_grad=False, 
            quant_type=lin4b.quant_type
        )
    elif hasattr(bnb.nn, "Int4Params"):
        # Older BnB API
        lin4b.weight = bnb.nn.Int4Params(
            w_gpu,
            requires_grad=False,
            quant_type=lin4b.quant_type
        )
    else:
        raise RuntimeError("BitsAndBytes version incompatible - no Params4bit or Int4Params")
    
    del w_gpu
    torch.cuda.empty_cache()


def dequant_unsloth_weight_cpu(checkpoint_path: str, layer_idx: int, 
                               expert_idx: int, proj_type: str,
                               out_features: int, in_features: int) -> torch.Tensor:
    """
    Dequantize Unsloth weight on CPU, return FP16 tensor.
    
    Args:
        checkpoint_path: Path to safetensors file
        layer_idx: Layer index (0-23)
        expert_idx: Expert index (0-31)
        proj_type: "gate_up" or "down"
        out_features: Output dimension
        in_features: Input dimension
    
    Returns:
        FP16 tensor of shape (out_features, in_features)
    """
    # Build key names
    prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
    
    if proj_type == "gate_up":
        weight_key = f"{prefix}.w13_weight"
    else:  # down
        weight_key = f"{prefix}.w2_weight"
    
    # Load from safetensors
    with safetensors.torch.safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        if weight_key not in f.keys():
            # Try alternate naming
            if proj_type == "gate_up":
                weight_key = f"{prefix}.gate_up_proj.qweight"
            else:
                weight_key = f"{prefix}.down_proj.qweight"
            
            if weight_key not in f.keys():
                return None
        
        qweight = f.get_tensor(weight_key)
        
        # Get quantization parameters
        absmax_key = weight_key.replace("qweight", "absmax") 
        if absmax_key not in f.keys():
            absmax_key = weight_key.replace("weight", "absmax")
        
        # Handle nested/double quantization
        absmax_data = f.get_tensor(absmax_key) if absmax_key in f.keys() else None
        
        # Check for double quantization
        nested_absmax_key = absmax_key.replace("absmax", "nested_absmax")
        nested_quant_map_key = absmax_key.replace("absmax", "nested_quant_map")
        
        if nested_absmax_key in f.keys():
            # Double quantized - dequantize absmax first
            nested_absmax = f.get_tensor(nested_absmax_key).float()
            nested_quant_map = f.get_tensor(nested_quant_map_key).float()
            
            # Dequantize absmax (it was quantized as int8)
            absmax_uint8 = absmax_data.to(torch.uint8)
            absmax = nested_quant_map[absmax_uint8.long()]
            absmax = absmax * nested_absmax.item()
        else:
            absmax = absmax_data.float() if absmax_data is not None else torch.ones(1)
    
    # Unpack 4-bit values - vectorized
    total_params = out_features * in_features
    qweight_unpacked = torch.zeros(total_params, dtype=torch.uint8)
    
    # Each byte contains 2 4-bit values
    qweight_bytes = qweight.view(-1)
    lower_nibbles = (qweight_bytes & 0x0F)
    upper_nibbles = ((qweight_bytes >> 4) & 0x0F)
    
    # Interleave
    if total_params % 2 == 0:
        qweight_unpacked[0::2] = lower_nibbles
        qweight_unpacked[1::2] = upper_nibbles
    else:
        qweight_unpacked[0::2] = lower_nibbles
        qweight_unpacked[1::2] = upper_nibbles[:total_params//2]
    
    # Apply NF4 dequantization
    weight_fp = NF4_QUANT_MAP[qweight_unpacked.long()]
    
    # Apply blockwise absmax scaling
    blocksize = 64
    num_blocks = (total_params + blocksize - 1) // blocksize
    
    for block_idx in range(min(num_blocks, len(absmax))):
        start_idx = block_idx * blocksize
        end_idx = min(start_idx + blocksize, total_params)
        weight_fp[start_idx:end_idx] *= absmax[block_idx]
    
    # Reshape and convert to FP16
    weight_matrix = weight_fp.reshape(out_features, in_features)
    return weight_matrix.to(torch.float16).contiguous()


@torch.inference_mode()
def convert_layer_streaming(layer_idx: int, checkpoint_paths: List[str],
                           mlp_layer, device: str = "cuda") -> None:
    """
    Convert one layer using streaming per-expert approach.
    
    Args:
        layer_idx: Layer index (0-23)
        checkpoint_paths: List of safetensors checkpoint paths
        mlp_layer: The MoE MLP layer to populate
        device: Target device
    """
    H = 2880   # hidden_size (square MLP variant)
    FF = 2880  # intermediate_size
    E = 32     # num_experts
    
    logger.info(f"Converting layer {layer_idx} with streaming pipeline")
    
    for e in range(E):
        if e % 8 == 0:
            logger.info(f"  Layer {layer_idx}: Converting expert {e}/{E}")
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        # Try each checkpoint file
        converted = False
        for ckpt_path in checkpoint_paths:
            # --- Process gate_up (w13) ---
            cpu_fp16_w13 = dequant_unsloth_weight_cpu(
                ckpt_path, layer_idx, e, "gate_up", 
                out_features=2*FF, in_features=H
            )
            
            if cpu_fp16_w13 is None:
                continue
                
            # Split into w1 (gate) and w3 (up)
            cpu_fp16_w1 = cpu_fp16_w13[:FF, :].contiguous()
            cpu_fp16_w3 = cpu_fp16_w13[FF:, :].contiguous()
            del cpu_fp16_w13
            
            # Quantize w1 (gate) immediately
            assert cpu_fp16_w1.shape == (FF, H), f"w1 shape mismatch: {cpu_fp16_w1.shape}"
            lin4b_w1 = make_bnb_linear4bit(out_features=FF, in_features=H)
            fp16_to_bnb_inplace(lin4b_w1, cpu_fp16_w1, device)
            
            # Attach to expert - gate projection
            if hasattr(mlp_layer.experts, 'w1'):
                if mlp_layer.experts.w1 is None:
                    mlp_layer.experts.w1 = [None] * E
                mlp_layer.experts.w1[e] = lin4b_w1
            
            del cpu_fp16_w1
            gc.collect()
            torch.cuda.empty_cache()
            
            # Quantize w3 (up) immediately  
            assert cpu_fp16_w3.shape == (FF, H), f"w3 shape mismatch: {cpu_fp16_w3.shape}"
            lin4b_w3 = make_bnb_linear4bit(out_features=FF, in_features=H)
            fp16_to_bnb_inplace(lin4b_w3, cpu_fp16_w3, device)
            
            # Attach to expert - up projection
            if hasattr(mlp_layer.experts, 'w3'):
                if mlp_layer.experts.w3 is None:
                    mlp_layer.experts.w3 = [None] * E
                mlp_layer.experts.w3[e] = lin4b_w3
            
            del cpu_fp16_w3
            gc.collect()
            torch.cuda.empty_cache()
            
            # --- Process down (w2) ---
            cpu_fp16_w2 = dequant_unsloth_weight_cpu(
                ckpt_path, layer_idx, e, "down",
                out_features=H, in_features=FF
            )
            
            if cpu_fp16_w2 is None:
                continue
                
            # Quantize w2 (down) immediately
            assert cpu_fp16_w2.shape == (H, FF), f"w2 shape mismatch: {cpu_fp16_w2.shape}"
            lin4b_w2 = make_bnb_linear4bit(out_features=H, in_features=FF)
            fp16_to_bnb_inplace(lin4b_w2, cpu_fp16_w2, device)
            
            # Attach to expert - down projection
            if hasattr(mlp_layer.experts, 'w2'):
                if mlp_layer.experts.w2 is None:
                    mlp_layer.experts.w2 = [None] * E
                mlp_layer.experts.w2[e] = lin4b_w2
            
            del cpu_fp16_w2
            gc.collect()
            torch.cuda.empty_cache()
            
            converted = True
            break
        
        if not converted:
            logger.warning(f"Could not find expert {e} in any checkpoint")
        
        # Log memory usage
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        logger.debug(f"  L{layer_idx} E{e}: peak VRAM ~{peak_mb:.1f} MiB")
    
    logger.info(f"Layer {layer_idx} conversion complete")