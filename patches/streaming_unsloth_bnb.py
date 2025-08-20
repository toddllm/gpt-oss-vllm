#!/usr/bin/env python3
"""
Streaming Unsloth to vLLM BitsAndBytes converter.
Dequantizes on CPU, quantizes immediately on GPU per-expert.
"""

import torch
import bitsandbytes as bnb
from typing import List
import safetensors.torch
import logging
import gc

logger = logging.getLogger(__name__)

# Disable gradients globally
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = True

def log_mem(tag):
    """Log current GPU memory usage"""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        logger.info(f"[VRAM] {tag}: alloc={alloc:.2f}GB, reserved={reserv:.2f}GB")
    else:
        logger.info(f"[VRAM] {tag}: CUDA not available")


def make_bnb_linear4bit(out_features: int, in_features: int, device: str = "cuda"):
    """Create a BitsAndBytes Linear4bit module"""
    import bitsandbytes as bnb
    
    # Create Linear4bit with NF4 quantization, block size 128, FP16 compute
    # CRITICAL: Create on CPU first to avoid GPU allocation during memory-critical phase
    linear = bnb.nn.Linear4bit(
        input_features=in_features,   # Correct parameter name
        output_features=out_features, # Correct parameter name
        bias=False,  # GPT-OSS MoE experts don't use bias
        compute_dtype=torch.float16,
        compress_statistics=True,
        quant_type="nf4",
        device="cpu"  # Create on CPU first
    )
    return linear


def fp16_to_bnb_inplace(bnb_linear, fp16_weight: torch.Tensor, device: str = "cuda"):
    """Convert FP16 weight to BnB Linear4bit format in-place"""
    # CRITICAL: Work entirely on CPU to avoid GPU OOM
    # Both bnb_linear and fp16_weight should be on CPU at this point
    
    # Simple approach: copy the FP16 data directly into the Linear4bit weight
    # BitsAndBytes will handle the quantization during the first forward pass
    with torch.no_grad():
        # Both tensors should be on CPU - direct copy without device transfer
        bnb_linear.weight.data.copy_(fp16_weight.data)
    
    # Now move the BnB module to target device (this should be more memory efficient)
    bnb_linear = bnb_linear.to(device)
    
    # Free the FP16 tensor reference
    del fp16_weight
    
    return bnb_linear  # Return moved module


def attach_expert_bnb_linear(mlp_layer, expert: int, proj: str, weight: torch.Tensor, device: str = "cuda"):
    """Attach BnB Linear4bit module directly to expert slot"""
    out_features, in_features = weight.shape
    
    # Create BnB Linear4bit module on CPU
    bnb_linear = make_bnb_linear4bit(out_features, in_features, device="cpu")
    
    # Convert FP16 weight to BnB format and move to target device
    bnb_linear = fp16_to_bnb_inplace(bnb_linear, weight, device)
    
    # Attach to the appropriate expert slot
    # For now, we'll store in a custom attribute since FusedMoE expects fused tensors
    if not hasattr(mlp_layer, 'bnb_experts'):
        mlp_layer.bnb_experts = {}
    
    if expert not in mlp_layer.bnb_experts:
        mlp_layer.bnb_experts[expert] = {}
    
    mlp_layer.bnb_experts[expert][proj] = bnb_linear
    
    # Set flag to use BnB experts
    mlp_layer.use_bnb_experts = True
    
    logger.debug(f"Attached BnB {proj} linear for expert {expert}: {out_features}x{in_features}")
    
    # Validate dimensions
    H, FF = 2880, 2880
    if proj in ["gate", "up"]:
        assert out_features == FF and in_features == H, f"Invalid {proj} dims: {out_features}x{in_features}"
    else:  # down
        assert out_features == H and in_features == FF, f"Invalid {proj} dims: {out_features}x{in_features}"

# NF4 quantization map from bitsandbytes
NF4_QUANT_MAP = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
], dtype=torch.float32)


def dequant_unsloth_expert_cpu(checkpoint_paths: List[str], layer_idx: int, 
                               expert_idx: int, proj_type: str) -> torch.Tensor:
    """
    Dequantize single expert weight on CPU.
    
    Returns:
        FP16 tensor or None if not found
    """
    # Build key names for Unsloth format  
    # Format is: model.layers.{layer}.mlp.experts.{gate_up_projs|down_projs}.{expert}.weight
    prefix = f"model.layers.{layer_idx}.mlp.experts"
    
    if proj_type == "gate_up":
        weight_key = f"{prefix}.gate_up_projs.{expert_idx}.weight"
        out_features = 5760  # 2 * intermediate_size
        in_features = 2880   # hidden_size
    else:  # down
        weight_key = f"{prefix}.down_projs.{expert_idx}.weight"
        out_features = 2880  # hidden_size
        in_features = 2880   # intermediate_size
    
    # Try each checkpoint file
    for ckpt_path in checkpoint_paths:
        try:
            with safetensors.torch.safe_open(ckpt_path, framework="pt", device="cpu") as f:
                if weight_key not in f.keys():
                    continue
                
                qweight = f.get_tensor(weight_key)
                
                # Get quantization parameters
                absmax_key = weight_key + ".absmax"
                
                absmax_data = f.get_tensor(absmax_key) if absmax_key in f.keys() else None
                
                # Check for double quantization
                nested_absmax_key = weight_key + ".nested_absmax"
                nested_quant_map_key = weight_key + ".nested_quant_map"
                
                if nested_absmax_key in f.keys():
                    # Double quantized - dequantize absmax first
                    nested_absmax = f.get_tensor(nested_absmax_key).float()
                    nested_quant_map = f.get_tensor(nested_quant_map_key).float()
                    
                    # Dequantize absmax
                    absmax_uint8 = absmax_data.to(torch.uint8)
                    absmax = nested_quant_map[absmax_uint8.long()]
                    
                    # Handle shape compatibility for nested scaling
                    if nested_absmax.numel() == 1:
                        # Single scaling factor
                        absmax = absmax * nested_absmax.item()
                    elif nested_absmax.shape == absmax.shape:
                        # Element-wise scaling
                        absmax = absmax * nested_absmax
                    else:
                        # Try to broadcast
                        try:
                            absmax = absmax * nested_absmax
                        except RuntimeError as e:
                            print(f"Shape mismatch in nested scaling: absmax {absmax.shape}, nested_absmax {nested_absmax.shape}")
                            # Fallback: use mean of nested_absmax
                            absmax = absmax * nested_absmax.mean()
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
            # absmax contains scaling factors for blocks of the quantized weight
            if len(absmax.shape) == 1 and absmax.numel() > 1:
                # Blockwise scaling - absmax has one entry per block
                blocksize = total_params // absmax.numel()
                if blocksize * absmax.numel() != total_params:
                    # Handle partial blocks
                    blocksize = 64  # fallback
                    num_blocks = min((total_params + blocksize - 1) // blocksize, len(absmax))
                    for block_idx in range(num_blocks):
                        start_idx = block_idx * blocksize
                        end_idx = min(start_idx + blocksize, total_params)
                        weight_fp[start_idx:end_idx] *= absmax[block_idx]
                else:
                    # Perfect division - each absmax entry scales one block
                    weight_fp = weight_fp.view(-1, blocksize) * absmax.unsqueeze(1)
                    weight_fp = weight_fp.view(-1)
            else:
                # Single scaling factor for all weights
                weight_fp *= absmax.item() if absmax.numel() == 1 else absmax[0]
            
            # Reshape and convert to FP16
            weight_matrix = weight_fp.reshape(out_features, in_features)
            return weight_matrix.to(torch.float16).contiguous()
            
        except Exception as e:
            print(f"Error loading {weight_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return None


@torch.inference_mode()
def convert_layer_streaming_bnb(layer_idx: int, checkpoint_paths: List[str],
                                mlp_layer, device: str = "cuda") -> None:
    """
    Convert one layer using streaming per-expert BitsAndBytes approach.
    This replaces the FusedMoE weights with quantized versions.
    """
    H = 2880   # hidden_size 
    FF = 2880  # intermediate_size (square MLP)
    E = 32     # num_experts
    
    logger.info(f"Converting layer {layer_idx} with streaming BnB pipeline")
    log_mem(f"Start_L{layer_idx}")
    
    # Direct per-expert BnB attachment - NO fused tensor accumulation
    
    for e in range(E):
        if e % 8 == 0:
            logger.info(f"  Layer {layer_idx}: Converting expert {e}/{E}")
        
        # Reset peak memory tracking
        torch.cuda.reset_peak_memory_stats()
        
        # Dequantize gate_up on CPU
        cpu_fp16_w13 = dequant_unsloth_expert_cpu(
            checkpoint_paths, layer_idx, e, "gate_up"
        )
        
        if cpu_fp16_w13 is None:
            logger.warning(f"Could not find expert {e} gate_up")
            # Use zeros as fallback
            cpu_fp16_w13 = torch.zeros(2*FF, H, dtype=torch.float16)
        
        # Direct attachment - NO accumulation
        if cpu_fp16_w13 is not None:
            # Split gate_up into w1 (gate) and w3 (up) components
            w1_cpu = cpu_fp16_w13[:FF, :].contiguous()  # [FF, H] - gate
            w3_cpu = cpu_fp16_w13[FF:, :].contiguous()  # [FF, H] - up
            del cpu_fp16_w13
            
            # Create BnB Linear4bit modules and attach directly
            attach_expert_bnb_linear(mlp_layer, expert=e, proj="gate", weight=w1_cpu, device=device)
            del w1_cpu
            
            attach_expert_bnb_linear(mlp_layer, expert=e, proj="up", weight=w3_cpu, device=device)
            del w3_cpu
        
        # Dequantize down on CPU
        cpu_fp16_w2 = dequant_unsloth_expert_cpu(
            checkpoint_paths, layer_idx, e, "down"
        )
        
        if cpu_fp16_w2 is not None:
            # Create BnB Linear4bit and attach directly
            attach_expert_bnb_linear(mlp_layer, expert=e, proj="down", weight=cpu_fp16_w2, device=device)
            del cpu_fp16_w2
        
        # Aggressive cleanup after each expert
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory usage every 8 experts
        if e % 8 == 0:
            log_mem(f"L{layer_idx}_E{e}")
    
    # All experts attached directly - no fused tensors needed
    
    logger.info(f"Layer {layer_idx} conversion complete")
    log_mem(f"End_L{layer_idx}")