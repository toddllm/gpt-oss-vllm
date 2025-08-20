#!/usr/bin/env python3
"""
BitsAndBytes 4-bit quantization for MoE experts only.
Targets GPT-OSS MoE weights on RTX 3090.
"""

import torch
from typing import Any, Dict, List, Optional
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    logger.warning("bitsandbytes not installed - BnB MoE quantization unavailable")


class BitsAndBytesMoE4bitConfig(QuantizationConfig):
    """4-bit quantization for MoE experts only, keeping router/attention FP16"""
    
    def __init__(self):
        self.weight_bits = 4
        self.group_size = 128
        self.compute_dtype = torch.float16
        self.use_nf4 = True  # Use NF4 quantization (better for 4-bit)
        
    @classmethod
    def get_name(cls) -> str:
        return "bnb_moe_4bit"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BitsAndBytesMoE4bitConfig":
        return cls()
    
    def get_quant_method(self, layer, prefix):
        """Return quantization method for this layer"""
        # Only quantize MoE expert weights
        if "experts" in prefix and any(x in prefix for x in ["w13", "w2", "gate", "up", "down"]):
            return "bnb_4bit"
        return None
    
    def get_scaled_act_names(self) -> List[str]:
        return []
    
    @classmethod
    def get_min_capability(cls) -> int:
        # Minimum compute capability for BitsAndBytes
        return 70  # SM 7.0 (Volta)
    
    def wrap_linear(self, linear_layer: torch.nn.Linear, 
                   layer_name: str = "") -> torch.nn.Module:
        """
        Wrap a linear layer with BitsAndBytes 4-bit quantization.
        Only used for MoE expert weights.
        """
        if not HAS_BNB:
            raise ImportError("bitsandbytes required for BnB MoE quantization")
        
        # Check if this is an MoE expert layer
        if not any(x in layer_name for x in ["experts", "w13", "w2"]):
            logger.debug(f"Skipping BnB quantization for non-expert layer: {layer_name}")
            return linear_layer
        
        logger.info(f"Applying BnB 4-bit quantization to MoE expert: {layer_name}")
        
        # Create BitsAndBytes 4-bit linear layer
        bnb_linear = bnb.nn.Linear4bit(
            linear_layer.in_features,
            linear_layer.out_features,
            bias=linear_layer.bias is not None,
            compute_dtype=self.compute_dtype,
            compress_statistics=True,
            quant_type='nf4',  # Use NF4 quantization
        )
        
        # If the original layer has weights, quantize them
        if linear_layer.weight is not None:
            bnb_linear.weight = bnb.nn.Params4bit(
                linear_layer.weight.data,
                requires_grad=False,
                compress_statistics=True,
                quant_type='nf4',
            )
        
        if linear_layer.bias is not None:
            bnb_linear.bias = linear_layer.bias
            
        return bnb_linear


def is_moe_weight(name: str) -> bool:
    """Check if a weight name corresponds to an MoE expert weight"""
    moe_patterns = [
        "experts.w13_weight",  # Gate and up projection
        "experts.w2_weight",   # Down projection
        "experts.w13_bias",
        "experts.w2_bias",
        "gate_up_proj",
        "down_proj",
    ]
    return any(pattern in name for pattern in moe_patterns)


def get_moe_weight_names(model) -> List[str]:
    """Get list of all MoE expert weight names in the model"""
    moe_weights = []
    for name, _ in model.named_parameters():
        if is_moe_weight(name):
            moe_weights.append(name)
    return moe_weights