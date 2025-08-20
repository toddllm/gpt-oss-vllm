#!/usr/bin/env python3
"""
Unsloth to vLLM weight converter for GPT-OSS MoE model.
This converter handles the format mismatch between Unsloth's individual expert storage
and vLLM's FusedMoE expectations.

Key approach: Create virtual fused tensors that dynamically dispatch to individual experts.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from safetensors import safe_open
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from bitsandbytes.functional import QuantState
from vllm.logger import init_logger
import os
import glob

logger = init_logger(__name__)


class UnslothToVLLMConverter:
    """Converter for Unsloth packed weights to vLLM FusedMoE format."""
    
    def __init__(self, checkpoint_paths: List[str], device: str = "cuda"):
        self.checkpoint_paths = checkpoint_paths
        self.device = device
        self.weight_cache = {}
        self._analyze_checkpoint_structure()
    
    def _analyze_checkpoint_structure(self):
        """Analyze the Unsloth checkpoint to understand structure."""
        self.structure = {
            'num_layers': 0,
            'num_experts': 32,  # GPT-OSS has 32 experts
            'hidden_size': 2880,
            'intermediate_size': 14400,
            'has_bias': False
        }
        
        # Scan first checkpoint to understand structure
        with safe_open(self.checkpoint_paths[0], framework='pt', device='cpu') as f:
            for key in f.keys():
                if 'mlp.experts' in key:
                    # Extract layer index
                    if 'layers.' in key:
                        layer_idx = int(key.split('layers.')[1].split('.')[0])
                        self.structure['num_layers'] = max(self.structure['num_layers'], layer_idx + 1)
                    
                    # Check for bias
                    if '.bias' in key:
                        self.structure['has_bias'] = True
        
        logger.info(f"Checkpoint structure: {self.structure}")
    
    def load_expert_weight(self, layer_idx: int, expert_idx: int, 
                          proj_type: str) -> Dict[str, torch.Tensor]:
        """Load a specific expert's weight from checkpoint."""
        
        cache_key = (layer_idx, expert_idx, proj_type)
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        
        weights = {}
        
        # Try each checkpoint file
        for ckpt_path in self.checkpoint_paths:
            with safe_open(ckpt_path, framework='pt', device='cpu') as f:
                if proj_type == "gate_up":
                    prefix = f"model.layers.{layer_idx}.mlp.experts.gate_up_projs.{expert_idx}"
                else:  # down
                    prefix = f"model.layers.{layer_idx}.mlp.experts.down_projs.{expert_idx}"
                
                # Main weight (packed 4-bit)
                weight_key = f"{prefix}.weight"
                if weight_key in f.keys():
                    weights['qweight'] = f.get_tensor(weight_key)
                    
                    # Quantization metadata
                    for suffix in ['absmax', 'nested_absmax', 'nested_quant_map', 'quant_map']:
                        meta_key = f"{weight_key}.{suffix}"
                        if meta_key in f.keys():
                            weights[suffix] = f.get_tensor(meta_key)
                    
                    # Bias if present
                    bias_key = f"{prefix}.bias"
                    if bias_key in f.keys():
                        weights['bias'] = f.get_tensor(bias_key)
                    
                    # Cache and return
                    self.weight_cache[cache_key] = weights
                    return weights
        
        return weights
    
    def create_fused_expert_tensor(self, layer_idx: int, 
                                  proj_type: str) -> Tuple[torch.Tensor, Dict]:
        """
        Create a virtual fused tensor for all experts in a layer.
        
        For vLLM FusedMoE:
        - w13_weight: [num_experts, 2*intermediate_size, hidden_size] 
        - w2_weight: [num_experts, hidden_size, intermediate_size]
        """
        num_experts = self.structure['num_experts']
        hidden_size = self.structure['hidden_size']
        intermediate_size = self.structure['intermediate_size']
        
        if proj_type == "w13":  # gate_up combined
            # Expected shape: [32, 28800, 2880]
            # But Unsloth stores packed 4-bit: [8294400, 1] per expert
            # 8294400 bytes * 2 nibbles/byte = 16588800 4-bit values
            # 16588800 / (28800 * 2880) * 4 = ~0.8, suggests additional compression
            
            # Create placeholder for fused tensor metadata
            fused_metadata = {
                'experts': [],
                'type': 'gate_up',
                'layer': layer_idx
            }
            
            # Collect all expert weights
            for expert_idx in range(num_experts):
                expert_weights = self.load_expert_weight(layer_idx, expert_idx, "gate_up")
                if 'qweight' in expert_weights:
                    fused_metadata['experts'].append({
                        'idx': expert_idx,
                        'weights': expert_weights
                    })
                else:
                    logger.warning(f"Missing weights for L{layer_idx}E{expert_idx} gate_up")
            
            # Create virtual fused tensor (will be handled specially)
            # For now, return a placeholder tensor with metadata attached
            placeholder = torch.zeros((1,), device=self.device)
            placeholder._unsloth_metadata = fused_metadata
            return placeholder, fused_metadata
            
        else:  # w2 (down projection)
            # Expected shape: [32, 2880, 14400]
            fused_metadata = {
                'experts': [],
                'type': 'down',
                'layer': layer_idx
            }
            
            for expert_idx in range(num_experts):
                expert_weights = self.load_expert_weight(layer_idx, expert_idx, "down")
                if 'qweight' in expert_weights:
                    fused_metadata['experts'].append({
                        'idx': expert_idx,
                        'weights': expert_weights
                    })
                else:
                    logger.warning(f"Missing weights for L{layer_idx}E{expert_idx} down")
            
            placeholder = torch.zeros((1,), device=self.device)
            placeholder._unsloth_metadata = fused_metadata
            return placeholder, fused_metadata
    
    def create_bnb_linear_for_expert(self, expert_weights: Dict[str, torch.Tensor],
                                    out_features: int, in_features: int) -> Linear4bit:
        """Create a BitsAndBytes Linear4bit module with Unsloth weights."""
        
        # Create Linear4bit module (note: BnB uses input_features, output_features order)
        linear = Linear4bit(
            input_features=in_features,
            output_features=out_features,
            bias=expert_weights.get('bias') is not None,
            compute_dtype=torch.float16,
            compress_statistics=False,
            quant_type='nf4'
        )
        
        # Extract quantization metadata
        qweight = expert_weights['qweight']
        absmax = expert_weights.get('absmax')
        
        # Handle packed weight shape
        if len(qweight.shape) == 2 and qweight.shape[1] == 1:
            qweight = qweight.squeeze(1)
        
        # Create quantization state
        quant_state = QuantState(
            absmax=absmax.to(self.device) if absmax is not None else None,
            shape=(out_features, in_features),
            code=expert_weights.get('quant_map'),
            blocksize=64,
            quant_type='nf4',
            dtype=torch.float16,
            offset=expert_weights.get('nested_absmax'),
            state2=expert_weights.get('nested_quant_map')
        )
        
        # Create Params4bit with the packed weight
        if hasattr(bnb.nn, 'Params4bit'):
            params = Params4bit(
                qweight.to(self.device).contiguous(),
                requires_grad=False,
                compress_statistics=False,
                quant_type='nf4'
            )
        else:
            # Fallback for older API
            params = bnb.nn.Int4Params(
                qweight.to(self.device).contiguous(),
                requires_grad=False,
                compress_statistics=False
            )
        
        params.quant_state = quant_state
        linear.weight = params
        
        # Set bias if present
        if expert_weights.get('bias') is not None:
            linear.bias = torch.nn.Parameter(expert_weights['bias'].to(self.device))
        
        return linear
    
    def convert_layer(self, layer_idx: int) -> Dict[str, Any]:
        """Convert all experts in a layer to vLLM-compatible format."""
        
        logger.info(f"Converting layer {layer_idx}")
        
        # Create virtual fused tensors with metadata
        w13_tensor, w13_meta = self.create_fused_expert_tensor(layer_idx, "w13")
        w2_tensor, w2_meta = self.create_fused_expert_tensor(layer_idx, "w2")
        
        # Create individual Linear4bit modules for each expert
        expert_modules = {
            'w13': [],  # gate_up projections
            'w2': []    # down projections
        }
        
        for expert_data in w13_meta['experts']:
            expert_idx = expert_data['idx']
            weights = expert_data['weights']
            
            # For gate_up: out=2*intermediate_size, in=hidden_size
            linear = self.create_bnb_linear_for_expert(
                weights,
                out_features=2 * self.structure['intermediate_size'],
                in_features=self.structure['hidden_size']
            )
            expert_modules['w13'].append(linear)
        
        for expert_data in w2_meta['experts']:
            expert_idx = expert_data['idx']
            weights = expert_data['weights']
            
            # For down: out=hidden_size, in=intermediate_size
            linear = self.create_bnb_linear_for_expert(
                weights,
                out_features=self.structure['hidden_size'],
                in_features=self.structure['intermediate_size']
            )
            expert_modules['w2'].append(linear)
        
        return {
            'w13_tensor': w13_tensor,
            'w2_tensor': w2_tensor,
            'w13_meta': w13_meta,
            'w2_meta': w2_meta,
            'expert_modules': expert_modules
        }


class UnslothMoEWrapper(torch.nn.Module):
    """
    Wrapper that makes Unsloth individual experts work with vLLM's FusedMoE interface.
    """
    
    def __init__(self, expert_modules: Dict[str, List[Linear4bit]], 
                 num_experts: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.w13_experts = torch.nn.ModuleList(expert_modules['w13'])
        self.w2_experts = torch.nn.ModuleList(expert_modules['w2'])
    
    def forward(self, hidden_states: torch.Tensor, 
                router_logits: torch.Tensor,
                top_k: int = 4) -> torch.Tensor:
        """
        Forward pass that mimics FusedMoE behavior but uses individual experts.
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get top-k experts for each token
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = torch.nn.functional.softmax(topk_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # Process each token
        for b in range(batch_size):
            for s in range(seq_len):
                token_hidden = hidden_states[b, s]
                token_output = torch.zeros_like(token_hidden)
                
                # Apply selected experts
                for k in range(top_k):
                    expert_idx = topk_ids[b, s, k].item()
                    weight = topk_weights[b, s, k]
                    
                    # Gate and up projection (w13)
                    gate_up_output = self.w13_experts[expert_idx](token_hidden)
                    
                    # Split gate and up
                    gate, up = gate_up_output.chunk(2, dim=-1)
                    
                    # Apply activation (SiLU for GPT-OSS)
                    intermediate = torch.nn.functional.silu(gate) * up
                    
                    # Down projection (w2)
                    expert_output = self.w2_experts[expert_idx](intermediate)
                    
                    # Weight and accumulate
                    token_output += weight * expert_output
                
                output[b, s] = token_output
        
        return output


def test_conversion():
    """Test the Unsloth to vLLM conversion."""
    
    # Find checkpoint files
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/")
    ckpt_pattern = os.path.join(cache_dir, "*/*.safetensors")
    ckpt_files = sorted(glob.glob(ckpt_pattern))
    
    if not ckpt_files:
        logger.error(f"No checkpoint files found in {cache_dir}")
        return False
    
    logger.info(f"Found {len(ckpt_files)} checkpoint files")
    
    # Create converter
    converter = UnslothToVLLMConverter(ckpt_files)
    
    # Test converting layer 0
    layer_data = converter.convert_layer(0)
    
    if layer_data['expert_modules']['w13'] and layer_data['expert_modules']['w2']:
        logger.info(f"✓ Successfully converted layer 0 with {len(layer_data['expert_modules']['w13'])} experts")
        
        # Create wrapper
        wrapper = UnslothMoEWrapper(layer_data['expert_modules'])
        logger.info("✓ Created MoE wrapper")
        
        # Test forward pass with dummy data
        batch_size, seq_len = 1, 10
        hidden_size = converter.structure['hidden_size']
        num_experts = converter.structure['num_experts']
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=converter.device)
        router_logits = torch.randn(batch_size, seq_len, num_experts, device=converter.device)
        
        with torch.no_grad():
            output = wrapper(hidden_states, router_logits)
        
        logger.info(f"✓ Forward pass successful, output shape: {output.shape}")
        return True
    else:
        logger.error("Failed to convert expert modules")
        return False


if __name__ == "__main__":
    success = test_conversion()
    if success:
        print("\n✓ Unsloth to vLLM conversion test passed!")
    else:
        print("\n✗ Conversion test failed")