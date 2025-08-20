#!/usr/bin/env python3
"""
Debug script to understand shape mismatches in weight loading.
"""

import os
import glob
from safetensors import safe_open
import torch

# Find checkpoint files
cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/")
ckpt_files = sorted(glob.glob(os.path.join(cache_dir, "*/*.safetensors")))

if ckpt_files:
    print(f"Found {len(ckpt_files)} checkpoint files")
    
    # Check shapes of various weights
    with safe_open(ckpt_files[0], framework='pt', device='cpu') as f:
        print("\n=== Weight Shapes ===")
        
        # Check attention weights
        for key in f.keys():
            if 'self_attn' in key and 'weight' in key and not 'absmax' in key:
                tensor = f.get_tensor(key)
                print(f"{key}: {tensor.shape}")
                break
        
        # Check MoE weights  
        print("\n=== MoE Expert Weights ===")
        gate_up_key = "model.layers.0.mlp.experts.gate_up_projs.0.weight"
        if gate_up_key in f.keys():
            tensor = f.get_tensor(gate_up_key)
            print(f"{gate_up_key}: {tensor.shape}")
            
            # Check metadata
            for suffix in ['absmax', 'nested_absmax', 'nested_quant_map', 'quant_map']:
                meta_key = f"{gate_up_key}.{suffix}"
                if meta_key in f.keys():
                    meta = f.get_tensor(meta_key)
                    print(f"  {suffix}: {meta.shape}")
        
        down_key = "model.layers.0.mlp.experts.down_projs.0.weight"
        if down_key in f.keys():
            tensor = f.get_tensor(down_key)
            print(f"{down_key}: {tensor.shape}")
    
    print("\n=== Expected Shapes (vLLM) ===")
    print("w13_weight (gate+up fused): [32, 28800, 2880]")
    print("w2_weight (down): [32, 2880, 14400]")
    
    print("\n=== Actual Unsloth Shapes ===")
    print("gate_up per expert: [8294400, 1] packed 4-bit")
    print("down per expert: [4147200, 1] packed 4-bit")
    
    print("\n=== After Unpacking ===")
    print("gate_up per expert: 8294400 * 2 = 16588800 4-bit values")
    print("Expected params: 28800 * 2880 = 82944000")
    print("Ratio: 82944000 / 16588800 = 5.0x")
    print("\nThis suggests 5x additional compression beyond 4-bit!")
    
else:
    print("No checkpoint files found")