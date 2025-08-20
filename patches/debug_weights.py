#!/usr/bin/env python3
"""
Debug weight loading to understand the mismatch
"""
import torch
from safetensors import safe_open
import glob
import os

cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/")
ckpt_files = sorted(glob.glob(os.path.join(cache_dir, "*", "*.safetensors")))

print(f"Found {len(ckpt_files)} checkpoint files")

# Count weights by category
categories = {
    'embeddings': [],
    'attention': [],
    'moe_experts': [],
    'moe_metadata': [],
    'layernorm': [],
    'other': []
}

for ckpt in ckpt_files:
    with safe_open(ckpt, framework='pt') as f:
        for key in f.keys():
            if 'embed' in key or 'lm_head' in key:
                categories['embeddings'].append(key)
            elif any(x in key for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
                categories['attention'].append(key)
            elif 'experts' in key and not any(x in key for x in ['absmax', 'quant', 'nested']):
                if '.weight' in key and key.endswith('.weight'):
                    categories['moe_experts'].append(key)
                elif '.bias' in key:
                    categories['moe_experts'].append(key)
                else:
                    categories['moe_metadata'].append(key)
            elif 'layernorm' in key or '.norm' in key:
                categories['layernorm'].append(key)
            else:
                categories['other'].append(key)

print("\nWeight categories:")
for cat, keys in categories.items():
    print(f"  {cat}: {len(keys)} weights")
    if cat == 'moe_experts' and keys:
        print(f"    Sample: {keys[0]}")

# Check what vLLM expects
print("\n\nvLLM expects for MoE:")
print("  - experts.w13_weight (gate+up combined)")
print("  - experts.w2_weight (down)")
print("\nUnsloth provides:")
print("  - experts.gate_up_projs.X.weight")
print("  - experts.down_projs.X.weight")
print("\nKey issue: Need to map Unsloth individual expert weights to vLLM's FusedMoE format")