#!/usr/bin/env python3
# test_fa_version.py
import sys
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')

from vllm.attention.utils.fa_utils import get_flash_attn_version
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

fa_version = get_flash_attn_version()
print(f"Selected FA version: {fa_version}")

if fa_version == 2:
    print("✓ FA2 correctly selected for RTX 3090")
elif fa_version == 3:
    print("✗ FA3 incorrectly selected - patch may not be working")
else:
    print(f"✗ Unexpected FA version: {fa_version}")