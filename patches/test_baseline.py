#!/usr/bin/env python3
# test_baseline.py - Verify vLLM works with a simple model
import sys
import os
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')

try:
    from vllm import LLM
    import torch
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    
    # Test with a small model first
    print("\nTesting vLLM with gpt2...")
    llm = LLM("gpt2", max_model_len=512)
    output = llm.generate("Hello world", max_tokens=10)
    print(f"Generated: {output[0].outputs[0].text}")
    print("✓ vLLM baseline test passed")
except Exception as e:
    print(f"✗ Baseline test failed: {e}")
    print("CRITICAL: Fix basic vLLM functionality before proceeding")
    import traceback
    traceback.print_exc()