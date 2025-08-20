#!/usr/bin/env python3
# test_vllm_launch.py
import sys
import os
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')

# Set environment variables
os.environ['VLLM_FLASH_ATTN_VERSION'] = '2'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6+PTX'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from vllm import LLM
import traceback
import torch

# Clear GPU memory
torch.cuda.empty_cache()

try:
    print("=" * 50)
    print("ATTEMPTING TO LOAD GPT-OSS:20B WITH PATCHES")
    print("=" * 50)
    
    llm = LLM(
        "openai/gpt-oss-20b",
        quantization="bitsandbytes",  # Use BitsAndBytes instead of MXFP4
        max_model_len=512,  # Start small
        gpu_memory_utilization=0.95,
        dtype="float16",
        enforce_eager=True,
        trust_remote_code=True,
    )
    
    print("✓ MODEL LOADED SUCCESSFULLY!")
    
    # Try a simple generation
    output = llm.generate("What is 2+2?", max_tokens=10)
    print(f"Generated: {output[0].outputs[0].text}")
    print("✓ GENERATION SUCCESSFUL!")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\n--- Full traceback ---")
    traceback.print_exc()
    
    # Analyze error
    error_str = str(e).lower()
    if "mxfp4" in error_str:
        print("\nISSUE: MXFP4 quantization still being used")
        print("ACTION: Need to force BitsAndBytes quantization")
    elif "memory" in error_str or "oom" in error_str:
        print("\nISSUE: Out of memory")
        print("ACTION: Reduce max_model_len or use 4-bit quantization")
    elif "attention" in error_str:
        print("\nISSUE: Attention mechanism issue")
        print("ACTION: Check if all patches applied correctly")
    else:
        print("\nISSUE: Unknown error")
        print("ACTION: Check logs for more details")