#!/usr/bin/env python3
# test_vllm_prequantized.py - Use pre-quantized 4-bit model
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
    print("LOADING PRE-QUANTIZED 4-BIT GPT-OSS MODEL")
    print("=" * 50)
    
    # Try the pre-quantized 4-bit model that works with unsloth
    llm = LLM(
        "unsloth/gpt-oss-20b-unsloth-bnb-4bit",  # Pre-quantized model
        max_model_len=512,  # Start small
        gpu_memory_utilization=0.95,
        dtype="float16",
        enforce_eager=True,
        trust_remote_code=True,
    )
    
    print("✓ PRE-QUANTIZED MODEL LOADED SUCCESSFULLY!")
    
    # Try a simple generation
    output = llm.generate("What is 2+2?", max_tokens=10)
    print(f"Generated: {output[0].outputs[0].text}")
    print("✓ GENERATION SUCCESSFUL!")
    
except Exception as e:
    print(f"✗ Failed: {e}")
    print("\n--- Trying alternative approach ---")
    
    # If that fails, try without specifying quantization
    try:
        llm = LLM(
            "unsloth/gpt-oss-20b",  # Let vLLM detect quantization
            load_format="auto",
            max_model_len=512,
            gpu_memory_utilization=0.95,
            dtype="float16",
            enforce_eager=True,
            trust_remote_code=True,
        )
        print("✓ Alternative approach worked!")
        
        output = llm.generate("What is 2+2?", max_tokens=10)
        print(f"Generated: {output[0].outputs[0].text}")
        
    except Exception as e2:
        print(f"✗ Alternative also failed: {e2}")
        traceback.print_exc()