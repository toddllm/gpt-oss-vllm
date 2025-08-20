#!/usr/bin/env python3
# test_gptoss_fail.py - Document the current failure
import sys
import os
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

from vllm import LLM
import traceback

try:
    print("Attempting to load GPT-OSS:20B...")
    llm = LLM("openai/gpt-oss-20b", max_model_len=512, enforce_eager=True)
    print("✓ Unexpected success - model loaded!")
except Exception as e:
    print(f"✗ Expected failure captured:")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\n--- Full traceback ---")
    traceback.print_exc()
    
    # Save error for reference
    with open("baseline_error.txt", "w") as f:
        f.write(f"Error type: {type(e).__name__}\n")
        f.write(f"Error message: {str(e)}\n\n")
        f.write(traceback.format_exc())