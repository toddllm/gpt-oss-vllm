#!/usr/bin/env python3
# test_no_assertion.py
import sys
import os
sys.path.insert(0, '/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages')
os.environ['VLLM_FLASH_ATTN_VERSION'] = '2'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'

try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    print("✓ FlashAttention backend imported without assertion error")
    
    # Try to instantiate with sinks=None (what gpt_oss will do on RTX 3090)
    impl = FlashAttentionImpl(
        num_heads=32,
        head_size=64,
        scale=1.0,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        sinks=None  # This is what RTX 3090 will pass
    )
    print("✓ FlashAttentionImpl instantiated with sinks=None")
    
except AssertionError as e:
    if "Sinks are only supported" in str(e):
        print("✗ CRITICAL: Assertion still firing - patch 1 failed")
        print("Check vllm/v1/attention/backends/flash_attn.py")
    else:
        print(f"✗ Different assertion: {e}")
except Exception as e:
    print(f"Other error (may be OK): {e}")