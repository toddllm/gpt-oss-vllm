#!/usr/bin/env python3
"""
Quick test to verify model loads completely without OOM.
Just tests loading, not generation.
"""

import os
import torch
import time
from vllm.logger import init_logger

logger = init_logger(__name__)

def quick_load_test():
    """Test model loading only."""
    
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    
    logger.info("=== Quick Load Test ===")
    logger.info("Testing GPT-OSS model loading...")
    
    start_time = time.time()
    
    try:
        from vllm import LLM
        
        # Initialize with conservative settings
        llm = LLM(
            model="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            quantization="bitsandbytes",
            dtype="float16",
            max_model_len=128,  # Minimal for loading test
            gpu_memory_utilization=0.35,  # Conservative
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
        )
        
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded successfully in {load_time:.1f} seconds")
        
        # Check VRAM usage
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserv = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✓ VRAM: alloc={alloc:.2f}GB, reserved={reserv:.2f}GB, total={total:.2f}GB")
        
        logger.info("✓ MODEL LOADING TEST PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_load_test()
    exit(0 if success else 1)