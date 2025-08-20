#!/usr/bin/env python3
"""
Smoke test for GPT-OSS with vLLM on RTX 3090.
Single token forward pass to verify everything works.
"""

import os
import torch
import time
from vllm.logger import init_logger

logger = init_logger(__name__)

def smoke_test():
    """Run minimal inference test."""
    
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    
    logger.info("=== GPT-OSS Smoke Test ===")
    logger.info("Loading model with vLLM...")
    
    start_time = time.time()
    
    try:
        from vllm import LLM, SamplingParams
        
        # Initialize with conservative settings for KV cache math
        llm = LLM(
            model="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            quantization="bitsandbytes",  # On-the-fly quantization
            dtype="float16",
            max_model_len=256,  # Conservative for first boot
            gpu_memory_utilization=0.40,  # Conservative: 0.40 < 0.43 free ratio
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
        )
        
        load_time = time.time() - start_time
        logger.info(f"✓ Model loaded in {load_time:.1f} seconds")
        
        # Single token test
        test_prompt = "The"
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic
            max_tokens=1,     # Single token only
            top_p=1.0,
        )
        
        logger.info(f"Testing single token generation from: '{test_prompt}'")
        
        start_gen = time.time()
        outputs = llm.generate([test_prompt], sampling_params)
        gen_time = time.time() - start_gen
        
        # Check output
        generated = outputs[0].outputs[0].text
        token_id = outputs[0].outputs[0].token_ids[0] if outputs[0].outputs[0].token_ids else None
        
        logger.info(f"✓ Generated token: '{generated}' (ID: {token_id})")
        logger.info(f"✓ Generation time: {gen_time:.3f} seconds")
        
        # Verify it's reasonable
        assert len(generated) > 0, "No output generated"
        assert token_id is not None, "No token ID returned"
        
        # Multi-token test
        logger.info("\nTesting 10-token generation...")
        sampling_params.max_tokens = 10
        
        start_gen = time.time()
        outputs = llm.generate(["The future of AI is"], sampling_params)
        gen_time = time.time() - start_gen
        
        generated = outputs[0].outputs[0].text
        token_count = len(outputs[0].outputs[0].token_ids)
        
        logger.info(f"✓ Generated: '{generated}'")
        logger.info(f"✓ Tokens: {token_count}, Time: {gen_time:.3f}s")
        logger.info(f"✓ Tokens/sec: {token_count/gen_time:.1f}")
        
        logger.info("\n=== SMOKE TEST PASSED ===")
        logger.info("GPT-OSS is working on RTX 3090 with vLLM!")
        
        return True
        
    except Exception as e:
        logger.error(f"Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = smoke_test()
    exit(0 if success else 1)