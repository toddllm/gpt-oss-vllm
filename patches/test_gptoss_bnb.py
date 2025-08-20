#!/usr/bin/env python3
"""
Test loading GPT-OSS with BitsAndBytes MoE quantization on RTX 3090
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.logger import init_logger

logger = init_logger(__name__)

def test_gptoss_bnb():
    """Test loading GPT-OSS with BnB MoE quantization"""
    
    model_id = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
    
    logger.info(f"Testing GPT-OSS with BnB MoE quantization")
    logger.info(f"Model: {model_id}")
    logger.info(f"GPU: RTX 3090 (Compute Capability 8.6)")
    
    try:
        # Initialize LLM with our patches
        # Note: We dequantize Unsloth to FP16, then let vLLM quantize on-the-fly
        llm = LLM(
            model=model_id,
            quantization="bitsandbytes",  # Quantize on-the-fly with BitsAndBytes
            # Don't use load_format="bitsandbytes" - we're providing FP16 tensors
            tensor_parallel_size=1,
            gpu_memory_utilization=0.95,
            max_model_len=2048,  # Start with shorter context
            trust_remote_code=True,
            enforce_eager=True,  # Disable compilation for testing
        )
        
        logger.info("✓ Model loaded successfully!")
        
        # Test inference
        prompt = "The future of artificial intelligence is"
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
        )
        
        logger.info(f"Testing inference with prompt: '{prompt}'")
        outputs = llm.generate([prompt], sampling_params)
        
        for output in outputs:
            generated_text = output.outputs[0].text
            logger.info(f"Generated text: {generated_text}")
        
        logger.info("✓ Inference test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gptoss_bnb()
    if success:
        print("\n✓ All tests passed! GPT-OSS is working with BnB MoE on RTX 3090")
    else:
        print("\n✗ Tests failed - check logs above for details")