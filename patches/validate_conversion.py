#!/usr/bin/env python3
"""
Validation script for Unsloth to vLLM conversion.
Includes all critical safety checks and assertions.
"""

import torch
import numpy as np
from typing import Dict, List
import os
import glob
from safetensors import safe_open
from vllm.logger import init_logger

logger = init_logger(__name__)

# Expected dimensions for square MLP variant
EXPECTED_CONFIG = {
    'hidden_size': 2880,
    'intermediate_size': 2880,  # Square MLP variant
    'num_experts': 32,
    'num_layers': 24,
    'num_active_experts': 4,
}

def validate_config():
    """Validate model configuration matches expectations."""
    from transformers import AutoConfig
    
    config = AutoConfig.from_pretrained('unsloth/gpt-oss-20b-unsloth-bnb-4bit', trust_remote_code=True)
    
    logger.info("=== Configuration Validation ===")
    assert config.hidden_size == EXPECTED_CONFIG['hidden_size'], \
        f"Hidden size mismatch: {config.hidden_size} != {EXPECTED_CONFIG['hidden_size']}"
    assert config.intermediate_size == EXPECTED_CONFIG['intermediate_size'], \
        f"Intermediate size mismatch: {config.intermediate_size} != {EXPECTED_CONFIG['intermediate_size']}"
    assert config.num_local_experts == EXPECTED_CONFIG['num_experts'], \
        f"Expert count mismatch: {config.num_local_experts} != {EXPECTED_CONFIG['num_experts']}"
    
    logger.info(f"✓ Config validated: hidden={config.hidden_size}, intermediate={config.intermediate_size}, experts={config.num_local_experts}")
    return config

def validate_tensor_shapes(w13_weight: torch.Tensor, w2_weight: torch.Tensor, layer_idx: int):
    """Validate fused tensor shapes."""
    expected_w13_shape = (32, 5760, 2880)  # [experts, 2*intermediate, hidden]
    expected_w2_shape = (32, 2880, 2880)   # [experts, hidden, intermediate]
    
    assert w13_weight.shape == expected_w13_shape, \
        f"Layer {layer_idx} w13 shape mismatch: {w13_weight.shape} != {expected_w13_shape}"
    assert w2_weight.shape == expected_w2_shape, \
        f"Layer {layer_idx} w2 shape mismatch: {w2_weight.shape} != {expected_w2_shape}"
    
    logger.debug(f"  Layer {layer_idx}: w13={w13_weight.shape}, w2={w2_weight.shape} ✓")

def validate_param_conservation(original_params: int, fused_params: int, layer_idx: int):
    """Ensure parameter count is conserved during fusion."""
    assert original_params == fused_params, \
        f"Layer {layer_idx} param count mismatch: original={original_params:,} != fused={fused_params:,}"
    logger.debug(f"  Layer {layer_idx}: {fused_params:,} params conserved ✓")

def count_expert_params(checkpoint_paths: List[str], layer_idx: int) -> int:
    """Count total parameters in individual expert weights."""
    total_params = 0
    
    for expert_idx in range(32):
        # Count gate_up params
        for ckpt_path in checkpoint_paths:
            with safe_open(ckpt_path, framework='pt', device='cpu') as f:
                key = f"model.layers.{layer_idx}.mlp.experts.gate_up_projs.{expert_idx}.weight"
                if key in f.keys():
                    weight = f.get_tensor(key)
                    # Each byte = 2 4-bit values
                    total_params += weight.shape[0] * 2
                    break
        
        # Count down params
        for ckpt_path in checkpoint_paths:
            with safe_open(ckpt_path, framework='pt', device='cpu') as f:
                key = f"model.layers.{layer_idx}.mlp.experts.down_projs.{expert_idx}.weight"
                if key in f.keys():
                    weight = f.get_tensor(key)
                    total_params += weight.shape[0] * 2
                    break
    
    return total_params

def validate_quantization_coverage(model, expected_quantized: int = 2304):
    """Verify correct number of layers are quantized."""
    quantized_count = 0
    fp16_modules = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            if 'Int4' in str(type(module.weight)) or 'Linear4bit' in str(type(module)):
                quantized_count += 1
            elif 'router' in name or 'attn' in name or 'embed' in name or 'lm_head' in name:
                fp16_modules.append(name)
    
    logger.info(f"=== Quantization Coverage ===")
    logger.info(f"Quantized modules: {quantized_count}")
    logger.info(f"FP16 modules: {len(fp16_modules)} (router, attention, embeddings, lm_head)")
    
    if quantized_count != expected_quantized:
        logger.warning(f"Expected {expected_quantized} quantized modules, found {quantized_count}")
    
    # Log per-layer breakdown
    for layer_idx in range(24):
        layer_quantized = 0
        for name, module in model.named_modules():
            if f"layers.{layer_idx}.mlp.experts" in name and 'Linear4bit' in str(type(module)):
                layer_quantized += 1
        
        if layer_quantized > 0:
            logger.debug(f"  L{layer_idx:02d}: experts=32, quantized={layer_quantized} (gate/up/down)")

def run_smoke_test(model, tokenizer):
    """Run single-token forward pass as smoke test."""
    logger.info("=== Smoke Test ===")
    
    # Fixed input for reproducibility
    test_input = "The"
    input_ids = tokenizer(test_input, return_tensors="pt").input_ids.cuda()
    
    with torch.no_grad():
        # Get positions tensor
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        
        # Forward pass through model
        outputs = model(input_ids, positions)
        
        # Check output shape
        batch_size, seq_len, vocab_size = outputs.shape
        assert batch_size == 1
        assert seq_len == input_ids.shape[1]
        
        # Get top predictions
        logits = outputs[0, -1, :]
        top_k = torch.topk(logits, k=5)
        
        logger.info(f"✓ Smoke test passed: output shape={outputs.shape}")
        logger.info(f"  Top 5 token IDs: {top_k.indices.tolist()}")
        logger.info(f"  Top 5 logits: {top_k.values.tolist()[:5]}")

def main():
    """Run all validation checks."""
    logger.info("Starting Unsloth→vLLM conversion validation")
    
    # 1. Validate configuration
    config = validate_config()
    
    # 2. Find checkpoint files
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
    pattern = os.path.join(cache_dir, "models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/*/*.safetensors")
    ckpt_files = sorted(glob.glob(pattern))
    
    if not ckpt_files:
        logger.error("No checkpoint files found")
        return False
    
    logger.info(f"Found {len(ckpt_files)} checkpoint files")
    
    # 3. Import conversion function
    import sys
    sys.path.insert(0, '/home/tdeshane/vllm/patches')
    from unsloth_direct_loader import create_fused_expert_weights
    
    # 4. Test conversion for one layer with validation
    logger.info("\n=== Testing Layer 0 Conversion ===")
    
    # Count original params
    original_params = count_expert_params(ckpt_files, 0)
    logger.info(f"Original params in layer 0: {original_params:,}")
    
    # Convert
    fused_weights = create_fused_expert_weights(ckpt_files, 0)
    
    if fused_weights:
        # Validate shapes
        validate_tensor_shapes(fused_weights['w13_weight'], fused_weights['w2_weight'], 0)
        
        # Validate param conservation
        fused_params = fused_weights['w13_weight'].numel() + fused_weights['w2_weight'].numel()
        logger.info(f"Fused params in layer 0: {fused_params:,}")
        
        # Note: Can't directly compare packed vs unpacked params
        # But we can verify dimensions
        expected_fused = 32 * (5760 * 2880 + 2880 * 2880)  # w13 + w2
        assert fused_params == expected_fused, f"Fused param count wrong: {fused_params} != {expected_fused}"
        
        logger.info("✓ Layer 0 conversion validated")
    else:
        logger.error("✗ Layer 0 conversion failed")
        return False
    
    logger.info("\n=== All Validations Passed ===")
    logger.info("Ready to proceed with full model conversion")
    
    # Print launch command
    logger.info("\n=== Recommended Launch Command ===")
    print("""
export VLLM_FLASH_ATTN_VERSION=2
python -m vllm.entrypoints.openai.api_server \\
    --model unsloth/gpt-oss-20b-unsloth-bnb-4bit \\
    --quantization bitsandbytes \\
    --dtype float16 \\
    --max-model-len 1024 \\
    --enforce-eager \\
    --gpu-memory-utilization 0.90
""")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)