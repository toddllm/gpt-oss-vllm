# GPT-OSS:20B on RTX 3090 - Implementation Status Report

## Project Overview
**Goal**: Enable GPT-OSS:20B (20.9B parameter Sparse MoE model) to run on NVIDIA RTX 3090 (24GB VRAM) using vLLM with BitsAndBytes 4-bit quantization

**Model**: `unsloth/gpt-oss-20b-unsloth-bnb-4bit` - Pre-quantized Unsloth format with double-quantized NF4 weights

**Hardware Constraint**: RTX 3090 with compute capability 8.6 (requires FlashAttention 2, not 3)

## Technical Architecture

### Model Specifications
- **Total Parameters**: 20.9B
- **Active Parameters**: 2.6B per token
- **Architecture**: Sparse Mixture of Experts (MoE)
- **Expert Configuration**: 32 experts total, 4 active per token
- **Layers**: 24 transformer blocks
- **Hidden Size**: 2880
- **Attention Heads**: 30
- **Key-Value Heads**: 5
- **Intermediate Size**: 2880 (square MLP variant, not standard 14400)

### Weight Format Details
- **Source Format**: Unsloth double-quantized NF4
  - Primary quantization: NF4 4-bit
  - Secondary quantization: absmax values also quantized
  - Nested structure: `absmax` and `nested_absmax` tensors
- **Target Format**: BitsAndBytes 4-bit for vLLM compatibility
- **Expert Weights**: 
  - Gate projection (w1): [2880, 2880]
  - Up projection (w3): [2880, 2880]  
  - Down projection (w2): [2880, 2880]
  - Fused format: w13 (gate+up) and w2 (down)

## Implementation Approach

### 1. Streaming Per-Expert Conversion Pipeline
**Problem**: Converting all 32 experts at once causes OOM on 24GB VRAM

**Solution**: Process one expert at a time per layer
- Dequantize Unsloth weights to FP16 on CPU
- Transfer single expert to GPU
- Apply BitsAndBytes quantization
- Attach to model immediately
- Free intermediate tensors

**Code Location**: `/home/tdeshane/vllm/patches/streaming_unsloth_bnb.py`

### 2. Custom Weight Loading Integration
**Problem**: vLLM doesn't support Unsloth format natively

**Solution**: Custom `_load_unsloth_weights` method in GPT-OSS model
- Detects Unsloth checkpoint files
- Triggers streaming conversion per layer
- Handles non-MoE weights separately
- Implements QKV fusion for attention layers

**Code Location**: `/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py`

### 3. BitsAndBytes Loader Modifications
**Problem**: BnB loader expects individual expert weights but streaming creates attached modules

**Solution**: Bypass MoE fusion when streaming conversion is used
- Added `streaming_conversion_used` flag to model
- Modified `_fuse_moe_quant_states` to skip fusion when flag is set
- Prevents KeyError for missing individual expert weights

**Code Location**: `/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/model_loader/bitsandbytes_loader.py`

## Current Status: CRITICAL FAILURE

### Latest Test Results (2025-08-19 19:36:45)

**Failure Point**: Model inference during KV cache profiling

**Error**: 
```
RuntimeError: mat1 and mat2 must have the same dtype, but got Half and BFloat16
```

**Location**: Router linear layer forward pass in MLP block

### Streaming Conversion Progress
✅ **Successful Elements**:
- All 24 layers converted successfully (layers 0-23)
- Streaming conversion completed without OOM
- 32 experts × 24 layers = 768 expert modules processed
- Shape mismatches handled (non-fatal warnings)
- BnB MoE fusion successfully bypassed

❌ **Failure Analysis**:
The error occurs AFTER successful weight loading during the model's first forward pass:
1. Streaming conversion completes all layers
2. Non-MoE weights loaded (attention, norms, embedding)
3. Model initialization succeeds
4. Failure during KV cache profiling when running dummy forward pass
5. Router receives Half (FP16) tensor but has BFloat16 weights

### Root Cause
**Dtype Mismatch**: The router linear layer weights are in BFloat16 but receive FP16 inputs
- Model configured with `dtype=torch.float16`
- Router weights somehow end up as BFloat16
- Likely issue in weight loading or dtype conversion

## Critical Issues Remaining

### 1. Dtype Inconsistency
- Router weights are BFloat16 instead of FP16
- Need to ensure all weights match model dtype
- May require explicit dtype conversion during loading

### 2. Memory Constraints
- Successfully avoided OOM during conversion
- Using 40% GPU memory utilization
- Max model length limited to 256 tokens

### 3. Performance Considerations
- Streaming conversion takes ~15-20 minutes
- Each layer processes 32 experts sequentially
- Shape mismatch warnings (non-fatal but concerning)

## Files Modified

### Core Implementation Files
1. `/home/tdeshane/vllm/patches/streaming_unsloth_bnb.py`
   - Streaming conversion pipeline
   - Unsloth dequantization logic
   - BnB attachment functions

2. `/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py`
   - Custom Unsloth weight loading
   - QKV fusion implementation
   - Streaming conversion flag

3. `/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/model_loader/bitsandbytes_loader.py`
   - MoE fusion bypass logic
   - RTX 3090 compatibility checks

### Support Files
- `/home/tdeshane/vllm/patches/gpt_oss_server.py` - Persistent server wrapper
- `/home/tdeshane/vllm/patches/quick_client.py` - Test client

## Next Steps Required

### Immediate Fix Needed
1. **Fix dtype mismatch in router weights**
   - Ensure router linear layer uses FP16
   - Check weight loading dtype conversion
   - May need explicit `.half()` conversion

2. **Verify all model components have consistent dtype**
   - Router weights
   - Expert weights  
   - Attention weights
   - Layer norms

### Testing Protocol
1. Fix dtype issue
2. Run server with streaming conversion
3. Verify model loads without errors
4. Test simple text generation
5. Monitor memory usage during inference

## Environment Configuration
```bash
# RTX 3090 Compatibility
export CUDA_VISIBLE_DEVICES=0
export VLLM_FLASH_ATTN_VERSION=2

# Server Launch Parameters
--gpu-memory-utilization 0.40
--max-model-len 256
--quantization bitsandbytes
--load-format bitsandbytes
--dtype float16
--enforce-eager
--trust-remote-code
```

## Summary
The implementation has successfully solved the primary challenge of loading a 20.9B parameter model on 24GB VRAM through streaming per-expert conversion. The BitsAndBytes integration and MoE fusion bypass work correctly. However, a critical dtype mismatch between router weights (BFloat16) and model tensors (Half/FP16) prevents the model from running inference. This appears to be the final blocker - once router weights are correctly set to FP16, the model should be functional on RTX 3090.

## Critical Finding
The dtype error suggests that despite all the complex weight conversion working, there's a simple but critical oversight in ensuring dtype consistency across all model components. The router linear layers are ending up with BFloat16 weights when everything else uses FP16.