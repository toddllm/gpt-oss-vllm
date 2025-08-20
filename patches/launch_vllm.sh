#!/bin/bash
# Launch script for vLLM with GPT-OSS on RTX 3090
# Uses correct flags for FP16 tensors with on-the-fly BnB quantization

set -e

echo "=== vLLM Launch Script for GPT-OSS on RTX 3090 ==="
echo "Configuration:"
echo "  - Model: unsloth/gpt-oss-20b-unsloth-bnb-4bit"
echo "  - Quantization: BitsAndBytes (on-the-fly)"
echo "  - FlashAttention: v2 (RTX 3090 compatible)"
echo "  - MLP: Square variant (2880x2880)"
echo ""

# Force FA2 for RTX 3090
export VLLM_FLASH_ATTN_VERSION=2

# Disable any MXFP4 attempts
export VLLM_NO_MXFP4=1

# Launch vLLM API server
python -m vllm.entrypoints.openai.api_server \
    --model unsloth/gpt-oss-20b-unsloth-bnb-4bit \
    --quantization bitsandbytes \
    --dtype float16 \
    --max-model-len 1024 \
    --enforce-eager \
    --gpu-memory-utilization 0.90 \
    --host 0.0.0.0 \
    --port 8000 \
    --disable-log-stats \
    --trust-remote-code

# Note: We do NOT use --load-format bitsandbytes
# because we're providing FP16 tensors after dequantization