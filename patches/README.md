# GPT-OSS:20B on RTX 3090 with vLLM

This setup enables running GPT-OSS:20B (20.9B parameter MoE model) on RTX 3090 GPU using vLLM with BitsAndBytes quantization.

## Quick Start

### 1. Start the Server (one-time setup)
```bash
# This loads the model and keeps it in memory (~15-20 minutes initial load)
python gpt_oss_server.py

# Optional: adjust memory usage for your RTX 3090
python gpt_oss_server.py --gpu-memory-utilization 0.45 --max-model-len 256
```

### 2. Test with Quick Client (instant responses)
```bash
# Run test prompts
python quick_client.py

# Interactive mode
python quick_client.py --interactive

# Single prompt
python quick_client.py --prompt "The future of AI is"
```

## Server Options

```bash
python gpt_oss_server.py --help

# Common configurations:
--gpu-memory-utilization 0.35-0.60  # RTX 3090 memory usage
--max-model-len 256-1024            # Context length vs memory tradeoff  
--port 8000                         # Server port
--disable-log-requests              # Cleaner output
```

## Client Options

```bash
python quick_client.py --help

# Examples:
--interactive                       # Chat mode
--prompt "Your text here"          # Single generation
--max-tokens 100                   # Longer responses
--temperature 0.8                  # More creative
```

## Technical Details

### Model Architecture
- **GPT-OSS:20B**: Sparse MoE transformer with 32 experts, 4 active per token
- **Total params**: ~20.9B (2.6B active per token)
- **Quantization**: BitsAndBytes 4-bit NF4 
- **Source**: Unsloth double-quantized format

### RTX 3090 Optimizations
- ✅ FlashAttention 2 (FA3 not supported on CC 8.6)
- ✅ Streaming per-expert weight conversion (prevents OOM)
- ✅ BitsAndBytes MoE quantization 
- ✅ Conservative memory allocation

### Memory Usage
- **Model loading**: ~12-15GB VRAM
- **Inference**: +2-6GB for KV cache (depends on context length)
- **Total**: Fits comfortably in 24GB RTX 3090

## Files

- `gpt_oss_server.py` - Persistent server with OpenAI-compatible API
- `quick_client.py` - Fast client for testing
- `client_test.py` - Original direct vLLM test (slower)  
- `streaming_unsloth_bnb.py` - Weight conversion pipeline
- `gpt_oss.py` - Modified vLLM model implementation

## Troubleshooting

### Server won't start
- Check CUDA_VISIBLE_DEVICES=0
- Reduce --gpu-memory-utilization to 0.35
- Ensure no other processes using GPU

### Client connection failed  
- Make sure server is running: `python gpt_oss_server.py`
- Check server logs for loading progress
- Verify port (default 8000)

### Out of memory
- Lower --gpu-memory-utilization (try 0.35-0.40)
- Reduce --max-model-len (try 256 or 128)
- Close other GPU applications

## Performance

- **Loading time**: 15-20 minutes (one-time per server start)
- **Generation speed**: ~5-15 tokens/sec (depends on prompt complexity)
- **Memory efficient**: Streaming conversion prevents OOM
- **Persistent**: Server keeps model loaded for instant responses