# GPT-OSS:20B Quick Usage Guide

## 🚀 Quick Start

### 1. Start Server (Terminal 1)
```bash
cd /home/tdeshane/vllm/patches
python gpt_oss_server.py
```

**Wait for**: `INFO: Application startup complete.` (~15-20 minutes)

### 2. Test Connection (Terminal 2)
```bash
python test_client_connection.py
```

### 3. Run Client Tests
```bash
# Test prompts
python quick_client.py

# Interactive chat
python quick_client.py --interactive

# Single prompt
python quick_client.py --prompt "The future of AI is"
```

## 📊 Server Options

```bash
# Conservative for RTX 3090 (recommended)
python gpt_oss_server.py --gpu-memory-utilization 0.35 --max-model-len 256

# Balanced performance
python gpt_oss_server.py --gpu-memory-utilization 0.45 --max-model-len 512

# Maximum utilization (use carefully)
python gpt_oss_server.py --gpu-memory-utilization 0.60 --max-model-len 1024
```

## 🔧 Troubleshooting

### Server won't start
- Check GPU is available: `nvidia-smi`
- Close other GPU processes
- Try lower memory: `--gpu-memory-utilization 0.30`

### Client can't connect
- Check server logs for "Application startup complete"
- Test connection: `python test_client_connection.py`
- Verify port: default is 8000

### Out of memory
- Reduce `--gpu-memory-utilization` to 0.30-0.35
- Reduce `--max-model-len` to 128-256
- Check VRAM usage: `nvidia-smi`

## 📝 Example Session

**Terminal 1 (Server):**
```bash
python gpt_oss_server.py --gpu-memory-utilization 0.40
# Wait 15-20 minutes for "Application startup complete"
```

**Terminal 2 (Client):**
```bash
python quick_client.py --interactive
📝 Enter prompt: Write a short story about AI
✅ Generated: In the year 2045, artificial intelligence had become...
⚡ Speed: 8.3 tokens/sec (42 tokens in 5.1s)
```

## 🎯 Performance Expectations

- **Loading**: 15-20 minutes (one-time per server restart)
- **Generation**: 5-15 tokens/sec (varies by prompt complexity)
- **Memory**: ~12-18GB VRAM (fits RTX 3090's 24GB)
- **Latency**: <100ms after model loads

## 📁 Key Files

- `gpt_oss_server.py` - Persistent server wrapper
- `quick_client.py` - Fast client for testing
- `test_client_connection.py` - Connection diagnostics
- `streaming_unsloth_bnb.py` - Weight conversion engine
- `gpt_oss.py` - Modified vLLM model (in vllm package)