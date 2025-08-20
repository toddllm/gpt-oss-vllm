# GPT-OSS on vLLM (RTX 3090)

Run `unsloth/gpt-oss-20b-unsloth-bnb-4bit` on an RTX 3090 (24 GB) using vLLM with BitsAndBytes 4-bit quantization, including a streaming per-expert conversion pipeline to avoid OOM.

## Hardware & constraints
- GPU: RTX 3090 (SM 8.6)
- FlashAttention: FA2 required
- VRAM: 24 GB; recommended `--gpu-memory-utilization` 0.40–0.50 to start

## Quick start (persistent API server)
```bash
# 1) Start server in background (first load ~15–20 minutes)
nohup python patches/gpt_oss_server.py \
  --gpu-memory-utilization 0.40 \
  --max-model-len 256 \
  --disable-log-requests \
  > patches/server_log.txt 2>&1 &

# 2) Monitor
python patches/check_server_status.py
# or
tail -f patches/server_log.txt

# 3) Test client
python patches/quick_client.py --prompt "The future of AI is"
```

## One-off smoke test (Python API)
```bash
python patches/smoke_test.py
```

## What this repo adds
- `patches/streaming_unsloth_bnb.py`: CPU dequant + per-expert BnB attach to avoid OOM
- `patches/gpt_oss_server.py`: Server launcher using vLLM OpenAI API
- `patches/check_server_status.py`: Readiness probe against logs and /v1/models
- `patches/quick_client.py`: Simple completions client
- Test utilities: FA2 checks, server connection checks, launch sanity tests

See `patches/GPT-OSS-RTX3090-STATUS.md` for the current integration status and known issues.

## Configuration details
- Force FA2: `export VLLM_FLASH_ATTN_VERSION=2`
- vLLM flags (recommended):
  - `--quantization bitsandbytes`
  - `--load-format bitsandbytes`
  - `--dtype float16`
  - `--gpu-memory-utilization 0.40`
  - `--max-model-len 256`
  - `--enforce-eager`
  - `--trust-remote-code`

## Endpoints
- Models: `GET http://localhost:8000/v1/models`
- Completions: `POST http://localhost:8000/v1/completions`

Example curl:
```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"unsloth/gpt-oss-20b-unsloth-bnb-4bit",
    "prompt":"Hello",
    "max_tokens":16
  }'
```

## Troubleshooting
- If logs show dtype mismatch (Half vs BFloat16), ensure router is FP16 in the model implementation. The provided environment patch sets the router to FP16 to match runtime dtype.
- If OOM during load: lower `--gpu-memory-utilization` and `--max-model-len`.
- If FA3 gets selected, export `VLLM_FLASH_ATTN_VERSION=2`.

## License
Apache-2.0 for vLLM; check upstream model license for `unsloth/gpt-oss-20b-unsloth-bnb-4bit`.
