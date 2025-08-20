# GPT-OSS:20B on RTX 3090 with vLLM — Consolidated Status

This document consolidates prior status and analysis and tracks progress via issues.

## Current Blockers
- Router BF16 vs FP16 mismatch (matmul crash). See issue #1.
- Engine core initialization failing during Unsloth streaming load. See issue #3.
- NF4 scaling blocksize uncertainty in Unsloth dequant. See issue #2.

## What Works
- FA2 forced on RTX 3090; sinks/sliding window disabled.
- BitsAndBytes MoE enabled; loader skips fusion when streaming conversion is used.
- Server/client tooling and smoke tests added (streaming + concurrency).

## Next Actions
- Verify router FP16 enforcement end-to-end under vLLM entrypoint. (#1)
- Add per-layer post-conversion dtype/shape checks; bail fast with clear error. (#3)
- Add more robust absmax blocksize inference and validation metrics. (#2)

## How to Run
- Start server: see `patches/gpt_oss_server.py` (loads Unsloth model with streaming conversion).
- Monitor: `python patches/check_server_status.py`.
- Client tests: `python patches/quick_client.py [--stream --endpoint chat --concurrency N]`.
- Suite: `python patches/smoke_suite.py`.

## Notes
- If Unsloth prequant remains incompatible long-term, consider unquantized base + on-the-fly BnB as a fallback path.
