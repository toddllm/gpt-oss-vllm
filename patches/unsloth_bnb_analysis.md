# Unsloth→vLLM BnB Weight Format Analysis

## Problem Summary
- **Unsloth 4-bit checkpoint**: Stores already-quantized expert weights (qweight packed nibbles + scales/zeros/g_idx)
- **vLLM BnB-MoE path**: Expects unquantized float weights to wrap into bnb.nn.Linear4bit
- **Result**: Shape mismatch when feeding packed tensors into path expecting floats

## Solution: Path A - Unsloth→BnB Loader Shim
Build adapter to directly construct Int4Params from Unsloth's stored tensors and attach to bnb.nn.Linear4bit modules.

## Key Components Needed

### 1. Weight Mapping
For each MoE block L and expert E, we have 3 Linear projections:
- `w_gate` (w1/gate_proj): [ffn, d_model] or [d_model, ffn]
- `w_up` (w3/up_proj): Same dimensions as gate
- `w_down` (w2/down_proj): Transposed dimensions

### 2. Unsloth Tensor Components
Each expert Linear needs:
- `qweight`: Packed 4-bit weights (uint8)
- `scales`: Quantization scales (float16/float32)
- `zeros`: Zero points (float16/float32)
- `g_idx`: Group indices (optional, int32)
- Metadata: `group_size`, `quant_type` (likely nf4)

### 3. Orientation Handling
- BitsAndBytes stores as (out_features, in_features)
- Unsloth may store pretransposed
- Need layout probe to detect and handle transpose

### 4. Version Compatibility
Handle different bitsandbytes API versions:
- Modern: `create_quant_state()`
- Older: Direct `Int4Params` construction

## Implementation Plan

1. **Create adapter module** (`unsloth_bnb_loader.py`)
   - `attach_unsloth_int4_to_linear()` function
   - Version detection and API compatibility
   - Layout probe for transpose detection

2. **Integrate into GPT-OSS loading**
   - Enumerate expert modules
   - Load Unsloth tensors for each expert
   - Attach using adapter
   - Keep router/attention in FP16

3. **Testing Strategy**
   - Single expert conversion test
   - Single block (all experts) test
   - Full 24-layer model test
   - Start with max_model_len=1024

## Critical Rules
- NEVER unpack qweight (keep packed format)
- Only quantize MoE expert weights
- Router/attention/embeddings stay FP16
- Use strict=False loading, drop expert weights from state_dict

## Environment Settings
- `VLLM_FLASH_ATTN_VERSION=2`
- Sinks/sliding disabled
- `--dtype float16 --enforce-eager`
- Start with `--max-model-len 1024`