# GPT-OSS MoE Architecture Documentation

## Model Architecture
- **Type**: Sparse Mixture of Experts (MoE) Transformer
- **Total Parameters**: 20.9B
- **Active Parameters per Token**: ~3.6B
- **Experts**: 32 total
- **Experts per Token**: Top-4 routing
- **Quantization Target**: MoE weights (the bulk of parameters)

## Weight Distribution
1. **MoE Expert Weights** (majority of params - needs 4-bit):
   - Gate projection (w_gate/w13)
   - Up projection (w_up/w13)  
   - Down projection (w_down/w2)
   
2. **Router Weights** (small - keep FP16):
   - Router linear layer per MoE block
   
3. **Attention Weights** (small - keep FP16):
   - QKV projections
   - Output projection
   
4. **Embeddings** (small - keep FP16):
   - Token embeddings
   - Position embeddings

## vLLM Implementation Details

### Current Structure in gpt_oss.py:
- `MLPBlock`: Contains the MoE implementation
  - `self.router`: Linear layer for expert selection (FP16)
  - `self.experts`: FusedMoE with expert weights (needs BnB 4-bit)
  
### Weight Names in vLLM:
- Expert weights: `model.layers.{layer_idx}.mlp.experts.{param_name}`
  - w13_weight (gate_up combined)
  - w2_weight (down)
  - w13_weight_scale, w2_weight_scale (for quantization)

## Quantization Strategy for RTX 3090
1. **MoE Experts**: BitsAndBytes 4-bit NF4 quantization
2. **Everything Else**: Keep FP16 (router, attention, embeddings)
3. **Reason**: MoE weights dominate memory usage; quantizing them gives 90% of benefit