# Ensure GPT-OSS router uses FP16 to avoid Half/BFloat16 matmul mismatches
try:
    import torch
    from vllm.model_executor.models import gpt_oss as _go
    if hasattr(_go, 'MLPBlock'):
        orig_init = _go.MLPBlock.__init__
        def _patched_init(self, config, layer_idx: int, quant_config, prefix: str = ""):
            orig_init(self, config, layer_idx, quant_config, prefix)
            # After init, make sure router is fp16
            try:
                if hasattr(self, 'router') and isinstance(self.router, torch.nn.Linear):
                    if self.router.weight.dtype != torch.float16:
                        with torch.no_grad():
                            self.router.to(dtype=torch.float16)
            except Exception:
                pass
        _go.MLPBlock.__init__ = _patched_init
except Exception:
    pass
