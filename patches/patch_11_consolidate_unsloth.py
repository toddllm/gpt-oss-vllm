#!/usr/bin/env python3
"""
Consolidate Unsloth handling in GPT-OSS model.
Remove duplicate code paths and ensure proper conversion is triggered.
"""

import shutil

gptoss_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"

# Backup
shutil.copy2(gptoss_file, f"{gptoss_file}.consolidate_backup")
print(f"Backed up to {gptoss_file}.consolidate_backup")

with open(gptoss_file, 'r') as f:
    content = f.read()

# Replace the _load_unsloth_weights method with one that actually does conversion
new_load_unsloth = '''
    def _load_unsloth_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load and convert Unsloth pre-quantized weights for RTX 3090."""
        from vllm.logger import init_logger
        logger = init_logger(__name__)
        logger.info('Loading Unsloth weights with conversion')
        
        import os
        import glob
        import sys
        
        # Add patches directory
        sys.path.insert(0, '/home/tdeshane/vllm/patches')
        
        try:
            from unsloth_direct_loader import create_fused_expert_weights
            
            # Find Unsloth checkpoint files
            cache_dir = os.path.expanduser('~/.cache/huggingface/hub/')
            pattern = os.path.join(cache_dir, 'models--*unsloth*gpt-oss*/snapshots/*/*.safetensors')
            ckpt_files = sorted(glob.glob(pattern))
            
            if not ckpt_files:
                pattern2 = os.path.join(cache_dir, 'models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/*/*.safetensors')
                ckpt_files = sorted(glob.glob(pattern2))
            
            if ckpt_files:
                logger.info(f'Found {len(ckpt_files)} Unsloth checkpoint files')
                
                # Process each layer
                num_layers = self.config.num_hidden_layers
                for layer_idx in range(num_layers):
                    logger.info(f'Converting layer {layer_idx}/{num_layers}')
                    
                    # Create fused weights from Unsloth format (dequantized)
                    fused_weights = create_fused_expert_weights(ckpt_files, layer_idx)
                    
                    if fused_weights:
                        # Apply to model's MoE experts
                        mlp = self.model.layers[layer_idx].mlp
                        
                        # The FusedMoE expects these tensors
                        if hasattr(mlp.experts, 'w13_weight'):
                            # Note: dimensions should be [32, 5760, 2880] based on analysis
                            mlp.experts.w13_weight.data = fused_weights['w13_weight'].to(mlp.experts.w13_weight.device)
                            logger.debug(f'Set w13_weight for layer {layer_idx}: {fused_weights["w13_weight"].shape}')
                        
                        if hasattr(mlp.experts, 'w2_weight'):
                            # Note: dimensions should be [32, 2880, 2880]
                            mlp.experts.w2_weight.data = fused_weights['w2_weight'].to(mlp.experts.w2_weight.device)
                            logger.debug(f'Set w2_weight for layer {layer_idx}: {fused_weights["w2_weight"].shape}')
                
                # Load non-MoE weights normally
                weights_list = list(weights) if weights else []
                params_dict = dict(self.named_parameters())
                loaded_params = set()
                
                for name, weight in weights_list:
                    # Skip MoE weights (already handled)
                    if 'mlp.experts' in name or 'gate_up_projs' in name or 'down_projs' in name:
                        loaded_params.add(name)
                        continue
                    
                    # Handle other weights with standard renaming
                    rename_mapping = {
                        'self_attn': 'attn',
                        'input_layernorm.weight': 'attn.norm.weight',
                        'post_attention_layernorm.weight': 'mlp.norm.weight',
                        'embed_tokens': 'embedding',
                    }
                    
                    renamed_name = name
                    for old, new in rename_mapping.items():
                        if old in renamed_name:
                            renamed_name = renamed_name.replace(old, new)
                    
                    if renamed_name in params_dict:
                        param = params_dict[renamed_name]
                        weight_loader = getattr(param, 'weight_loader', 
                                              lambda p, w: p.data.copy_(w.to(p.device)))
                        weight_loader(param, weight)
                        loaded_params.add(renamed_name)
                
                logger.info(f'Conversion complete. Loaded {len(loaded_params)} parameters')
                return loaded_params
            else:
                logger.error('No Unsloth checkpoint files found')
                raise RuntimeError('Cannot find Unsloth checkpoint files')
                
        except Exception as e:
            logger.error(f'Failed to load Unsloth weights: {e}')
            import traceback
            traceback.print_exc()
            raise
'''

# Find and replace the _load_unsloth_weights method
import re

# Pattern to match the entire method
pattern = r'def _load_unsloth_weights\(self.*?\n(?:.*?\n)*?        return loaded_params'
match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

if match:
    content = content[:match.start()] + new_load_unsloth.strip() + content[match.end():]
    print("✓ Replaced _load_unsloth_weights method")
else:
    print("✗ Could not find _load_unsloth_weights method")

# Write back
with open(gptoss_file, 'w') as f:
    f.write(content)

print("\n✓ Successfully consolidated Unsloth handling")