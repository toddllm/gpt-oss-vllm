#!/usr/bin/env python3
"""
Integrate Unsloth weight loading into GPT-OSS model.
This patch modifies the model to detect and handle Unsloth checkpoints.
"""

import shutil

gptoss_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"

# Backup
shutil.copy2(gptoss_file, f"{gptoss_file}.integrate_backup")
print(f"Backed up to {gptoss_file}.integrate_backup")

with open(gptoss_file, 'r') as f:
    lines = f.readlines()

# Find the load_weights method
for i, line in enumerate(lines):
    if "def load_weights(self," in line:
        # Find the end of the method signature
        method_start = i
        brace_count = 0
        for j in range(i, len(lines)):
            if "{" in lines[j]:
                brace_count += lines[j].count("{")
            if "}" in lines[j]:
                brace_count += lines[j].count("}")
            if ":" in lines[j] and brace_count == 0:
                # Found the colon at end of method signature
                insert_idx = j + 1
                break
        
        # Add Unsloth detection and handling
        indent = "        "
        insert_lines = [
            f"\n{indent}# Check if this is an Unsloth model\n",
            f"{indent}model_path = str(getattr(self, 'model', ''))\n",
            f"{indent}if 'unsloth' in model_path.lower() or 'unsloth' in str(params).lower():\n",
            f"{indent}    from vllm.logger import init_logger\n",
            f"{indent}    _unsloth_logger = init_logger(__name__)\n",
            f"{indent}    _unsloth_logger.info('Detected Unsloth model, using special weight conversion')\n",
            f"{indent}    \n",
            f"{indent}    # Special handling for Unsloth weights\n",
            f"{indent}    # These weights are double-quantized and need special processing\n",
            f"{indent}    import os\n",
            f"{indent}    import glob\n",
            f"{indent}    import sys\n",
            f"{indent}    \n",
            f"{indent}    # Add patches directory to path\n",
            f"{indent}    sys.path.insert(0, '/home/tdeshane/vllm/patches')\n",
            f"{indent}    \n",
            f"{indent}    try:\n",
            f"{indent}        from unsloth_direct_loader import create_fused_expert_weights\n",
            f"{indent}        \n",
            f"{indent}        # Find Unsloth checkpoint files\n",
            f"{indent}        cache_dir = os.path.expanduser('~/.cache/huggingface/hub/')\n",
            f"{indent}        # Look for any unsloth gpt-oss model\n",
            f"{indent}        pattern = os.path.join(cache_dir, 'models--*unsloth*gpt-oss*/snapshots/*/*.safetensors')\n",
            f"{indent}        ckpt_files = sorted(glob.glob(pattern))\n",
            f"{indent}        \n",
            f"{indent}        if not ckpt_files:\n",
            f"{indent}            # Try more specific path\n",
            f"{indent}            pattern2 = os.path.join(cache_dir, 'models--unsloth--gpt-oss-20b-unsloth-bnb-4bit/snapshots/*/*.safetensors')\n",
            f"{indent}            ckpt_files = sorted(glob.glob(pattern2))\n",
            f"{indent}        \n",
            f"{indent}        if ckpt_files:\n",
            f"{indent}            _unsloth_logger.info(f'Found {{len(ckpt_files)}} Unsloth checkpoint files')\n",
            f"{indent}            \n",
            f"{indent}            # Process MoE weights layer by layer\n",
            f"{indent}            for name, loaded_weight in params.items():\n",
            f"{indent}                if 'mlp.experts.w13_weight' in name:\n",
            f"{indent}                    # Extract layer index\n",
            f"{indent}                    layer_idx = int(name.split('layers.')[1].split('.')[0])\n",
            f"{indent}                    _unsloth_logger.info(f'Converting Unsloth weights for layer {{layer_idx}}')\n",
            f"{indent}                    \n",
            f"{indent}                    # Create fused weights from Unsloth format\n",
            f"{indent}                    fused_weights = create_fused_expert_weights(ckpt_files, layer_idx)\n",
            f"{indent}                    \n",
            f"{indent}                    if fused_weights:\n",
            f"{indent}                        # Find the parameter to update\n",
            f"{indent}                        for name2, param in params.items():\n",
            f"{indent}                            if f'layers.{{layer_idx}}.mlp.experts.w13_weight' in name2:\n",
            f"{indent}                                param.data = fused_weights['w13_weight'].to(param.device)\n",
            f"{indent}                                _unsloth_logger.debug(f'Updated w13_weight for layer {{layer_idx}}')\n",
            f"{indent}                            elif f'layers.{{layer_idx}}.mlp.experts.w2_weight' in name2:\n",
            f"{indent}                                param.data = fused_weights['w2_weight'].to(param.device)\n",
            f"{indent}                                _unsloth_logger.debug(f'Updated w2_weight for layer {{layer_idx}}')\n",
            f"{indent}            \n",
            f"{indent}            _unsloth_logger.info('Unsloth weight conversion completed')\n",
            f"{indent}            # Continue with regular weight loading for non-MoE weights\n",
            f"{indent}        else:\n",
            f"{indent}            _unsloth_logger.warning('No Unsloth checkpoint files found')\n",
            f"{indent}    except Exception as e:\n",
            f"{indent}        _unsloth_logger.error(f'Failed to load Unsloth weights: {{e}}')\n",
            f"{indent}        import traceback\n",
            f"{indent}        traceback.print_exc()\n",
            f"\n",
        ]
        
        lines[insert_idx:insert_idx] = insert_lines
        print(f"✓ Added Unsloth weight loading to load_weights method")
        break

# Write the modified file
with open(gptoss_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Successfully integrated Unsloth weight loading into GPT-OSS model")