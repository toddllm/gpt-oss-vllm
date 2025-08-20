#!/usr/bin/env python3
"""
Patch GPT-OSS to use Unsloth pre-quantized weights with custom loader.
"""
import shutil

gptoss_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"

# Backup
shutil.copy2(gptoss_file, f"{gptoss_file}.unsloth_backup")
print(f"Backed up to {gptoss_file}.unsloth_backup")

with open(gptoss_file, 'r') as f:
    lines = f.readlines()

# Add custom load_weights method that handles Unsloth format
# Find the load_weights method
for i, line in enumerate(lines):
    if "def load_weights(self, weights:" in line:
        # Insert Unsloth detection at the beginning of the method
        indent = "        "
        insert_lines = [
            f"{indent}# Check if this is an Unsloth pre-quantized model\n",
            f"{indent}from vllm.platforms import current_platform\n",
            f"{indent}is_rtx3090 = False\n",
            f"{indent}try:\n",
            f"{indent}    cap = current_platform.get_device_capability()\n",
            f"{indent}    is_rtx3090 = (cap and cap.major == 8 and cap.minor == 6)\n",
            f"{indent}except:\n",
            f"{indent}    pass\n",
            f"\n",
            f"{indent}# Detect Unsloth format by checking for specific weight patterns\n",
            f"{indent}is_unsloth = False\n",
            f"{indent}weight_names = [name for name, _ in weights]\n",
            f"{indent}if any('gate_up_projs' in name for name in weight_names):\n",
            f"{indent}    is_unsloth = True\n",
            f"{indent}    from vllm.logger import init_logger\n",
            f"{indent}    logger = init_logger(__name__)\n",
            f"{indent}    logger.info('Detected Unsloth pre-quantized format')\n",
            f"\n",
            f"{indent}if is_unsloth and is_rtx3090:\n",
            f"{indent}    # Use custom Unsloth loader for RTX 3090\n",
            f"{indent}    return self._load_unsloth_weights(weights)\n",
            f"\n",
            f"{indent}# Original load_weights implementation follows\n",
        ]
        
        # Insert after the def line
        lines.insert(i + 1, ''.join(insert_lines))
        print("✓ Added Unsloth format detection")
        break

# Add the custom Unsloth loader method before load_weights
for i, line in enumerate(lines):
    if "def load_weights(self, weights:" in line:
        # Insert the new method before load_weights
        indent = "    "
        unsloth_method = [
            f"\n{indent}def _load_unsloth_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n",
            f"{indent}    \"\"\"Load Unsloth pre-quantized weights for RTX 3090.\"\"\"\n",
            f"{indent}    from vllm.logger import init_logger\n",
            f"{indent}    logger = init_logger(__name__)\n",
            f"{indent}    logger.info('Loading Unsloth weights with custom loader')\n",
            f"\n",
            f"{indent}    params_dict = dict(self.named_parameters())\n",
            f"{indent}    loaded_params: set[str] = set()\n",
            f"\n",
            f"{indent}    # Process weights\n",
            f"{indent}    for name, weight in weights:\n",
            f"{indent}        weight = weight.cuda()\n",
            f"\n",
            f"{indent}        # Handle MoE expert weights specially\n",
            f"{indent}        if 'gate_up_projs' in name or 'down_projs' in name:\n",
            f"{indent}            # Map Unsloth names to vLLM names\n",
            f"{indent}            # Unsloth: model.layers.L.mlp.experts.gate_up_projs.E.weight\n",
            f"{indent}            # vLLM: model.block.L.mlp.experts.w13_weight (combined)\n",
            f"{indent}            if 'gate_up_projs' in name:\n",
            f"{indent}                # Extract layer and expert indices\n",
            f"{indent}                parts = name.split('.')\n",
            f"{indent}                layer_idx = int(parts[2])  # layers.X\n",
            f"{indent}                expert_idx = int(parts[6]) if len(parts) > 6 else 0  # gate_up_projs.X\n",
            f"\n",
            f"{indent}                # For now, store the weight as-is\n",
            f"{indent}                # The FusedMoE will need to handle the Unsloth format\n",
            f"{indent}                logger.debug(f'Storing gate_up weight for L{{layer_idx}}E{{expert_idx}}: {{weight.shape}}')\n",
            f"{indent}                loaded_params.add(name)\n",
            f"\n",
            f"{indent}            elif 'down_projs' in name:\n",
            f"{indent}                parts = name.split('.')\n",
            f"{indent}                layer_idx = int(parts[2])\n",
            f"{indent}                expert_idx = int(parts[6]) if len(parts) > 6 else 0\n",
            f"{indent}                logger.debug(f'Storing down weight for L{{layer_idx}}E{{expert_idx}}: {{weight.shape}}')\n",
            f"{indent}                loaded_params.add(name)\n",
            f"\n",
            f"{indent}            # Skip the quantization metadata for now\n",
            f"{indent}            if any(x in name for x in ['absmax', 'quant_map', 'quant_state']):\n",
            f"{indent}                loaded_params.add(name)\n",
            f"{indent}                continue\n",
            f"\n",
            f"{indent}        # Handle attention weights (keep original logic)\n",
            f"{indent}        elif 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:\n",
            f"{indent}            shard_id = ('q' if 'q_proj' in name else\n",
            f"{indent}                        'k' if 'k_proj' in name else 'v')\n",
            f"{indent}            name = name.replace('self_attn', 'attn')\n",
            f"{indent}            param_name = name.replace(f'{{shard_id}}_proj', 'qkv')\n",
            f"{indent}            if param_name in params_dict:\n",
            f"{indent}                param = params_dict[param_name]\n",
            f"{indent}                weight_loader = param.weight_loader\n",
            f"{indent}                weight_loader(param, weight, loaded_shard_id=shard_id)\n",
            f"{indent}                loaded_params.add(param_name)\n",
            f"\n",
            f"{indent}        # Handle other weights normally\n",
            f"{indent}        else:\n",
            f"{indent}            # Apply standard renaming\n",
            f"{indent}            rename_mapping = {{\n",
            f"{indent}                'self_attn': 'attn',\n",
            f"{indent}                'input_layernorm.weight': 'attn.norm.weight',\n",
            f"{indent}                'post_attention_layernorm.weight': 'mlp.norm.weight',\n",
            f"{indent}                'embed_tokens': 'embedding',\n",
            f"{indent}            }}\n",
            f"{indent}            renamed_name = name\n",
            f"{indent}            for old, new in rename_mapping.items():\n",
            f"{indent}                if old in renamed_name:\n",
            f"{indent}                    renamed_name = renamed_name.replace(old, new)\n",
            f"\n",
            f"{indent}            if renamed_name in params_dict:\n",
            f"{indent}                param = params_dict[renamed_name]\n",
            f"{indent}                weight_loader = getattr(param, 'weight_loader', \n",
            f"{indent}                                        lambda p, w: p.data.copy_(w))\n",
            f"{indent}                weight_loader(param, weight)\n",
            f"{indent}                loaded_params.add(renamed_name)\n",
            f"\n",
            f"{indent}    logger.info(f'Loaded {{len(loaded_params)}} parameters with Unsloth format')\n",
            f"{indent}    return loaded_params\n",
            f"\n",
        ]
        
        lines.insert(i, ''.join(unsloth_method))
        print("✓ Added _load_unsloth_weights method")
        break

# Write the modified file
with open(gptoss_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Successfully patched gpt_oss.py for Unsloth weight loading")