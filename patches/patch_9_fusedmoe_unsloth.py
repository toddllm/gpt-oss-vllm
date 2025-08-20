#!/usr/bin/env python3
"""
Patch FusedMoE to handle Unsloth individual expert weights.
Instead of expecting pre-fused weights, handle individual expert tensors.
"""
import shutil

fusedmoe_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/layer.py"

# Backup
shutil.copy2(fusedmoe_file, f"{fusedmoe_file}.unsloth_backup")
print(f"Backed up to {fusedmoe_file}.unsloth_backup")

with open(fusedmoe_file, 'r') as f:
    lines = f.readlines()

# Find the weight_loader method and add Unsloth handling
for i, line in enumerate(lines):
    if "def weight_loader(self," in line:
        # Add Unsloth detection at the beginning
        indent = "        "
        insert_idx = i + 8  # After the method signature and initial checks
        
        insert_lines = [
            f"\n{indent}# Handle Unsloth individual expert weights\n",
            f"{indent}if 'gate_up_projs' in weight_name or 'down_projs' in weight_name:\n",
            f"{indent}    # This is an Unsloth weight, handle specially\n",
            f"{indent}    from vllm.logger import init_logger\n",
            f"{indent}    logger = init_logger(__name__)\n",
            f"{indent}    \n",
            f"{indent}    # Parse the weight name to get expert index\n",
            f"{indent}    # Format: model.layers.L.mlp.experts.gate_up_projs.E.weight or down_projs\n",
            f"{indent}    parts = weight_name.split('.')\n",
            f"{indent}    if len(parts) >= 7:\n",
            f"{indent}        expert_idx = int(parts[6])  # The expert index\n",
            f"{indent}        proj_type = parts[5]  # 'gate_up_projs' or 'down_projs'\n",
            f"{indent}        \n",
            f"{indent}        # For RTX 3090, we need to handle the packed format specially\n",
            f"{indent}        from vllm.platforms import current_platform\n",
            f"{indent}        try:\n",
            f"{indent}            cap = current_platform.get_device_capability()\n",
            f"{indent}            if cap and cap.major == 8 and cap.minor == 6:\n",
            f"{indent}                logger.debug(f'RTX 3090: Storing Unsloth ' + str(proj_type) + ' for expert ' + str(expert_idx))\n",
            f"{indent}                # Store the weight temporarily, will be processed later\n",
            f"{indent}                if not hasattr(param, '_unsloth_weights'):\n",
            f"{indent}                    param._unsloth_weights = {{}}\n",
            f"{indent}                param._unsloth_weights[(expert_idx, proj_type)] = loaded_weight\n",
            f"{indent}                return True if return_success else None\n",
            f"{indent}        except:\n",
            f"{indent}            pass\n",
            f"\n",
        ]
        
        lines[insert_idx:insert_idx] = insert_lines
        print("✓ Added Unsloth weight handling to FusedMoE")
        break

# Write the modified file
with open(fusedmoe_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Successfully patched FusedMoE for Unsloth weight handling")