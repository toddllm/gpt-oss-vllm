#!/usr/bin/env python3
"""
Add BitsAndBytes MoE support to GPT-OSS model for RTX 3090
"""
import shutil

gptoss_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"

# Backup
shutil.copy2(gptoss_file, f"{gptoss_file}.bnb_backup")
print(f"Backed up to {gptoss_file}.bnb_backup")

with open(gptoss_file, 'r') as f:
    lines = f.readlines()

# 1. Add import for BnB MoE config at the top
import_idx = None
for i, line in enumerate(lines):
    if "from vllm.model_executor.layers.quantization import QuantizationConfig" in line:
        import_idx = i + 1
        break

if import_idx:
    lines.insert(import_idx, "from vllm.model_executor.layers.quantization.bnb_moe_4bit import BitsAndBytesMoE4bitConfig, is_moe_weight\n")
    print("✓ Added BnB MoE import")

# 2. Add packed_modules_mapping to GptOssForCausalLM class
class_start = None
for i, line in enumerate(lines):
    if "class GptOssForCausalLM" in line:
        class_start = i
        break

if class_start:
    # Find __init__ method
    for i in range(class_start, min(class_start + 50, len(lines))):
        if "def __init__" in lines[i]:
            # Find super().__init__() call and add after it
            for j in range(i, min(i + 20, len(lines))):
                if "super().__init__()" in lines[j]:
                    # Add packed_modules_mapping after super().__init__()
                    indent = "        "
                    insert_lines = [
                        f"{indent}# BitsAndBytes MoE support for RTX 3090\n",
                        f"{indent}from vllm.platforms import current_platform\n",
                        f"{indent}self.use_bnb_moe = False\n",
                        f"{indent}try:\n",
                        f"{indent}    cap = current_platform.get_device_capability()\n",
                        f"{indent}    if cap and cap.major == 8 and cap.minor == 6:\n",
                        f"{indent}        from vllm.logger import init_logger\n",
                        f"{indent}        logger = init_logger(__name__)\n",
                        f"{indent}        logger.info('RTX 3090: Enabling BitsAndBytes MoE quantization')\n",
                        f"{indent}        self.use_bnb_moe = True\n",
                        f"{indent}        # Define packed modules mapping for MoE experts only\n",
                        f"{indent}        self.packed_modules_mapping = {{\n",
                        f"{indent}            'experts': ['w13_weight', 'w2_weight'],  # MoE expert projections\n",
                        f"{indent}        }}\n",
                        f"{indent}except:\n",
                        f"{indent}    pass\n",
                        "\n"
                    ]
                    for k, new_line in enumerate(insert_lines):
                        lines.insert(j + k + 1, new_line)
                    print("✓ Added packed_modules_mapping to GptOssForCausalLM")
                    break
            break

# 3. Modify MLPBlock to use BnB for MoE experts on RTX 3090
mlp_block_idx = None
for i, line in enumerate(lines):
    if "class MLPBlock" in line:
        mlp_block_idx = i
        break

if mlp_block_idx:
    # Find the FusedMoE initialization
    for i in range(mlp_block_idx, min(mlp_block_idx + 50, len(lines))):
        if "self.experts = FusedMoE(" in lines[i]:
            # Insert RTX 3090 check before FusedMoE
            indent = "        "
            insert_lines = [
                f"{indent}# Check for RTX 3090 and use BnB quantization for MoE\n",
                f"{indent}from vllm.platforms import current_platform\n",
                f"{indent}moe_quant_config = quant_config\n",
                f"{indent}try:\n",
                f"{indent}    cap = current_platform.get_device_capability()\n",
                f"{indent}    if cap and cap.major == 8 and cap.minor == 6:\n",
                f"{indent}        # Override with BnB MoE quantization for RTX 3090\n",
                f"{indent}        from vllm.model_executor.layers.quantization.bnb_moe_4bit import BitsAndBytesMoE4bitConfig\n",
                f"{indent}        from vllm.logger import init_logger\n",
                f"{indent}        logger = init_logger(__name__)\n",
                f"{indent}        logger.info(f'Layer {{layer_idx}}: Using BnB 4-bit for MoE experts')\n",
                f"{indent}        moe_quant_config = BitsAndBytesMoE4bitConfig()\n",
                f"{indent}except:\n",
                f"{indent}    pass\n",
                "\n"
            ]
            for j, new_line in enumerate(insert_lines):
                lines.insert(i + j, new_line)
            
            # Update the FusedMoE line to use moe_quant_config
            lines[i + len(insert_lines)] = lines[i + len(insert_lines)].replace(
                "quant_config=quant_config,",
                "quant_config=moe_quant_config,"
            )
            print("✓ Modified MLPBlock to use BnB MoE on RTX 3090")
            break

# Write the modified file
with open(gptoss_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Successfully patched gpt_oss.py for BnB MoE support")