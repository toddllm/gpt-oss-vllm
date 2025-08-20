#!/usr/bin/env python3
"""
Patch BitsAndBytes loader to allow MoE models on RTX 3090
"""
import shutil

bnb_loader_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/model_loader/bitsandbytes_loader.py"

# Backup
shutil.copy2(bnb_loader_file, f"{bnb_loader_file}.moe_backup")
print(f"Backed up to {bnb_loader_file}.moe_backup")

with open(bnb_loader_file, 'r') as f:
    lines = f.readlines()

# Find and comment out the FusedMoE prequant restriction for RTX 3090
for i, line in enumerate(lines):
    if "Prequant BitsAndBytes models with FusedMoE is not" in line:
        # Add RTX 3090 check before the error
        indent = "                "
        new_lines = [
            f"{indent}# RTX 3090 compatibility: Allow MoE with custom BnB\n",
            f"{indent}from vllm.platforms import current_platform\n",
            f"{indent}try:\n",
            f"{indent}    cap = current_platform.get_device_capability()\n",
            f"{indent}    if not (cap and cap.major == 8 and cap.minor == 6):\n",
            f"{indent}        raise ValueError(\n",
            f"{indent}            \"Prequant BitsAndBytes models with FusedMoE is not \"\n",
            f"{indent}            \"supported yet.\")\n",
            f"{indent}    else:\n",
            f"{indent}        from vllm.logger import init_logger\n",
            f"{indent}        logger = init_logger(__name__)\n",
            f"{indent}        logger.info(\"RTX 3090: Allowing BnB MoE (experimental)\")\n",
            f"{indent}except:\n",
            f"{indent}    raise ValueError(\n",
            f"{indent}        \"Prequant BitsAndBytes models with FusedMoE is not \"\n",
            f"{indent}        \"supported yet.\")\n",
        ]
        # Replace the original lines
        lines[i-1:i+2] = new_lines
        print("✓ Patched FusedMoE prequant check for RTX 3090")
        break

# Write the modified file
with open(bnb_loader_file, 'w') as f:
    f.writelines(lines)

print("\n✓ Successfully patched bitsandbytes_loader.py for MoE support")