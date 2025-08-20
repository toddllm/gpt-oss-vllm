#!/usr/bin/env python3
# patch_2_force_fa2.py
import shutil

fa_utils_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/attention/utils/fa_utils.py"

# Backup
shutil.copy2(fa_utils_file, f"{fa_utils_file}.orig")
print(f"Backed up to {fa_utils_file}.orig")

with open(fa_utils_file, 'r') as f:
    lines = f.readlines()

# Find the insertion point (after line setting fa_version)
insert_idx = None
for i, line in enumerate(lines):
    if "fa_version = envs.VLLM_FLASH_ATTN_VERSION" in line:
        insert_idx = i + 1
        break

if insert_idx:
    # Add RTX 3090 check after environment override
    patch_lines = [
        "\n",
        "        # RTX 3090 compatibility override\n",
        "        if device_capability.major == 8 and device_capability.minor == 6:\n",
        "            logger.warning_once(\"RTX 3090 detected: Forcing FA2 for GPT-OSS compatibility\")\n",
        "            fa_version = 2\n"
    ]
    
    # Insert the patch
    for j, patch_line in enumerate(patch_lines):
        lines.insert(insert_idx + j, patch_line)
    
    with open(fa_utils_file, 'w') as f:
        f.writelines(lines)
    print("✓ CRITICAL PATCH 2: FA2 forced for RTX 3090")
else:
    print("✗ Could not find insertion point - manual patch needed")
    print("Look for: fa_version = envs.VLLM_FLASH_ATTN_VERSION")