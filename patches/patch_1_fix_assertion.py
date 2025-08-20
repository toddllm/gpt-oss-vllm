#!/usr/bin/env python3
# patch_1_fix_assertion.py
import shutil
import re

flash_attn_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/v1/attention/backends/flash_attn.py"

# Backup
shutil.copy2(flash_attn_file, f"{flash_attn_file}.orig")
print(f"Backed up to {flash_attn_file}.orig")

with open(flash_attn_file, 'r') as f:
    lines = f.readlines()

# Find and replace the assertion (lines 416-418)
modified = False
for i in range(len(lines)):
    if i >= 415 and "self.sinks is not None:" in lines[i]:
        # Found the check, now modify the assertion
        if i+1 < len(lines) and "assert self.vllm_flash_attn_version == 3" in lines[i+1]:
            # Replace the assertion with a conditional check
            indent = len(lines[i+1]) - len(lines[i+1].lstrip())
            new_lines = [
                lines[i],  # Keep the if statement
                " " * indent + "# Only require FA3 if sinks are actually being used\n",
                " " * indent + "if self.vllm_flash_attn_version != 3:\n",
                " " * indent + "    raise RuntimeError(\n",
                " " * indent + '        "Sinks require FlashAttention 3, but FA%d is being used. "\n',
                " " * indent + '        "Disable sinks or use a GPU that supports FA3."\n',
                " " * indent + "        % self.vllm_flash_attn_version)\n"
            ]
            # Replace lines[i] through lines[i+2] (the if, assert, and message)
            lines = lines[:i] + new_lines + lines[i+3:]
            modified = True
            break

if modified:
    with open(flash_attn_file, 'w') as f:
        f.writelines(lines)
    print("✓ CRITICAL PATCH 1: Assertion fixed to be conditional")
else:
    print("✗ Could not find assertion - manual patch needed")
    print("Look for: assert self.vllm_flash_attn_version == 3")