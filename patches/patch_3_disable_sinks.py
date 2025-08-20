#!/usr/bin/env python3
# patch_3_disable_sinks.py
import shutil

gptoss_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/model_executor/models/gpt_oss.py"

# Backup
shutil.copy2(gptoss_file, f"{gptoss_file}.orig")
print(f"Backed up to {gptoss_file}.orig")

with open(gptoss_file, 'r') as f:
    lines = f.readlines()

# Find the sliding window initialization and add RTX 3090 check
insert_idx = None
for i, line in enumerate(lines):
    if "# Only apply sliding window to every other layer" in line:
        # Found the comment, insert patch after sliding_window assignment
        for j in range(i, min(i+5, len(lines))):
            if "2 == 0 else None)" in lines[j]:
                insert_idx = j + 1
                break
        break

if insert_idx:
    # Add RTX 3090 compatibility check
    patch_lines = [
        "\n",
        "        # RTX 3090 Compatibility: Disable advanced attention features\n",
        "        from vllm.platforms import current_platform\n",
        "        try:\n",
        "            cap = current_platform.get_device_capability()\n",
        "            if cap and cap.major == 8 and cap.minor == 6:\n",
        "                from vllm.logger import init_logger\n",
        "                logger = init_logger(__name__)\n",
        "                logger.warning(\"RTX 3090: Disabling sinks & sliding window (setting to 0/None)\")\n",
        "                sliding_window = None  # Disable sliding window\n",
        "                self.sinks = None  # Disable sinks\n",
        "        except:\n",
        "            pass  # If platform detection fails, continue with defaults\n",
        "\n"
    ]
    
    # Insert the patch
    for j, patch_line in enumerate(patch_lines):
        lines.insert(insert_idx + j, patch_line)
    
    with open(gptoss_file, 'w') as f:
        f.writelines(lines)
    print("✓ CRITICAL PATCH 3: Sinks/sliding disabled for RTX 3090")
else:
    print("✗ Could not find sliding window init - manual patch needed")
    print("Look for: # Only apply sliding window to every other layer")