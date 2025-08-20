#!/usr/bin/env python3
# patch_4_override_quant.py - Allow BitsAndBytes override for RTX 3090
import shutil

config_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/config.py"

# Backup
shutil.copy2(config_file, f"{config_file}.orig")
print(f"Backed up to {config_file}.orig")

with open(config_file, 'r') as f:
    lines = f.readlines()

# Find the quantization check and add RTX 3090 exception
modified = False
for i, line in enumerate(lines):
    if '"Quantization method specified in the model config "' in line:
        # Found the error message, now we need to add a check before the raise
        # Go back to find the elif statement
        for j in range(max(0, i-3), i):
            if "elif self.quantization != quant_method:" in lines[j]:
                # Add RTX 3090 check before the raise
                indent = len(lines[j]) - len(lines[j].lstrip())
                new_lines = [
                    lines[j],  # Keep the elif
                    " " * (indent + 4) + "# RTX 3090 override: Allow BitsAndBytes instead of MXFP4\n",
                    " " * (indent + 4) + "from vllm.platforms import current_platform\n",
                    " " * (indent + 4) + "try:\n",
                    " " * (indent + 8) + "cap = current_platform.get_device_capability()\n",
                    " " * (indent + 8) + "if cap and cap.major == 8 and cap.minor == 6:\n",
                    " " * (indent + 12) + "if quant_method == 'mxfp4' and self.quantization == 'bitsandbytes':\n",
                    " " * (indent + 16) + "from vllm.logger import init_logger\n",
                    " " * (indent + 16) + "logger = init_logger(__name__)\n",
                    " " * (indent + 16) + "logger.warning('RTX 3090: Overriding MXFP4 with BitsAndBytes')\n",
                    " " * (indent + 16) + "# Keep bitsandbytes, ignore mxfp4\n",
                    " " * (indent + 12) + "else:\n",
                    " " * (indent + 16) + "raise ValueError(\n",
                    " " * (indent + 20) + '"Quantization method specified in the model config "\n',
                    " " * (indent + 20) + 'f"({quant_method}) does not match the quantization "\n',
                    " " * (indent + 20) + 'f"method specified in the `quantization` argument "\n',
                    " " * (indent + 20) + 'f"({self.quantization}).")\n',
                    " " * (indent + 8) + "else:\n",
                    " " * (indent + 12) + "raise ValueError(\n",
                    " " * (indent + 16) + '"Quantization method specified in the model config "\n',
                    " " * (indent + 16) + 'f"({quant_method}) does not match the quantization "\n',
                    " " * (indent + 16) + 'f"method specified in the `quantization` argument "\n',
                    " " * (indent + 16) + 'f"({self.quantization}).")\n',
                    " " * (indent + 4) + "except:\n",
                    " " * (indent + 8) + "raise ValueError(\n",
                    " " * (indent + 12) + '"Quantization method specified in the model config "\n',
                    " " * (indent + 12) + 'f"({quant_method}) does not match the quantization "\n',
                    " " * (indent + 12) + 'f"method specified in the `quantization` argument "\n',
                    " " * (indent + 12) + 'f"({self.quantization}).")\n',
                ]
                # Replace the original lines with our patched version
                lines = lines[:j] + new_lines + lines[i+4:]  # Skip original error lines
                modified = True
                break
        if modified:
            break

if modified:
    with open(config_file, 'w') as f:
        f.writelines(lines)
    print("✓ PATCH 4: Quantization override enabled for RTX 3090")
else:
    print("✗ Could not find quantization check - manual patch needed")