#!/usr/bin/env python3
# patch_5_fix_logger.py - Fix logger scope issue
import shutil

config_file = "/home/tdeshane/miniconda3/envs/vllm_gptoss/lib/python3.12/site-packages/vllm/config.py"

with open(config_file, 'r') as f:
    content = f.read()

# Find and fix the logger import issue
# Move logger import outside the if block
old_pattern = """                    if cap and cap.major == 8 and cap.minor == 6:
                        if quant_method == 'mxfp4' and self.quantization == 'bitsandbytes':
                            from vllm.logger import init_logger
                            logger = init_logger(__name__)
                            logger.warning('RTX 3090: Overriding MXFP4 with BitsAndBytes')"""

new_pattern = """                    if cap and cap.major == 8 and cap.minor == 6:
                        if quant_method == 'mxfp4' and self.quantization == 'bitsandbytes':
                            from vllm.logger import init_logger
                            _logger = init_logger(__name__)
                            _logger.warning('RTX 3090: Overriding MXFP4 with BitsAndBytes')"""

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open(config_file, 'w') as f:
        f.write(content)
    print("✓ PATCH 5: Fixed logger scope issue")
else:
    print("Pattern not found, trying alternative fix...")
    
    # Alternative: check if we need to fix the pre-quantized model case
    # The issue is in line 1221 where logger is referenced but not defined
    import re
    
    # Find the _verify_quantization method and ensure logger is defined
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'def _verify_quantization' in line:
            # Look for the method and add logger import at the beginning
            for j in range(i+1, min(i+10, len(lines))):
                if 'logger.warning' in lines[j] and 'logger = ' not in '\n'.join(lines[i:j]):
                    # Add logger import right after the method definition
                    indent = '        '  # Assuming method body indentation
                    lines.insert(i+1, f'{indent}from vllm.logger import init_logger')
                    lines.insert(i+2, f'{indent}logger = init_logger(__name__)')
                    content = '\n'.join(lines)
                    with open(config_file, 'w') as f:
                        f.write(content)
                    print("✓ PATCH 5: Added logger import to _verify_quantization")
                    break
            break