#!/usr/bin/env python3
"""
GPT-OSS:20B vLLM Server
Keeps the model loaded in memory and serves requests via OpenAI-compatible API.
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS:20B vLLM Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.50, 
                       help="GPU memory utilization (0.35-0.60 for RTX 3090)")
    parser.add_argument("--max-model-len", type=int, default=512,
                       help="Maximum model context length")
    parser.add_argument("--disable-log-requests", action="store_true",
                       help="Disable request logging for cleaner output")
    
    args = parser.parse_args()
    
    # Set environment for RTX 3090 compatibility
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"
    
    print("🚀 Starting GPT-OSS:20B Server for RTX 3090")
    print(f"📊 GPU Memory Utilization: {args.gpu_memory_utilization}")
    print(f"📏 Max Model Length: {args.max_model_len}")
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("🔄 Loading GPT-OSS:20B model... (this will take ~15-20 minutes)")
    
    # Build command line arguments for vLLM server
    vllm_args = [
        "--model", "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "--host", args.host,
        "--port", str(args.port),
        "--quantization", "bitsandbytes",
        "--load-format", "bitsandbytes",
        "--dtype", "float16",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--enforce-eager",
    ]
    
    if args.disable_log_requests:
        vllm_args.append("--disable-log-requests")
    
    # Execute vLLM server
    try:
        import subprocess
        import sys
        
        # Use current Python executable and run as module
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"] + vllm_args
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed with exit code: {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"❌ Server failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())