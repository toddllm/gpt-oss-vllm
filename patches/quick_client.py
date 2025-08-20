#!/usr/bin/env python3
"""
Quick client for testing GPT-OSS server.
Fast requests against persistent server.
"""

import requests
import json
import time
import argparse
import sys

def test_server(host="localhost", port=8000, prompt=None, interactive=False):
    """Test the GPT-OSS server with quick requests."""
    
    url = f"http://{host}:{port}/v1/completions"
    
    # Test server availability first
    try:
        health_response = requests.get(f"http://{host}:{port}/v1/models", timeout=5)
        if health_response.status_code != 200:
            print(f"❌ Server not responding properly (status: {health_response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {host}:{port}")
        print(f"   Make sure to run: python gpt_oss_server.py")
        return False
    
    print(f"✅ Connected to GPT-OSS server at {host}:{port}")
    
    if interactive:
        print("🎯 Interactive mode - type 'quit' to exit")
        while True:
            try:
                user_prompt = input("\n📝 Enter prompt: ").strip()
                if user_prompt.lower() in ['quit', 'exit', 'q']:
                    break
                if not user_prompt:
                    continue
                
                generate_text(url, user_prompt)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
    else:
        # Use provided prompt or default test prompts
        if prompt:
            test_prompts = [prompt]
        else:
            test_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology advances rapidly,",
                "The key to solving climate change lies in",
                "Once upon a time, in a distant galaxy,",
                "The most important lesson I've learned is"
            ]
        
        print("🧪 Running test prompts...")
        for i, test_prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: '{test_prompt}'")
            generate_text(url, test_prompt)
            if i < len(test_prompts):
                time.sleep(0.5)  # Brief pause between requests
    
    return True

def generate_text(url, prompt, max_tokens=50, temperature=0.7):
    """Generate text for a single prompt."""
    
    payload = {
        "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "stop": ["\n\n"]
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload, timeout=60)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['text']
            
            # Calculate tokens/sec if available
            usage = result.get('usage', {})
            completion_tokens = usage.get('completion_tokens', 0)
            tokens_per_sec = completion_tokens / response_time if response_time > 0 and completion_tokens > 0 else 0
            
            print(f"✅ Generated: {generated_text}")
            if tokens_per_sec > 0:
                print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec ({completion_tokens} tokens in {response_time:.2f}s)")
            else:
                print(f"⏱️  Time: {response_time:.2f}s")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (>60s)")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Quick GPT-OSS client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port") 
    parser.add_argument("--prompt", help="Single prompt to test")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Interactive mode")
    parser.add_argument("--max-tokens", type=int, default=50,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Temperature for generation")
    
    args = parser.parse_args()
    
    print("🚀 GPT-OSS Quick Client")
    print("=" * 40)
    
    success = test_server(
        host=args.host,
        port=args.port, 
        prompt=args.prompt,
        interactive=args.interactive
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())