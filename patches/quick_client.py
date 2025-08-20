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
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_server(host="localhost", port=8000, prompt=None, interactive=False,
                endpoint="completions", stream=False, concurrency=1,
                max_tokens=50, temperature=0.7, timeout=60):
    """Test the GPT-OSS server with quick requests."""
    
    if endpoint not in {"completions", "chat"}:
        raise ValueError("endpoint must be 'completions' or 'chat'")
    url = f"http://{host}:{port}/v1/{'chat/completions' if endpoint=='chat' else 'completions'}"
    
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
        test_prompts = [prompt] if prompt else [
            "The future of artificial intelligence is",
            "In a world where technology advances rapidly,",
            "The key to solving climate change lies in",
            "Once upon a time, in a distant galaxy,",
            "The most important lesson I've learned is",
        ]

        print("🧪 Running test prompts...")
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = []
                for i, test_prompt in enumerate(test_prompts, 1):
                    futures.append(pool.submit(
                        generate_text,
                        url,
                        test_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                        endpoint=endpoint,
                        timeout=timeout,
                    ))
                for fut in as_completed(futures):
                    fut.result()
        else:
            for i, test_prompt in enumerate(test_prompts, 1):
                print(f"\n📝 Test {i}: '{test_prompt}'")
                generate_text(url, test_prompt, max_tokens=max_tokens,
                              temperature=temperature, stream=stream,
                              endpoint=endpoint, timeout=timeout)
                if i < len(test_prompts):
                    time.sleep(0.5)
    
    return True

def generate_text(url, prompt, max_tokens=50, temperature=0.7, stream=False,
                  endpoint="completions", timeout=60):
    """Generate text for a single prompt."""

    if endpoint == "chat":
        payload = {
            "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": stream,
        }
    else:
        payload = {
            "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stop": ["\n\n"],
            "stream": stream,
        }

    try:
        start_time = time.time()
        if stream:
            with requests.post(url, json=payload, timeout=timeout, stream=True) as resp:
                if resp.status_code != 200:
                    print(f"❌ Error {resp.status_code}: {resp.text}")
                    return
                print("🔊 Streaming chunks:")
                accumulated = []
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            if endpoint == "chat":
                                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            else:
                                delta = obj.get("choices", [{}])[0].get("text", "")
                            if delta:
                                accumulated.append(delta)
                                print(delta, end="", flush=True)
                        except json.JSONDecodeError:
                            continue
                print()
                total_time = time.time() - start_time
                text = "".join(accumulated)
                print(f"✅ Done in {total_time:.2f}s; {len(text)} chars")
        else:
            response = requests.post(url, json=payload, timeout=timeout)
            response_time = time.time() - start_time
            if response.status_code == 200:
                result = response.json()
                if endpoint == "chat":
                    generated_text = result['choices'][0]['message']['content']
                else:
                    generated_text = result['choices'][0]['text']
                usage = result.get('usage', {})
                completion_tokens = usage.get('completion_tokens', 0)
                tokens_per_sec = (completion_tokens / response_time) if response_time > 0 and completion_tokens > 0 else 0
                print(f"✅ Generated: {generated_text}")
                if tokens_per_sec > 0:
                    print(f"⚡ Speed: {tokens_per_sec:.1f} tok/s ({completion_tokens} in {response_time:.2f}s)")
                else:
                    print(f"⏱️  Time: {response_time:.2f}s")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
    except requests.exceptions.Timeout:
        print(f"⏰ Request timed out (>{timeout}s)")
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
    parser.add_argument("--endpoint", choices=["completions", "chat"], default="completions",
                       help="Use legacy completions or chat completions")
    parser.add_argument("--stream", action="store_true", help="Enable server-side streaming")
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent requests for non-interactive mode")
    parser.add_argument("--timeout", type=int, default=60, help="Per-request timeout in seconds")
    
    args = parser.parse_args()
    
    print("🚀 GPT-OSS Quick Client")
    print("=" * 40)
    
    success = test_server(
        host=args.host,
        port=args.port, 
        prompt=args.prompt,
        interactive=args.interactive,
        endpoint=args.endpoint,
        stream=args.stream,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())