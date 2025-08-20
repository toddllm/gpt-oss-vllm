#!/usr/bin/env python3
"""
Client code to test GPT-OSS:20B running on RTX 3090 with vLLM.
Run this after the model has finished loading.
"""

import requests
import json
import time

def test_vllm_server():
    """Test the vLLM OpenAI-compatible API server."""
    
    # vLLM server endpoint
    url = "http://localhost:8000/v1/completions"
    
    # Test prompts
    test_prompts = [
        "The future of artificial intelligence is",
        "In a world where technology has advanced beyond imagination,",
        "The key to solving climate change lies in",
        "Once upon a time, in a distant galaxy,",
        "The most important lesson I've learned is"
    ]
    
    print("🚀 Testing GPT-OSS:20B on RTX 3090 with vLLM")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n📝 Test {i}: '{prompt}'")
        
        # Request payload
        payload = {
            "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "stop": ["\n\n"]
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['text']
                tokens_generated = result['usage']['completion_tokens']
                tokens_per_sec = tokens_generated / response_time if response_time > 0 else 0
                
                print(f"✅ Generated: {generated_text}")
                print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec ({tokens_generated} tokens in {response_time:.2f}s)")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
        
        time.sleep(1)  # Brief pause between requests

def test_direct_vllm():
    """Test direct vLLM usage (if server isn't running)."""
    
    print("🔄 Testing direct vLLM usage...")
    
    try:
        from vllm import LLM, SamplingParams
        
        # Create LLM instance
        llm = LLM(
            model="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
            quantization="bitsandbytes",
            load_format="bitsandbytes",  # Required for BnB quantization
            dtype="float16",
            max_model_len=512,  # Increase for testing
            gpu_memory_utilization=0.50,  # Increase after successful loading
            trust_remote_code=True,
            enforce_eager=True,
        )
        
        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=50,
            stop=["\n\n"]
        )
        
        # Test prompts
        prompts = [
            "The future of AI is",
            "In a world where",
            "The key to success"
        ]
        
        print("✅ Model loaded successfully!")
        print("🧪 Running generation tests...")
        
        for prompt in prompts:
            print(f"\n📝 Prompt: '{prompt}'")
            
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            gen_time = time.time() - start_time
            
            generated_text = outputs[0].outputs[0].text
            tokens_generated = len(outputs[0].outputs[0].token_ids)
            tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
            
            print(f"✅ Generated: {generated_text}")
            print(f"⚡ Speed: {tokens_per_sec:.1f} tokens/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct vLLM test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Checking if vLLM server is running...")
    
    # Try server first
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM server detected!")
            test_vllm_server()
        else:
            print("❌ Server not responding properly")
            test_direct_vllm()
    except:
        print("📡 No server detected, trying direct vLLM...")
        test_direct_vllm()