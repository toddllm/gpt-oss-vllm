#!/usr/bin/env python3
"""
Comprehensive smoke suite to validate server readiness and endpoint behavior.
"""
import time
import subprocess
import sys
import json
import requests

SERVER_URL = "http://localhost:8000"


def wait_for_server(timeout_s: int = 1800, poll_interval: float = 10.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            r = requests.get(f"{SERVER_URL}/v1/models", timeout=5)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(poll_interval)
    return False


def test_basic_completion() -> bool:
    payload = {
        "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "prompt": "The",
        "max_tokens": 1,
        "temperature": 0.0,
    }
    r = requests.post(f"{SERVER_URL}/v1/completions", json=payload, timeout=60)
    if r.status_code != 200:
        print("❌ basic completion status:", r.status_code, r.text)
        return False
    data = r.json()
    text = data["choices"][0]["text"]
    return isinstance(text, str) and len(text) > 0


def test_chat_stream() -> bool:
    payload = {
        "model": "unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        "messages": [{"role": "user", "content": "Say hello succinctly."}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
    }
    with requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, stream=True, timeout=60) as resp:
        if resp.status_code != 200:
            print("❌ chat stream status:", resp.status_code, resp.text)
            return False
        any_chunk = False
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: "):]
                if data.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    delta = obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        any_chunk = True
                except json.JSONDecodeError:
                    continue
        return any_chunk


def main():
    print("🔎 Waiting for server availability...")
    ready = wait_for_server(timeout_s=60*30, poll_interval=10)
    if not ready:
        print("❌ Server not ready in time")
        sys.exit(1)

    print("🧪 Testing /v1/completions (single token)")
    assert test_basic_completion(), "basic completion failed"

    print("🧪 Testing /v1/chat/completions (stream)")
    assert test_chat_stream(), "chat streaming failed"

    print("✅ Smoke suite passed")


if __name__ == "__main__":
    main()
