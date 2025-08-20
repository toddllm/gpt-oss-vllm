#!/usr/bin/env python3
"""
Check server loading status and test when ready.
"""

import subprocess
import time
import requests
import re

def check_server_status():
    """Check if server is loaded and ready."""
    
    print("🔍 Checking GPT-OSS server status...")
    
    # Check if server process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'gpt_oss_server.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip()
            print(f"✅ Server process running (PID: {pid})")
        else:
            print("❌ Server process not found")
            return False
    except:
        print("❌ Could not check server process")
        return False
    
    # Check server logs for completion
    try:
        with open('server_log.txt', 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
            
        if 'Application startup complete' in log_content:
            print("🎉 Server fully loaded and ready!")
            return True
        elif 'Converting layer' in log_content:
            # Extract last converted layer
            matches = re.findall(r"Converting layer\s+(\d+)/(\d+)", log_content)
            if matches:
                last = matches[-1]
                print(f"⏳ Server loading: layer {last[0]}/{last[1]}")
            else:
                print("⏳ Server loading...")
            return False
        elif any(err in log_content for err in ['ERROR', 'ValueError', 'RuntimeError', 'AssertionError']):
            print("❌ Server encountered errors:")
            lines = log_content.split('\n')
            error_lines = [line for line in lines if any(err in line for err in ['ERROR', 'ValueError', 'RuntimeError', 'AssertionError'])]
            for line in error_lines[-5:]:  # Show last 5
                print(f"   {line}")
            return False
        else:
            print("⏳ Server starting up...")
            return False
            
    except FileNotFoundError:
        print("⚠️  Server log not found")
        return False

def test_connection():
    """Test server connection."""
    try:
        response = requests.get('http://localhost:8000/v1/models', timeout=5)
        if response.status_code == 200:
            print("✅ Server is responding to requests!")
            return True
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            return False
    except:
        print("❌ Cannot connect to server")
        return False

if __name__ == "__main__":
    print("=" * 50)
    ready = check_server_status()
    
    if ready:
        print("\n🧪 Testing connection...")
        if test_connection():
            print("\n🚀 Ready for client tests!")
            print("   Run: python quick_client.py")
        else:
            print("\n⚠️  Server loaded but not responding")
    else:
        print("\n⏱️  Server still loading... check again in 5-10 minutes")
        print("   Monitor: tail -f server_log.txt")
    
    print("=" * 50)