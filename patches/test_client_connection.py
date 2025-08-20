#!/usr/bin/env python3
"""
Quick connection test for GPT-OSS server without waiting for full model load.
"""

import requests
import time
import sys

def test_connection(host="localhost", port=8000, timeout=5):
    """Test if server is responding."""
    
    try:
        # Test basic connection
        url = f"http://{host}:{port}/v1/models"
        print(f"🔍 Testing connection to {url}...")
        
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            print("✅ Server is responding!")
            models = response.json()
            print(f"📋 Available models: {models}")
            return True
        else:
            print(f"⚠️  Server responded with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - server not running or still loading")
        return False
    except requests.exceptions.Timeout:
        print("⏰ Connection timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_health_check(host="localhost", port=8000):
    """Test health endpoint."""
    
    try:
        url = f"http://{host}:{port}/health"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"⚠️  Health check failed: {response.status_code}")
            return False
    except:
        print("❌ Health check unavailable")
        return False

if __name__ == "__main__":
    print("🧪 Testing GPT-OSS Server Connection")
    print("=" * 40)
    
    # Test connection
    connected = test_connection()
    
    if connected:
        # Test health
        test_health_check()
        print("\n🎉 Server is ready for requests!")
    else:
        print("\n💡 Server may still be loading the model.")
        print("   Wait for 'Application startup complete' message.")
        print("   This takes ~15-20 minutes for initial load.")
    
    sys.exit(0 if connected else 1)