#!/usr/bin/env python3
"""
Script để test ứng dụng FastAPI
"""
import requests
import json

def test_ping():
    """Test health check endpoint"""
    try:
        response = requests.get("http://localhost:80/ping")
        print(f"Ping Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing ping: {e}")
        return False

def test_ask():
    """Test ask endpoint"""
    try:
        data = {
            "question": "Hello, how are you?",
            "role": "doctor",
            "responseWithAudio": "false"
        }
        response = requests.post("http://localhost:80/ask", data=data)
        print(f"Ask Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing ask: {e}")
        return False

if __name__ == "__main__":
    print("Testing FastAPI Healthcare Chatbot...")
    print("=" * 50)
    
    print("\n1. Testing /ping endpoint:")
    ping_ok = test_ping()
    
    print("\n2. Testing /ask endpoint:")
    ask_ok = test_ask()
    
    print("\n" + "=" * 50)
    if ping_ok and ask_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")