#!/usr/bin/env python3
"""Test script to verify Ollama connection and model availability."""

import requests
import json
import sys

def test_ollama_connection():
    """Test connection to Ollama and list available models."""
    ollama_host = "http://localhost:11434"
    
    print(f"Testing connection to Ollama at {ollama_host}...")
    
    try:
        # Test basic connectivity
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        
        if response.status_code == 200:
            print("✓ Successfully connected to Ollama service")
            
            # List available models
            models = response.json().get('models', [])
            if models:
                print(f"\nAvailable models ({len(models)}):")
                for model in models:
                    name = model.get('name', 'unknown')
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    print(f"  - {name} ({size:.2f} GB)")
                    
                # Check specifically for Mistral
                mistral_models = [m['name'] for m in models if 'mistral' in m['name'].lower()]
                if mistral_models:
                    print(f"\n✓ Mistral models found: {', '.join(mistral_models)}")
                else:
                    print("\n⚠ No Mistral models found. You may need to pull one:")
                    print("  ollama pull mistral:7b")
            else:
                print("\n⚠ No models found. Pull a model first:")
                print("  ollama pull mistral:7b")
                
        else:
            print(f"✗ Failed to connect: HTTP {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to Ollama. Make sure the service is running:")
        print("  ollama serve")
        return False
    except requests.exceptions.Timeout:
        print("✗ Connection timed out")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False
    
    # Test generation endpoint
    print("\nTesting generation endpoint...")
    try:
        test_payload = {
            "model": "mistral:7b",
            "prompt": "Say 'Hello, negotiation test!'",
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 10
            }
        }
        
        response = requests.post(
            f"{ollama_host}/api/generate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generation test successful")
            print(f"  Response: {result.get('response', 'No response')[:100]}")
        else:
            print(f"⚠ Generation failed: HTTP {response.status_code}")
            if response.text:
                error = json.loads(response.text)
                print(f"  Error: {error.get('error', 'Unknown error')}")
                if 'not found' in error.get('error', '').lower():
                    print("\n  Try pulling the model first:")
                    print("    ollama pull mistral:7b")
    except Exception as e:
        print(f"⚠ Generation test failed: {e}")
    
    return True

if __name__ == "__main__":
    success = test_ollama_connection()
    sys.exit(0 if success else 1)