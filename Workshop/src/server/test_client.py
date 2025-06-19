#!/usr/bin/env python
# This file provides a simple test client for the FastAPI inference server.
# It demonstrates how to interact with the SSE streaming API.
#
# Example usage:
# python -m src.server.test_client --message "Hello, how are you?"

import asyncio
import json
import aiohttp
import argparse


async def test_streaming_api(base_url: str, message: str, prompt_name: str = None):
    """
    Test the streaming chat API endpoint.
    """
    url = f"{base_url}/chat/stream"
    
    payload = {
        "message": message,
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    if prompt_name:
        payload["prompt_name"] = prompt_name
    
    print(f"Testing streaming API at: {url}")
    print(f"Message: {message}")
    if prompt_name:
        print(f"Prompt name: {prompt_name}")
    print("-" * 50)
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                print(f"Error: HTTP {response.status}")
                text = await response.text()
                print(f"Response: {text}")
                return
            
            print("Streaming response:")
            full_response = ""
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove 'data: ' prefix
                    try:
                        data = json.loads(data_str)
                        
                        if data.get('type') == 'token':
                            token = data.get('token', '')
                            print(token, end='', flush=True)
                            full_response += token
                        elif data.get('type') == 'complete':
                            print("\n" + "-" * 50)
                            print("Generation complete!")
                            break
                        elif data.get('type') == 'error':
                            print(f"\nError: {data.get('error')}")
                            break
                    except json.JSONDecodeError:
                        print(f"Could not parse JSON: {data_str}")
            
            print(f"\nFull response: {full_response}")


async def test_health_api(base_url: str):
    """
    Test the health check endpoint.
    """
    url = f"{base_url}/health"
    
    print(f"Testing health API at: {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Health check response: {json.dumps(data, indent=2)}")
            else:
                print(f"Error: HTTP {response.status}")
                text = await response.text()
                print(f"Response: {text}")


async def test_models_api(base_url: str):
    """
    Test the models info endpoint.
    """
    url = f"{base_url}/models"
    
    print(f"Testing models API at: {url}")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                print(f"Model info response: {json.dumps(data, indent=2)}")
            else:
                print(f"Error: HTTP {response.status}")
                text = await response.text()
                print(f"Response: {text}")


async def main():
    parser = argparse.ArgumentParser(
        description="Test client for the FastAPI inference server"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Hello! How are you today?",
        help="Message to send to the chat API",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        default=None,
        help="Name of the system prompt to use",
    )
    parser.add_argument(
        "--test-health",
        action="store_true",
        help="Test the health endpoint",
    )
    parser.add_argument(
        "--test-models",
        action="store_true",
        help="Test the models endpoint",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FastAPI Inference Server Test Client")
    print("=" * 60)
    
    # Test health endpoint
    if args.test_health:
        await test_health_api(args.base_url)
        print()
    
    # Test models endpoint
    if args.test_models:
        await test_models_api(args.base_url)
        print()
    
    # Test streaming chat endpoint
    await test_streaming_api(args.base_url, args.message, args.prompt_name)


if __name__ == "__main__":
    asyncio.run(main()) 