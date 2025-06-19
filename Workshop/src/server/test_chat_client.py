#!/usr/bin/env python
# Simple test client for the chat server with session management
# 
# Example usage:
# python -m src.server.test_chat_client
#
# This demonstrates:
# - Sending messages with session_id for conversation history
# - Resetting conversation history 
# - Multiple turns in the same session

import asyncio
import aiohttp
import json
import sys
from typing import Optional


async def send_chat_message(
    message: str, 
    session_id: Optional[str] = None,
    server_url: str = "http://localhost:8000"
) -> None:
    """Send a chat message and stream the response."""
    url = f"{server_url}/chat/stream"
    payload = {
        "message": message,
        "max_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    
    if session_id:
        payload["session_id"] = session_id
    
    print(f"\n🧑 USER: {message}")
    if session_id:
        print(f"   (Session: {session_id})")
    print("🤖 ASSISTANT: ", end="", flush=True)
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    print(f"Error: {response.status}")
                    text = await response.text()
                    print(f"Response: {text}")
                    return
                
                # Process SSE stream
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_json = line[6:]  # Remove 'data: ' prefix
                        try:
                            data = json.loads(data_json)
                            if data.get('type') == 'token':
                                print(data['token'], end='', flush=True)
                            elif data.get('type') == 'complete':
                                print()  # Newline when complete
                                break
                            elif data.get('type') == 'error':
                                print(f"\nError: {data.get('error')}")
                                break
                        except json.JSONDecodeError:
                            continue
                            
    except Exception as e:
        print(f"Request failed: {e}")


async def reset_chat_session(
    session_id: str = "default",
    server_url: str = "http://localhost:8000"
) -> None:
    """Reset/clear conversation history for a session."""
    url = f"{server_url}/chat/reset"
    payload = {"session_id": session_id}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ {result['message']}")
                else:
                    print(f"Error resetting session: {response.status}")
                    
    except Exception as e:
        print(f"Reset request failed: {e}")


async def main():
    """Demonstrate chat functionality with session management."""
    print("=== Chat Server Test Client ===")
    print("Testing conversation history and session management...")
    
    session_id = "test-session-123"
    
    # Test 1: Start a conversation
    print("\n--- Test 1: Starting a conversation ---")
    await send_chat_message("Hello! What's your name?", session_id)
    
    # Test 2: Continue the conversation (should remember context)
    print("\n--- Test 2: Continuing conversation (with history) ---")
    await send_chat_message("What did I just ask you?", session_id)
    
    # Test 3: Another turn
    print("\n--- Test 3: Another turn ---")
    await send_chat_message("Can you count to 3?", session_id)
    
    # Test 4: Reset the conversation
    print("\n--- Test 4: Resetting conversation ---")
    await reset_chat_session(session_id)
    
    # Test 5: Send a message after reset (should not remember previous context)
    print("\n--- Test 5: After reset (no history) ---")
    await send_chat_message("What did I ask you before?", session_id)
    
    # Test 6: Test without session_id (should work but without persistence)
    print("\n--- Test 6: Without session_id (default session) ---")
    await send_chat_message("This is a message without explicit session_id")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1) 