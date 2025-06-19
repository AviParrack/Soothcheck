# This file provides an adapter for local LLM inference through the queue manager.
# It mimics the litellm.completion interface to maintain compatibility with existing
# augmentation code while routing requests to a local inference server.
#
# Key functions:
# - local_completion: HTTP client that mimics litellm.completion
# - LocalLLMConfig: Configuration for the local inference server
# - Error handling and retry logic for robust operation
#
# Example usage:
# from src.inference.local_llm_adapter import local_completion
# response = local_completion(model="local", messages=messages, tools=tools)

import asyncio
import json
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LocalLLMConfig:
    """Configuration for local LLM server connection"""
    host: str = "127.0.0.1"
    port: int = 8001
    timeout: int = 300  # 5 minutes for large models
    max_retries: int = 3
    retry_delay: float = 1.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

# Global config - can be modified by calling code
LOCAL_LLM_CONFIG = LocalLLMConfig()

class LocalLLMError(Exception):
    """Custom exception for local LLM adapter errors"""
    pass

class MockLiteLLMResponse:
    """Mock object that mimics litellm response structure"""
    def __init__(self, response_data: Dict[str, Any]):
        self.choices = []
        
        # Convert response data to choices format
        if "choices" in response_data:
            for choice_data in response_data["choices"]:
                choice = MockChoice(choice_data)
                self.choices.append(choice)
        
        self.model = response_data.get("model", "local")
        self.usage = response_data.get("usage", {})

class MockChoice:
    """Mock choice object that mimics litellm choice structure"""
    def __init__(self, choice_data: Dict[str, Any]):
        self.message = MockMessage(choice_data.get("message", {}))
        self.finish_reason = choice_data.get("finish_reason", "stop")

class MockMessage:
    """Mock message object that mimics litellm message structure"""
    def __init__(self, message_data: Dict[str, Any]):
        self.role = message_data.get("role", "assistant")
        self.content = message_data.get("content")
        self.tool_calls = message_data.get("tool_calls")
        
        # Convert tool_calls to mock objects if they exist
        if self.tool_calls:
            self.tool_calls = [MockToolCall(tc) for tc in self.tool_calls]

class MockToolCall:
    """Mock tool call object that mimics litellm tool call structure"""
    def __init__(self, tool_call_data: Dict[str, Any]):
        self.type = tool_call_data.get("type", "function")
        self.function = MockFunction(tool_call_data.get("function", {}))

class MockFunction:
    """Mock function object that mimics litellm function structure"""
    def __init__(self, function_data: Dict[str, Any]):
        self.name = function_data.get("name", "")
        self.arguments = function_data.get("arguments", "{}")

def local_completion(
    model: str,
    messages: List[Dict[str, str]], 
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = "auto",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 4096,
    **kwargs
) -> MockLiteLLMResponse:
    """
    HTTP client that mimics litellm.completion interface for local inference.
    
    Args:
        model: Model identifier (always routes to local for this adapter)
        messages: List of chat messages
        tools: Optional list of tool definitions for function calling
        tool_choice: Tool choice strategy ("auto", "required", etc.)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter  
        max_new_tokens: Maximum tokens to generate
        **kwargs: Additional parameters (logged but ignored)
    
    Returns:
        MockLiteLLMResponse: Response object that mimics litellm structure
    
    Raises:
        LocalLLMError: If the request fails after retries
    """
    
    # Log any unused kwargs
    if kwargs:
        logger.debug(f"Ignoring additional parameters: {kwargs}")
    
    # Prepare request payload
    request_data = {
        "model": "local",  # Always local for this adapter
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    
    # Make HTTP request with retry logic
    for attempt in range(LOCAL_LLM_CONFIG.max_retries):
        try:
            logger.debug(f"Making request to {LOCAL_LLM_CONFIG.base_url}/completion (attempt {attempt + 1})")
            
            response = requests.post(
                f"{LOCAL_LLM_CONFIG.base_url}/completion",
                json=request_data,
                timeout=LOCAL_LLM_CONFIG.timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            logger.debug(f"Received response: {json.dumps(response_data, indent=2)}")
            
            # Return mock litellm response
            return MockLiteLLMResponse(response_data)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request attempt {attempt + 1} failed: {e}")
            
            if attempt < LOCAL_LLM_CONFIG.max_retries - 1:
                # Wait before retry (with exponential backoff)
                wait_time = LOCAL_LLM_CONFIG.retry_delay * (2 ** attempt)
                logger.info(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                error_msg = f"Local LLM request failed after {LOCAL_LLM_CONFIG.max_retries} attempts: {e}"
                logger.error(error_msg)
                raise LocalLLMError(error_msg)
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise LocalLLMError(f"Invalid JSON response from local LLM server: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error in local completion: {e}")
            raise LocalLLMError(f"Unexpected error: {e}")

def check_local_llm_health() -> bool:
    """
    Check if the local LLM server is healthy and ready to accept requests.
    
    Returns:
        bool: True if server is healthy, False otherwise
    """
    try:
        response = requests.get(
            f"{LOCAL_LLM_CONFIG.base_url}/health",
            timeout=10
        )
        response.raise_for_status()
        
        health_data = response.json()
        is_healthy = (
            health_data.get("status") == "ok" and 
            health_data.get("model_loaded", False)
        )
        
        if is_healthy:
            logger.info(f"Local LLM server is healthy. Model: {health_data.get('model_spec')}")
        else:
            logger.warning(f"Local LLM server is not ready: {health_data}")
            
        return is_healthy
        
    except Exception as e:
        logger.error(f"Failed to check local LLM health: {e}")
        return False

def wait_for_local_llm_ready(timeout: int = 180) -> bool:
    """
    Wait for the local LLM server to become ready.
    
    Args:
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if server becomes ready, False if timeout
    """
    logger.info(f"Waiting up to {timeout} seconds for local LLM server at {LOCAL_LLM_CONFIG.base_url} to become ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if check_local_llm_health():
            logger.info("Local LLM server is ready!")
            return True
        
        logger.debug("Server not ready yet, waiting...")
        time.sleep(2)
    
    logger.error(f"Local LLM server did not become ready within {timeout}s")
    return False

def configure_local_llm(host: str = None, port: int = None, timeout: int = None, max_retries: int = None):
    """
    Configure the local LLM connection parameters.
    
    Args:
        host: Server host (default: 127.0.0.1)  
        port: Server port (default: 8001)
        timeout: Request timeout in seconds (default: 300)
        max_retries: Maximum retry attempts (default: 3)
    """
    global LOCAL_LLM_CONFIG
    
    if host is not None:
        LOCAL_LLM_CONFIG.host = host
    if port is not None:
        LOCAL_LLM_CONFIG.port = port
    if timeout is not None:
        LOCAL_LLM_CONFIG.timeout = timeout
    if max_retries is not None:
        LOCAL_LLM_CONFIG.max_retries = max_retries
    
    logger.info(f"Local LLM adapter configured: {LOCAL_LLM_CONFIG.base_url}")

# Example usage and testing
if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.DEBUG)
    
    # Check if server is available
    if not check_local_llm_health():
        print("Local LLM server is not available. Start it with:")
        print("python -m src.inference.queue_manager llama-3.3-70b-instruct --port 8001")
        exit(1)
    
    # Test function calling
    test_messages = [
        {"role": "user", "content": "What's the weather like?"}
    ]
    
    test_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    try:
        response = local_completion(
            model="local",
            messages=test_messages,
            tools=test_tools
        )
        
        print("Response received:")
        print(f"Model: {response.model}")
        print(f"Content: {response.choices[0].message.content}")
        if response.choices[0].message.tool_calls:
            print(f"Tool calls: {len(response.choices[0].message.tool_calls)}")
            
    except Exception as e:
        print(f"Test failed: {e}") 