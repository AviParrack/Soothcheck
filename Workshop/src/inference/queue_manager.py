# This file implements a FastAPI server for managing local inference requests.
# It provides a queue-based approach for handling multiple function calling requests
# from the augmentation pipeline while reusing a loaded model.
#
# Key features:
# - FastAPI server that loads a single model once
# - Supports Llama 3.2/3.3 function calling via TemplateManager
# - Accurate pythonic function call parsing: [func(param=value)]
# - litellm-compatible API interface
# - Simple request queue (no batching initially)
# - Robust error handling and logging
#
# Example usage:
# python -m src.inference.queue_manager llama-3.3-70b-instruct --port 8001

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Add project root to path for imports
QUEUE_MANAGER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = QUEUE_MANAGER_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.inference import preload_model_and_tokenizer, generate_text_for_api
from src.utils.model_resolver import resolve_model_specification
from src.utils.template_manager import TemplateManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request/Response Models
class FunctionCallRequest(BaseModel):
    """Request model that mimics litellm.completion interface"""
    model: str = "local"  # Always local for this server
    messages: List[Dict[str, str]]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[str] = "auto"
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 4096

class FunctionCallResponse(BaseModel):
    """Response model that mimics litellm response format"""
    choices: List[Dict[str, Any]]
    model: str
    usage: Optional[Dict[str, int]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_spec: Optional[str] = None

# Global state for the server
class QueueManagerState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_spec = None
        self.model_resolution = None

server_state = QueueManagerState()

# Create FastAPI app
app = FastAPI(
    title="Local LLM Queue Manager",
    description="FastAPI server for managing local model inference with function calling",
    version="1.0.0",
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.post("/completion", response_model=FunctionCallResponse)
async def function_call_completion(request: FunctionCallRequest):
    """
    Main endpoint that mimics litellm.completion for function calling.
    Handles the conversion between litellm format and Llama 3.3 function calling.
    """
    if not server_state.model or not server_state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert litellm format to Llama 3.3 format
        messages = request.messages
        tools = request.tools or []
        
        # Use TemplateManager for consistent formatting (includes Llama fixes)
        input_ids = TemplateManager.format_for_model(
            data=messages,
            tokenizer=server_state.tokenizer,
            model_type="chat",
            purpose="inference",
            tools=tools
        )

        # Generate response
        response_text = generate_text_for_api(
            prompt=input_ids,
            model=server_state.model,
            tokenizer=server_state.tokenizer,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        # Parse function calls from response if tools were provided
        tool_calls = None
        if tools:
            tool_calls = _parse_function_calls_from_response(response_text, tools)

        # Format response in litellm-compatible format
        choice = {
            "message": {
                "role": "assistant",
                "content": response_text if not tool_calls else None,
                "tool_calls": tool_calls if tool_calls else None
            },
            "finish_reason": "stop"
        }

        return FunctionCallResponse(
            choices=[choice],
            model=server_state.model_spec,
            usage={"prompt_tokens": len(input_ids), "completion_tokens": 0}  # Simplified
        )

    except Exception as e:
        logger.error(f"Error in completion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=server_state.model is not None,
        model_spec=server_state.model_spec,
    )

def _parse_function_calls_from_response(response_text: str, tools: List[Dict]) -> Optional[List[Dict]]:
    """
    Parse function calls from Llama 3.2/3.3 pythonic format.
    
    Llama 3.2/3.3 outputs function calls in pythonic format:
    [func1(param1=value1, param2=value2), func2(param3=value3)]
    
    This parser handles:
    - Single and multiple function calls
    - Proper parameter extraction with param=value syntax
    - String values with or without quotes
    - Robust error handling
    """
    import re
    
    try:
        # Look for pythonic function calls in brackets: [func(...)]
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, response_text)
        
        if not matches:
            logger.debug("No function call brackets found in response")
            return None
        
        function_calls = []
        
        for match in matches:
            try:
                # Parse each function call within the brackets
                # Handle multiple calls: func1(...), func2(...)
                # Use regex to find function_name(parameters) patterns
                call_pattern = r'(\w+)\(([^)]*)\)'
                calls = re.findall(call_pattern, match)
                
                if not calls:
                    logger.debug(f"No function calls found in match: {match}")
                    continue
                
                for func_name, params_str in calls:
                    logger.debug(f"Parsing function call: {func_name}({params_str})")
                    
                    # Parse parameters like param1=value1, param2=value2
                    params = {}
                    
                    if params_str.strip():
                        # Split by comma, handling nested quotes and commas
                        param_parts = []
                        current_part = ""
                        paren_depth = 0
                        quote_char = None
                        
                        for char in params_str:
                            if quote_char:
                                current_part += char
                                if char == quote_char and (not current_part.endswith('\\' + quote_char)):
                                    quote_char = None
                            elif char in ['"', "'"]:
                                quote_char = char
                                current_part += char
                            elif char == '(':
                                paren_depth += 1
                                current_part += char
                            elif char == ')':
                                paren_depth -= 1
                                current_part += char
                            elif char == ',' and paren_depth == 0:
                                param_parts.append(current_part.strip())
                                current_part = ""
                            else:
                                current_part += char
                        
                        if current_part.strip():
                            param_parts.append(current_part.strip())
                        
                        # Parse each parameter
                        for param in param_parts:
                            if '=' in param:
                                key, value = param.split('=', 1)
                                key = key.strip()
                                value = value.strip()
                                
                                # Remove outer quotes if present
                                if ((value.startswith('"') and value.endswith('"')) or 
                                    (value.startswith("'") and value.endswith("'"))):
                                    value = value[1:-1]
                                
                                params[key] = value
                                logger.debug(f"Extracted parameter: {key}={value}")
                    
                    # Validate function name exists in tools
                    valid_function = False
                    for tool in tools:
                        if (tool.get("type") == "function" and 
                            tool.get("function", {}).get("name") == func_name):
                            valid_function = True
                            break
                    
                    if not valid_function:
                        logger.warning(f"Function '{func_name}' not found in available tools")
                        continue
                    
                    function_calls.append({
                        "type": "function",
                        "function": {
                            "name": func_name,
                            "arguments": json.dumps(params)
                        }
                    })
                    
                    logger.debug(f"Successfully parsed function call: {func_name} with {len(params)} parameters")
                    
            except Exception as e:
                logger.warning(f"Failed to parse function call from match '{match}': {e}")
                continue
        
        if function_calls:
            logger.info(f"Successfully parsed {len(function_calls)} function call(s)")
            return function_calls
        else:
            logger.debug("No valid function calls found after parsing")
            return None
            
    except Exception as e:
        logger.error(f"Failed to parse function calls from response: {e}")
        return None

async def load_model_on_startup(model_spec: str, device: str = "auto"):
    """Load the specified model during server startup."""
    logger.info(f"Loading model: {model_spec}")
    logger.info(f"Device: {device}")

    try:
        # Resolve model specification
        server_state.model_resolution = resolve_model_specification(
            model_spec, project_root=PROJECT_ROOT
        )
        server_state.model_spec = model_spec

        logger.info(f"✓ Model resolution successful:")
        logger.info(f"  - Model type: {server_state.model_resolution.model_type}")
        logger.info(f"  - Base model: {server_state.model_resolution.base_model_hf_id}")
        if server_state.model_resolution.peft_adapter_path:
            logger.info(f"  - PEFT adapter: {server_state.model_resolution.peft_adapter_path}")

        # Load model and tokenizer
        server_state.model, server_state.tokenizer = preload_model_and_tokenizer(
            model_spec, device=device
        )

        logger.info(f"✓ Model loaded successfully and ready for function calling")

    except Exception as e:
        logger.error(f"✗ ERROR: Failed to load model '{model_spec}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    """Main entry point for the queue manager server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Local LLM Queue Manager for function calling with augmentation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_spec",
        type=str,
        help="Model specification (experiment name or config name)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (127.0.0.1 for local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for model inference",
    )

    args = parser.parse_args()

    # Load model before starting server
    logger.info("Loading model before starting server...")
    asyncio.run(load_model_on_startup(args.model_spec, args.device))

    # Start server
    logger.info(f"Starting Local LLM Queue Manager on {args.host}:{args.port}")
    logger.info(f"Model: {args.model_spec}")
    logger.info(f"Device: {args.device}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )

if __name__ == "__main__":
    main()

