#!/usr/bin/env python
# This file implements the FastAPI server for serving language models via SSE streaming.
# It provides endpoints for:
# - POST /chat/stream: Streaming inference via Server-Sent Events (with conversation history)
# - POST /chat/reset: Reset conversation history for a session
# - GET /health: Health check endpoint
# - GET /models: List available models (optional)
#
# The server integrates with the existing inference pipeline:
# - Uses preload_model_and_tokenizer() to load models
# - Uses generate_text_yield_tokens() for streaming generation
# - Uses ChatManager for conversation history management
# - Supports model resolution via resolve_model_specification()
#
# Chat features:
# - Conversation history is maintained per session_id
# - Sessions are created automatically when first accessed
# - History can be reset via the /chat/reset endpoint
# - Backward compatible: requests without session_id work as before
#
# Example usage:
# python -m src.server.main gemma-3-27b-it --host 0.0.0.0 --port 8000 --prompt_name test-prompt --request-timeout 600

# and then to serve on ngrok:
# ngrok http 8000 --domain=workshop-labs.ngrok.io --log stdout




import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Optional
import time
import threading

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add project root to path for imports
SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.inference import (
    preload_model_and_tokenizer,
)
from src.inference.inference_utils import load_and_format_prompt as load_and_format_prompt_from_utils
from src.inference.chat_manager import ChatManager
from src.utils.model_resolver import resolve_model_specification


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    prompt_name: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_spec: Optional[str] = None


class ModelInfo(BaseModel):
    model_spec: str
    model_type: str
    base_model_hf_id: str
    is_experiment: bool
    peft_adapter_path: Optional[str] = None


# Global state
class ServerState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_spec = None
        self.model_type = None
        self.model_resolution = None
        self.prompt_name: Optional[str] = None  # Default system prompt
        self.chat_sessions: dict[str, ChatManager] = {}  # session_id -> ChatManager
        self.request_timeout: float = 600.0  # Default timeout for streamer


server_state = ServerState()


# Create FastAPI app
app = FastAPI(
    title="Language Model Inference Server",
    description="FastAPI server for streaming language model inference",
    version="1.0.0",
)

# CORS Configuration
# Regex for allowed origins:
# - http(s)://localhost or 127.0.0.1 with any port
# - https for any subdomain of ngrok.app or ngrok.io
# - https for any subdomain of github.io
# - http(s) for workshoplabs.ai
allow_origin_regex = r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.ngrok\.(io|app)|https://.*\.github\.io|https?://workshoplabs\.ai"

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    Initiate streaming inference using Server-Sent Events.
    This version correctly handles KV caching and chat history.
    """
    if not server_state.model or not server_state.tokenizer:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Get or create chat session
    session_id = request.session_id or "default"
    if session_id not in server_state.chat_sessions:
        server_state.chat_sessions[session_id] = ChatManager(
            max_history_turns=12, max_sequence_length=4096
        )

    chat_manager = server_state.chat_sessions[session_id]

    async def generate():
        # This async generator function contains the core logic for streaming.
        from transformers import TextIteratorStreamer, DynamicCache
        from threading import Thread
        import torch
        from src.utils.template_manager import TemplateManager
        from src.inference.inference_utils import _load_system_prompt_content

        # Container to get the final generated sequence from the thread
        generation_output_container = {}

        try:
            print(f"DEBUG: Starting streaming for session_id: {request.session_id or 'default'}")
            print(f"DEBUG: Request message length: {len(request.message)}")
            print(f"DEBUG: Model device: {server_state.model.device}")
            print(f"DEBUG: Tokenizer type: {type(server_state.tokenizer)}")
            
            # Check GPU memory if available
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"DEBUG: GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                except Exception as mem_e:
                    print(f"DEBUG: Could not get GPU memory info: {mem_e}")

            # Add user message and handle system prompt
            chat_manager.messages.append({"role": "user", "content": request.message})
            
            # Use server-defined prompt, fall back to request's prompt
            prompt_name_to_use = server_state.prompt_name or request.prompt_name
            if len(chat_manager.messages) == 1 and prompt_name_to_use:
                system_prompt_content = _load_system_prompt_content(prompt_name_to_use)
                if system_prompt_content:
                    chat_manager.messages.insert(0, {"role": "system", "content": system_prompt_content})

            print(f"DEBUG: Total messages in conversation: {len(chat_manager.messages)}")

            # Format conversation and handle potential trimming
            inputs = server_state.tokenizer.apply_chat_template(
                chat_manager.messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
            ).to(server_state.model.device)
            input_length = inputs["input_ids"].shape[1]
            print(f"DEBUG: Input sequence length: {input_length}")

            if input_length > chat_manager.max_sequence_length:
                print(f"Warning: Sequence length ({input_length}) exceeds max. Trimming history.")
                chat_manager._trim_history()
                chat_manager.kv_cache = DynamicCache()  # Reset cache after trimming
                inputs = server_state.tokenizer.apply_chat_template(
                    chat_manager.messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
                ).to(server_state.model.device)
                input_length = inputs["input_ids"].shape[1]
                print(f"DEBUG: Input sequence length after trimming: {input_length}")

            # Setup streamer and generation params
            print(f"DEBUG: Setting up TextIteratorStreamer with timeout: {server_state.request_timeout}")
            streamer = TextIteratorStreamer(
                server_state.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=server_state.request_timeout,
            )
            pad_token_id = server_state.tokenizer.pad_token_id or server_state.tokenizer.eos_token_id
            termination_tokens = TemplateManager.get_termination_tokens(server_state.tokenizer)
            
            print(f"DEBUG: Pad token ID: {pad_token_id}")
            print(f"DEBUG: Termination tokens: {termination_tokens}")

            # Generation function to run in a background thread
            def _generate_with_no_grad():
                thread_id = id(threading.current_thread())
                print(f"DEBUG: Generation thread {thread_id} started")
                
                with torch.no_grad():
                    try:
                        print(f"DEBUG: Thread {thread_id} - Starting model.generate()")
                        generation_start_time = time.time()
                        
                        outputs = server_state.model.generate(
                            **inputs,
                            max_new_tokens=request.max_tokens,
                            do_sample=True,
                            temperature=request.temperature,
                            top_p=request.top_p,
                            pad_token_id=pad_token_id,
                            eos_token_id=termination_tokens,
                            past_key_values=chat_manager.kv_cache,
                            use_cache=True,
                            streamer=streamer,
                            return_dict_in_generate=True,
                        )
                        
                        generation_end_time = time.time()
                        print(f"DEBUG: Thread {thread_id} - model.generate() completed in {generation_end_time - generation_start_time:.2f}s")
                        generation_output_container["outputs"] = outputs
                        
                    except Exception as e:
                        print(f"ERROR: Thread {thread_id} - Error during generation: {e}")
                        import traceback
                        traceback.print_exc()
                        generation_output_container["error"] = e
                    finally:
                        print(f"DEBUG: Thread {thread_id} - Calling streamer.end()")
                        try:
                            streamer.end()
                            print(f"DEBUG: Thread {thread_id} - streamer.end() completed")
                        except Exception as end_e:
                            print(f"ERROR: Thread {thread_id} - Error in streamer.end(): {end_e}")

            # Start generation and stream tokens
            print("DEBUG: Starting generation thread")
            thread = Thread(target=_generate_with_no_grad)
            thread.start()
            print(f"DEBUG: Thread started with ID: {id(thread)}")

            # Stream tokens with detailed logging
            token_count = 0
            stream_start_time = time.time()
            print("DEBUG: Starting to iterate over streamer tokens")
            
            try:
                for token in streamer:
                    token_count += 1
                    if token_count == 1:
                        print(f"DEBUG: Received first token after {time.time() - stream_start_time:.2f}s")
                    elif token_count % 10 == 0:
                        print(f"DEBUG: Received {token_count} tokens so far")
                    
                    yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
                    # CRITICAL: Yield control back to event loop to allow immediate streaming
                    await asyncio.sleep(0)
                    
            except Exception as stream_e:
                print(f"ERROR: Exception while iterating over streamer: {stream_e}")
                import traceback
                traceback.print_exc()
                raise stream_e
            
            print(f"DEBUG: Finished streaming {token_count} tokens")
            print("DEBUG: Waiting for generation thread to complete")
            thread.join(timeout=60.0)  # Add timeout to thread.join()
            
            if thread.is_alive():
                print("WARNING: Generation thread is still alive after 60s timeout")
            else:
                print("DEBUG: Generation thread completed successfully")

            # Check for errors from the generation thread
            if "error" in generation_output_container:
                print(f"DEBUG: Found error in generation_output_container: {generation_output_container['error']}")
                raise generation_output_container["error"]

            if "outputs" not in generation_output_container:
                print("ERROR: No outputs found in generation_output_container")
                raise RuntimeError("Generation completed but no outputs were produced")

            # Correctly update history using the raw tokens
            print("DEBUG: Processing generation outputs for history")
            generated_tokens = generation_output_container["outputs"].sequences
            response_token_ids = generated_tokens[0][input_length:]
            
            response_for_history = server_state.tokenizer.decode(
                response_token_ids, skip_special_tokens=False
            ).strip()
            
            print(f"DEBUG: Adding response to history (length: {len(response_for_history)})")
            chat_manager.messages.append({"role": "assistant", "content": response_for_history})
            chat_manager._trim_history()

            print("DEBUG: Sending completion signal")
            yield f"data: {json.dumps({'type': 'complete'})}\n\n"

        except Exception as e:
            print(f"ERROR: Exception in streaming generate(): {e}")
            import traceback
            traceback.print_exc()
            error_data = {
                'type': 'error',
                'error': str(e),
                'message': 'An error occurred during generation'
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",  # Correct media type for SSE
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Transfer-Encoding": "chunked",  # Force chunked encoding
            "X-Content-Type-Options": "nosniff",  # Prevent content sniffing buffering
            "Content-Encoding": "identity",  # Prevent compression buffering
            "Access-Control-Allow-Origin": "*",  # Prevent CORS-related buffering
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthResponse(
        status="ok",
        model_loaded=server_state.model is not None,
        model_spec=server_state.model_spec,
    )


@app.get("/models", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the currently loaded model.
    """
    if not server_state.model_resolution:
        raise HTTPException(status_code=503, detail="No model loaded")

    return ModelInfo(
        model_spec=server_state.model_spec,
        model_type=server_state.model_resolution.model_type,
        base_model_hf_id=server_state.model_resolution.base_model_hf_id,
        is_experiment=server_state.model_resolution.is_experiment,
        peft_adapter_path=server_state.model_resolution.peft_adapter_path,
    )


@app.post("/chat/reset")
async def reset_chat_session(request: dict):
    """
    Reset/clear conversation history for a given session.
    """
    session_id = request.get("session_id", "default")
    
    if session_id in server_state.chat_sessions:
        server_state.chat_sessions[session_id].clear_history()
        return {"message": f"Session '{session_id}' history cleared"}
    else:
        # Create new empty session
        server_state.chat_sessions[session_id] = ChatManager(max_history_turns=10)
        return {"message": f"New session '{session_id}' created"}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    print(f"Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()
    
    return {
        "error": str(exc),
        "type": "server_error",
        "detail": "An internal server error occurred"
    }


async def load_model_on_startup(
    model_spec: str, device: str = "auto", prompt_name: Optional[str] = None
):
    """
    Load the specified model during server startup.
    """
    print(f"\n{'='*80}")
    print(f"LOADING MODEL FOR SERVER")
    print(f"{'='*80}")
    print(f"Model specification: {model_spec}")
    print(f"Device: {device}")

    try:
        # Resolve model specification
        server_state.model_resolution = resolve_model_specification(
            model_spec, project_root=PROJECT_ROOT
        )
        server_state.model_type = server_state.model_resolution.model_type
        server_state.model_spec = model_spec
        server_state.prompt_name = prompt_name

        print(f"✓ Model resolution successful:")
        print(f"  - Model type: {server_state.model_type}")
        print(f"  - Base model: {server_state.model_resolution.base_model_hf_id}")
        if server_state.model_resolution.peft_adapter_path:
            print(f"  - PEFT adapter: {server_state.model_resolution.peft_adapter_path}")

        # Load model and tokenizer
        server_state.model, server_state.tokenizer = preload_model_and_tokenizer(
            model_spec, device=device
        )

        print(f"✓ Model loaded successfully and ready for inference")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"\n✗ ERROR: Failed to load model '{model_spec}'")
        print(f"Details: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """
    Main entry point for the server.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="FastAPI server for language model inference via SSE streaming",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_spec",
        type=str,
        help="Model specification (experiment name or config name)",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Optional: Name of the system prompt file from 'prompts' to use as a default for all sessions.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for model inference",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=600,
        help="Request timeout in seconds for the streamer.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Set timeout
    server_state.request_timeout = float(args.request_timeout)

    # Load model before starting server
    print("Loading model before starting server...")
    asyncio.run(load_model_on_startup(args.model_spec, args.device, args.prompt_name))

    # Start server
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {args.model_spec}")
    print(f"Device: {args.device}")
    print(f"API Documentation: http://{args.host}:{args.port}/docs")

    # Use the app object directly instead of module string to preserve state
    uvicorn.run(
        app,  # Pass app object directly, not string
        host=args.host,
        port=args.port,
        reload=False,  # Disable reload to preserve state
        log_level="info",
    )


if __name__ == "__main__":
    main() 