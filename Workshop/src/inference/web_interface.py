#!/usr/bin/env python
# This script launches a Gradio web interface for comparing three language models:
# 1. A locally run open-source model (e.g., google/gemini-3-27b-it).
# 2. A fine-tuned model (PEFT adapter) specified by an experiment name.
# 3. A closed-source model via API (e.g., Gemini Flash), with API key from .env.
#
# It preloads the local models for fast inference and streams outputs to the UI.
#
# Usage:
# python -m src.inference.web_interface <your_experiment_name> \\
#                                     [--base_local_model_id google/gemma-3-27b-it] \\
#                                     [--gemini_model_id gemini-2.0-flash] \\
#                                     [--ui_password YOUR_PASSWORD] \\
#                                     [--prompt_name PROMPT_NAME]
#
# Ensure GEMINI_API_KEY is set in a .env file in the project root.
#
# Arguments:
#   your_experiment_name (str): Required. Name of the fine-tuned experiment.
#                               Model adapter is expected in 'experiments/[name]'.
#   --base_local_model_id (str): Optional. HF ID of the base local model.
#                                Default: "google/gemma-3-27b-it".
#   --gemini_model_id (str): Optional. Model ID for the Gemini API.
#                            Default: "gemini-2.0-flash".
#   --ui_password (str): Optional. Password to protect the Gradio interface.
#                        If not set, auth is disabled (or uses env var if set).
#   --prompt_name (str): Optional. Default system prompt to use. Default: None.
#   --device (str): Device for local models ('auto', 'cuda', 'cpu'). Default: 'auto'.
#   --max_new_tokens (int): Max new tokens for generation.
#   --temperature (float): Generation temperature. Default: 0.7.
#   --top_p (float): Top-p for sampling. Default: 0.9.


import os
import asyncio
import gradio as gr
import argparse
from pathlib import Path
import sys
from typing import AsyncGenerator, Tuple, Optional
from dotenv import load_dotenv  # Added for .env support

# Ensure src is in path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Assuming src/inference/web_interface.py
sys.path.append(str(PROJECT_ROOT))

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

try:
    from src.inference.inference import (
        preload_model_and_tokenizer,
        generate_text_yield_tokens,
        load_and_format_prompt_from_utils,  # Renamed from load_and_format_prompt for clarity
    )
    from src.inference.inference_utils import (
        _load_system_prompt_content,
    )  # Added import

    # For Gemini API
    from google import genai  # Updated import
    from google.genai import types  # Added for GenerateContentConfig
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(
        "Please ensure your environment is set up correctly and all dependencies are installed."
    )
    print(f"PROJECT_ROOT: {PROJECT_ROOT}, SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


# --- Globals for preloaded models and generation parameters ---
# These will be populated by command-line arguments
CLI_ARGS = None

# Preloaded local models and tokenizers
LOCAL_MODEL_1 = None
TOKENIZER_1 = None
LOCAL_MODEL_2_FINETUNED = None
TOKENIZER_2_FINETUNED = None

# Gemini API client
GEMINI_CLIENT = None

# Model types for prompt formatting
MODEL_TYPE_1 = None
MODEL_TYPE_2 = None


def initialize_models_and_client(args: argparse.Namespace):
    """Load local models and initialize Gemini client based on CLI args."""
    global LOCAL_MODEL_1, TOKENIZER_1, LOCAL_MODEL_2_FINETUNED, TOKENIZER_2_FINETUNED, GEMINI_CLIENT
    global CLI_ARGS, MODEL_TYPE_1, MODEL_TYPE_2
    CLI_ARGS = args

    print(f"Initializing Web UI with args: {args}")

    # 1. Load base local model (e.g., gemma-3-27b-it)
    if args.base_local_model_id:
        print(
            f"Loading base local model: {args.base_local_model_id} on device: {args.device}"
        )
        try:
            LOCAL_MODEL_1, TOKENIZER_1 = preload_model_and_tokenizer(
                args.base_local_model_id, device=args.device
            )
            MODEL_TYPE_1 = "chat"  # Assume all models are chat models

            # Debug: Verify this is NOT a PEFT model
            is_peft_model = hasattr(LOCAL_MODEL_1, "base_model")
            is_actual_peft_model = "peft" in str(type(LOCAL_MODEL_1)).lower()
            print(
                f"DEBUG: Base model '{args.base_local_model_id}' has base_model attr: {is_peft_model}"
            )
            print(
                f"DEBUG: Base model '{args.base_local_model_id}' is actual PEFT model: {is_actual_peft_model}"
            )
            if is_actual_peft_model:
                print(
                    "WARNING: Base model appears to be a PEFT model! This should not happen."
                )
            else:
                print(
                    "✅ Base model is correctly loaded as non-PEFT (may have PEFT infrastructure but not adapted)"
                )

            print(
                f"Successfully loaded base local model '{args.base_local_model_id}' (using chat model type)."
            )
        except Exception as e:
            print(f"Error loading base local model {args.base_local_model_id}: {e}")
            LOCAL_MODEL_1, TOKENIZER_1 = None, None
            MODEL_TYPE_1 = None
    else:
        print("No base_local_model_id provided. Skipping its load.")

    # 2. Load fine-tuned model
    if args.finetuned_experiment_name:
        finetuned_model_path = (
            PROJECT_ROOT / "experiments" / args.finetuned_experiment_name
        )
        print(
            f"Loading fine-tuned model from: {finetuned_model_path} on device: {args.device}"
        )
        try:
            # Reuse the base model and tokenizer to save memory
            if LOCAL_MODEL_1 and TOKENIZER_1:
                print(
                    "DEBUG: Reusing base model and tokenizer for PEFT adapter to save memory"
                )
                LOCAL_MODEL_2_FINETUNED, TOKENIZER_2_FINETUNED = (
                    preload_model_and_tokenizer(
                        str(finetuned_model_path),
                        device=args.device,
                        base_model_to_reuse=LOCAL_MODEL_1,
                        tokenizer_to_reuse=TOKENIZER_1,
                    )
                )
            else:
                print("DEBUG: Loading PEFT model without base model reuse")
                LOCAL_MODEL_2_FINETUNED, TOKENIZER_2_FINETUNED = (
                    preload_model_and_tokenizer(
                        str(finetuned_model_path),
                        device=args.device,
                    )
                )
            MODEL_TYPE_2 = "chat"  # Assume all models are chat models

            # Debug: Verify this IS a PEFT model
            is_peft_model = hasattr(LOCAL_MODEL_2_FINETUNED, "base_model")
            is_actual_peft_model = "peft" in str(type(LOCAL_MODEL_2_FINETUNED)).lower()
            print(
                f"DEBUG: Fine-tuned model '{args.finetuned_experiment_name}' has base_model attr: {is_peft_model}"
            )
            print(
                f"DEBUG: Fine-tuned model '{args.finetuned_experiment_name}' is actual PEFT model: {is_actual_peft_model}"
            )
            if not is_actual_peft_model:
                print(
                    "WARNING: Fine-tuned model does not appear to be a PEFT model! This might be unexpected."
                )
            else:
                print("✅ Fine-tuned model is correctly loaded as PEFT")

            print(
                f"Successfully loaded fine-tuned model '{args.finetuned_experiment_name}' (using chat model type)."
            )
        except Exception as e:
            print(
                f"Error loading fine-tuned model {args.finetuned_experiment_name} from {finetuned_model_path}: {e}"
            )
            LOCAL_MODEL_2_FINETUNED, TOKENIZER_2_FINETUNED = None, None
            MODEL_TYPE_2 = None
    else:
        print("No finetuned_experiment_name provided. Skipping its load.")

    # 3. Initialize Gemini API client
    gemini_api_key_env = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key_env:
        print("Initializing Gemini API client using GEMINI_API_KEY from environment...")
        try:
            GEMINI_CLIENT = genai.Client(
                api_key=gemini_api_key_env
            )  # New: create client instance
            print("Gemini API client initialized.")
        except Exception as e:
            print(f"Error initializing Gemini API client: {e}")
            GEMINI_CLIENT = None
    else:
        print(
            "GEMINI_API_KEY not found in environment. Gemini API model will not be available."
        )

    # Clean up GPU memory after model loading
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("DEBUG: Cleared GPU cache after model loading")
    except Exception as e:
        print(f"DEBUG: Could not clear GPU cache: {e}")


async def run_local_model_stream(
    model, tokenizer, full_prompt: str
) -> AsyncGenerator[str, None]:
    """Helper to stream from a local model using generate_text_yield_tokens."""
    if not model or not tokenizer:
        yield "[Model not loaded or error during initialization]"
        return

    # Add debugging information
    print(f"DEBUG: Starting generation with model on device: {model.device}")
    print(f"DEBUG: Model type: {type(model)}")
    print(f"DEBUG: Prompt length: {len(full_prompt)} characters")

    # LOG THE FULL PROMPT BEFORE INFERENCE
    print("=" * 80)
    print("PROMPT LOGGING - LOCAL MODEL")
    print(f"Model: {type(model)}")
    print(f"Is PEFT: {'peft' in str(type(model)).lower()}")
    print(f"Prompt length: {len(full_prompt)} characters")
    print("FULL PROMPT CONTENT:")
    print("-" * 40)
    print(repr(full_prompt))  # Using repr to show escape characters
    print("-" * 40)
    print("FULL PROMPT CONTENT (readable):")
    print(full_prompt)
    print("=" * 80)

    # Check if PEFT model and print adapter info
    if hasattr(model, "base_model"):
        print(
            f"DEBUG: PEFT model detected, base model device: {model.base_model.device}"
        )
        if hasattr(model, "active_adapters"):
            print(f"DEBUG: Active adapters: {model.active_adapters}")

        # Ensure PEFT model is in eval mode
        model.eval()
        if hasattr(model.base_model, "eval"):
            model.base_model.eval()
        print(f"DEBUG: Set PEFT model to eval mode")
    else:
        # Ensure regular model is in eval mode
        model.eval()
        print(f"DEBUG: Set model to eval mode")

    try:
        token_count = 0
        async for token in generate_text_yield_tokens(
            prompt=full_prompt,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=CLI_ARGS.max_new_tokens,
            temperature=CLI_ARGS.temperature,
            top_p=CLI_ARGS.top_p,
        ):
            token_count += 1
            if token_count == 1:
                print(f"DEBUG: First token received from model")
            elif token_count % 10 == 0:
                print(f"DEBUG: Received {token_count} tokens so far")
            yield token
        print(f"DEBUG: Generation completed, total tokens: {token_count}")
    except Exception as e:
        print(f"Error during local model streaming: {e}")
        import traceback

        traceback.print_exc()
        yield f"[Error generating from local model: {e}]"


async def run_gemini_api_stream(
    prompt_content: str,  # Just the user prompt
) -> AsyncGenerator[str, None]:
    """Helper to stream from Gemini API, using system_instruction if available."""
    if not GEMINI_CLIENT:  # GEMINI_CLIENT is now an instance of genai.Client
        yield "[Gemini API client not initialized or API key missing]"
        return

    # Helper function to run blocking iterator next() in a thread
    # and handle StopIteration by returning a sentinel (None).
    def _blocking_next_chunk(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return None  # Sentinel to indicate iterator exhaustion

    try:
        # Prepare parameters for types.GenerateContentConfig
        config_params = {
            "max_output_tokens": CLI_ARGS.max_new_tokens,
            "temperature": CLI_ARGS.temperature,
            "top_p": CLI_ARGS.top_p,
        }

        system_prompt_text = None
        if CLI_ARGS.prompt_name:
            system_prompt_text = _load_system_prompt_content(
                prompt_name=CLI_ARGS.prompt_name,
                prompts_dir="prompts",
            )

        if system_prompt_text and system_prompt_text.strip():
            config_params["system_instruction"] = system_prompt_text
            print(
                f"Using system instruction for Gemini API: '{system_prompt_text[:100]}...'"
            )

        # LOG THE FULL PROMPT BEFORE INFERENCE
        print("=" * 80)
        print("PROMPT LOGGING - GEMINI API MODEL")
        print(f"Model: {CLI_ARGS.gemini_model_id}")
        print(f"User prompt length: {len(prompt_content)} characters")
        print(
            f"System prompt length: {len(system_prompt_text) if system_prompt_text else 0} characters"
        )
        print("SYSTEM PROMPT CONTENT:")
        print("-" * 40)
        if system_prompt_text:
            print(repr(system_prompt_text))  # Using repr to show escape characters
            print("-" * 40)
            print("SYSTEM PROMPT CONTENT (readable):")
            print(system_prompt_text)
        else:
            print("No system prompt")
        print("-" * 40)
        print("USER PROMPT CONTENT:")
        print("-" * 40)
        print(repr(prompt_content))  # Using repr to show escape characters
        print("-" * 40)
        print("USER PROMPT CONTENT (readable):")
        print(prompt_content)
        print("=" * 80)

        final_config = types.GenerateContentConfig(**config_params)

        sync_iterator = GEMINI_CLIENT.models.generate_content_stream(
            model=CLI_ARGS.gemini_model_id,
            contents=[prompt_content],
            config=final_config,
        )

        while True:
            try:
                # Run the blocking _blocking_next_chunk helper in a separate thread.
                # This helper catches StopIteration and returns None.
                chunk = await asyncio.to_thread(_blocking_next_chunk, sync_iterator)

                if chunk is None:  # Sentinel received, iterator is exhausted.
                    break

                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                # Some stream parts might be empty responses or metadata,
                # so we only yield if there's actual text.

            except Exception as e_inner:
                # Catch other errors during the iteration (e.g., from chunk processing if any)
                error_message = f"[Error processing Gemini API stream chunk: {e_inner}]"
                print(error_message)
                yield error_message
                break  # Stop trying to process further chunks on error

            await asyncio.sleep(0.01)  # Yield control to allow other tasks to run

    except Exception as e:
        # Catch errors from the initial API call or setup
        error_message = f"[Error generating from Gemini API: {e}]"
        # Keep original print for server log, yield error_message for UI
        print(f"Error during Gemini API streaming: {e}")
        yield error_message


async def compare_models_streamed(
    user_prompt: str,
) -> AsyncGenerator[Tuple[Optional[str], Optional[str], Optional[str]], None]:
    """
    Handles streaming generation for all three models.
    The API model and the base local model stream concurrently from the start.
    The fine-tuned local model starts streaming only after the base local model has finished.
    Yields updates for Gradio textboxes.
    """

    text_outputs = ["", "", ""]
    formatted_prompt_local_model1 = None
    formatted_prompt_local_model2 = None

    # LOG THE ORIGINAL USER PROMPT
    print("=" * 80)
    print("PROMPT FORMATTING - ORIGINAL USER INPUT")
    print(f"User prompt length: {len(user_prompt)} characters")
    print("ORIGINAL USER PROMPT:")
    print("-" * 40)
    print(repr(user_prompt))  # Using repr to show escape characters
    print("-" * 40)
    print("ORIGINAL USER PROMPT (readable):")
    print(user_prompt)
    print("=" * 80)

    if TOKENIZER_1 and MODEL_TYPE_1:
        formatted_prompt_local_model1 = load_and_format_prompt_from_utils(
            user_input=user_prompt,
            tokenizer=TOKENIZER_1,
            model_type=MODEL_TYPE_1,
            prompt_name=CLI_ARGS.prompt_name,
        )
        # LOG FORMATTED PROMPT FOR MODEL 1
        print("=" * 80)
        print("PROMPT FORMATTING - BASE LOCAL MODEL")
        print(f"Model type: {MODEL_TYPE_1}")
        print(f"Prompt name: {CLI_ARGS.prompt_name}")
        print(
            f"Formatted prompt length: {len(formatted_prompt_local_model1)} characters"
        )
        print("FORMATTED PROMPT FOR BASE LOCAL MODEL:")
        print("-" * 40)
        print(
            repr(formatted_prompt_local_model1)
        )  # Using repr to show escape characters
        print("-" * 40)
        print("FORMATTED PROMPT FOR BASE LOCAL MODEL (readable):")
        print(formatted_prompt_local_model1)
        print("=" * 80)
    else:
        formatted_prompt_local_model1 = user_prompt
        print("=" * 80)
        print("PROMPT FORMATTING - BASE LOCAL MODEL (FALLBACK)")
        print("Using unformatted user prompt as fallback")
        print("=" * 80)

    if TOKENIZER_2_FINETUNED and MODEL_TYPE_2:
        formatted_prompt_local_model2 = load_and_format_prompt_from_utils(
            user_input=user_prompt,
            tokenizer=TOKENIZER_2_FINETUNED,
            model_type=MODEL_TYPE_2,
            prompt_name=CLI_ARGS.prompt_name,
        )
        # LOG FORMATTED PROMPT FOR MODEL 2
        print("=" * 80)
        print("PROMPT FORMATTING - FINE-TUNED LOCAL MODEL")
        print(f"Model type: {MODEL_TYPE_2}")
        print(f"Prompt name: {CLI_ARGS.prompt_name}")
        print(
            f"Formatted prompt length: {len(formatted_prompt_local_model2)} characters"
        )
        print("FORMATTED PROMPT FOR FINE-TUNED LOCAL MODEL:")
        print("-" * 40)
        print(
            repr(formatted_prompt_local_model2)
        )  # Using repr to show escape characters
        print("-" * 40)
        print("FORMATTED PROMPT FOR FINE-TUNED LOCAL MODEL (readable):")
        print(formatted_prompt_local_model2)
        print("=" * 80)
    else:
        formatted_prompt_local_model2 = user_prompt
        print("=" * 80)
        print("PROMPT FORMATTING - FINE-TUNED LOCAL MODEL (FALLBACK)")
        print("Using unformatted user prompt as fallback")
        print("=" * 80)

    # Debug logging to verify model separation
    print(f"DEBUG: Base model type: {type(LOCAL_MODEL_1)}")
    print(f"DEBUG: Fine-tuned model type: {type(LOCAL_MODEL_2_FINETUNED)}")
    if LOCAL_MODEL_1:
        is_actual_peft_1 = "peft" in str(type(LOCAL_MODEL_1)).lower()
        print(f"DEBUG: Base model is actual PEFT: {is_actual_peft_1}")
    if LOCAL_MODEL_2_FINETUNED:
        is_actual_peft_2 = "peft" in str(type(LOCAL_MODEL_2_FINETUNED)).lower()
        print(f"DEBUG: Fine-tuned model is actual PEFT: {is_actual_peft_2}")

    stream1 = run_local_model_stream(
        LOCAL_MODEL_1, TOKENIZER_1, formatted_prompt_local_model1
    )
    # stream2 will be initialized later
    stream3 = run_gemini_api_stream(user_prompt)

    active_streams = [
        True,
        False,
        True,
    ]  # stream1 (base local), stream2 (FT local - initially false), stream3 (API)
    stream2_initialized = False
    stream2_iterator = None  # Initialize to None

    # Memory monitoring
    try:
        import torch

        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
            print(f"DEBUG: GPU memory before generation: {memory_before:.2f} GB")
    except:
        pass

    async def get_next_or_none(stream_iter):
        try:
            return await stream_iter.__anext__()
        except StopAsyncIteration:
            return None

    while any(active_streams):
        tasks = []

        # Task for stream1 (base local model)
        if active_streams[0]:
            tasks.append(get_next_or_none(stream1))
        else:
            tasks.append(
                asyncio.sleep(0, result=None)
            )  # Placeholder if stream1 is done

        # Task for stream2 (fine-tuned local model)
        if (
            active_streams[1] and stream2_iterator
        ):  # Only if it has been activated AND initialized
            tasks.append(get_next_or_none(stream2_iterator))
        else:
            tasks.append(
                asyncio.sleep(0, result=None)
            )  # Placeholder if stream2 is not active or done

        # Task for stream3 (API model)
        if active_streams[2]:
            tasks.append(get_next_or_none(stream3))
        else:
            tasks.append(
                asyncio.sleep(0, result=None)
            )  # Placeholder if stream3 is done

        results = await asyncio.gather(*tasks, return_exceptions=True)

        token1, token2, token3 = None, None, None

        # Process stream1 (base local)
        if active_streams[0]:
            if isinstance(results[0], Exception) or results[0] is None:
                active_streams[0] = False  # Mark stream1 as done
                print("DEBUG: Base local model stream finished")
                if isinstance(results[0], Exception):
                    text_outputs[0] += f" [Stream Error: {results[0]}]"
            else:
                text_outputs[0] += results[0]
                token1 = text_outputs[0]

        # Process stream2 (fine-tuned local)
        if active_streams[1]:
            if isinstance(results[1], Exception) or results[1] is None:
                active_streams[1] = False  # Mark stream2 as done
                print("DEBUG: Fine-tuned local model stream finished")
                if isinstance(results[1], Exception):
                    text_outputs[1] += f" [Stream Error: {results[1]}]"
            else:
                text_outputs[1] += results[1]
                token2 = text_outputs[1]

        # Process stream3 (API model)
        if active_streams[2]:
            if isinstance(results[2], Exception) or results[2] is None:
                active_streams[2] = False  # Mark stream3 as done
                if isinstance(results[2], Exception):
                    text_outputs[2] += f" [Stream Error: {results[2]}]"
            else:
                text_outputs[2] += results[2]
                token3 = text_outputs[2]

        # Check if stream1 has finished, and if so, initialize and activate stream2
        if (
            not active_streams[0]
            and not stream2_initialized
            and LOCAL_MODEL_2_FINETUNED
            and TOKENIZER_2_FINETUNED
        ):
            print(
                "DEBUG: Base local model finished. Starting fine-tuned local model generation..."
            )

            # Clear GPU cache before starting fine-tuned model
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    memory_after_base = torch.cuda.memory_allocated() / 1024**3  # GB
                    print(
                        f"DEBUG: GPU memory after base model: {memory_after_base:.2f} GB"
                    )
            except:
                pass

            stream2_iterator = run_local_model_stream(
                LOCAL_MODEL_2_FINETUNED,
                TOKENIZER_2_FINETUNED,
                formatted_prompt_local_model2,
            )
            active_streams[1] = True  # Activate stream2
            stream2_initialized = True
            print("DEBUG: Fine-tuned model stream initialized and activated")
        elif (
            not active_streams[0]
            and not stream2_initialized
            and (not LOCAL_MODEL_2_FINETUNED or not TOKENIZER_2_FINETUNED)
        ):
            # If FT model isn't available, ensure it's marked as done and never starts
            active_streams[1] = False
            stream2_initialized = True  # Prevent re-evaluation
            if not text_outputs[1]:  # Add a message if it never ran
                text_outputs[1] = "[Fine-tuned model not available or not loaded]"
                token2 = text_outputs[1]  # Ensure this message is yielded
            print("DEBUG: Fine-tuned model not available, marked as done")

        # Always yield the full current state of all text outputs
        yield (text_outputs[0], text_outputs[1], text_outputs[2])

        if not any(active_streams):  # All streams finished
            print("DEBUG: All streams finished")
            break

        await asyncio.sleep(0)


def launch_gradio_interface():
    """Sets up and launches the Gradio UI."""
    global CLI_ARGS  # Access the parsed CLI arguments

    auth_config = None
    if CLI_ARGS.ui_password:
        auth_config = ("user", CLI_ARGS.ui_password)  # Simple username "user"
        print(f"Gradio UI will be password protected. User: 'user'")
    elif os.environ.get(
        "UI_PASSWORD"
    ):  # Fallback to env variable if --ui_password not set
        auth_config = ("user", os.environ["UI_PASSWORD"])
        print(
            f"Gradio UI will be password protected using UI_PASSWORD env var. User: 'user'"
        )
    else:
        print("Gradio UI will not be password protected.")

    with gr.Blocks(
        title="Multi-Model Comparison Interface", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown("# Language Model Comparison")
        gr.Markdown(
            "Enter a prompt to see responses from a base local model, a fine-tuned local model, and a Gemini API model."
        )

        with gr.Accordion("Model Configuration:", open=False):
            gr.Markdown(
                f"""
            - **Base Local Model:** {CLI_ARGS.base_local_model_id if LOCAL_MODEL_1 else 'Not Loaded'}
            - **Fine-tuned Model:** {CLI_ARGS.finetuned_experiment_name if LOCAL_MODEL_2_FINETUNED else 'Not Loaded'}
            - **API Model:** {CLI_ARGS.gemini_model_id} (via API Key {'Provided' if GEMINI_CLIENT else 'Not Provided'})
            - **System Prompt Template:** {CLI_ARGS.prompt_name}
            """
            )

        prompt_textbox = gr.Textbox(
            label="Your Prompt",
            lines=4,
            autofocus=True,
            placeholder="Type your query here...",
        )

        with gr.Row():
            output_local_base = gr.Textbox(
                label=f"Local: {Path(CLI_ARGS.base_local_model_id).name if CLI_ARGS.base_local_model_id else 'Base Model'}",
                lines=15,
                interactive=False,
            )
            output_finetuned = gr.Textbox(
                label=f"Local FT: {CLI_ARGS.finetuned_experiment_name if CLI_ARGS.finetuned_experiment_name else 'Fine-tuned'}",
                lines=15,
                interactive=False,
            )
            output_gemini_api = gr.Textbox(
                label=f"Gemini API ({CLI_ARGS.gemini_model_id})",
                lines=15,
                interactive=False,
            )

        submit_button = gr.Button("Generate Responses")

        # Connect the button to the streaming comparison function
        submit_button.click(
            fn=compare_models_streamed,
            inputs=[prompt_textbox],
            outputs=[output_local_base, output_finetuned, output_gemini_api],
        )
        prompt_textbox.submit(  # Also allow submitting with Enter key
            fn=compare_models_streamed,
            inputs=[prompt_textbox],
            outputs=[output_local_base, output_finetuned, output_gemini_api],
        )

    print("Launching Gradio interface...")
    demo.queue()  # Allow multiple requests, though streaming handles concurrency per request

    # Launch the demo
    # Need to handle share=True potentially being problematic in some environments
    # Server name and port can also be made configurable
    demo.launch(
        share=CLI_ARGS.share_gradio,  # Make this a CLI arg
        auth=auth_config,
        server_name="0.0.0.0",  # Listen on all interfaces
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gradio Web Interface for Model Comparison"
    )
    parser.add_argument(
        "finetuned_experiment_name",
        type=str,
        help="Name of the fine-tuned experiment (e.g., 'my_adam_clone_run'). Model expected in 'experiments/[name]'.",
    )
    parser.add_argument(
        "--base_local_model_id",
        type=str,
        default="google/gemma-3-27b-it",
        help="Hugging Face ID of the base local model to run (e.g., 'mistralai/Mistral-7B-Instruct-v0.2'). Default: 'google/gemma-3-27b-it'.",
    )
    parser.add_argument(
        "--gemini_model_id",
        type=str,
        default="gemini-2.0-flash",
        help="Model ID for the Gemini API (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro-latest'). Default: 'gemini-2.0-flash'.",
    )
    parser.add_argument(
        "--ui_password",
        type=str,
        default=None,  # Default to no password, but can check env var
        help="Password to protect the Gradio UI. If not set, checks UI_PASSWORD env var or disables auth.",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Name of the system prompt file from 'prompts' dir. Default: None.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for local models (default: 'auto').",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens for generation (default: 256).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7).",
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p for sampling (default: 0.9)."
    )
    parser.add_argument(
        "--share_gradio",
        action="store_true",  # Makes it a flag, presence means True
        help="Enable Gradio sharing (creates a public link). Use with caution.",
    )

    args = parser.parse_args()

    # Initialize models and client first
    initialize_models_and_client(args)

    # Then launch the Gradio UI
    launch_gradio_interface()
