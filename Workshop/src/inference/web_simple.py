#!/usr/bin/env python
# This script launches a simplified Gradio web interface for interacting with a single
# locally run open-source language model (e.g., google/gemma-3-27b-it).
#
# It preloads the local model for fast inference and streams outputs to the UI.
#
# Usage:
# python -m src.inference.web_simple \
#                                     [--local_model_id google/gemma-3-27b-it] \
#                                     [--prompt_name PROMPT_NAME] \
#                                     [--device auto] \
#                                     [--max_new_tokens 4096] \
#                                     [--temperature 0.7] \
#                                     [--top_p 0.9] \
#                                     [--share_gradio]
#
# Arguments:
#   --local_model_id (str): Optional. HF ID of the local model.
#                             Default: "google/gemma-3-27b-it".
#   --prompt_name (str): Optional. Default system prompt to use. Default: None.
#   --device (str): Device for local models ('auto', 'cuda', 'cpu'). Default: 'auto'.
#   --max_new_tokens (int): Max new tokens for generation. Default: 4096.
#   --temperature (float): Generation temperature. Default: 0.7.
#   --top_p (float): Top-p for sampling. Default: 0.9.
#   --share_gradio (bool): Optional. Enable Gradio sharing (creates a public link). Default: False.

import os
import asyncio
import gradio as gr
import argparse
from pathlib import Path
import sys
from typing import AsyncGenerator, Optional

# Ensure src is in path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from src.inference.inference import (
        preload_model_and_tokenizer,
        generate_text_yield_tokens,
        load_and_format_prompt_from_utils,
    )
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print(
        "Please ensure your environment is set up correctly and all dependencies are installed."
    )
    sys.exit(1)

# --- Globals for preloaded model and generation parameters ---
CLI_ARGS = None
LOCAL_MODEL = None
TOKENIZER = None


def initialize_model(args: argparse.Namespace):
    """Load local model based on CLI args."""
    global LOCAL_MODEL, TOKENIZER, CLI_ARGS
    CLI_ARGS = args

    print(f"Initializing Web UI with args: {args}")

    if args.local_model_id:
        print(f"Loading local model: {args.local_model_id} on device: {args.device}")
        try:
            LOCAL_MODEL, TOKENIZER = preload_model_and_tokenizer(
                args.local_model_id, device=args.device
            )
            print(f"Successfully loaded local model '{args.local_model_id}'.")
        except Exception as e:
            print(f"Error loading local model {args.local_model_id}: {e}")
            LOCAL_MODEL, TOKENIZER = None, None
    else:
        print("No local_model_id provided. Cannot proceed.")
        sys.exit(1)


async def run_model_stream(
    user_prompt: str,
) -> AsyncGenerator[str, None]:
    """Helper to stream from the local model."""
    if not LOCAL_MODEL or not TOKENIZER:
        yield "[Model not loaded or error during initialization]"
        return

    formatted_prompt = load_and_format_prompt_from_utils(
        user_input=user_prompt,
        prompt_name=CLI_ARGS.prompt_name,
        tokenizer=TOKENIZER,
    )

    current_text = ""  # Initialize an accumulator string
    try:
        async for token in generate_text_yield_tokens(
            prompt=formatted_prompt,
            model=LOCAL_MODEL,
            tokenizer=TOKENIZER,
            max_new_tokens=CLI_ARGS.max_new_tokens,
            temperature=CLI_ARGS.temperature,
            top_p=CLI_ARGS.top_p,
        ):
            current_text += token  # Append the new token
            yield current_text  # Yield the entire accumulated text
            await asyncio.sleep(
                0
            )  # Allow event loop to process other tasks (e.g., UI updates)
    except Exception as e:
        print(f"Error during model streaming: {e}")
        # Optionally, display the error in the UI as well
        error_message = f" [Error generating from model: {e}]"
        if not current_text:  # If error happened before any output
            yield error_message
        else:  # Append error to existing text
            current_text += error_message
            yield current_text


def launch_gradio_interface():
    """Sets up and launches the Gradio UI."""
    global CLI_ARGS

    with gr.Blocks(title="Simple Model Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Simple Language Model Interaction")
        gr.Markdown(f"Interact with the **{CLI_ARGS.local_model_id}** model.")

        with gr.Accordion("Model Configuration:", open=False):
            gr.Markdown(
                f"""
            - **Local Model:** {CLI_ARGS.local_model_id if LOCAL_MODEL else 'Not Loaded'}
            - **System Prompt Template:** {CLI_ARGS.prompt_name if CLI_ARGS.prompt_name else 'None'}
            """
            )

        prompt_textbox = gr.Textbox(
            label="Your Prompt",
            lines=4,
            autofocus=True,
            placeholder="Type your query here...",
        )

        output_textbox = gr.Textbox(
            label=f"Model: {Path(CLI_ARGS.local_model_id).name if CLI_ARGS.local_model_id else 'Output'}",
            lines=15,
            interactive=False,
        )

        submit_button = gr.Button("Generate Response")

        submit_button.click(
            fn=run_model_stream,
            inputs=[prompt_textbox],
            outputs=[output_textbox],
        )
        prompt_textbox.submit(
            fn=run_model_stream,
            inputs=[prompt_textbox],
            outputs=[output_textbox],
        )

    print("Launching Gradio interface...")
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        share=CLI_ARGS.share_gradio,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplified Gradio Web Interface for a Single Model"
    )
    parser.add_argument(
        "--local_model_id",
        type=str,
        default="google/gemma-3-27b-it",
        help="Hugging Face ID of the local model to run. Default: 'google/gemma-3-27b-it'.",
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
        default=4096,
        help="Max new tokens for generation (default: 4096).",
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
        action="store_true",
        help="Enable Gradio sharing (creates a public link). Use with caution.",
    )

    args = parser.parse_args()

    initialize_model(args)
    launch_gradio_interface()
