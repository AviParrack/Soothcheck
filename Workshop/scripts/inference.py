#!/usr/bin/env python
# This script is the command-line interface for running inference with language models.
# It handles:
# - Parsing command-line arguments for model selection, model configuration, prompt, and generation parameters.
# - Loading the specified model configuration from `configs/model/`. If --model is not given,
#   it's inferred from the `model_identifier` positional argument.
# - Determining the model type (e.g., "chat" or "text_generation") from the model config.
# - Loading and preloading the specified model using `src.inference.inference`.
# - Running in either interactive chat mode or single prompt mode, respecting the model_type.
#
# Example usage (interactive mode, inferring config name):
# python -m scripts.inference gemma-3-27b-it
# (Assumes configs/model/gemma-3-27b-it.json exists; loads model from that config's model_name_or_path)
#
# Example usage (interactive mode with an experiment, inferring config name):
# python -m scripts.inference my_experiment_name
# (Assumes configs/model/my_experiment_name.json exists; loads experiments/my_experiment_name)
#
# Example usage (single prompt with config name):
# python -m scripts.inference gemma-3-27b-pt --prompt "Translate this"
# (Uses configs/model/gemma-3-27b-pt.json)
#
# Example usage (explicitly setting config name for an experiment):
# python -m scripts.inference my_experiment_name --model gemma-3-27b-it
#
# Arguments:
#   model_identifier (str): Positional. Identifier for the model to load. Can be:
#                             - The name of an experiment (e.g., 'my_model_adapter'), expected in 'experiments/[name]'.
#                             - A name that matches a model config in `configs/model/` (e.g., 'gemma-3-27b-it').
#   --model (str): Optional. Name of the model config file (without .json) from `configs/model/`.
#                              If not provided, the script attempts to infer it from `model_identifier`.
#   --model_path (str): Optional. Direct path or Hugging Face model ID. If provided, this overrides
#                       the model path derived from `model_identifier` or the model config.
#   --prompt_name (str): Optional. Name of the system prompt file from the 'prompts' directory. Default: None.
#                        Only used if model_type from model is "chat".
#   --device (str): Optional. Device for inference ('auto', 'cpu', 'cuda'). Default: 'auto'.
#   --prompt (str): Optional. If provided, runs in single prompt mode with this text.
#                   Otherwise, runs in interactive mode.
#   --max_length (int): Optional. Maximum new tokens to generate. Default: 4096.
#   --temperature (float): Optional. Generation temperature. Default: 0.7.
#   --top_p (float): Optional. Top-p for sampling. Default: 0.9.

import argparse
import os
import sys
from pathlib import Path
import json

# Ensure src is in path for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.inference import (
    preload_model_and_tokenizer,
    interactive_mode,
    interactive_chat_mode,
    single_prompt_mode,
)
from src.utils.model_resolver import resolve_model_specification

# from src.inference.inference_utils import load_model # Not directly used in main


def main():
    parser = argparse.ArgumentParser(
        description="Command-line interface for language model inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Interactive mode (config inferred from identifier 'gemma-3-27b-it'):\n"
            "    python -m scripts.inference gemma-3-27b-it\n"
            "  Interactive chat mode with history:\n"
            "    python -m scripts.inference gemma-3-27b-it --chat\n"
            "  Interactive mode (experiment 'my_exp', config 'my_exp.json' inferred):\n"
            "    python -m scripts.inference my_exp --prompt_name some_chat_prompt\n"
            "  Single prompt with config name:\n"
            """    python -m scripts.inference gemma-3-27b-pt --prompt "Translate this to French: Hello"\n"""
            "  Explicit config for an experiment:\n"
            '''    python -m scripts.inference my_exp --model base_model_config --prompt "Hi"'''
        ),
    )
    parser.add_argument(
        "model_identifier",
        type=str,
        help="Identifier for the model (experiment name or config name). "
        "Used to infer --model if not provided, and to determine the model to load.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,  # Now optional
        help="Optional: Name of the model configuration file (e.g., 'gemma-3-27b-pt') from 'configs/model/'. "
        "If not provided, attempts to infer from `model_identifier`.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional: Explicit path to the model or Hugging Face model ID. Overrides model path derived from "
        "`model_identifier` or its configuration.",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default=None,
        help="Optional: Name of the system prompt file (without .md) from the 'prompts' directory. "
        "Only relevant for 'chat' model types.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="A single prompt to generate a response for. If not provided, interactive mode is used.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter for text generation.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Enable chat mode with conversation history in interactive mode.",
    )

    args = parser.parse_args()

    # --- Determine Model Configuration and Path Using Unified Resolution ---
    model_spec = args.model_identifier if not args.model else args.model
    override_model_path = args.model_path

    print(f"Resolving model specification: {model_spec}")
    try:
        model_resolution = resolve_model_specification(
            model_spec, project_root=PROJECT_ROOT
        )

        model_type = model_resolution.model_type
        model_config_data = model_resolution.model_config.model_dump()

        # Override with explicit model path if provided
        if override_model_path:
            actual_model_path_to_load = override_model_path
            print(f"Using explicit --model_path override: {actual_model_path_to_load}")
        else:
            # Use PEFT adapter path if available, otherwise base model
            actual_model_path_to_load = (
                model_resolution.peft_adapter_path or model_resolution.base_model_hf_id
            )

        print(f"✓ Resolved model specification '{model_spec}':")
        print(
            f"  - Model config source: {'experiment' if model_resolution.is_experiment else 'config'}"
        )
        print(f"  - Model type: {model_type}")
        print(f"  - Base model HF ID: {model_resolution.base_model_hf_id}")
        if model_resolution.peft_adapter_path:
            print(f"  - PEFT adapter path: {model_resolution.peft_adapter_path}")
        print(f"  - Model path to load: {actual_model_path_to_load}")
        if model_resolution.experiment_path:
            print(f"  - Experiment path: {model_resolution.experiment_path}")

    except ValueError as e:
        print(f"Error: Could not resolve model specification '{model_spec}'")
        print(f"Details: {e}")
        sys.exit(1)

    # --- Preload Model and Tokenizer ---
    print(
        f"Attempting to preload model using resolved specification on device: {args.device}"
    )
    try:
        # Create a temporary specification that preload_model_and_tokenizer can resolve
        # We'll pass the original model_spec since preload_model_and_tokenizer now handles resolution
        model, tokenizer = preload_model_and_tokenizer(model_spec, device=args.device)
    except Exception as e:
        print(
            f"Fatal Error: Could not preload model from '{model_spec}'. Exception: {e}"
        )
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("Model and tokenizer preloaded successfully.")

    # --- Run Inference ---
    if args.prompt:
        single_prompt_mode(
            prompt_text=args.prompt,
            model=model,
            tokenizer=tokenizer,
            prompt_name_arg=args.prompt_name,
            max_length_arg=args.max_length,
            temperature_arg=args.temperature,
            top_p_arg=args.top_p,
            model_type=model_type,
        )
    else:
        if args.chat:
            interactive_chat_mode(
                model=model,
                tokenizer=tokenizer,
                prompt_name_arg=args.prompt_name,
                max_length_arg=args.max_length,
                temperature_arg=args.temperature,
                top_p_arg=args.top_p,
                model_type=model_type,
            )
        else:
            interactive_mode(
                model=model,
                tokenizer=tokenizer,
                prompt_name_arg=args.prompt_name,
                max_length_arg=args.max_length,
                temperature_arg=args.temperature,
                top_p_arg=args.top_p,
                model_type=model_type,
            )


if __name__ == "__main__":
    main()
