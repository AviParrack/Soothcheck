#!/usr/bin/env python
# This file contains utility functions for the inference pipelines.
# Key functions:
# - load_model: Loads PEFT-adapted models from local paths or base Hugging Face models.
#               Can reuse an already loaded base model for PEFT adapters.
# - load_and_format_prompt: Loads an optional system prompt and formats it with user input.
#                           For "chat" models, it uses the model's chat template via the tokenizer.
#                           For "text_generation" models, it prepends BOS if necessary and uses the raw prompt.
# These utilities are designed to be used by both command-line and web-based inference interfaces.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pathlib import Path
import os
import warnings
from ..utils.template_manager import TemplateManager
from typing import List


def load_model(
    base_model_hf_id: str,
    peft_adapter_path: str | None = None,
    device: str | None = None,
    base_model_to_reuse: AutoModelForCausalLM | None = None,
    tokenizer_to_reuse: AutoTokenizer | None = None,
    quantization_config: dict | None = None,
):
    """
    Load a model and tokenizer with clean separation of base model and PEFT adapter.

    Args:
        base_model_hf_id: HuggingFace model identifier for the base model.
        peft_adapter_path: Optional path to PEFT adapter directory.
        device: Device to load the model on ('cpu', 'cuda', 'auto').
        base_model_to_reuse: Optional. An already loaded base model to use for PEFT adapters.
        tokenizer_to_reuse: Optional. An already loaded tokenizer to use with the PEFT adapter's base.
        quantization_config: Optional quantization configuration dict.

    Returns:
        model: The loaded model.
        tokenizer: The tokenizer for the model.

    Raises:
        ValueError: If the model cannot be loaded.
    """
    if device is None or device == "auto":
        device_map = "auto"
    else:
        device_map = device

    try:
        # Prepare quantization config if provided
        quantization_config_obj = None
        if quantization_config:
            from transformers import BitsAndBytesConfig

            print(f"  → Applying quantization config")
            print(f"    4-bit: {quantization_config.get('load_in_4bit', False)}")
            print(f"    8-bit: {quantization_config.get('load_in_8bit', False)}")

            # Create BitsAndBytesConfig with proper dtype handling
            bnb_config_params = quantization_config.copy()

            # Convert compute dtype string to torch dtype
            if "bnb_4bit_compute_dtype" in bnb_config_params:
                compute_dtype_str = bnb_config_params["bnb_4bit_compute_dtype"]
                dtype_map = {
                    "bfloat16": torch.bfloat16,
                    "float16": torch.float16,
                    "float32": torch.float32,
                }
                bnb_config_params["bnb_4bit_compute_dtype"] = dtype_map.get(
                    compute_dtype_str, torch.bfloat16
                )

            quantization_config_obj = BitsAndBytesConfig(**bnb_config_params)

        if peft_adapter_path:
            print(f"\n→ Loading PEFT-adapted model")
            print(f"  PEFT adapter directory: {peft_adapter_path}")
            adapter_path_obj = Path(peft_adapter_path)

            if not (
                adapter_path_obj.is_dir()
                and (adapter_path_obj / "adapter_config.json").exists()
            ):
                raise ValueError(
                    f"PEFT adapter path '{peft_adapter_path}' is not a valid adapter directory"
                )

            config = PeftConfig.from_pretrained(peft_adapter_path)
            print(f"  Base model from PEFT config: {config.base_model_name_or_path}")
            print(f"  PEFT type: {config.peft_type}")
            if hasattr(config, "target_modules"):
                print(f"  Target modules: {config.target_modules}")

            current_base_model = None
            if base_model_to_reuse and tokenizer_to_reuse:
                print(
                    f"  → Reusing provided base model: {type(base_model_to_reuse).__name__}"
                )
                current_base_model = base_model_to_reuse
                tokenizer = tokenizer_to_reuse
            else:
                print(f"  → Loading base model: {base_model_hf_id}")
                print(f"    Device map: {device_map}")
                print(f"    Dtype: bfloat16")

                model_kwargs = {
                    "torch_dtype": torch.bfloat16,
                    "device_map": device_map,
                    "trust_remote_code": False,
                }
                if quantization_config_obj:
                    model_kwargs["quantization_config"] = quantization_config_obj

                current_base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_hf_id, **model_kwargs
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_hf_id)
                print(f"  ✓ Base model loaded successfully")

            print(f"  → Applying PEFT adapters...")
            model = PeftModel.from_pretrained(current_base_model, peft_adapter_path)
            print(f"  ✓ PEFT adapters applied successfully")
            print(f"  ✓ PEFT model ready: {base_model_hf_id} + {peft_adapter_path}")

        else:
            print(f"\n→ Loading base model directly (no PEFT)")
            print(f"  Model ID: {base_model_hf_id}")
            print(f"  Device map: {device_map}")
            print(f"  Dtype: bfloat16")

            model_kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device_map,
                "trust_remote_code": False,
            }
            if quantization_config_obj:
                model_kwargs["quantization_config"] = quantization_config_obj

            model = AutoModelForCausalLM.from_pretrained(
                base_model_hf_id, **model_kwargs
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model_hf_id)
            print(f"  ✓ Model loaded successfully: {base_model_hf_id}")

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print("  → Set tokenizer.pad_token to tokenizer.eos_token")
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))
                print("  → Added new pad_token '[PAD]' and resized model embeddings")

        return model, tokenizer

    except Exception as e:
        print(
            f"\n✗ ERROR loading model: Base model: '{base_model_hf_id}', PEFT: '{peft_adapter_path}'"
        )
        print(f"  Error details: {e}")
        import traceback

        traceback.print_exc()
        raise ValueError(
            f"Could not load model. Base model: '{base_model_hf_id}', PEFT adapter: '{peft_adapter_path}'. "
            f"Original error: {e}"
        )


def _load_system_prompt_content(
    prompt_name: str | None = None, prompts_dir: str = "prompts"
) -> str:
    """
    Loads a system prompt from a .md file.

    Args:
        prompt_name: Optional. The filename (without .md extension) of the prompt
                     in the prompts_dir. If None, or file not found, an empty string is returned.
        prompts_dir: The directory where prompt files are stored, relative to project root.

    Returns:
        The content of the system prompt file as a string, or an empty string if not found/specified.
    """
    if not prompt_name:
        try:
            from tqdm import tqdm

            tqdm.write(
                "No prompt_name provided for system prompt. No system prompt will be loaded."
            )
        except ImportError:
            print(
                "No prompt_name provided for system prompt. No system prompt will be loaded."
            )
        return ""

    try:
        # Determine project root robustly
        utils_file_path = Path(__file__).resolve()
        project_root = (
            utils_file_path.parent.parent.parent
        )  # src/inference/inference_utils.py -> project_root
    except Exception:
        project_root = Path.cwd()
        warnings.warn(
            f"Could not reliably determine project root from __file__. Using current working "
            f"directory '{project_root}' as project root. This might be incorrect if not running from project root."
        )

    prompt_file_path = project_root / prompts_dir / f"{prompt_name}.md"

    if prompt_file_path.exists():
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            print(f"Loaded system prompt from: {prompt_file_path}")
            return content
        except Exception as e:
            warnings.warn(f"Could not read prompt file {prompt_file_path}: {e}")
            return ""
    else:
        warnings.warn(
            f"System prompt file specified ('{prompt_name}' at '{prompt_file_path}') but not found. "
            "No system prompt will be used."
        )
        return ""


def load_and_format_prompt(
    user_input: str,
    tokenizer: AutoTokenizer,
    model_type: str,
    prompt_name: str | None = None,
    prompts_dir: str = "prompts",
) -> List[int]:
    """
    Loads an optional system prompt and formats it with the user's input,
    respecting the model_type.

    For "chat" models, it applies the tokenizer's chat template.
    For "text_generation" models, it tokenizes the user_input directly.

    Args:
        user_input: The user's input text.
        tokenizer: The tokenizer corresponding to the model.
        model_type: The type of model ("chat" or "text_generation").
        prompt_name: Optional. The filename (without .md extension) of the system prompt
                     in the prompts_dir. Relevant only for "chat" models.
        prompts_dir: The directory where prompt files are stored.

    Returns:
        input_ids: Token IDs ready for the model.
    """
    if model_type == "chat":
        # Build messages list
        messages = []

        # Add system prompt if specified
        if prompt_name:
            system_prompt_content = _load_system_prompt_content(
                prompt_name, prompts_dir
            )
            if system_prompt_content:
                messages.append({"role": "system", "content": system_prompt_content})
                print(f"Added system prompt from: {prompt_name}")
            else:
                warnings.warn(
                    f"System prompt '{prompt_name}' was specified but not loaded. User prompt: {user_input[:50]}..."
                )

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Use the centralized TemplateManager
        input_ids = TemplateManager.format_for_model(
            data=messages,
            tokenizer=tokenizer,
            model_type=model_type,
            purpose="inference",  # Explicitly specify inference purpose
        )

        return input_ids

    elif model_type == "text_generation":
        if prompt_name:
            warnings.warn(
                f"A system prompt ('{prompt_name}') was provided, but model_type is 'text_generation'. "
                "System prompts are typically ignored by text_generation models. The prompt will be the direct user input."
            )

        # Use the unified function for text generation too
        return TemplateManager.format_for_model(
            data=user_input,
            tokenizer=tokenizer,
            model_type="text_generation",
            purpose="inference",
        )

    else:
        warnings.warn(
            f"Unknown model_type: '{model_type}'. Using raw user input as prompt."
        )
        # Fallback to text generation behavior
        return TemplateManager.format_for_model(
            data=user_input,
            tokenizer=tokenizer,
            model_type="text_generation",
            purpose="inference",
        )
