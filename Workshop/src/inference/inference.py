#!/usr/bin/env python
# This file implements the core inference pipeline for language models.
# It provides functions for:
# - Loading models (fine-tuned with PEFT or base Hugging Face models) and caching them.
# - Formatting prompts via `src.inference.inference_utils.load_and_format_prompt`,
#   which handles chat templates, BOS tokens, and system prompts based on model_type.
# - Generating text with support for token streaming (for CLI and async use).
# - Running interactive command-line chats or single-prompt generation.
#
# This module is intended to be used as a library by other scripts or applications.
#
# Key functions:
# - preload_model_and_tokenizer: Loads and caches the model and tokenizer.
# - generate_text_stream: Generates text and prints tokens to stdout (for CLI).
# - generate_text_yield_tokens: Async generator for yielding tokens (for UI/API).
# - generate_text_for_api: Generates full text response (for non-streaming API).
# - generate_batch_responses: Processes multiple prompts for evaluation/benchmarking.
# - interactive_mode: Handles interactive command-line chat logic.
# - single_prompt_mode: Handles single prompt command-line logic.

import torch
from transformers import (
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
    AutoModelForCausalLM,
)
from threading import Thread
from tqdm import tqdm

# Assuming inference_utils is in the same directory or accessible via python path adjustments by caller
from .inference_utils import (
    load_model as load_model_from_utils,
    load_and_format_prompt as load_and_format_prompt_from_utils,
)
from .chat_manager import ChatManager
from ..utils.template_manager import TemplateManager
from ..utils.model_resolver import resolve_model_specification

# Global cache for preloaded model and tokenizer
# PRELOADED_MODEL = None # Removed global cache
# PRELOADED_TOKENIZER = None # Removed global cache


def preload_model_and_tokenizer(
    model_spec: str,
    device: str | None = None,
    base_model_to_reuse: AutoModelForCausalLM | None = None,
    tokenizer_to_reuse: AutoTokenizer | None = None,
):
    """
    Loads the model and tokenizer using model resolver.
    Uses the robust loader from inference_utils.
    This version does not use global caching to support loading multiple distinct models.
    Can reuse an already loaded base model and tokenizer for PEFT adapters.
    """

    print(f"\n{'='*80}")
    print(f"UNIFIED MODEL LOADING PIPELINE")
    print(f"{'='*80}")
    print(f"Model specification: {model_spec}")
    print(f"Target device: {device or 'auto'}")

    # Resolve model specification
    try:
        model_resolution = resolve_model_specification(model_spec)
        base_model_hf_id = model_resolution.base_model_hf_id
        peft_adapter_path = model_resolution.peft_adapter_path

        print(f"\nModel Resolution Details:")
        print(f"  ✓ Base model HF ID: {base_model_hf_id}")
        print(f"  ✓ Model type: {model_resolution.model_type}")
        print(f"  ✓ Is experiment: {model_resolution.is_experiment}")

        if peft_adapter_path:
            print(f"  ✓ PEFT adapter path: {peft_adapter_path}")
            print(f"    → This is a FINE-TUNED model with LoRA/PEFT adapters")
        else:
            print(f"  ✓ No PEFT adapters - loading BASE MODEL directly")

        if model_resolution.experiment_path:
            print(f"  ✓ Experiment path: {model_resolution.experiment_path}")

    except ValueError as e:
        print(f"\n✗ ERROR: Could not resolve model specification '{model_spec}'")
        print(f"  Details: {e}")
        raise

    print(f"\n{'='*40}")
    print(f"LOADING MODEL COMPONENTS")
    print(f"{'='*40}")

    # Log what's happening
    if base_model_to_reuse and tokenizer_to_reuse:
        print(f"→ Reusing already loaded base model and tokenizer")
    else:
        print(f"→ Loading fresh model instance from HuggingFace")

    model, tokenizer = load_model_from_utils(
        base_model_hf_id=base_model_hf_id,
        peft_adapter_path=peft_adapter_path,
        device=device,
        base_model_to_reuse=base_model_to_reuse,
        tokenizer_to_reuse=tokenizer_to_reuse,
        quantization_config=model_resolution.model_config.quantization_config,
    )

    if device and device != "auto" and hasattr(model, "to") and not base_model_to_reuse:
        try:
            if not hasattr(model, "base_model"):
                model.to(torch.device(device))
                print(f"✓ Model moved to device: {device}")
        except Exception as e:
            print(f"⚠ Warning: Could not explicitly move model to {device}: {e}")

    print(f"\n{'='*40}")
    print(f"MODEL LOADING SUCCESSFUL")
    print(f"{'='*40}")
    print(f"✓ Model spec '{model_spec}' loaded successfully")
    print(f"✓ Base model: {base_model_hf_id}")
    if peft_adapter_path:
        print(f"✓ With PEFT adapters from: {peft_adapter_path}")
    print(f"✓ Model type: {model_resolution.model_type}")
    print(f"✓ Ready for inference")
    print(f"{'='*80}\n")

    return model, tokenizer


def generate_text_stream(
    prompt,  # Can be str or List[int] (pre-tokenized)
    model,  # Expected to be preloaded
    tokenizer,  # Expected to be preloaded
    max_new_tokens=4096,
    temperature=0.7,
    top_p=0.9,
):
    """
    Generate text using the model and stream the output to STDOUT via TextStreamer.
    Suitable for direct CLI display.
    """
    from ..utils.template_manager import TemplateManager
    
    # Handle both string prompts and pre-tokenized input
    if isinstance(prompt, str):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
    elif isinstance(prompt, list):
        # Already tokenized - create tensor directly
        inputs = {
            "input_ids": torch.tensor([prompt], dtype=torch.long).to(model.device),
            "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long).to(model.device)
        }
    else:
        raise ValueError(f"Prompt must be string or list of token IDs, got {type(prompt)}")
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    
    # Get proper termination tokens for this model (critical for Llama 3+)
    termination_tokens = TemplateManager.get_termination_tokens(tokenizer)
    
    # Log termination tokens for debugging (especially important for Llama models)
    if "llama" in tokenizer.name_or_path.lower():
        # print(f"DEBUG: Using Llama termination tokens: {termination_tokens}")
        try:
            special_tokens = [tokenizer.decode([t]) for t in termination_tokens if t is not None]
            # print(f"DEBUG: Termination token strings: {special_tokens}")
        except:
            # print("DEBUG: Could not decode termination tokens for display")
            pass

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": pad_token_id,
        "eos_token_id": termination_tokens,  # Use proper termination tokens
    }
    with torch.no_grad():
        model.generate(**generation_kwargs)


async def generate_text_yield_tokens(
    prompt,  # Can be str or List[int] (pre-tokenized)
    model,
    tokenizer,
    max_new_tokens=4096,
    temperature=0.7,
    top_p=0.9,
):
    """
    Generate text using the model and yield tokens one by one asynchronously.
    Suitable for use with asyncio and frameworks like Gradio.
    """
    from ..utils.template_manager import TemplateManager
    
    # Handle both string prompts and pre-tokenized input
    if isinstance(prompt, str):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
    elif isinstance(prompt, list):
        # Already tokenized - create tensor directly
        inputs = {
            "input_ids": torch.tensor([prompt], dtype=torch.long).to(model.device),
            "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long).to(model.device)
        }
    else:
        raise ValueError(f"Prompt must be string or list of token IDs, got {type(prompt)}")
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    
    # Get proper termination tokens for this model (critical for Llama 3+)
    termination_tokens = TemplateManager.get_termination_tokens(tokenizer)
    
    # Log termination tokens for debugging (especially important for Llama models)
    if "llama" in tokenizer.name_or_path.lower():
        # print(f"DEBUG: Using Llama termination tokens: {termination_tokens}")
        try:
            special_tokens = [tokenizer.decode([t]) for t in termination_tokens if t is not None]
            # print(f"DEBUG: Termination token strings: {special_tokens}")
        except:
            print("DEBUG: Could not decode termination tokens for display")

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=120.0,  # Longer timeout for PEFT
    )

    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "pad_token_id": pad_token_id,
        "eos_token_id": termination_tokens,  # Use proper termination tokens
    }

    # Add memory-efficient settings for PEFT models
    is_peft_model = "peft" in str(type(model)).lower()
    if is_peft_model:
        print("DEBUG: Using memory-efficient settings for PEFT model")
        # Use smaller batch size and enable memory optimizations
        generation_kwargs.update(
            {
                "use_cache": True,  # Enable KV cache for efficiency
            }
        )

    # Use torch.no_grad() context for the generation thread
    def _generate_with_no_grad():
        with torch.no_grad():
            try:
                # Clear cache before generation to free up memory
                if hasattr(torch.cuda, "empty_cache"):
                    torch.cuda.empty_cache()

                model.generate(**generation_kwargs)
            except Exception as e:
                print(f"Error in generation thread: {e}")
                # Signal end of stream properly
                streamer.end()

    thread = Thread(target=_generate_with_no_grad)
    thread.start()

    try:
        # Iterate through the streamer with timeout handling
        for new_text in streamer:
            yield new_text
    except Exception as e:
        print(f"Error in streamer iteration: {e}")
        yield f"[Streaming error: {e}]"
    finally:
        # Ensure thread cleanup
        if thread.is_alive():
            print("Waiting for generation thread to complete...")
            thread.join(timeout=30.0)  # Longer timeout for PEFT models
            if thread.is_alive():
                print("Warning: Generation thread did not complete within timeout")


def generate_text_for_api(
    prompt,  # Can be str or List[int] (pre-tokenized)
    model,
    tokenizer,
    max_new_tokens=4096,
    temperature=0.7,
    top_p=0.9,
):
    """
    Generate text using the model and return the full response.
    This version is for non-streaming uses.
    """
    from ..utils.template_manager import TemplateManager
    
    # Handle both string prompts and pre-tokenized input
    if isinstance(prompt, str):
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            model.device
        )
    elif isinstance(prompt, list):
        # Already tokenized - create tensor directly
        inputs = {
            "input_ids": torch.tensor([prompt], dtype=torch.long).to(model.device),
            "attention_mask": torch.ones((1, len(prompt)), dtype=torch.long).to(model.device)
        }
    else:
        raise ValueError(f"Prompt must be string or list of token IDs, got {type(prompt)}")
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    
    # Get proper termination tokens for this model (critical for Llama 3+)
    termination_tokens = TemplateManager.get_termination_tokens(tokenizer)
    
    # Log termination tokens for debugging (especially important for Llama models)
    if "llama" in tokenizer.name_or_path.lower():
        print(f"DEBUG: Using Llama termination tokens: {termination_tokens}")
        try:
            special_tokens = [tokenizer.decode([t]) for t in termination_tokens if t is not None]
            print(f"DEBUG: Termination token strings: {special_tokens}")
        except:
            print("DEBUG: Could not decode termination tokens for display")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=termination_tokens,  # Use proper termination tokens
        )

    full_generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return full_generated_text.strip()


def generate_batch_responses(
    prompts: list[str],
    model,
    tokenizer,
    model_type: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.9,
    prompt_name: str | None = None,
    show_progress: bool = True,
) -> list[str]:
    """
    Generate responses for a batch of prompts.

    This is the main function for evaluation/benchmarking that processes multiple
    prompts and returns their responses. It handles prompt formatting based on
    model type and provides clear progress tracking.

    Args:
        prompts: List of user prompts to process
        model: The loaded model
        tokenizer: The tokenizer
        model_type: Type of model ("chat" or "text_generation")
        max_new_tokens: Maximum tokens to generate per response
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        prompt_name: Optional system prompt name (for chat models)
        show_progress: Whether to show progress bar

    Returns:
        List of generated responses (same length as prompts)
    """

    responses = []

    # Set model to eval mode
    model.eval()

    # Create progress iterator
    prompt_iterator = prompts
    if show_progress:
        prompt_iterator = tqdm(prompts, desc="Generating responses", unit="prompt")

    for i, user_prompt in enumerate(prompt_iterator):
        try:
            # Format the prompt according to model type
            formatted_prompt = load_and_format_prompt_from_utils(
                user_input=user_prompt,
                tokenizer=tokenizer,
                model_type=model_type,
                prompt_name=prompt_name,
            )

            # Generate response using the non-streaming API function
            response = generate_text_for_api(
                prompt=formatted_prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            responses.append(response)

        except Exception as e:
            error_msg = f"ERROR: Failed to generate response - {str(e)}"
            print(f"\nError on prompt {i+1}: {error_msg}")
            responses.append(error_msg)

    return responses


def interactive_mode(
    model,
    tokenizer,
    prompt_name_arg: str | None,
    max_length_arg: int,
    temperature_arg: float,
    top_p_arg: float,
    model_type: str,
):
    """
    Run the model in interactive mode, accepting user inputs.
    Uses preloaded model and tokenizer.
    Prompt formatting is delegated to load_and_format_prompt_from_utils.
    """
    print("\n=== Language Model Interactive Mode ===")
    print(f"Model type: {model_type}")
    print("Model is preloaded. Type 'exit' or 'quit' to quit.")

    while True:
        try:
            user_input = input("\nYour prompt: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            print("\nGenerating response (streaming):")
            formatted_prompt = load_and_format_prompt_from_utils(
                user_input=user_input,
                tokenizer=tokenizer,
                model_type=model_type,
                prompt_name=prompt_name_arg,
            )

            generate_text_stream(
                formatted_prompt,
                model,
                tokenizer,
                max_new_tokens=max_length_arg,
                temperature=temperature_arg,
                top_p=top_p_arg,
            )
            print()  # Newline after streaming finishes

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def interactive_chat_mode(
    model,
    tokenizer,
    prompt_name_arg: str | None,
    max_length_arg: int,
    temperature_arg: float,
    top_p_arg: float,
    model_type: str,
):
    """
    Run the model in interactive chat mode with efficient conversation history.
    Uses persistent KV cache to avoid recomputing previous context.
    """
    print("\n=== Language Model Chat Mode (with persistent KV cache) ===")
    print(f"Model type: {model_type}")
    print("Model is preloaded. Type 'clear' to reset conversation, 'exit' or 'quit' to quit.")
    print("This mode uses persistent KV cache - each turn builds on previous efficiently!")
    
    # Create efficient chat manager with persistent KV cache
    chat_manager = ChatManager(
        max_history_turns=12,  # 6 user + 6 assistant messages
        max_sequence_length=4096  # Allow much longer conversations
    )
    
    while True:
        try:
            print(f"\n{chat_manager.get_history_summary()}")
            user_input = input("Your prompt: ")
            
            if user_input.lower() in ["exit", "quit"]:
                break
            elif user_input.lower() == "clear":
                chat_manager.clear_history()
                print("Conversation history and cache cleared.")
                continue

            print("\nGenerating response:")
            
            # Use the efficient generation with persistent KV cache
            try:
                response = chat_manager.generate_response(
                    user_input=user_input,
                    model=model,
                    tokenizer=tokenizer,
                    prompt_name=prompt_name_arg,
                    max_new_tokens=max_length_arg,
                    temperature=temperature_arg,
                    top_p=top_p_arg,
                )
                print()  # Extra newline after response
                
            except Exception as e:
                print(f"Error in KV cache generation: {e}")
                # Fallback to simple approach
                print("Falling back to simple generation...")
                
                formatted_prompt = load_and_format_prompt_from_utils(
                    user_input=user_input,
                    tokenizer=tokenizer,
                    model_type=model_type,
                    prompt_name=prompt_name_arg,
                )
                
                generate_text_stream(
                    formatted_prompt,
                    model,
                    tokenizer,
                    max_new_tokens=max_length_arg,
                    temperature=temperature_arg,
                    top_p=top_p_arg,
                )
                print()

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


def single_prompt_mode(
    prompt_text: str,
    model,
    tokenizer,
    prompt_name_arg: str | None,
    max_length_arg: int,
    temperature_arg: float,
    top_p_arg: float,
    model_type: str,
):
    """
    Generate a response for a single prompt using preloaded model.
    Prompt formatting is delegated to load_and_format_prompt_from_utils.
    """
    print(
        f"\nGenerating response for single prompt (streaming), model_type: {model_type}"
    )

    formatted_prompt = load_and_format_prompt_from_utils(
        user_input=prompt_text,
        tokenizer=tokenizer,
        model_type=model_type,
        prompt_name=prompt_name_arg,
    )

    generate_text_stream(
        formatted_prompt,
        model,
        tokenizer,
        max_new_tokens=max_length_arg,
        temperature=temperature_arg,
        top_p=top_p_arg,
    )
    print()  # Newline after streaming finishes


# Removed run_inference_cli and main() along with argparse from this file.
# This file is now purely a library.
