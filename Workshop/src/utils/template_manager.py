# This file provides centralized chat template management for both training and inference.
# It ensures consistent template application across the entire codebase based on model type.
#
# Key functions:
# - format_for_model: Universal function that handles all formatting cases
# - get_termination_tokens: Returns proper stopping tokens for different models

from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer, AutoTokenizer
import warnings
import logging
import re

logger = logging.getLogger(__name__)


class TemplateManager:
    """Centralized manager for chat templates across training and inference."""

    # Track warnings to avoid spam
    _assistant_mask_warning_shown = False

    # Known model patterns and their template behaviors
    MODEL_PATTERNS = {
        "gemma": {
            "pattern": ["gemma"],
            "template_type": "gemma",
            "supports_system": True,
            "add_generation_prompt_training": False,
            "add_generation_prompt_inference": True,
            "special_stop_tokens": [
                "<end_of_turn>"
            ],  # Gemma should stop at end of turn
        },
        "llama": {
            "pattern": ["llama", "Llama", "llama-3", "Llama-3"],
            "template_type": "llama",
            "supports_system": True,
            "add_generation_prompt_training": False,
            "add_generation_prompt_inference": True,
            "special_stop_tokens": ["<|eot_id|>"],  # Critical for Llama 3+ models
        },
        "mistral": {
            "pattern": ["mistral", "Mistral"],
            "template_type": "mistral",
            "supports_system": False,
            "add_generation_prompt_training": False,
            "add_generation_prompt_inference": True,
            "special_stop_tokens": [],  # Mistral uses standard EOS
        },
        "flan-t5": {
            "pattern": ["flan-t5", "Flan-T5"],
            "template_type": "flan-t5",
            "supports_system": False,
            "add_generation_prompt_training": False,
            "add_generation_prompt_inference": True,
            "special_stop_tokens": [],
        },
    }

    @classmethod
    def get_model_template_config(cls, model_name_or_path: str) -> Dict[str, Any]:
        """Get template configuration based on model name."""
        model_name_lower = model_name_or_path.lower()

        for template_name, config in cls.MODEL_PATTERNS.items():
            if any(
                pattern.lower() in model_name_lower for pattern in config["pattern"]
            ):
                return config

        # Default configuration for unknown models
        logger.warning(
            f"Unknown model pattern: {model_name_or_path}. Using default template config."
        )
        return {
            "template_type": "default",
            "supports_system": True,
            "add_generation_prompt_training": False,
            "add_generation_prompt_inference": True,
            "special_stop_tokens": [],
        }

    @classmethod
    def _get_llama_template_without_default_system(cls) -> str:
        """
        Returns a Llama chat template that doesn't inject default system messages.
        This is a simplified version that only adds system messages when explicitly provided.
        """
        return """{{- bos_token }}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {{- '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'user' %}
        {{- '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- elif message['role'] == 'assistant' %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}"""

    @classmethod
    def get_termination_tokens(cls, tokenizer: PreTrainedTokenizer) -> List[int]:
        """
        Get proper termination token IDs for generation based on model type.

        For Llama 3+ models, this includes both the standard EOS token and <|eot_id|>.
        This is critical to prevent infinite generation.

        Args:
            tokenizer: The model tokenizer

        Returns:
            List of token IDs to use as stopping criteria
        """
        termination_tokens = []

        # Always include the standard EOS token if available - this is the most reliable
        if tokenizer.eos_token_id is not None:
            termination_tokens.append(tokenizer.eos_token_id)
            logger.debug(f"Added EOS token ID: {tokenizer.eos_token_id}")

        # Get model-specific configuration
        template_config = cls.get_model_template_config(tokenizer.name_or_path)

        # Add model-specific stop tokens
        for special_token in template_config.get("special_stop_tokens", []):
            try:
                # First try convert_tokens_to_ids (works for properly registered special tokens)
                token_id = tokenizer.convert_tokens_to_ids(special_token)

                # Verify it's actually a single token and not UNK
                if (
                    isinstance(token_id, int)
                    and token_id != tokenizer.unk_token_id
                    and token_id not in termination_tokens
                ):
                    termination_tokens.append(token_id)
                    logger.debug(
                        f"Added special stop token '{special_token}' (ID: {token_id})"
                    )
                elif isinstance(token_id, list):
                    # If it returns multiple tokens, it's not a proper special token
                    logger.warning(
                        f"Special stop token '{special_token}' tokenized to multiple tokens: {token_id}. Skipping."
                    )
                else:
                    logger.warning(
                        f"Special stop token '{special_token}' not found in tokenizer vocabulary or is UNK token"
                    )

            except Exception as e:
                logger.warning(
                    f"Could not convert special stop token '{special_token}' to ID: {e}"
                )

        # For Llama models, also try to get the eot_id directly from tokenizer attributes
        # as a fallback (though this should be covered by eos_token_id)
        if template_config["template_type"] == "llama":
            for attr_name in ["eot_id", "end_of_text_id"]:
                if hasattr(tokenizer, attr_name):
                    attr_value = getattr(tokenizer, attr_name)
                    if (
                        isinstance(attr_value, int)
                        and attr_value not in termination_tokens
                    ):
                        termination_tokens.append(attr_value)
                        logger.debug(
                            f"Added {attr_name} from tokenizer attribute: {attr_value}"
                        )

        # Remove duplicates while preserving order
        unique_tokens = []
        for token in termination_tokens:
            if token not in unique_tokens:
                unique_tokens.append(token)

        logger.debug(
            f"Final termination tokens for {tokenizer.name_or_path}: {unique_tokens}"
        )
        return unique_tokens

    @classmethod
    def format_for_model(
        cls,
        data: Union[str, List[Dict[str, str]]],
        tokenizer: PreTrainedTokenizer,
        model_type: str,
        purpose: str = "training",
        max_length: Optional[int] = None,
        truncation: bool = False,
        padding: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Union[List[int], Dict[str, List[int]]]:
        """
        Universal formatting function that handles all cases.

        Args:
            data: Either a text string or list of message dicts
            tokenizer: The tokenizer to use
            model_type: "chat" or "text_generation"
            purpose: "training" or "inference"
            max_length: Maximum sequence length
            truncation: Whether to truncate
            padding: Whether to pad
            tools: Optional list of tool definitions for function calling

        Returns:
            Token IDs (list) or dict with input_ids/attention_mask/labels for training
        """
        # Validate input/model compatibility
        is_text_data = isinstance(data, str)
        is_messages_data = isinstance(data, list)
        is_chat_model = model_type == "chat"
        is_text_model = model_type == "text_generation"

        # Error on mismatches
        if is_chat_model and is_text_data:
            raise ValueError(
                f"Chat model '{tokenizer.name_or_path}' requires messages format, "
                f"but got text string. Convert your text data to messages format: "
                f'[{{"role": "user", "content": "your text"}}] or use a text_generation model.'
            )

        if is_text_model and is_messages_data:
            raise ValueError(
                f"Text generation model '{tokenizer.name_or_path}' requires text format, "
                f"but got messages list. Use model_type='chat' or convert messages to text."
            )

        # Handle text generation models
        if is_text_model:
            if purpose == "inference":
                # Add BOS if needed
                if tokenizer.bos_token and not data.startswith(tokenizer.bos_token):
                    data = tokenizer.bos_token + data

            result = tokenizer(
                data,
                max_length=max_length,
                truncation=truncation,
                padding=padding,
                return_tensors=None,
                **kwargs,
            )
            return result["input_ids"] if purpose == "inference" else result

        # Handle chat models
        if is_chat_model:
            # Verify tokenizer has chat template
            if (
                not hasattr(tokenizer, "chat_template")
                or tokenizer.chat_template is None
            ):
                raise ValueError(
                    f"Chat model '{tokenizer.name_or_path}' tokenizer lacks chat template. "
                    f"This is required for chat formatting."
                )

            # Get template config
            template_config = cls.get_model_template_config(tokenizer.name_or_path)

            # Determine if we need generation prompt
            add_generation_prompt = (
                template_config["add_generation_prompt_inference"]
                if purpose == "inference"
                else template_config["add_generation_prompt_training"]
            )

            # For Llama models, use custom template to avoid default system message
            original_template = None
            if template_config["template_type"] == "llama":
                original_template = tokenizer.chat_template
                tokenizer.chat_template = (
                    cls._get_llama_template_without_default_system()
                )

            try:
                # For training, use the new HuggingFace standard approach
                if purpose == "training":
                    # Try to use return_assistant_tokens_mask if available (HF transformers >= 4.34)
                    try:
                        template_kwargs = {
                            "tokenize": True,
                            "add_generation_prompt": add_generation_prompt,
                            "return_assistant_tokens_mask": True,
                            "return_dict": True,
                            "max_length": max_length,
                            "truncation": truncation,
                            "padding": padding,
                            **kwargs,
                        }
                        
                        # Add tools if provided
                        if tools is not None:
                            template_kwargs["tools"] = tools
                        
                        result = tokenizer.apply_chat_template(
                            data,
                            **template_kwargs,
                        )

                        # Convert assistant mask to labels
                        input_ids = result["input_ids"]
                        assistant_mask = result["assistant_tokens_mask"]
                        attention_mask = result.get(
                            "attention_mask", [1] * len(input_ids)
                        )

                        # Create labels: -100 for non-assistant tokens, actual token IDs for assistant tokens
                        labels = []
                        for i, (token_id, is_assistant) in enumerate(
                            zip(input_ids, assistant_mask)
                        ):
                            if is_assistant:
                                labels.append(token_id)
                            else:
                                labels.append(-100)

                        final_result = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }

                        logger.debug(
                            f"Using HuggingFace assistant tokens mask for training"
                        )

                    except Exception as e:
                        # Fallback for older HF versions or tokenizers that don't support assistant masks
                        # This catches various errors including KeyError for 'assistant_tokens_mask'
                        if not cls._assistant_mask_warning_shown:
                            logger.warning(
                                f"Assistant tokens mask not available ({e}), falling back to training on all tokens"
                            )
                            cls._assistant_mask_warning_shown = True

                        fallback_kwargs = {
                            "tokenize": True,
                            "add_generation_prompt": add_generation_prompt,
                            "max_length": max_length,
                            "truncation": truncation,
                            "padding": padding,
                            "return_tensors": None,
                            **kwargs,
                        }
                        
                        # Add tools if provided
                        if tools is not None:
                            fallback_kwargs["tools"] = tools
                        
                        result = tokenizer.apply_chat_template(
                            data,
                            **fallback_kwargs,
                        )

                        if isinstance(result, dict):
                            input_ids = result["input_ids"]
                            attention_mask = result.get(
                                "attention_mask", [1] * len(input_ids)
                            )
                        else:
                            input_ids = result
                            attention_mask = [1] * len(input_ids)

                        # Fallback: train on all tokens
                        labels = input_ids.copy()

                        final_result = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }
                else:
                    # For inference, return just input_ids
                    inference_kwargs = {
                        "tokenize": True,
                        "add_generation_prompt": add_generation_prompt,
                        "max_length": max_length,
                        "truncation": truncation,
                        "padding": padding,
                        "return_tensors": None,
                        **kwargs,
                    }
                    
                    # Add tools if provided
                    if tools is not None:
                        inference_kwargs["tools"] = tools
                    
                    result = tokenizer.apply_chat_template(
                        data,
                        **inference_kwargs,
                    )

                    if isinstance(result, dict):
                        final_result = result["input_ids"]
                    else:
                        final_result = result

            except Exception as e:
                raise ValueError(
                    f"Failed to apply chat template: {e}. "
                    f"Check that messages have correct 'role'/'content' format."
                ) from e
            finally:
                # Restore original template if we modified it
                if original_template is not None:
                    tokenizer.chat_template = original_template

            return final_result

        raise ValueError(f"Unknown model_type: {model_type}")
