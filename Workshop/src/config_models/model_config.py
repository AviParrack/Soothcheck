# This file defines the ModelConfig class, used for specifying parameters
# related to model loading, quantization, and attention mechanisms.
#
# Class: ModelConfig
# Inherits from: BaseConfig
# Methods:
#  - load(cls, config_name: str) -> "ModelConfig": (classmethod)
#    Loads a ModelConfig by its name (e.g., 'gemma-2b').

from pydantic import Field
from typing import Optional, Union, Dict, Any, Type

from .base_config import BaseConfig


class ModelConfig(BaseConfig):
    model_name_or_path: str = Field(
        default="google/gemma-2b",
        description="Path to pretrained model or model identifier from Hugging Face Hub.",
    )
    model_type: Optional[str] = Field(
        default=None,
        description="Type of model: 'chat' for instruction-tuned models or 'text_generation' for base models. If None, will be inferred from model name.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code when loading the model.",
    )
    torch_dtype: Optional[str] = Field(
        default="bfloat16",
        description="Torch dtype to use for model loading (e.g., 'float16', 'bfloat16', 'auto').",
    )
    # For 4-bit/8-bit quantization with bitsandbytes
    # E.g. bnb_4bit_compute_dtype: "float16"
    # bnb_4bit_quant_type: "nf4"
    # bnb_4bit_use_double_quant: False
    # load_in_4bit: True
    # load_in_8bit: False
    quantization_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="BitsAndBytes quantization config. E.g., {'load_in_4bit': True, 'bnb_4bit_quant_type': 'nf4'}.",
    )
    attn_implementation: Optional[str] = Field(
        default="eager",
        description="Attention implementation to use (e.g., 'eager', 'sdpa'). If None, Transformers default or model-specific logic applies.",
    )
    # Placeholder for device_map, though often handled by Trainer or accelerator
    # device_map: Optional[Union[str, Dict]] = Field(default=None, description="Device map for model loading.")

    chat_template_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional custom chat template configuration. E.g., {'add_generation_prompt_training': False, 'supports_system': True}. If None, defaults based on model_name_or_path will be used.",
    )

    @classmethod
    def load(cls, config_name: str) -> "ModelConfig":
        """Loads a ModelConfig by its name (e.g., 'gemma-2b')."""
        return super().load_from_name(config_name, "model")
