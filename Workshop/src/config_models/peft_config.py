# This file defines the PeftConfig class, used for specifying parameters
# for Parameter-Efficient Fine-Tuning (PEFT) methods like LoRA.
#
# Class: PeftConfig
# Inherits from: BaseConfig
# Methods:
#  - load(cls, config_name: str) -> "PeftConfig": (classmethod)
#    Loads a PeftConfig by its name (e.g., 'lora_default').

from pydantic import Field
from typing import Optional, List, Literal, Union, Type

from .base_config import BaseConfig

# Literal types for PEFT TaskType and PeftType, if we want to be strict.
# from peft import TaskType, PeftType # Would require peft to be installed during config definition time
# For now, using strings and user needs to ensure they are valid.


class PeftConfig(BaseConfig):
    peft_type: str = Field(
        default="LORA",
        description="PEFT type, e.g., 'LORA', 'PROMPT_TUNING'. User must ensure this matches PeftType enum from peft library.",
    )
    task_type: str = Field(
        default="CAUSAL_LM",
        description="Task type for PEFT, e.g., 'CAUSAL_LM', 'SEQ_CLS'. User must ensure this matches TaskType enum from peft library.",
    )
    r: int = Field(default=8, description="LoRA attention dimension (rank).")
    lora_alpha: int = Field(default=16, description="LoRA alpha parameter.")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout probability.")
    target_modules: Optional[Union[List[str], str]] = Field(
        default=None,
        description='List of module names to apply LoRA to (e.g., ["q_proj", "v_proj"]). If \'auto\', will try to find common ones for LLMs. If None/empty, might apply to all linear layers based on underlying peft lib defaults or error if not inferable.',
    )
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias type for LoRA. Can be 'none', 'all' or 'lora_only'.",
    )
    # Other PEFT parameters can be added here as needed
    # fan_in_fan_out: bool = Field(default=False, description="Set this to True if the layer to replace stores weight like (fan_in, fan_out)")
    # modules_to_save: Optional[List[str]] = Field(default=None, description="List of modules apart from PEFT layers to be set as trainable and saved in the final checkpoint.")

    @classmethod
    def load(cls, config_name: str) -> "PeftConfig":
        """Loads a PeftConfig by its name (e.g., 'lora_default')."""
        return super().load_from_name(config_name, "peft")
