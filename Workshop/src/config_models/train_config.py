# This file defines the TrainConfig class, which orchestrates multi-stage training pipelines.
# It contains a list of training stages (SFT, DPO, etc.) and global pipeline settings.
# Dataset names are now passed as command-line arguments rather than being embedded in configs.
# Stage configs are now referenced by name and loaded from configs/stage_configs/
#
# Class: TrainConfig
# Inherits from: BaseConfig
# Properties:
#  - computed_model_config_path: Optional[str]
#  - computed_peft_config_path: Optional[str]
# Methods:
#  - load(cls, config_name: str) -> "TrainConfig": (classmethod)
#    Loads a TrainConfig by its name (e.g., 'my_pipeline').
#  - get_stage_configs() -> List[BaseStageConfig]: Load and return typed stage configuration objects

from pydantic import Field, validator
from typing import Optional, List, Union, Type, Dict, Any
import os
import json
from pathlib import Path

from .base_config import BaseConfig
from .stage_configs import BaseStageConfig, SFTStageConfig, DPOStageConfig


class StageReference(BaseConfig):
    """Reference to a stage configuration file."""

    type: str = Field(..., description="Type of stage ('sft', 'dpo', etc.)")

    config_name: str = Field(
        ..., description="Name of the stage config file (without .json extension)"
    )


class TrainConfig(BaseConfig):
    # Pipeline orchestration
    pipeline_name: str = Field(
        ...,
        description="Name of this training pipeline (e.g., 'sft_then_dpo'). Used for organizing outputs and identification.",
    )

    stages: List[StageReference] = Field(
        ...,
        description="List of stage references to execute in order. Each stage references a config file by name.",
    )

    # Global config references - paths will be derived from these
    model: Optional[str] = Field(
        default=None,
        description="Name of the model configuration (e.g., 'gemma-2b'). Path will be configs/model/[name].json. If not specified, must be provided via model override.",
    )

    peft_config_name: Optional[str] = Field(
        default=None,
        description="Name of the PEFT configuration (e.g., 'lora_default'). Path will be configs/peft/[name].json.",
    )

    @property
    def computed_model_config_path(self) -> Optional[str]:
        if self.model is None:
            return None
        name = self.model
        if not name.endswith(".json"):
            name += ".json"
        return os.path.join("configs", "model", name)

    @property
    def computed_peft_config_path(self) -> Optional[str]:
        if self.peft_config_name:
            name = self.peft_config_name
            if not name.endswith(".json"):
                name += ".json"
            return os.path.join("configs", "peft", name)
        return None

    # Global pipeline settings
    output_dir: str = Field(
        default="experiments",
        description="Base directory for experiment outputs. Final path will be output_dir/[dataset_name]/[pipeline_name]/",
    )
    overwrite_output_dir: bool = Field(
        default=False,
        description="If True, overwrite the content of the output directory. Be cautious with this.",
    )

    # W&B Logging
    report_to: List[str] = Field(
        default_factory=lambda: ["wandb"],
        description="The list of integrations to report results and logs to.",
    )
    wandb_project: Optional[str] = Field(
        default="pollux", description="Weights & Biases project name."
    )

    # Miscellaneous
    seed: int = Field(
        default=42, description="Random seed for initialization for reproducibility."
    )

    def get_stage_configs(
        self, dataset_name: Optional[str] = None, stage_config_selections: Optional[Dict[str, str]] = None
    ) -> List[BaseStageConfig]:
        """Parse and return typed stage configuration objects by loading referenced config files."""
        parsed_stages = []

        for stage_ref in self.stages:
            # Determine which config file to use - either from selections or default
            if stage_config_selections and stage_ref.type in stage_config_selections:
                config_name = stage_config_selections[stage_ref.type]
                print(f"Using selected config '{config_name}' for stage '{stage_ref.type}'")
            else:
                config_name = stage_ref.config_name

            # Load the stage config file
            config_path = os.path.join(
                "configs", "stage_configs", f"{config_name}.json"
            )

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Stage config file not found: {config_path}")

            with open(config_path, "r") as f:
                stage_config_dict = json.load(f)

            # Create the appropriate stage config object
            if stage_ref.type == "sft":
                stage_config = SFTStageConfig(**stage_config_dict)
            elif stage_ref.type == "dpo":
                stage_config = DPOStageConfig(**stage_config_dict)
            else:
                raise ValueError(f"Unknown stage type: {stage_ref.type}")

            # Set processed dataset name if we have dataset context
            if dataset_name and stage_config.processed_dataset_name is None:
                stage_config.processed_dataset_name = (
                    f"{dataset_name}_{stage_config.stage_name}"
                )

            parsed_stages.append(stage_config)

        return parsed_stages

    def get_output_dir_for_dataset(self, dataset_name: str) -> str:
        """Get the output directory path for a given dataset."""
        return os.path.join(self.output_dir, dataset_name, self.pipeline_name)

    @classmethod
    def load(cls, config_name: str) -> "TrainConfig":
        """Loads a TrainConfig by its name (e.g., 'my_pipeline')."""
        return super().load_from_name(config_name, "train")
