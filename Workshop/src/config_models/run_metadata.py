# This file defines the unified run metadata system for tracking training experiments.
# It provides a single source of truth for all information needed to understand,
# reproduce, and chain training runs.
#
# Classes:
# - RunMetadata: Complete metadata for any training run
# - StageInfo: Information about individual training stages
# - ChainInfo: Information about chaining from previous runs
#
# This replaces the fragmented approach of separate config files with a single,
# comprehensive metadata system.

from pydantic import Field, validator
from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from pathlib import Path

from .base_config import BaseConfig


class StageInfo(BaseConfig):
    """Information about a training stage within a run."""

    stage_type: str = Field(..., description="Type of stage (sft, dpo, etc.)")
    stage_config_name: str = Field(..., description="Name of stage config used")
    dataset_name: str = Field(..., description="Processed dataset used for this stage")
    starting_model: str = Field(..., description="Model this stage started from")

    # Training results
    final_loss: Optional[float] = Field(default=None, description="Final training loss")
    eval_loss: Optional[float] = Field(
        default=None, description="Final evaluation loss"
    )
    total_steps: Optional[int] = Field(default=None, description="Total training steps")

    # Hyperparameters (most important ones)
    learning_rate: Optional[float] = Field(
        default=None, description="Learning rate used"
    )
    num_epochs: Optional[float] = Field(default=None, description="Number of epochs")
    batch_size: Optional[int] = Field(default=None, description="Effective batch size")


class ChainInfo(BaseConfig):
    """Information about chaining from previous runs."""

    parent_run: str = Field(
        ..., description="Parent run this was chained from (dataset/run_name)"
    )
    parent_stage: Optional[str] = Field(
        default=None, description="Specific stage if from pipeline"
    )
    chain_type: Literal["manual", "pipeline", "suite"] = Field(
        ..., description="How this chain was created"
    )


class RunMetadata(BaseConfig):
    """Complete metadata for a training run."""

    # Identity
    run_name: str = Field(..., description="Name of this run")
    dataset_name: str = Field(..., description="Base dataset name")
    run_type: Literal["single_stage", "pipeline", "suite_experiment"] = Field(
        ..., description="Type of run"
    )

    # Timing
    created_at: str = Field(..., description="When run was created (ISO format)")
    completed_at: Optional[str] = Field(
        default=None, description="When run completed (ISO format)"
    )

    # Configuration
    model: str = Field(..., description="Model configuration used")
    peft_config_name: Optional[str] = Field(
        default=None, description="PEFT configuration used"
    )

    # Training stages (list even for single stage for consistency)
    stages: List[StageInfo] = Field(..., description="Training stages in this run")

    # Chaining information
    chain_info: Optional[ChainInfo] = Field(
        default=None, description="Chaining information if applicable"
    )

    # Pipeline information (for multi-stage runs)
    pipeline_config_name: Optional[str] = Field(
        default=None, description="Pipeline config if from pipeline"
    )

    # Suite information (for systematic experiments)
    suite_name: Optional[str] = Field(
        default=None, description="Experiment suite name if applicable"
    )
    experiment_id: Optional[str] = Field(
        default=None, description="Experiment ID within suite"
    )

    # W&B tracking
    wandb_project: Optional[str] = Field(default=None, description="W&B project")
    wandb_run_urls: List[str] = Field(
        default_factory=list, description="W&B run URLs for each stage"
    )

    # Tags for organization
    tags: List[str] = Field(
        default_factory=list, description="Tags for organization and filtering"
    )

    # Status
    status: Literal["running", "completed", "failed"] = Field(
        default="running", description="Current status"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )

    def get_final_stage(self) -> StageInfo:
        """Get the final training stage."""
        return self.stages[-1]

    def get_stage_by_type(self, stage_type: str) -> Optional[StageInfo]:
        """Get a stage by its type."""
        for stage in self.stages:
            if stage.stage_type == stage_type:
                return stage
        return None

    def get_run_path(self) -> str:
        """Get the path to this run."""
        return f"experiments/{self.dataset_name}/{self.run_name}"

    def get_display_name(self) -> str:
        """Get a human-readable display name."""
        if self.run_type == "single_stage":
            return f"{self.dataset_name}/{self.run_name} ({self.stages[0].stage_type})"
        elif self.run_type == "pipeline":
            stage_types = [s.stage_type for s in self.stages]
            return f"{self.dataset_name}/{self.run_name} ({' → '.join(stage_types)})"
        else:
            return f"{self.dataset_name}/{self.run_name} (suite: {self.suite_name})"

    def can_be_chained_to(self, stage_type: str) -> bool:
        """Check if this run can be used as input for a specific stage type."""
        # Basic compatibility checks
        if self.status != "completed":
            return False

        final_stage = self.get_final_stage()

        # SFT can chain to DPO
        if final_stage.stage_type == "sft" and stage_type == "dpo":
            return True

        # More chaining rules can be added here
        return False

    @classmethod
    def from_run_directory(cls, run_path: str) -> "RunMetadata":
        """Load run metadata from a run directory."""
        run_dir = Path(run_path)
        metadata_file = run_dir / "run_metadata.json"

        if metadata_file.exists():
            return cls.from_json(str(metadata_file))

        # Try to construct from legacy files for backward compatibility
        return cls._construct_from_legacy_files(run_dir)

    @classmethod
    def _construct_from_legacy_files(cls, run_dir: Path) -> "RunMetadata":
        """Construct metadata from legacy file structure."""
        # This would analyze existing files to build metadata
        # Implementation depends on current file structure
        raise NotImplementedError("Legacy file migration not implemented yet")

    def save_to_directory(self, run_dir: Path) -> None:
        """Save metadata to run directory."""
        metadata_file = run_dir / "run_metadata.json"
        self.to_json(str(metadata_file))

    def generate_model_card(self) -> str:
        """Generate a model card README for this run."""

        # Get base model info
        final_stage = self.get_final_stage()

        # Determine model type based on stages
        if len(self.stages) == 1:
            model_description = f"{self.stages[0].stage_type.upper()} fine-tuned model"
        else:
            stage_names = [s.stage_type.upper() for s in self.stages]
            model_description = (
                f"Multi-stage fine-tuned model ({' → '.join(stage_names)})"
            )

        card = f"""---
base_model: {self.model}
library_name: peft
tags:
{chr(10).join(f"- {tag}" for tag in self.tags)}
pipeline_tag: text-generation
---

# {self.run_name}

{model_description} trained on the {self.dataset_name} dataset.

## Model Details

- **Model Type**: {model_description}
- **Base Model**: {self.model}
- **Dataset**: {self.dataset_name}
- **Training Type**: {self.run_type.replace('_', ' ').title()}
"""

        if self.chain_info:
            card += f"- **Chained From**: {self.chain_info.parent_run}\n"

        if self.pipeline_config_name:
            card += f"- **Pipeline**: {self.pipeline_config_name}\n"

        if self.suite_name:
            card += f"- **Experiment Suite**: {self.suite_name}\n"

        card += f"""
- **Created**: {self.created_at}
- **Status**: {self.status}

## Training Stages

"""

        for i, stage in enumerate(self.stages, 1):
            card += f"""### Stage {i}: {stage.stage_type.upper()}

- **Dataset**: {stage.dataset_name}
- **Starting Model**: {stage.starting_model}
"""
            if stage.learning_rate:
                card += f"- **Learning Rate**: {stage.learning_rate}\n"
            if stage.num_epochs:
                card += f"- **Epochs**: {stage.num_epochs}\n"
            if stage.final_loss:
                card += f"- **Final Loss**: {stage.final_loss:.4f}\n"
            if stage.eval_loss:
                card += f"- **Eval Loss**: {stage.eval_loss:.4f}\n"
            card += "\n"

        card += f"""## Usage

```python
# Load the model
python -m scripts.inference {self.dataset_name}/{self.run_name}
```

## Files

- `adapter_model.safetensors`: LoRA adapter weights
- `model_config.json`: Model configuration  
- `peft_config.json`: PEFT/LoRA configuration
- `run_metadata.json`: Complete training metadata
- `trainer_state.json`: Detailed training logs
"""

        if self.wandb_run_urls:
            card += "\n## Training Logs\n\n"
            for i, url in enumerate(self.wandb_run_urls):
                stage_name = (
                    self.stages[i].stage_type
                    if i < len(self.stages)
                    else f"Stage {i+1}"
                )
                card += f"- [{stage_name.upper()}]({url})\n"

        return card
