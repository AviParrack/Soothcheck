# This file contains the main pipeline orchestration logic for multi-stage training.
# It manages the execution of training stages in sequence, handling model passing
# between stages, dataset loading, and output management.
#
# Classes:
# - PipelineRunner: Main orchestrator for multi-stage training pipelines
#
# Functions:
# - run_training_pipeline: Main entry point for running a complete training pipeline

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from transformers import set_seed
from datasets import load_from_disk

from ..config_models import TrainConfig, ModelConfig, PeftConfig as PydanticPeftConfig
from ..config_models.stage_configs import BaseStageConfig
from ..config_models.run_metadata import RunMetadata, StageInfo
from .stage_runner import StageRunner
from .run_manager import RunManager


class PipelineRunner:
    """Main orchestrator for multi-stage training pipelines."""

    def __init__(
        self,
        train_config: TrainConfig,
        model_config: ModelConfig,
        peft_config: Optional[PydanticPeftConfig],
        dataset_name: str,
        run_name: Optional[str] = None,
        model_name: Optional[str] = None,
        stage_config_selections: Optional[Dict[str, str]] = None,
        resolved_peft_adapter_path: Optional[str] = None,
    ):
        self.train_config = train_config
        self.model_config = model_config
        self.peft_config = peft_config
        self.dataset_name = dataset_name
        self.stage_config_selections = stage_config_selections
        self.resolved_peft_adapter_path = resolved_peft_adapter_path

        # Use RunManager for unified directory structure
        self.run_manager = RunManager()

        # Determine effective model name for metadata and run naming
        self.effective_model_name = (
            model_name
            or train_config.model
            or model_config.model_name_or_path.split("/")[
                -1
            ]  # Extract model name from path
        )

        # Get stage types for run name generation
        # For run name generation, we only need stage types, so we can use the stage references directly
        stage_types = [stage_ref.type for stage_ref in train_config.stages]

        # Generate run name if not provided
        self.run_name = self.run_manager.create_run_name(
            base_name=run_name,
            run_type="pipeline",
            stage_types=stage_types,
            model=self.effective_model_name,
            dataset_name=dataset_name,
            include_timestamp=True,
        )

        # Set up output directory using unified structure
        self.output_dir = self.run_manager.create_run_directory(
            dataset_name=dataset_name, run_name=self.run_name, run_type="pipeline"
        )

        # Track stages for metadata
        self.completed_stages: List[StageInfo] = []
        self.start_time = datetime.now()

    def run(self, inspect_batches: int = 0) -> None:
        """Run the complete training pipeline."""
        try:
            # Set random seed
            set_seed(self.train_config.seed)
            print(f"Random seed set to: {self.train_config.seed}")

            # Get parsed stage configurations with dataset context
            stage_configs = self.train_config.get_stage_configs(self.dataset_name, self.stage_config_selections)
            print(
                f"Running pipeline '{self.train_config.pipeline_name}' with {len(stage_configs)} stages:"
            )
            for i, stage_config in enumerate(stage_configs):
                print(
                    f"  Stage {i+1}: {stage_config.stage_name} (dataset: {stage_config.processed_dataset_name})"
                )

            # Run each stage in sequence
            for stage_idx, stage_config in enumerate(stage_configs):
                print(f"\n{'='*50}")
                print(
                    f"Starting Stage {stage_idx + 1}/{len(stage_configs)}: {stage_config.stage_name}"
                )
                print(f"{'='*50}")

                # Determine model path for this stage
                if stage_idx == 0:
                    # First stage: check if we have a resolved PEFT adapter path from model specification
                    base_model_hf_id = self.model_config.model_name_or_path
                    peft_adapter_path = self.resolved_peft_adapter_path
                    starting_model_path = peft_adapter_path or base_model_hf_id
                    if peft_adapter_path:
                        print(f"✓ First stage will load PEFT adapter from: {peft_adapter_path}")
                    else:
                        print(f"✓ First stage will load base model: {base_model_hf_id}")
                else:
                    # Subsequent stages use the previous stage output (which is a PEFT adapter)
                    previous_stage_dir = (
                        self.output_dir
                        / f"stage_{stage_configs[stage_idx-1].stage_name}"
                    )
                    base_model_hf_id = self.model_config.model_name_or_path
                    peft_adapter_path = str(previous_stage_dir)
                    starting_model_path = peft_adapter_path

                # For pipelines, stages save to subdirectories within the main run directory
                # But the final model is saved at the run root level
                stage_work_dir = self.output_dir / f"stage_{stage_config.stage_name}"
                stage_work_dir.mkdir(parents=True, exist_ok=True)

                # Create stage-specific run name for wandb
                stage_run_name = f"{self.run_name}_stage_{stage_config.stage_name}"

                # Create and run the stage
                stage_runner = StageRunner(
                    stage_config=stage_config,
                    model_config=self.model_config,
                    peft_config=self.peft_config,
                    base_model_hf_id=base_model_hf_id,
                    peft_adapter_path=peft_adapter_path,
                    output_dir=stage_work_dir,
                    run_name=stage_run_name,
                    report_to=self.train_config.report_to,
                    wandb_project=self.train_config.wandb_project,
                    pipeline_info={
                        "pipeline_name": self.train_config.pipeline_name,
                        "dataset_name": self.dataset_name,
                        "stage_index": stage_idx,
                        "total_stages": len(stage_configs),
                    },
                )

                stage_runner.run(
                    inspect_batches=inspect_batches if stage_idx == 0 else 0
                )

                # Create stage info for metadata with actual training results
                training_results = stage_runner.get_training_results()
                stage_info = StageInfo(
                    stage_type=stage_config.stage_name,
                    stage_config_name=f"{stage_config.stage_name}_default",  # TODO: Get actual config name
                    dataset_name=stage_config.processed_dataset_name,
                    starting_model=starting_model_path,
                    final_loss=(
                        training_results.get("final_loss") if training_results else None
                    ),
                    total_steps=(
                        training_results.get("total_steps")
                        if training_results
                        else None
                    ),
                    learning_rate=(
                        stage_config.learning_rate
                        if hasattr(stage_config, "learning_rate")
                        else None
                    ),
                    num_epochs=(
                        stage_config.num_train_epochs
                        if hasattr(stage_config, "num_train_epochs")
                        else None
                    ),
                )
                self.completed_stages.append(stage_info)

                print(
                    f"Stage {stage_idx + 1} completed. Working files saved to: {stage_work_dir}"
                )

                # If we're only inspecting batches, stop after first stage
                if inspect_batches > 0:
                    print("Batch inspection complete. Stopping pipeline early.")
                    return

            # Copy final model to the main run directory
            final_stage_dir = self.output_dir / f"stage_{stage_configs[-1].stage_name}"
            for file in final_stage_dir.glob("adapter_model.*"):
                import shutil

                shutil.copy2(file, self.output_dir / file.name)
            for file in final_stage_dir.glob("*.json"):
                if file.name not in [
                    "run_metadata.json"
                ]:  # Don't copy stage-specific metadata
                    import shutil

                    shutil.copy2(file, self.output_dir / file.name)

            print(f"\n{'='*50}")
            print(
                f"Pipeline '{self.train_config.pipeline_name}' completed successfully!"
            )
            print(f"Final model saved to: {self.output_dir}")
            print(f"{'='*50}")

            # Save unified metadata and generate model card
            self._save_unified_metadata()

        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _save_unified_metadata(self) -> None:
        """Save unified run metadata and model card."""
        try:
            # Create unified run metadata
            run_metadata = RunMetadata(
                run_name=self.run_name,
                dataset_name=self.dataset_name,
                run_type="pipeline",
                created_at=self.start_time.isoformat(),
                completed_at=datetime.now().isoformat(),
                model=self.effective_model_name,
                peft_config_name=self.train_config.peft_config_name,
                pipeline_config_name=self.train_config.pipeline_name,
                stages=self.completed_stages,
                status="completed",
                tags=[
                    "pipeline",
                    self.train_config.pipeline_name,
                    self.effective_model_name,
                    *[stage.stage_type for stage in self.completed_stages],
                ],
            )

            # Save unified metadata and generate model card
            self.run_manager.save_run_metadata(run_metadata, self.output_dir)

            # Save legacy configs for compatibility
            self.model_config.to_json(self.output_dir / "model_config.json")
            if self.peft_config:
                self.peft_config.to_json(self.output_dir / "peft_config.json")

            print(f"Run metadata and model card saved to: {self.output_dir}")
            print(
                f"Inference command: python -m scripts.inference {self.dataset_name}/{self.run_name}"
            )

        except Exception as e:
            print(f"Warning: Could not save run metadata: {e}")
            import traceback

            traceback.print_exc()


def run_training_pipeline(
    train_config: TrainConfig,
    model_config: ModelConfig,
    peft_config: Optional[PydanticPeftConfig],
    dataset_name: str,
    run_name: Optional[str] = None,
    inspect_batches: int = 0,
    model_name: Optional[str] = None,
    stage_config_selections: Optional[Dict[str, str]] = None,
    resolved_peft_adapter_path: Optional[str] = None,
) -> None:
    """
    Main entry point for running a complete training pipeline.

    Args:
        train_config: Pipeline configuration
        model_config: Model configuration
        peft_config: PEFT configuration (optional)
        dataset_name: Name of the dataset (raw dataset name, e.g., 'adam')
        run_name: Custom run name (optional)
        inspect_batches: Number of batches to inspect instead of training
        model_name: Optional model name for metadata (if different from model config)
        stage_config_selections: Optional stage config file selections
        resolved_peft_adapter_path: Optional PEFT adapter path from model resolution
    """
    runner = PipelineRunner(
        train_config=train_config,
        model_config=model_config,
        peft_config=peft_config,
        dataset_name=dataset_name,
        run_name=run_name,
        model_name=model_name,
        stage_config_selections=stage_config_selections,
        resolved_peft_adapter_path=resolved_peft_adapter_path,
    )

    runner.run(inspect_batches=inspect_batches)
