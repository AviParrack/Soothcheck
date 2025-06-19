#!/usr/bin/env python
# This script is the main entry point for running multi-stage training pipelines.
# It loads a pipeline configuration and executes all stages in sequence,
# passing models between stages and managing outputs.
#
# Example usage:
# python -m scripts.train_pipeline sft_then_dpo --dataset adam --model llama-3.3-70B-Instruct
# python -m scripts.train_pipeline sft_then_dpo --dataset adam --run_name my_custom_run
# python -m scripts.train_pipeline sft_then_dpo --dataset adam --inspect_batches 2
#
# Arguments:
#   config_name (str): Name of the pipeline configuration file (without .json extension)
#                      located in `configs/train/`. The pipeline config can optionally specify
#                      the model config, or it can be provided via --model argument.
#   --dataset (str): Name of the raw dataset (e.g., 'adam'). The pipeline will look for
#                    stage-specific processed datasets like 'adam_sft', 'adam_dpo'.
#   --model (str, optional): Model specification override. Can be config name, experiment path,
#                            or HF model ID. If not provided, uses model from pipeline config.
#   --run_name (str, optional): Custom name for this run. If not provided, will be
#                               auto-generated from model, dataset, and pipeline names.
#   --inspect_batches (int, optional): If > 0, inspects the specified number of training
#                                     batches and exits instead of training.

import argparse
import os
import sys
from pathlib import Path
import json
from typing import Optional, Dict, Any

# Set PyTorch CUDA allocation conf to potentially reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure src is in path for imports if running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config_models import (
    TrainConfig,
    ModelConfig,
    PeftConfig as PydanticPeftConfig,
)
from src.config_models.run_metadata import RunMetadata, StageInfo
from src.training.pipeline_runner import run_training_pipeline
from src.training.run_manager import RunManager
from src.utils.model_resolver import resolve_model_specification


def load_pipeline_configs(
    config_name: str,
    model_override: Optional[str] = None,
    peft_config_override: Optional[str] = None,
    project_root: Optional[Path] = None,
) -> tuple[TrainConfig, ModelConfig, Optional[PydanticPeftConfig], Optional[str]]:
    """
    Load and resolve pipeline configurations.

    Args:
        config_name: Name of the pipeline configuration (without .json extension)
        model_override: Optional model specification override
        peft_config_override: Optional PEFT config name override
        project_root: Optional project root path

    Returns:
        Tuple of (train_config, model_config, peft_config, resolved_peft_adapter_path)
    """
    config_file_name = config_name
    if not config_file_name.endswith(".json"):
        config_file_name += ".json"

    # Determine project root and config path
    if project_root is None:
        current_working_dir = Path.cwd()
        if (current_working_dir / "scripts").exists() and (
            current_working_dir / "src"
        ).exists():
            project_root = current_working_dir
        else:
            project_root = Path(__file__).resolve().parent.parent
            if not (project_root / "configs").exists():
                raise RuntimeError(
                    f"Could not reliably determine project root. Looked for configs directory from {project_root}"
                )

    base_config_dir = project_root / "configs" / "train"
    actual_config_path = base_config_dir / config_file_name

    if not actual_config_path.exists():
        raise FileNotFoundError(
            f"Pipeline configuration file not found at {actual_config_path}. "
            f"Please ensure '{config_file_name}' exists in '{base_config_dir}'."
        )

    # Load pipeline configuration
    train_config = TrainConfig.from_json(str(actual_config_path))

    # Handle model override
    resolved_peft_adapter_path: Optional[str] = None
    if model_override:
        model_resolution = resolve_model_specification(
            model_override, project_root=project_root
        )
        model_config = model_resolution.model_config
        resolved_peft_adapter_path = model_resolution.peft_adapter_path
        if resolved_peft_adapter_path:
            print(f"✓ Resolved PEFT adapter path from model specification: {resolved_peft_adapter_path}")
    elif train_config.computed_model_config_path:
        model_config = ModelConfig.from_json(train_config.computed_model_config_path)
    else:
        raise ValueError(
            f"No model specified. Either set 'model' in pipeline config '{config_name}' "
            f"or provide a model_override parameter."
        )

    # Handle PEFT config override
    pydantic_peft_config: PydanticPeftConfig | None = None
    if peft_config_override:
        peft_config_name = peft_config_override
        if not peft_config_name.endswith(".json"):
            peft_config_name += ".json"
        peft_config_path = project_root / "configs" / "peft" / peft_config_name

        if not peft_config_path.exists():
            raise FileNotFoundError(
                f"PEFT configuration file not found at {peft_config_path}"
            )

        pydantic_peft_config = PydanticPeftConfig.from_json(str(peft_config_path))
    elif train_config.computed_peft_config_path:
        pydantic_peft_config = PydanticPeftConfig.from_json(
            train_config.computed_peft_config_path
        )

    return train_config, model_config, pydantic_peft_config, resolved_peft_adapter_path


def validate_datasets(train_config: TrainConfig, dataset_name: str, stage_config_selections: Optional[Dict[str, str]] = None) -> None:
    """
    Validate that all required processed datasets exist.

    Args:
        train_config: The training configuration
        dataset_name: Base dataset name
        stage_config_selections: Optional stage config file selections
    """
    stage_configs = train_config.get_stage_configs(dataset_name, stage_config_selections)
    missing_datasets = []

    for stage_config in stage_configs:
        dataset_path = (
            Path("datasets") / "processed" / stage_config.processed_dataset_name
        )
        if not dataset_path.exists():
            missing_datasets.append(stage_config.processed_dataset_name)

    if missing_datasets:
        raise FileNotFoundError(
            f"The following processed datasets are missing: {missing_datasets}. "
            f"Please run prepare_dataset.py to create these datasets first."
        )


def run_pipeline_programmatic(
    config_name: str,
    dataset_name: str,
    run_name: Optional[str] = None,
    inspect_batches: int = 0,
    model_override: Optional[str] = None,
    peft_config_override: Optional[str] = None,
    stage_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    stage_config_selections: Optional[Dict[str, str]] = None,
    project_root: Optional[Path] = None,
) -> None:
    """
    Run a training pipeline programmatically without command line arguments.

    Args:
        config_name: Name of the pipeline configuration
        dataset_name: Base dataset name
        run_name: Optional custom run name
        inspect_batches: Number of batches to inspect (0 = run training)
        model_override: Optional model specification override
        peft_config_override: Optional PEFT config name override
        stage_overrides: Optional stage-specific parameter overrides
        stage_config_selections: Optional stage config file selections
        project_root: Optional project root path
    """
    # Load configurations
    train_config, model_config, pydantic_peft_config, resolved_peft_adapter_path = load_pipeline_configs(
        config_name=config_name,
        model_override=model_override,
        peft_config_override=peft_config_override,
        project_root=project_root,
    )

    # Validate datasets
    validate_datasets(train_config, dataset_name, stage_config_selections)

    # TODO: Apply stage_overrides when implemented in pipeline runner
    if stage_overrides:
        print(
            f"Warning: stage_overrides not yet implemented, ignoring: {stage_overrides}"
        )
    
    if stage_config_selections:
        print(f"Using stage config selections: {stage_config_selections}")

    # Run the pipeline
    run_training_pipeline(
        train_config=train_config,
        model_config=model_config,
        peft_config=pydantic_peft_config,
        dataset_name=dataset_name,
        run_name=run_name,
        inspect_batches=inspect_batches,
        model_name=model_override,  # Pass the override as the effective model name
        stage_config_selections=stage_config_selections,
        resolved_peft_adapter_path=resolved_peft_adapter_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run a multi-stage training pipeline.",
        epilog="Example usage:\n"
        "  python -m scripts.train_pipeline sft_then_dpo --dataset adam\n"
        "  python -m scripts.train_pipeline sft_then_dpo --dataset adam --run_name my_run\n"
        "  python -m scripts.train_pipeline sft_then_dpo --dataset adam --inspect_batches 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the pipeline configuration (e.g., 'sft_then_dpo'). The script will look for configs/train/[config_name].json",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the raw dataset (e.g., 'adam'). The pipeline will infer stage-specific processed datasets like 'adam_sft', 'adam_dpo'.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Custom name for this run. If not provided, will be auto-generated.",
    )
    parser.add_argument(
        "--inspect_batches",
        type=int,
        default=0,
        help="If > 0, inspects the specified number of training batches and exits instead of training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Optional: Override model specification. Can be config name, experiment path, or HF model ID. If not provided, uses model from pipeline config.",
    )
    parser.add_argument(
        "--peft_config",
        type=str,
        help="Optional: Override PEFT configuration name (e.g., 'lora_aggressive'). If not provided, uses PEFT config from pipeline config.",
    )
    parser.add_argument(
        "--stage_overrides",
        type=str,
        help='Optional: JSON string with stage-specific parameter overrides. Format: \'{"sft": {"learning_rate": 0.0001}}\'',
    )

    args = parser.parse_args()

    # Parse stage overrides if provided
    stage_overrides = None
    if args.stage_overrides:
        try:
            stage_overrides = json.loads(args.stage_overrides)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in stage_overrides: {e}")
            print('Expected format: \'{"sft": {"learning_rate": 0.0001}}\'')
            sys.exit(1)

    try:
        print(f"Starting pipeline '{args.config_name}' for dataset '{args.dataset}'...")

        run_pipeline_programmatic(
            config_name=args.config_name,
            dataset_name=args.dataset,
            run_name=args.run_name,
            inspect_batches=args.inspect_batches,
            model_override=args.model,
            peft_config_override=args.peft_config,
            stage_overrides=stage_overrides,
        )

        print("Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
