#!/usr/bin/env python
# This script runs a single training stage using the new stage-based configuration system.
# It can run SFT, DPO, or other training stages independently, useful for testing
# or running individual stages without going through a full pipeline.
#
# Example usage:
# python -m scripts.train_stage sft_default --dataset_name adam_sft --model gemma-3-1b-it
# python -m scripts.train_stage dpo_default --dataset_name adam_dpo --model adam/baseline_sft
# python -m scripts.train_stage sft_default --dataset_name adam_sft --inspect_batches 2
#
# Arguments:
#   config_name (str): Name of the stage configuration file (without .json extension)
#                      located in `configs/stage_configs/`. (e.g., 'sft_default', 'dpo_default')
#   --dataset_name (str): Name of the processed dataset to use (e.g., 'adam_sft', 'adam_dpo').
#   --model (str, optional): Either a model config name (e.g., 'gemma-3-1b-it') or
#                                       path to an experimental run (e.g., 'adam/baseline_sft').
#   --run_name (str, optional): Custom name for this run (auto-generated if not provided).
#   --peft_config_name (str, optional): Name of PEFT config (default: 'lora_default').
#   --inspect_batches (int, optional): If > 0, inspects batches instead of training.

import argparse
import os

# Set PyTorch CUDA allocation conf to potentially reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
from pathlib import Path
from datetime import datetime

# Ensure src is in path for imports if running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config_models import (
    TrainConfig,
    ModelConfig,
    PeftConfig as PydanticPeftConfig,
    SFTStageConfig,
    DPOStageConfig,
)
from src.config_models.run_metadata import RunMetadata, StageInfo
from src.training.stage_runner import StageRunner
from src.training.run_manager import RunManager
from src.utils.model_resolver import resolve_model_specification


def main():
    parser = argparse.ArgumentParser(
        description="Run a single training stage.",
        epilog="Example usage:\n"
        "  python -m scripts.train_stage sft_default --dataset_name adam_sft --model gemma-3-1b-it\n"
        "  python -m scripts.train_stage dpo_default --dataset_name adam_dpo --model adam/my_sft_run\n"
        "  python -m scripts.train_stage sft_default --dataset_name adam_sft --model gemma-3-1b-it --inspect_batches 2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the stage configuration (e.g., 'sft_default'). The script will look for configs/stage_configs/[config_name].json",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the processed dataset to use (e.g., 'adam_sft', 'adam_dpo').",
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
        required=True,
        help="Either a model config name (e.g., 'gemma-3-1b-it') or path to an experimental run (e.g., 'adam/baseline_sft').",
    )
    parser.add_argument(
        "--peft_config_name",
        type=str,
        default="lora_default",
        help="Name of the PEFT configuration (e.g., 'lora_default'). Path will be configs/peft/[name].json.",
    )

    args = parser.parse_args()

    config_file_name = args.config_name
    if not config_file_name.endswith(".json"):
        config_file_name += ".json"

    # Determine project root and config path
    current_working_dir = Path.cwd()
    if (current_working_dir / "scripts").exists() and (
        current_working_dir / "src"
    ).exists():
        project_root_path = current_working_dir
    else:
        project_root_path = Path(__file__).resolve().parent.parent
        if not (project_root_path / "configs").exists():
            print(
                f"Error: Could not reliably determine project root. Looked for configs directory from {project_root_path}"
            )
            sys.exit(1)

    base_config_dir = project_root_path / "configs" / "stage_configs"
    actual_config_path = base_config_dir / config_file_name

    if not actual_config_path.exists():
        print(f"Error: Stage configuration file not found at {actual_config_path}")
        print(f"Please ensure '{config_file_name}' exists in '{base_config_dir}'.")
        sys.exit(1)

    # --- 1. Load Configurations ---
    print(f"Loading stage configuration from: {actual_config_path}")

    # Load the stage config directly as JSON first to determine stage type
    import json

    with open(actual_config_path, "r") as f:
        stage_config_dict = json.load(f)

    # Determine stage type and create appropriate config object
    stage_name = stage_config_dict.get("stage_name", "")
    if stage_name == "sft":
        stage_config = SFTStageConfig(**stage_config_dict)
    elif stage_name == "dpo":
        stage_config = DPOStageConfig(**stage_config_dict)
    else:
        raise ValueError(f"Unknown stage type: {stage_name}. Expected 'sft' or 'dpo'.")

    # Resolve model config and starting model using unified resolution
    print(f"Resolving model specification: {args.model}")
    try:
        model_resolution = resolve_model_specification(
            args.model, project_root=project_root_path
        )

        model_config = model_resolution.model_config
        base_model_hf_id = model_resolution.base_model_hf_id
        peft_adapter_path = model_resolution.peft_adapter_path

        print(f"✓ Resolved model specification '{args.model}':")
        print(
            f"  - Model config source: {'experiment' if model_resolution.is_experiment else 'config'}"
        )
        print(f"  - Model type: {model_resolution.model_type}")
        print(f"  - Base model HF ID: {base_model_hf_id}")
        if peft_adapter_path:
            print(f"  - PEFT adapter path: {peft_adapter_path}")
        if model_resolution.experiment_path:
            print(f"  - Experiment path: {model_resolution.experiment_path}")

    except ValueError as e:
        print(f"Error: Could not resolve model specification '{args.model}'")
        print(f"Details: {e}")
        sys.exit(1)

    # Override the dataset name from command line
    stage_config.processed_dataset_name = args.dataset_name

    # Build PEFT config path
    pydantic_peft_config: PydanticPeftConfig | None = None
    if args.peft_config_name:
        peft_config_name = args.peft_config_name
        if not peft_config_name.endswith(".json"):
            peft_config_name += ".json"
        peft_config_path = project_root_path / "configs" / "peft" / peft_config_name

        print(f"Loading PEFT configuration from: {peft_config_path}")
        pydantic_peft_config = PydanticPeftConfig.from_json(str(peft_config_path))

    # --- 2. Setup Run with Unified System ---

    # Extract dataset base name from processed dataset name (e.g., 'adam_sft' -> 'adam')
    if "_" in args.dataset_name:
        dataset_base_name = args.dataset_name.rsplit("_", 1)[0]
    else:
        dataset_base_name = args.dataset_name

    # Use RunManager to create run name and directory
    run_manager = RunManager()
    run_name = run_manager.create_run_name(
        base_name=args.run_name,
        run_type="single_stage",
        stage_types=[stage_config.stage_name],
        model=args.model,
        dataset_name=dataset_base_name,
        include_timestamp=True,
    )

    output_dir = run_manager.create_run_directory(
        dataset_name=dataset_base_name, run_name=run_name, run_type="single_stage"
    )

    print(f"Run name: {run_name}")
    print(f"Output directory: {output_dir}")

    # --- 3. Validate Dataset Exists ---
    dataset_path = Path("datasets") / "processed" / args.dataset_name
    if not dataset_path.exists():
        print(f"Error: Processed dataset not found at {dataset_path}")
        print(f"Please run prepare_dataset.py to create '{args.dataset_name}' first.")
        sys.exit(1)

    # --- 4. Run Stage ---
    try:
        print(f"Starting single-stage training: {stage_config.stage_name}")

        stage_runner = StageRunner(
            stage_config=stage_config,
            model_config=model_config,
            peft_config=pydantic_peft_config,
            base_model_hf_id=base_model_hf_id,
            peft_adapter_path=peft_adapter_path,
            output_dir=output_dir,
            run_name=run_name,
            report_to=["wandb"],
            wandb_project="pollux",
            pipeline_info={
                "pipeline_name": f"single_stage_{stage_config.stage_name}",
                "dataset_name": dataset_base_name,
                "stage_index": 0,
                "total_stages": 1,
            },
        )

        stage_runner.run(inspect_batches=args.inspect_batches)

        if args.inspect_batches == 0:
            print("Single-stage training completed successfully!")
            print(f"Model saved to: {output_dir}")

            # Create unified run metadata with actual training results
            training_results = stage_runner.get_training_results()
            stage_info = StageInfo(
                stage_type=stage_config.stage_name,
                stage_config_name=args.config_name,
                dataset_name=args.dataset_name,
                starting_model=peft_adapter_path or base_model_hf_id,
                final_loss=(
                    training_results.get("final_loss") if training_results else None
                ),
                total_steps=(
                    training_results.get("total_steps") if training_results else None
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

            run_metadata = RunMetadata(
                run_name=run_name,
                dataset_name=dataset_base_name,
                run_type="single_stage",
                created_at=datetime.now().isoformat(),
                completed_at=datetime.now().isoformat(),
                model=args.model,
                peft_config_name=args.peft_config_name,
                stages=[stage_info],
                status="completed",
                tags=[stage_config.stage_name, args.model],
            )

            # Save unified metadata and generate model card
            run_manager.save_run_metadata(run_metadata, output_dir)

            # Save legacy configs for compatibility
            model_config.to_json(output_dir / "model_config.json")
            if pydantic_peft_config:
                pydantic_peft_config.to_json(output_dir / "peft_config.json")

            print(f"Run metadata and model card saved to: {output_dir}")
            print(
                f"Inference command: python -m scripts.inference {dataset_base_name}/{run_name}"
            )
        else:
            print("Batch inspection completed.")

    except Exception as e:
        print(f"Single-stage training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
