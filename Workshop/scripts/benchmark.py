#!/usr/bin/env python
# This script runs evaluation benchmarks on trained models from experiment suites
# or individual model runs. It loads evaluation datasets by name from
# evals/eval_datasets/ and saves responses to evals/model_responses/.
#
# Example usage:
# python -m scripts.benchmark example_basic --suite model_comparison
# python -m scripts.benchmark coding_test --suite suite1 --suite suite2 --model adam/my_run
# python -m scripts.benchmark math_problems --model gemma-3-27b-it --model qwen-32b-4bit --model experiments/rudolf/exp1_stage_sft
#
# Arguments:
#   eval_dataset_name: Name of the evaluation dataset (without .json extension)
#   --suite: Name(s) of experiment suites to evaluate (can specify multiple)
#   --model: Model to evaluate - can be config name (e.g. gemma-3-27b-it) or experiment path (e.g. rudolf/my_run)
#   --max-new-tokens: Maximum tokens to generate (default: 4096)
#   --temperature: Sampling temperature (default: 0.7)
#   --top-p: Nucleus sampling threshold (default: 0.9)
#   --run-name: Optional name for this evaluation run (for organizing outputs)

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List

from tqdm import tqdm

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_models.eval_configs import (
    EvalDataset,
    ModelSpec,
)
from src.eval.run_models import evaluate_model_on_dataset, expand_suite_to_model_specs
from src.utils.model_resolver import resolve_model_specification


def create_model_spec(model_str: str) -> ModelSpec:
    """
    Create a ModelSpec from a model specification string using unified model resolution.

    The model_str can be:
    - Config name (e.g., "gemma-3-27b-it-4bit")
    - Experiment path (e.g., "rudolf/my_run" or "experiments/rudolf/my_run")
    - HuggingFace model ID (if resolution fails, treated as direct ID)

    Args:
        model_str: Model specification string

    Returns:
        ModelSpec object with appropriate source_type and display_name
    """
    try:
        # Try to resolve using the unified model resolver
        resolution = resolve_model_specification(model_str)

        if resolution.is_experiment:
            return ModelSpec(
                source_type="run_path",
                source_path=model_str,
                display_name=f"Experiment: {model_str}",
            )
        else:
            return ModelSpec(
                source_type="model_config",
                source_path=model_str,
                display_name=f"Config: {model_str}",
            )

    except ValueError:
        # If resolution fails, treat as direct HuggingFace model ID
        print(
            f"Warning: Could not resolve '{model_str}' as config/experiment, treating as HuggingFace model ID"
        )
        return ModelSpec(
            source_type="base_model",
            source_path=model_str,
            display_name=f"HF Model: {model_str}",
        )


def load_eval_dataset(dataset_name: str) -> EvalDataset:
    """Load evaluation dataset from evals/eval_datasets/ by name."""
    # Follow project convention: dataset name without extension
    dataset_path = PROJECT_ROOT / "evals" / "eval_datasets" / f"{dataset_name}.json"

    print(f"\nLoading evaluation dataset: {dataset_name}")
    print(f"From path: {dataset_path}")

    if not dataset_path.exists():
        # List available datasets for helpful error message
        eval_datasets_dir = PROJECT_ROOT / "evals" / "eval_datasets"
        available_datasets = [f.stem for f in eval_datasets_dir.glob("*.json")]

        error_msg = f"Dataset '{dataset_name}' not found."
        if available_datasets:
            error_msg += (
                f"\nAvailable datasets: {', '.join(sorted(available_datasets))}"
            )
        else:
            error_msg += f"\nNo datasets found in {eval_datasets_dir}"

        raise FileNotFoundError(error_msg)

    with open(dataset_path, "r") as f:
        data = json.load(f)

    # Create EvalDataset from the JSON data
    eval_dataset = EvalDataset.from_simple_json(data, dataset_name)

    print(f"Loaded {len(eval_dataset.prompts)} prompts from '{dataset_name}'")

    return eval_dataset


def get_output_directory(dataset_name: str) -> Path:
    """
    Create and return the output directory path following project conventions.

    Structure: evals/model_responses/{dataset_name}/
    """
    output_dir = PROJECT_ROOT / "evals" / "model_responses" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def collect_all_model_specs(
    suite_names: List[str], additional_model_strs: List[str]
) -> List[ModelSpec]:
    """
    Collect all model specifications from suites and additional models.

    Args:
        suite_names: List of experiment suite names
        additional_model_strs: List of model specification strings

    Returns:
        List of all ModelSpec objects to evaluate
    """
    all_specs = []

    # Expand experiment suites
    for suite_name in suite_names:
        try:
            suite_specs = expand_suite_to_model_specs(suite_name)
            all_specs.extend(suite_specs)
            print(f"Added {len(suite_specs)} models from suite '{suite_name}'")
        except Exception as e:
            print(f"Warning: Failed to expand suite '{suite_name}': {e}")

    # Add additional models using unified resolution
    for model_str in additional_model_strs:
        try:
            spec = create_model_spec(model_str)
            all_specs.append(spec)
            print(f"Added model: {spec.display_name}")
        except Exception as e:
            print(f"Warning: Failed to create spec for '{model_str}': {e}")

    return all_specs


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation benchmarks on trained models. Evaluates models from experiment suites or individual runs on a dataset of prompts.",
        epilog="""Examples:
  python -m scripts.benchmark example_basic --suite model_comparison
  python -m scripts.benchmark coding_test --suite suite1 --model adam/my_run
  python -m scripts.benchmark math_problems --model gemma-3-27b-it --model qwen-32b-4bit --model experiments/rudolf/exp1_stage_sft
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "eval_dataset_name",
        type=str,
        help="Name of the evaluation dataset (without .json extension)",
    )

    # Model selection
    parser.add_argument(
        "--suite",
        type=str,
        action="append",
        default=[],
        help="Name of experiment suite to evaluate (can specify multiple)",
    )
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        default=[],
        help="Model to evaluate. Can be config name (e.g. 'gemma-3-27b-it'), experiment path (e.g. 'rudolf/my_run'), or HuggingFace model ID (can specify multiple)",
    )

    # Generation settings
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold (default: 0.9)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.suite and not args.model:
        parser.error("At least one --suite or --model must be specified")

    # Load evaluation dataset
    try:
        eval_dataset = load_eval_dataset(args.eval_dataset_name)
    except Exception as e:
        print(f"Error loading evaluation dataset: {e}")
        return 1

    # Collect all model specifications
    print("\nCollecting models to evaluate...")
    all_model_specs = collect_all_model_specs(args.suite, args.model)

    if not all_model_specs:
        print("No models found to evaluate!")
        return 1

    print(f"\nTotal models to evaluate: {len(all_model_specs)}")

    # Create output directory
    output_dir = get_output_directory(args.eval_dataset_name)

    # Prepare evaluation config
    eval_config = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "batch_size": 1,  # Fixed at 1 for now
        "eval_dataset_path": args.eval_dataset_name,
        "output_dir": str(output_dir),
        "suites": args.suite,
        "additional_models": args.model,
    }

    # Save evaluation configuration
    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)
    print(f"\nSaved evaluation config to: {config_path}")

    # Run evaluation for each model
    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)

    results_summary = {
        "eval_dataset": args.eval_dataset_name,
        "num_prompts": len(eval_dataset.prompts),
        "timestamp": datetime.now().isoformat(),
        "models_evaluated": [],
    }

    # Progress bar for models
    model_progress = tqdm(all_model_specs, desc="Evaluating models", unit="model")

    for i, model_spec in enumerate(model_progress):
        model_name = model_spec.display_name or model_spec.source_path
        model_progress.set_description(f"Evaluating {model_name}")

        print(f"\n\n{'='*80}")
        print(f"MODEL {i+1}/{len(all_model_specs)}: {model_name}")
        print(f"{'='*80}")

        try:
            # Evaluate the model
            model_responses = evaluate_model_on_dataset(
                model_spec=model_spec,
                eval_dataset=eval_dataset,
                eval_config=eval_config,
                output_dir=str(output_dir),
            )

            # Add to summary
            results_summary["models_evaluated"].append(
                {
                    "model_id": model_responses.model_id,
                    "model_path": model_responses.model_path,
                    "base_model": model_responses.base_model,
                    "peft_applied": model_responses.peft_applied,
                    "successful_responses": model_responses.successful_responses,
                    "total_prompts": model_responses.total_prompts,
                    "eval_time_seconds": model_responses.total_time_seconds,
                }
            )

            print(f"\n✓ Successfully evaluated {model_name}")

        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()

            # Add error to summary
            results_summary["models_evaluated"].append(
                {
                    "model_path": model_spec.source_path,
                    "error": str(e),
                    "status": "failed",
                }
            )

    # Save summary
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_path}")
    print(
        f"Successfully evaluated: {len([m for m in results_summary['models_evaluated'] if 'error' not in m])}/{len(all_model_specs)} models"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
