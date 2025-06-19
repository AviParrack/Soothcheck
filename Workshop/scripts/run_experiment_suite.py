#!/usr/bin/env python
# This script runs experiment suites for systematic parameter exploration.
# It allows users to define parameter grids and launch multiple experiments
# with different combinations of models, pipelines, and datasets.
#
# Example usage:
# python -m scripts.run_experiment_suite model_comparison --dry_run
# python -m scripts.run_experiment_suite dataset_ablation --max_concurrent 2
# python -m scripts.run_experiment_suite hyperparameter_sweep
#
# Arguments:
#   suite_name (str): Name of the experiment suite configuration file (without .json extension)
#                     located in `configs/experiment_suites/`.
#   --dry_run (flag): Show experiment plan without running experiments.
#   --max_concurrent (int): Override max concurrent experiments.

import argparse
import os
import sys
from pathlib import Path

# Set PyTorch CUDA allocation conf to potentially reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Ensure src is in path for imports if running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config_models.experiment_config import ExperimentSuite
from src.training.experiment_suite_runner import run_experiment_suite


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment suites for systematic parameter exploration.",
        epilog="Example usage:\n"
        "  python -m scripts.run_experiment_suite model_comparison --dry_run\n"
        "  python -m scripts.run_experiment_suite dataset_ablation --max_concurrent 2\n"
        "  python -m scripts.run_experiment_suite hyperparameter_sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "suite_name",
        type=str,
        help="Name of the experiment suite configuration (e.g., 'model_comparison'). The script will look for configs/experiment_suites/[suite_name].json",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show experiment plan without running experiments.",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        help="Override maximum number of concurrent experiments.",
    )

    args = parser.parse_args()

    config_file_name = args.suite_name
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

    base_config_dir = project_root_path / "configs" / "experiment_suites"
    actual_config_path = base_config_dir / config_file_name

    if not actual_config_path.exists():
        print(
            f"Error: Experiment suite configuration file not found at {actual_config_path}"
        )
        print(f"Please ensure '{config_file_name}' exists in '{base_config_dir}'.")
        sys.exit(1)

    # --- 1. Load Configuration ---
    print(f"Loading experiment suite configuration from: {actual_config_path}")
    suite_config = ExperimentSuite.from_json(str(actual_config_path))

    # Apply command-line overrides
    if args.dry_run:
        suite_config.dry_run = True

    if args.max_concurrent is not None:
        suite_config.max_concurrent_runs = args.max_concurrent

    # --- 2. Run Experiment Suite ---
    try:
        run_experiment_suite(suite_config)

    except Exception as e:
        print(f"Experiment suite failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
