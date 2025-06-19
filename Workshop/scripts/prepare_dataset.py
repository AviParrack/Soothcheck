#!/usr/bin/env python
# This script prepares a dataset for fine-tuning by processing raw data files
# (text or JSONL) into Arrow format, applying chat templates as defined
# by the tokenizer specified in the dataset configuration.
# It reads a DatasetConfig JSON file (e.g., from configs/data/) which must
# contain the 'tokenizer_name_or_path' field for this process.
# The script deletes any existing processed dataset with the same name before creating a new one.
#
# Note: If you want to run data augmentations (which create additional raw files),
# run the augment_dataset script first: python -m scripts.augment_dataset [config_name] [raw_dataset_name]
#
# Example usage:
# python -m scripts.prepare_dataset general_config adam sft
# (where general_config.json in configs/data/ defines tokenizer and builds,
#  adam is the raw dataset name, and sft is the build name)
# python -m scripts.prepare_dataset general_config adam
# (builds all builds defined in the configuration)
#
# Arguments:
#   config_name (str): Name of the dataset configuration file (without .json extension)
#                      located in `configs/data/`. This config file must specify
#                      the `tokenizer_name_or_path` and `builds` dictionary.
#   raw_dataset_name (str): Name of the raw dataset directory under datasets/raw/
#   build_name (str, optional): Name of the build to create (must exist in config builds dictionary).
#                               If not provided, all builds will be created.

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Set TOKENIZERS_PARALLELISM to false to avoid warnings when forking processes
# after tokenizers have been used (e.g., by datasets.save_to_disk).
# This should be set before any transformers/tokenizers imports if possible,
# or at least before they are heavily used.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# When running `python -m scripts.prepare_dataset` from the project root,
# the project root is added to sys.path, allowing absolute imports from src.
from src.config_models.data_config import DatasetConfig
from src.data.data import orchestrate_dataset_preparation
from src.data.data_utils import delete_processed_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for finetuning. Processes raw data (text or JSONL) into Arrow format, applying chat templates. Deletes any existing processed dataset with the same name first. The [config_name].json file must specify 'tokenizer_name_or_path' and 'builds'.",
        epilog="Example usage: python -m scripts.prepare_dataset general_config adam sft \n(This implies 'configs/data/general_config.json' exists, defines tokenizer and builds, raw data is in 'datasets/raw/adam', and creates 'adam_sft' processed dataset)\nOr: python -m scripts.prepare_dataset general_config adam \n(This builds all builds defined in the configuration)",
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the dataset configuration (e.g., 'general_config'). The script will look for 'configs/data/[config_name].json'. This JSON file must define 'tokenizer_name_or_path' and 'builds'.",
    )
    parser.add_argument(
        "raw_dataset_name",
        type=str,
        help="Name of the raw dataset directory under 'datasets/raw/' (e.g., 'adam').",
    )
    parser.add_argument(
        "build_name",
        type=str,
        nargs="?",
        help="Name of the build to create (e.g., 'sft', 'dpo'). Must exist in the config's builds dictionary. If not provided, all builds will be created.",
    )
    args = parser.parse_args()

    config_file_name = args.config_name
    if not config_file_name.endswith(".json"):
        config_file_name += ".json"

    project_root_path = os.getcwd()
    actual_config_path = os.path.join(
        project_root_path, "configs", "data", config_file_name
    )

    if not os.path.exists(actual_config_path):
        print(f"Error: Configuration file not found at {actual_config_path}")
        print(f"Looked in: {os.path.join('configs', 'data', config_file_name)}")
        print("Please create the dataset configuration file or check the name.")
        return

    print(f"Loading dataset configuration from: {actual_config_path}")
    try:
        dataset_config = DatasetConfig.from_json(actual_config_path)
    except Exception as e:
        print(f"Error loading dataset configuration from {actual_config_path}: {e}")
        return

    # Determine which builds to process
    if args.build_name is not None:
        # Validate the build name exists in configuration
        if args.build_name not in dataset_config.builds:
            print(f"Error: Build '{args.build_name}' not found in configuration.")
            print(f"Available builds: {list(dataset_config.builds.keys())}")
            return
        builds_to_process = [args.build_name]
    else:
        # Process all builds
        builds_to_process = list(dataset_config.builds.keys())
        print(f"No build name specified. Will process all builds: {builds_to_process}")

    # Check if raw dataset directory exists
    raw_data_dir = dataset_config.get_absolute_raw_data_path(args.raw_dataset_name)
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw dataset directory not found: {raw_data_dir}")
        print(
            f"Please ensure the raw dataset '{args.raw_dataset_name}' exists under datasets/raw/"
        )
        return

    # Process each build
    for build_name in builds_to_process:
        output_name = (
            f"{args.raw_dataset_name}{dataset_config.builds[build_name].output_suffix}"
        )
        print(f"\n{'='*60}")
        print(
            f"Processing build '{build_name}' for raw dataset '{args.raw_dataset_name}'"
        )
        print(f"Output will be saved as: {output_name}")
        print(f"{'='*60}")

        print("Step 1: Checking for and deleting any existing processed dataset...")
        if not delete_processed_dataset(
            dataset_config, args.raw_dataset_name, build_name
        ):
            print(
                f"Halting processing of build '{build_name}' because deletion of existing processed dataset '{output_name}' failed."
            )
            continue
        print("Step 1: Completed (Existing dataset deleted or none found).")

        print(
            f"Step 2: Starting dataset preparation for output: {output_name} from raw data at {args.raw_dataset_name}..."
        )
        try:
            orchestrate_dataset_preparation(
                dataset_config, args.raw_dataset_name, build_name
            )
            print(
                f"Step 2: Dataset preparation script finished successfully for: {output_name}"
            )
        except Exception as e:
            print(
                f"Step 2: An error occurred during dataset preparation for '{output_name}': {e}"
            )
            import traceback

            traceback.print_exc()
            print(f"Dataset preparation failed for build '{build_name}'.")
            continue

    print(f"\n{'='*60}")
    print("All requested builds have been processed.")


if __name__ == "__main__":
    if "src" not in sys.path[0] and "src" not in "".join(sys.path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Adjusted sys.path to include project root: {project_root}")
    main()
