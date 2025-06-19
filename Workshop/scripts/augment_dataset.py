#!/usr/bin/env python
# This script runs data augmentations on raw data files to create additional
# raw data files. Augmentations act on raw data to create more raw data,
# unlike pipelines which process raw data during conversion to processed data.
# Augmented files are saved with "aug.{something}" suffixes in an "augmented"
# folder within the raw dataset directory.
#
# Example usage:
# python -m scripts.augment_dataset general adam
# (where general.json in configs/data/ defines data_augmentations
#  and adam is the raw dataset name under datasets/raw/)
#
# To run only a specific augmentation:
# python -m scripts.augment_dataset general adam --augmentation qa
# (runs only the 'qa' augmentation, regardless of config)
#
# To control threading (useful for rate limiting or faster processing):
# python -m scripts.augment_dataset general adam --max-threads 5
# (processes up to 5 files concurrently, leading to 5 parallel LLM calls)
#
# To process files in deterministic order (useful for debugging):
# python -m scripts.augment_dataset general adam --no-randomize-files
# (processes files in lexicographical order, same every run)
#
# Arguments:
#   config_name (str): Name of the dataset configuration file (without .json extension)
#                      located in `configs/data/`. This config file should specify
#                      the `data_augmentations` list.
#   raw_dataset_name (str): Name of the raw dataset directory under datasets/raw/
#   --augmentation (str, optional): Name of a specific augmentation to run.
#                                   When specified, only this augmentation will run,
#                                   ignoring the data_augmentations in the config.
#   --max-threads (int, optional): Maximum number of files to process concurrently.
#                                  Controls parallel LLM calls. Default is 2.
#   --randomize-files (flag, optional): Randomize file processing order to avoid
#                                       processing the same files if runs fail.

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# When running `python -m scripts.augment_dataset` from the project root,
# the project root is added to sys.path, allowing absolute imports from src.
from src.config_models.data_config import DatasetConfig
from src.data.augment.augmentation_registry import get_augmentation_by_name
from src.data.augment.file_processor import run_augmentations


def main():
    parser = argparse.ArgumentParser(
        description="Run data augmentations on raw data files. Augmentations create additional raw files that can then be processed by pipelines.",
        epilog="Example usage: python -m scripts.augment_dataset general adam \n(This implies 'configs/data/general.json' exists, defines 'data_augmentations', and 'datasets/raw/adam' exists)\n\nTo run only a specific augmentation: python -m scripts.augment_dataset general adam --augmentation qa\n\nTo control threading: python -m scripts.augment_dataset general adam --max-threads 5\n\nTo disable file randomization: python -m scripts.augment_dataset general adam --no-randomize-files",
    )
    parser.add_argument(
        "config_name",
        type=str,
        help="Name of the dataset configuration (e.g., 'general_config'). The script will look for 'configs/data/[config_name].json'.",
    )
    parser.add_argument(
        "raw_dataset_name",
        type=str,
        help="Name of the raw dataset directory under 'datasets/raw/' (e.g., 'adam').",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        help="Name of a specific augmentation to run (e.g., 'qa', 'takes'). When specified, only this augmentation will run, ignoring the data_augmentations list in the config file.",
    )
    parser.add_argument(
        "--author_name",
        type=str,
        help="Name of the author to use in conversation augmentations (e.g., 'Adam Smith', 'Jane Doe'). This personalizes the generated conversations.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gemini", "local"],
        default="gemini",
        help="Model type to use for augmentation: 'gemini' for Gemini API or 'local' for local Llama 3.3 inference server.",
    )
    parser.add_argument(
        "--max-threads",
        type=int,
        default=2,
        help="Maximum number of files to process concurrently. Controls parallel LLM calls. Default is 2.",
    )
    parser.add_argument(
        "--randomize-files",
        action="store_true",
        default=True,
        help="Randomize file processing order to avoid processing the same files if runs fail. Default is True.",
    )
    parser.add_argument(
        "--no-randomize-files",
        dest="randomize_files",
        action="store_false",
        help="Process files in deterministic order (lexicographical).",
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

    # Check if raw dataset directory exists
    raw_data_dir = dataset_config.get_absolute_raw_data_path(args.raw_dataset_name)
    if not os.path.isdir(raw_data_dir):
        print(f"Error: Raw dataset directory not found: {raw_data_dir}")
        print(
            f"Please ensure the raw dataset '{args.raw_dataset_name}' exists under datasets/raw/"
        )
        return

    # Override data_augmentations if a specific augmentation is requested
    if args.augmentation:
        print(
            f"Running only the '{args.augmentation}' augmentation (overriding config)"
        )

        # Look for the augmentation in the existing config first
        augmentation_config_to_use = None
        for existing_aug in dataset_config.data_augmentations:
            if existing_aug.name == args.augmentation:
                augmentation_config_to_use = existing_aug
                print(
                    f"Using sources from config for augmentation '{args.augmentation}'"
                )
                break

        if augmentation_config_to_use is None:
            # Not found in config, fail with helpful error message
            print(
                f"Error: Augmentation '{args.augmentation}' is not defined in the config file."
            )
            print(
                f"Please add it to the 'data_augmentations' list in {config_file_name}"
            )
            print("Available augmentations in config:")
            for aug in dataset_config.data_augmentations:
                print(f"  - {aug.name}")
            return

        # Validate that the augmentation exists with its config parameters
        try:
            # Pass CLI parameters plus config parameters to the validation call
            validation_params = {}
            if args.author_name:
                validation_params["author_name"] = args.author_name

            # Add config parameters (exclude 'name' and 'sources')
            config_dict = augmentation_config_to_use.model_dump()
            config_params = {
                k: v for k, v in config_dict.items() if k not in ["name", "sources"]
            }
            validation_params.update(config_params)

            get_augmentation_by_name(args.augmentation, **validation_params)
        except ValueError as e:
            print(f"Error: {e}")
            return

        # Use only the specified augmentation
        dataset_config.data_augmentations = [augmentation_config_to_use]

    print(
        f"Running augmentations for raw dataset '{args.raw_dataset_name}' with config from {config_file_name}"
    )

    try:
        # Run augmentations using the file processor
        # Collect all available parameters
        available_params = {}
        if args.author_name:
            available_params["author_name"] = args.author_name
        available_params["model_type"] = args.model
        available_params["max_concurrent_files"] = args.max_threads
        available_params["randomize_files"] = args.randomize_files
        # Future parameters can be added here as needed

        result = run_augmentations(
            dataset_config, args.raw_dataset_name, **available_params
        )
        print("Augmentation script finished successfully.")
        print(f"Final stats: {result}")
    except Exception as e:
        print(f"An error occurred during augmentation: {e}")
        import traceback

        traceback.print_exc()
        print("Augmentation failed.")


if __name__ == "__main__":
    if "src" not in sys.path[0] and "src" not in "".join(sys.path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Adjusted sys.path to include project root: {project_root}")
    main()
