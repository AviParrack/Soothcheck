# This file handles file processing logic for data augmentations.
# It contains functions for finding files, processing single files, and running
# augmentations in parallel.
#
# Key functions:
# - find_files_for_augmentation: Finds files to process for a specific augmentation
# - process_single_file_for_augmentation: Processes a single file with one augmentation
# - process_augmentation: Processes all files for a single augmentation in parallel
# - run_augmentations: Main orchestration function for running all augmentations

import os
import glob
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import inspect
import random

from src.config_models.data_config import DatasetConfig
from src.data.augment.augmentation_registry import get_augmentation_by_name
from src.data.data_utils import (
    read_file_contents,
    find_files_for_component,
    validate_sources_exist,
)


def find_files_for_augmentation(
    config: DatasetConfig, raw_dataset_name: str, augmentation_config_dict: dict
) -> List[str]:
    """
    Finds files for a specific augmentation based on its sources.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        augmentation_config_dict: Dictionary containing augmentation configuration.

    Returns:
        List of absolute file paths for the augmentation to process.
    """
    return find_files_for_component(
        config, raw_dataset_name, augmentation_config_dict, "augmentation"
    )


def process_single_file_for_augmentation(
    file_path: str,
    augmentation_name: str,
    augmentation_instance,
    augmentation_config_dict: dict,
    raw_data_dir_abs: str,
    **kwargs,
) -> Tuple[str, int]:
    """
    Process a single file with a single augmentation.

    Args:
        file_path: Path to the file being processed
        augmentation_name: Name of the augmentation
        augmentation_instance: Instance of the augmentation
        augmentation_config_dict: Configuration for the augmentation
        raw_data_dir_abs: Absolute path to raw data directory
        **kwargs: Additional parameters to pass to augmentation (e.g., model_type)

    Returns:
        tuple: (file_path, created_files_count)
    """
    file_content = read_file_contents(file_path)
    if file_content is None:
        return file_path, 0

    # Prepare augmentation arguments
    augmentation_args = {
        "file_path": file_path,
        "file_content": file_content,
        "raw_data_dir": raw_data_dir_abs,
    }

    # Add all other parameters from the augmentation_config_dict
    # Filter out 'name' and 'sources' as they are not augment() args.
    specific_augmentation_params = {
        k: v
        for k, v in augmentation_config_dict.items()
        if k not in ["name", "sources"]
    }
    augmentation_args.update(specific_augmentation_params)

    # Add any additional kwargs (like model_type)
    augmentation_args.update(kwargs)

    created_files = augmentation_instance.augment(**augmentation_args)
    return file_path, len(created_files) if created_files else 0


def process_augmentation(
    augmentation_config_dict: dict,
    config: DatasetConfig,
    raw_dataset_name: str,
    raw_data_dir_abs: str,
    max_concurrent_files: int = 2,
    randomize_files: bool = True,
    **augmentation_kwargs,
) -> Dict[str, int]:
    """
    Process all files for a single augmentation in parallel.

    Args:
        augmentation_config_dict: Configuration for the augmentation
        config: The dataset configuration
        raw_dataset_name: Name of the raw dataset
        raw_data_dir_abs: Absolute path to raw data directory
        max_concurrent_files: Maximum concurrent files to process
        randomize_files: Whether to randomize file processing order (helps ensure
                        failed runs don't always process the same files)
        **augmentation_kwargs: Additional parameters for augmentation creation

    Returns:
        Dictionary with processing statistics for this augmentation
    """
    augmentation_name = augmentation_config_dict["name"]
    print(f"\n{'='*60}")
    print(f"Processing augmentation: {augmentation_name}")
    print(f"{'='*60}")

    # Find files for this augmentation
    files_to_process = find_files_for_augmentation(
        config, raw_dataset_name, augmentation_config_dict
    )

    if not files_to_process:
        print(f"No files found for {augmentation_name} augmentation.")
        return {"processed_files": 0, "skipped_files": 0, "created_files": 0}

    # Randomize file order if requested
    if randomize_files:
        random.shuffle(files_to_process)
        print(f"Randomized processing order for {len(files_to_process)} files")

    print(f"Found {len(files_to_process)} files to process for {augmentation_name}")

    # Create augmentation instance
    try:
        augmentation_instance = create_single_augmentation_instance(
            augmentation_config_dict, **augmentation_kwargs
        )
        if augmentation_instance is None:
            print(f"Failed to create instance for {augmentation_name}, skipping.")
            return {
                "processed_files": 0,
                "skipped_files": len(files_to_process),
                "created_files": 0,
            }
    except Exception as e:
        print(f"Error creating {augmentation_name} instance: {e}")
        return {
            "processed_files": 0,
            "skipped_files": len(files_to_process),
            "created_files": 0,
        }

    # Process files in parallel
    total_created_files = 0
    processed_files = 0
    skipped_files = 0

    print(f"Using {max_concurrent_files} workers for parallel processing")

    with tqdm(
        total=len(files_to_process), desc=f"Processing {augmentation_name}", unit="file"
    ) as progress_bar:
        with ThreadPoolExecutor(max_workers=max_concurrent_files) as executor:
            # Submit all file processing tasks
            future_to_file = {
                executor.submit(
                    process_single_file_for_augmentation,
                    file_path,
                    augmentation_name,
                    augmentation_instance,
                    augmentation_config_dict,
                    raw_data_dir_abs,
                    **augmentation_kwargs,
                ): file_path
                for file_path in files_to_process
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                filename = os.path.basename(file_path)
                try:
                    _, created_files_count = future.result()
                    total_created_files += created_files_count
                    processed_files += 1
                    progress_bar.set_postfix({"created": total_created_files})
                except Exception as e:
                    progress_bar.write(f"    ✗ Error processing {filename}: {e}")
                    skipped_files += 1
                finally:
                    progress_bar.update(1)

    # Finalize this augmentation
    finalized_files_count = 0
    if hasattr(augmentation_instance, "finalize"):
        try:
            print(f"Finalizing {augmentation_name} augmentation...")
            # Check if finalize method accepts parameters
            finalize_signature = inspect.signature(augmentation_instance.finalize)
            if "raw_data_dir" in finalize_signature.parameters:
                finalized_files = augmentation_instance.finalize(
                    raw_data_dir=raw_data_dir_abs
                )
            else:
                finalized_files = augmentation_instance.finalize()

            if finalized_files:
                finalized_files_count = len(finalized_files)
                total_created_files += finalized_files_count
                print(
                    f"✓ Finalized {augmentation_name}, created {finalized_files_count} additional files"
                )
            else:
                print(f"✓ Finalized {augmentation_name}, no additional files created")
        except Exception as e:
            print(f"✗ Error finalizing {augmentation_name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{augmentation_name} complete:")
    print(f"  Processed: {processed_files} files")
    print(f"  Skipped: {skipped_files} files")
    print(f"  Created: {total_created_files} augmented files")

    return {
        "processed_files": processed_files,
        "skipped_files": skipped_files,
        "created_files": total_created_files,
    }


def create_single_augmentation_instance(
    augmentation_config_dict: dict, **available_params
) -> Any:
    """
    Create a single augmentation instance from configuration dictionary.

    Args:
        augmentation_config_dict: Augmentation configuration
        **available_params: Any parameters that might be needed by augmentations
                           (e.g., author_name, model_name, temperature, etc.)

    Returns:
        Augmentation instance or None if creation failed
    """
    from src.data.augment.augmentation_registry import AVAILABLE_AUGMENTATIONS

    augmentation_name = augmentation_config_dict["name"]
    try:
        # Extract constructor parameters from config (exclude 'name' and 'sources')
        constructor_params = {
            k: v
            for k, v in augmentation_config_dict.items()
            if k not in ["name", "sources"]
        }

        # Dynamically determine what parameters this augmentation needs
        aug_class = AVAILABLE_AUGMENTATIONS.get(augmentation_name)
        if aug_class:
            # Check what parameters this augmentation class requires
            required_params = aug_class.get_required_params()

            # Add any matching available parameters
            for param_name in required_params:
                if param_name in available_params:
                    constructor_params[param_name] = available_params[param_name]
                    print(
                        f"Adding {param_name} '{available_params[param_name]}' to {augmentation_name} augmentation"
                    )

        return get_augmentation_by_name(augmentation_name, **constructor_params)
    except ValueError as e:
        print(f"Warning: {e}")
        return None


def run_augmentations(
    config: DatasetConfig, raw_dataset_name: str, **kwargs
) -> Dict[str, int]:
    """
    Runs all configured augmentations sequentially, processing files in parallel for each augmentation.

    Args:
        config: The dataset configuration
        raw_dataset_name: Name of the raw dataset
        **kwargs: Any parameters that might be needed by augmentations
                  (e.g., author_name, model_name, temperature, randomize_files, etc.)

    Returns:
        Dictionary with overall processing statistics
    """
    raw_data_dir_abs = config.get_absolute_raw_data_path(raw_dataset_name)

    if not os.path.isdir(raw_data_dir_abs):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir_abs}")

    if not config.data_augmentations:
        print("No data augmentations configured. Nothing to do.")
        return {"processed_files": 0, "skipped_files": 0, "created_files": 0}

    # Convert augmentation configs to dicts for easier handling
    augmentations_to_run_config_dicts = [
        a.model_dump() for a in config.data_augmentations
    ]

    # Validate that all source folders exist before processing
    print("Validating source folders...")
    if not validate_sources_exist(
        config, raw_dataset_name, augmentations_to_run_config_dicts, "augmentation"
    ):
        raise ValueError("Aborting augmentation due to missing source folders.")

    # Check if augmented directory exists and has files
    augmented_dir = os.path.join(raw_data_dir_abs, "augmented")
    existing_augmented_files = []
    if os.path.exists(augmented_dir):
        existing_augmented_files = glob.glob(os.path.join(augmented_dir, "*.aug.*"))

    if existing_augmented_files:
        print(
            f"Found {len(existing_augmented_files)} existing augmented files. They will be overwritten as needed."
        )

    print(
        f"\nStarting augmentation pipeline with {len(augmentations_to_run_config_dicts)} augmentations..."
    )

    # Process each augmentation sequentially, files in parallel
    total_stats = {"processed_files": 0, "skipped_files": 0, "created_files": 0}

    max_concurrent_files = kwargs.pop("max_concurrent_files", 10)
    randomize_files = kwargs.pop("randomize_files", True)

    for i, augmentation_config_dict in enumerate(augmentations_to_run_config_dicts, 1):
        augmentation_name = augmentation_config_dict["name"]
        print(
            f"\n[{i}/{len(augmentations_to_run_config_dicts)}] Starting {augmentation_name}..."
        )

        try:
            stats = process_augmentation(
                augmentation_config_dict,
                config,
                raw_dataset_name,
                raw_data_dir_abs,
                max_concurrent_files,
                randomize_files,
                **kwargs,
            )

            # Accumulate stats
            for key in total_stats:
                total_stats[key] += stats[key]

        except Exception as e:
            print(f"✗ Failed to process {augmentation_name}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("AUGMENTATION PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Overall results:")
    print(f"  Total processed: {total_stats['processed_files']} files")
    print(f"  Total skipped: {total_stats['skipped_files']} files")
    print(f"  Total created: {total_stats['created_files']} augmented files")

    return total_stats
