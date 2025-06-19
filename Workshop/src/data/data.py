# This file is responsible for orchestrating dataset preparation.
# It finds raw data files, then uses a DataPipeline to process their contents.
# For "chat" format, the pipeline produces a list of role/content dictionaries in a "messages" field.
# For "text" format, it produces a string in a "text" field.
# The processed data is then split and saved.
#
# Key Functions:
# - orchestrate_dataset_preparation: Main entry point.
# - find_dataset_files: Locates raw data files based on extensions.
# - find_files_for_pipeline: Locates files for a specific pipeline based on its sources.
# - create_sft_dataset: Uses a DataPipeline to process file contents for SFT training.
# - create_dpo_dataset: Uses a DataPipeline to process file contents for DPO training.
# - split_and_format_dataset: Splits the dataset and gathers statistics.

import os
import glob
import shutil
import json  # Added for JSONL parsing
import re  # Added for regex matching of files for pipelines
from typing import List, Tuple, Dict, Any, Optional
from datasets import (
    DatasetDict,
    Dataset,
)  # Removed load_dataset as we construct manually
from transformers import AutoTokenizer, PreTrainedTokenizer

from ..config_models.data_config import DatasetConfig
from .data_utils import (
    get_token_count_for_split,
    read_file_contents,
    is_file_extension_allowed,
    find_files_in_source,
    find_files_for_component,
    validate_sources_exist,
)  # Import new function
from .pipelines import (
    BasicPipeline,
    get_pipeline_by_name,
)  # Import the pipeline and getter

# from ..config_models.model_config import ModelConfig # No longer needed here

# Removed load_tokenizer_for_dataset function as it's no longer needed.


def find_dataset_files(raw_data_dir_abs: str, file_extensions: List[str]) -> List[str]:
    """
    Finds all files in the raw_data_dir with the given extensions.
    """
    all_file_paths = []
    print(
        f"Searching for files in: {raw_data_dir_abs} with extensions: {file_extensions}"
    )
    for ext in file_extensions:
        # Ensure extensions start with a dot for consistent matching
        normalized_ext = ext if ext.startswith(".") else "." + ext
        pattern = os.path.join(raw_data_dir_abs, f"**/*{normalized_ext}")
        found_files = glob.glob(pattern, recursive=True)
        all_file_paths.extend(found_files)
        if found_files:
            print(f"Found {len(found_files)} file(s) with extension '{normalized_ext}'")
    if not all_file_paths:
        print(f"No files found with extensions {file_extensions} in {raw_data_dir_abs}")
    return all_file_paths


def find_files_for_pipeline(
    config: DatasetConfig, raw_dataset_name: str, pipeline_config_dict: Dict[str, Any]
) -> List[str]:
    """
    Finds files for a specific pipeline based on its sources.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        pipeline_config_dict: Dictionary containing pipeline configuration.

    Returns:
        List of absolute file paths for the pipeline to process.
    """
    return find_files_for_component(
        config, raw_dataset_name, pipeline_config_dict, "pipeline"
    )


# Removed process_single_file_with_chunking as its logic is integrated into create_sft_dataset and create_dpo_dataset


def create_sft_dataset(
    config: DatasetConfig,
    raw_dataset_name: str,
    build_name: str,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """
    Uses a DataPipeline to process file contents for SFT training.
    The output Dataset will have a "source" column and either a "messages" column (for chat format)
    or a "text" column (for text format). It will also include a "pipeline_name" column.
    """
    processed_items: List[Dict[str, Any]] = []

    print(
        f"Processing files using tokenizer: {tokenizer.name_or_path} (for chunking/pipeline ops)"
    )

    # Get the build configuration
    if build_name not in config.builds:
        print(f"Error: Build '{build_name}' not found in configuration.")
        return Dataset.from_dict(
            {"messages": [], "source": [], "pipeline_name": [], "augmentation_name": []}
        )

    build_config = config.builds[build_name]
    pipelines_to_run_config_dicts = [
        p.model_dump() for p in build_config.data_pipelines
    ]

    if not pipelines_to_run_config_dicts:
        print(
            f"No data_pipelines specified for build '{build_name}'. Dataset will be empty."
        )
        if config.get_effective_prompt_format(build_name) == "chat":
            return Dataset.from_dict(
                {
                    "messages": [],
                    "source": [],
                    "pipeline_name": [],
                    "augmentation_name": [],
                }
            )
        else:  # "text"
            return Dataset.from_dict(
                {"text": [], "source": [], "pipeline_name": [], "augmentation_name": []}
            )

    # Validate that all source folders exist before processing
    print("Validating source folders...")
    if not validate_sources_exist(
        config, raw_dataset_name, pipelines_to_run_config_dicts, "pipeline"
    ):
        print("Aborting dataset creation due to missing source folders.")
        if config.get_effective_prompt_format(build_name) == "chat":
            return Dataset.from_dict(
                {
                    "messages": [],
                    "source": [],
                    "pipeline_name": [],
                    "augmentation_name": [],
                }
            )
        else:  # "text"
            return Dataset.from_dict(
                {"text": [], "source": [], "pipeline_name": [], "augmentation_name": []}
            )

    for pipeline_config_item_dict in pipelines_to_run_config_dicts:
        pipeline_name = pipeline_config_item_dict["name"]

        # Find files for this pipeline
        pipeline_files = find_files_for_pipeline(
            config, raw_dataset_name, pipeline_config_item_dict
        )

        if not pipeline_files:
            print(f"  No files found for pipeline '{pipeline_name}'. Skipping.")
            continue

        try:
            pipeline_instance = get_pipeline_by_name(pipeline_name)
        except ValueError as e:
            print(f"  Warning: {e}. Skipping pipeline {pipeline_name}.")
            continue

        print(
            f"  Processing {len(pipeline_files)} files with {type(pipeline_instance).__name__}"
        )

        for file_path in pipeline_files:
            filename = os.path.basename(file_path)

            file_content = read_file_contents(file_path)
            if file_content is None:
                print(f"    Skipping file {filename} due to read error.")
                continue

            # Prepare pipeline arguments using build-specific settings
            pipeline_args = {
                "file_path": file_path,
                "file_content": file_content,
                "tokenizer": tokenizer,
                "prompt_format": config.get_effective_prompt_format(build_name),
                "enable_chunking": config.get_effective_enable_chunking(build_name),
                "max_length": config.get_effective_max_length(build_name),
                "chunk_overlap_ratio": config.get_effective_chunk_overlap_ratio(build_name),
                "dataset_split_seed": config.split_seed,  # This stays global
            }

            # Add all other parameters from the pipeline_config_item_dict
            # These are the pipeline-specific parameters like min_grab_length, etc.
            # Filter out 'name' and 'sources' as they are not process() args.
            specific_pipeline_params = {
                k: v
                for k, v in pipeline_config_item_dict.items()
                if k not in ["name", "sources"]
            }
            pipeline_args.update(specific_pipeline_params)

            try:
                items_from_file = pipeline_instance.process(**pipeline_args)
                # Add pipeline name to each item
                for item in items_from_file:
                    item["pipeline_name"] = pipeline_name
                processed_items.extend(items_from_file)
                if items_from_file:
                    print(
                        f"    Successfully processed {filename} with {pipeline_name}, yielded {len(items_from_file)} items."
                    )
                elif (
                    file_content.strip()
                ):  # Pipeline ran but no items, and file was not empty
                    print(
                        f"    Note: {filename} yielded no items from pipeline {pipeline_name} despite having content."
                    )
            except Exception as e:
                print(
                    f"    Warning: Error processing file content from {filename} with pipeline {type(pipeline_instance).__name__}: {e}. Skipping."
                )
                import traceback

                traceback.print_exc()  # More detailed error for debugging
                continue

    if not processed_items:
        print("Warning: No items were processed. Resulting dataset will be empty.")
        if config.get_effective_prompt_format(build_name) == "chat":
            return Dataset.from_dict(
                {
                    "messages": [],
                    "source": [],
                    "pipeline_name": [],
                    "augmentation_name": [],
                }
            )
        else:  # "text"
            return Dataset.from_dict(
                {"text": [], "source": [], "pipeline_name": [], "augmentation_name": []}
            )

    if config.get_effective_prompt_format(build_name) == "chat":
        dict_for_dataset = {
            "messages": [
                item["messages"] for item in processed_items if "messages" in item
            ],
            "source": [
                item["source"] for item in processed_items if "messages" in item
            ],
            "pipeline_name": [
                item["pipeline_name"] for item in processed_items if "messages" in item
            ],
            "augmentation_name": [
                item.get("augmentation_name", "unknown")
                for item in processed_items
                if "messages" in item
            ],
        }
        if len(dict_for_dataset["messages"]) != len(processed_items):
            print(
                "Warning: Some items were dropped when constructing 'chat' dataset, possibly due to missing 'messages' field or other inconsistencies."
            )
        # Verify all lists have the same length
        list_lengths = [len(v) for v in dict_for_dataset.values()]
        if len(set(list_lengths)) > 1:
            print(
                f"ERROR: Inconsistent list lengths in chat dataset construction: {dict(zip(dict_for_dataset.keys(), list_lengths))}"
            )
            # Fallback to empty to prevent crash
            return Dataset.from_dict(
                {
                    "messages": [],
                    "source": [],
                    "pipeline_name": [],
                    "augmentation_name": [],
                }
            )
        return Dataset.from_dict(dict_for_dataset)
    else:  # "text"
        dict_for_dataset = {
            "text": [item["text"] for item in processed_items if "text" in item],
            "source": [item["source"] for item in processed_items if "text" in item],
            "pipeline_name": [
                item["pipeline_name"] for item in processed_items if "text" in item
            ],
            "augmentation_name": [
                item.get("augmentation_name", "unknown")
                for item in processed_items
                if "text" in item
            ],
        }
        if len(dict_for_dataset["text"]) != len(processed_items):
            print(
                "Warning: Some items were dropped when constructing 'text' dataset, possibly due to missing 'text' field or other inconsistencies."
            )
        # Verify all lists have the same length
        list_lengths = [len(v) for v in dict_for_dataset.values()]
        if len(set(list_lengths)) > 1:
            print(
                f"ERROR: Inconsistent list lengths in text dataset construction: {dict(zip(dict_for_dataset.keys(), list_lengths))}"
            )
            # Fallback to empty
            return Dataset.from_dict(
                {"text": [], "source": [], "pipeline_name": [], "augmentation_name": []}
            )
        return Dataset.from_dict(dict_for_dataset)


def create_dpo_dataset(
    config: DatasetConfig,
    raw_dataset_name: str,
    build_name: str,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    """
    Uses a DataPipeline to process file contents for DPO training.
    The output Dataset will have "prompt", "response_accepted", "response_rejected",
    "source", and "pipeline_name" columns.
    """
    processed_items: List[Dict[str, Any]] = []

    print(
        f"Processing files using tokenizer: {tokenizer.name_or_path} (for DPO dataset)"
    )

    # Get the build configuration
    if build_name not in config.builds:
        print(f"Error: Build '{build_name}' not found in configuration.")
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    build_config = config.builds[build_name]
    pipelines_to_run_config_dicts = [
        p.model_dump() for p in build_config.data_pipelines
    ]

    if not pipelines_to_run_config_dicts:
        print(
            f"No data_pipelines specified for build '{build_name}'. Dataset will be empty."
        )
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    # Validate that all source folders exist before processing
    print("Validating source folders...")
    if not validate_sources_exist(
        config, raw_dataset_name, pipelines_to_run_config_dicts, "pipeline"
    ):
        print("Aborting dataset creation due to missing source folders.")
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    for pipeline_config_item_dict in pipelines_to_run_config_dicts:
        pipeline_name = pipeline_config_item_dict["name"]

        # Find files for this pipeline
        pipeline_files = find_files_for_pipeline(
            config, raw_dataset_name, pipeline_config_item_dict
        )

        if not pipeline_files:
            print(f"  No files found for pipeline '{pipeline_name}'. Skipping.")
            continue

        try:
            pipeline_instance = get_pipeline_by_name(pipeline_name)
        except ValueError as e:
            print(f"  Warning: {e}. Skipping pipeline {pipeline_name}.")
            continue

        print(
            f"  Processing {len(pipeline_files)} files with {type(pipeline_instance).__name__}"
        )

        for file_path in pipeline_files:
            filename = os.path.basename(file_path)

            file_content = read_file_contents(file_path)
            if file_content is None:
                print(f"    Skipping file {filename} due to read error.")
                continue

            # Prepare pipeline arguments using build-specific settings
            pipeline_args = {
                "file_path": file_path,
                "file_content": file_content,
                "tokenizer": tokenizer,
                "prompt_format": config.get_effective_prompt_format(build_name),
                "enable_chunking": config.get_effective_enable_chunking(build_name),
                "max_length": config.get_effective_max_length(build_name),
                "chunk_overlap_ratio": config.get_effective_chunk_overlap_ratio(build_name),
                "dataset_split_seed": config.split_seed,
            }

            # Add all other parameters from the pipeline_config_item_dict
            specific_pipeline_params = {
                k: v
                for k, v in pipeline_config_item_dict.items()
                if k not in ["name", "sources"]
            }
            pipeline_args.update(specific_pipeline_params)

            try:
                items_from_file = pipeline_instance.process(**pipeline_args)
                # Add pipeline name to each item
                for item in items_from_file:
                    item["pipeline_name"] = pipeline_name
                processed_items.extend(items_from_file)
                if items_from_file:
                    print(
                        f"    Successfully processed {filename} with {pipeline_name}, yielded {len(items_from_file)} items."
                    )
                elif file_content.strip():
                    print(
                        f"    Note: {filename} yielded no items from pipeline {pipeline_name} despite having content."
                    )
            except Exception as e:
                print(
                    f"    Warning: Error processing file content from {filename} with pipeline {type(pipeline_instance).__name__}: {e}. Skipping."
                )
                import traceback

                traceback.print_exc()
                continue

    if not processed_items:
        print("Warning: No items were processed. Resulting dataset will be empty.")
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    # Construct DPO dataset
    dpo_items = [
        item
        for item in processed_items
        if "prompt" in item
        and "response_accepted" in item
        and "response_rejected" in item
    ]

    if not dpo_items:
        print(
            "Warning: No valid DPO items found (missing prompt/response_accepted/response_rejected fields)."
        )
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    if len(dpo_items) != len(processed_items):
        print(
            f"Warning: {len(processed_items) - len(dpo_items)} items were dropped due to missing DPO fields."
        )

    dict_for_dataset = {
        "prompt": [item["prompt"] for item in dpo_items],
        "response_accepted": [item["response_accepted"] for item in dpo_items],
        "response_rejected": [item["response_rejected"] for item in dpo_items],
        "source": [item["source"] for item in dpo_items],
        "pipeline_name": [item["pipeline_name"] for item in dpo_items],
        "augmentation_name": [
            item.get("augmentation_name", "unknown") for item in dpo_items
        ],
    }

    # Verify all lists have the same length
    list_lengths = [len(v) for v in dict_for_dataset.values()]
    if len(set(list_lengths)) > 1:
        print(
            f"ERROR: Inconsistent list lengths in DPO dataset construction: {dict(zip(dict_for_dataset.keys(), list_lengths))}"
        )
        return Dataset.from_dict(
            {
                "prompt": [],
                "response_accepted": [],
                "response_rejected": [],
                "source": [],
                "pipeline_name": [],
                "augmentation_name": [],
            }
        )

    return Dataset.from_dict(dict_for_dataset)


def split_and_format_dataset(
    full_dataset: Dataset,
    config: DatasetConfig,
    build_name: str,
    tokenizer: PreTrainedTokenizer,
) -> DatasetDict:
    """
    Splits the dataset into train/validation if configured, and returns a DatasetDict.
    Also prints token counts for each split using the provided tokenizer, if applicable (for text format).
    """
    validation_split_fraction = config.get_effective_validation_split_fraction(
        build_name
    )
    prompt_format = config.get_effective_prompt_format(build_name)

    dataset_dict_to_save: DatasetDict
    if validation_split_fraction and 0 < validation_split_fraction < 1:
        print(
            f"Splitting dataset into train/validation ({validation_split_fraction*100:.1f}% for validation)..."
        )
        print(f"  Split seed: {config.split_seed}")
        # Ensure the dataset has more than 1 example for splitting if validation_split_fraction is small
        if len(full_dataset) <= 1 and validation_split_fraction > 0:
            print(
                f"  Warning: Dataset has only {len(full_dataset)} item(s). Cannot create a validation split. Using all data for training."
            )
            dataset_dict_to_save = DatasetDict({"train": full_dataset})
        else:
            try:
                split_dataset_dict = full_dataset.train_test_split(
                    test_size=validation_split_fraction,
                    seed=config.split_seed,
                    shuffle=True,  # Good practice to shuffle before splitting
                )
                dataset_dict_to_save = split_dataset_dict
            except (
                ValueError
            ) as e:  # Handle cases like test_size too small for the dataset
                print(
                    f"  Warning: Could not split dataset (e.g., dataset too small for split): {e}. Using all data for training."
                )
                dataset_dict_to_save = DatasetDict({"train": full_dataset})
    else:
        print(
            "No validation split requested or split fraction is invalid. Using all data for 'train' split."
        )
        dataset_dict_to_save = DatasetDict({"train": full_dataset})

    # Print split sizes and token counts
    print("Dataset split summary:")
    for split_name, split_dataset in dataset_dict_to_save.items():
        num_items = len(split_dataset)
        token_count_str = (
            "(Token count not applicable for this format or tokenizer unavailable)"
        )

        if prompt_format == "text":
            if (
                "text" in split_dataset.column_names
            ):  # Check for hardcoded "text" column
                token_count_str = get_token_count_for_split(
                    split_dataset=split_dataset,
                    text_column="text",  # Hardcode to "text"
                    tokenizer=tokenizer,
                    split_name=split_name,
                )
            else:
                token_count_str = f"(Text column 'text' not found for token count)"
        elif prompt_format == "chat":
            token_count_str = "(Token counting for 'chat' messages format is TBD/handled at training time)"
        # else: handled by initial value of token_count_str

        print(
            f"  - '{split_name}' split: {num_items:,} items/chunks, {token_count_str}"
        )

    return dataset_dict_to_save


def orchestrate_dataset_preparation(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
):
    """
    Main orchestrator function for preparing a dataset.
    Does not handle deletion of prior processed datasets; that should be handled by the caller.
    """
    raw_data_dir_abs = config.get_absolute_raw_data_path(raw_dataset_name)
    processed_data_dir_abs = config.get_absolute_processed_data_path(
        raw_dataset_name, build_name
    )

    if not os.path.isdir(raw_data_dir_abs):
        print(f"Error: Raw data directory not found: {raw_data_dir_abs}")
        return

    # Load the tokenizer for this build
    tokenizer_name = config.get_effective_tokenizer_name_or_path(build_name)
    print(f"Loading tokenizer for build '{build_name}': {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(
                    f"Set tokenizer.pad_token to eos_token ('{tokenizer.eos_token}') for tokenizer '{tokenizer_name}'"
                )
        print(f"Tokenizer '{tokenizer_name}' loaded successfully.")
    except Exception as e:
        print(
            f"Fatal Error: Could not load tokenizer '{tokenizer_name}'. Exception: {e}"
        )
        return

    parent_of_processed_dir = os.path.dirname(processed_data_dir_abs)
    if not os.path.exists(parent_of_processed_dir):
        os.makedirs(parent_of_processed_dir, exist_ok=True)
        print(f"Created parent directory for processed data: {parent_of_processed_dir}")

    # Determine dataset type based on build name
    if build_name.lower() == "dpo":
        full_dataset = create_dpo_dataset(
            config=config,
            raw_dataset_name=raw_dataset_name,
            build_name=build_name,
            tokenizer=tokenizer,
        )
    else:
        full_dataset = create_sft_dataset(
            config=config,
            raw_dataset_name=raw_dataset_name,
            build_name=build_name,
            tokenizer=tokenizer,
        )

    if (
        full_dataset is None or len(full_dataset) == 0
    ):  # create_sft_dataset and create_dpo_dataset now return empty Datasets instead of None
        output_name = f"{raw_dataset_name}{config.builds[build_name].output_suffix}"
        print(
            f"Warning: Dataset for '{output_name}' is empty after processing. No data will be saved."
        )
        return

    # Pass the loaded tokenizer for token counting in splits
    dataset_to_save = split_and_format_dataset(
        full_dataset, config, build_name, tokenizer
    )

    output_name = f"{raw_dataset_name}{config.builds[build_name].output_suffix}"
    print(f"Saving processed dataset '{output_name}' to {processed_data_dir_abs}...")
    try:
        dataset_to_save.save_to_disk(processed_data_dir_abs)
        print(
            f"Processed dataset '{output_name}' saved successfully to {processed_data_dir_abs}."
        )
        print(f"  Splits saved: {list(dataset_to_save.keys())}")
    except Exception as e:
        print(
            f"Fatal Error: Could not save dataset to disk at '{processed_data_dir_abs}'. Exception: {e}"
        )
        import traceback

        traceback.print_exc()
