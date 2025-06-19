import os
import shutil
import glob
import re
from typing import Optional, List, Dict, Any, Tuple
from datasets import Dataset
from transformers import PreTrainedTokenizer
from ..config_models.data_config import (
    DatasetConfig,
    SourceConfig,
)  # Relative import for config

# This file contains utility functions for dataset manipulation,
# such as deleting processed datasets, calculating token counts,
# and reading file contents.
# These functions are generally independent of specific data processing pipelines.
#
# Functions:
# - delete_processed_dataset: Deletes a processed dataset directory.
# - get_token_count_for_split: Estimates token counts for a dataset split.
# - read_file_contents: Reads the content of a single file into a string.
# - is_file_extension_allowed: Checks if a file's extension is in a list of allowed extensions.
# - find_files_in_source: Finds files in a specific folder with optional regex filtering.
# - validate_sources_exist: Validates that source folders exist before processing.
# - find_files_for_component: Generic function to find files for pipelines or augmentations.

# This file is for utility functions related to dataset manipulation that are
# not part of the main data processing orchestration or specific chunking logic.

# Functions:
# - delete_processed_dataset: Deletes a processed dataset directory based on config.


def delete_processed_dataset(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
) -> bool:
    """
    Deletes the processed dataset directory for the given DatasetConfig, raw dataset name, and build.

    Args:
        config: The DatasetConfig object specifying the dataset configuration.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build.

    Returns:
        True if deletion was successful or the directory didn't exist.
        False if an error occurred during deletion.
    """
    processed_data_dir_abs = config.get_absolute_processed_data_path(
        raw_dataset_name, build_name
    )

    if os.path.isdir(processed_data_dir_abs):
        print(
            f"Attempting to delete existing processed data directory: {processed_data_dir_abs}"
        )
        try:
            shutil.rmtree(processed_data_dir_abs)
            print(f"Successfully deleted: {processed_data_dir_abs}")
            return True
        except OSError as e:
            print(f"Error deleting directory {processed_data_dir_abs}: {e.strerror}")
            print("Please check permissions or if files are in use, then try again.")
            return False
    else:
        print(
            f"Processed data directory not found (no deletion needed): {processed_data_dir_abs}"
        )
        return True  # Directory didn't exist, so considered a success in terms of precondition for saving.


def read_file_contents(file_path: str, encoding: str = "utf-8") -> Optional[str]:
    """
    Reads the content of a single file into a string.

    Args:
        file_path: The absolute path to the file.
        encoding: The encoding to use when reading the file.

    Returns:
        The file content as a string, or None if an error occurs.
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except (IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {file_path} with encoding {encoding}: {e}")
        return None


def is_file_extension_allowed(file_path: str, allowed_extensions: List[str]) -> bool:
    """
    Checks if a file's extension is in a list of allowed extensions.
    Comparison is case-insensitive and normalizes extensions to start with a dot.

    Args:
        file_path: The path to the file.
        allowed_extensions: A list of allowed extensions (e.g., [".txt", "md"]).

    Returns:
        True if the file's extension is allowed, False otherwise.
    """
    if not allowed_extensions:
        return False  # Or True, depending on desired behavior for empty allow list. False seems safer.

    filename = os.path.basename(file_path)
    file_ext_tuple = os.path.splitext(filename)
    if not file_ext_tuple[1]:  # No extension
        return False

    current_file_ext = file_ext_tuple[1].lower()
    if not current_file_ext.startswith("."):
        current_file_ext = (
            "." + current_file_ext
        )  # Should already have dot from splitext, but good practice

    normalized_allowed_extensions = set()
    for ext in allowed_extensions:
        normalized_ext = ext.lower()
        if not normalized_ext.startswith("."):
            normalized_ext = "." + normalized_ext
        normalized_allowed_extensions.add(normalized_ext)

    return current_file_ext in normalized_allowed_extensions


def find_files_in_source(
    config: DatasetConfig,
    raw_dataset_name: str,
    source: SourceConfig,
) -> List[str]:
    """
    Finds files in a specific source (folder with optional regex pattern).

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        source: The SourceConfig specifying folder and optional file regex.

    Returns:
        List of absolute file paths that match the source criteria.
    """
    folder_abs_path = config.get_absolute_folder_path(raw_dataset_name, source.folder)

    if not os.path.isdir(folder_abs_path):
        print(
            f"Warning: Source folder '{source.folder}' not found at {folder_abs_path}"
        )
        return []

    # Find all files in the folder (no extension filtering)
    all_file_paths = []
    pattern = os.path.join(folder_abs_path, "**/*")
    found_files = glob.glob(pattern, recursive=True)
    # Filter out directories
    all_file_paths = [f for f in found_files if os.path.isfile(f)]

    # Filter by file regex if specified
    if source.file_regex:
        filtered_files = []
        for file_path in all_file_paths:
            filename = os.path.basename(file_path)
            if re.match(source.file_regex, filename):
                filtered_files.append(file_path)
        return filtered_files

    return all_file_paths


def get_token_count_for_split(
    split_dataset: Dataset,
    text_column: str,
    tokenizer: PreTrainedTokenizer,
    split_name: str,
    sample_size_for_token_estimation: int = 1000,
) -> str:
    """
    Estimates token counts for a given dataset split using the provided tokenizer.
    """
    num_items = len(split_dataset)
    token_count_str = ""

    if not tokenizer:
        return "(tokenizer not available for token count)"
    if text_column not in split_dataset.column_names:
        return f"(text column '{text_column}' not found for token count)"
    if num_items == 0:
        return "0 tokens (empty split)"

    try:
        text_to_estimate = ""
        # Estimate tokens by joining a sample if the dataset is very large.
        if num_items > sample_size_for_token_estimation:
            print(
                f"    Estimating tokens for '{split_name}' split from a sample of {sample_size_for_token_estimation} items (total {num_items})."
            )
            # Ensure the sample is actually drawn if num_items > sample_size
            sample_texts = split_dataset.select(
                range(min(num_items, sample_size_for_token_estimation))
            )[text_column]
            text_to_estimate = " ".join(sample_texts)
        else:
            text_to_estimate = " ".join(split_dataset[text_column])

        if not text_to_estimate.strip():
            total_tokens = 0
        else:
            # Directly use tokenizer for a more accurate count if possible,
            # or fallback to a simpler estimation if needed.
            # For now, using the length of tokenized IDs.
            total_tokens = len(tokenizer.encode(text_to_estimate))

        # Adjust if sampling was used for a rough full estimate
        if (
            num_items > sample_size_for_token_estimation
            and total_tokens > 0
            and sample_size_for_token_estimation > 0
        ):
            estimated_full_tokens = int(
                total_tokens
                * (num_items / min(num_items, sample_size_for_token_estimation))
            )
            token_count_str = f"~{estimated_full_tokens:,} tokens (estimated from sample, using '{tokenizer.name_or_path}')"
        else:
            token_count_str = (
                f"{total_tokens:,} tokens (estimated using '{tokenizer.name_or_path}')"
            )

    except Exception as e:
        token_count_str = f"(error calculating tokens for '{split_name}': {e})"

    return token_count_str


def validate_sources_exist(
    config: DatasetConfig,
    raw_dataset_name: str,
    component_configs: List[Dict[str, Any]],
    component_type: str = "component",
) -> bool:
    """
    Validates that all source folders referenced in component configurations exist.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        component_configs: List of component configuration dictionaries.
        component_type: Type of component ("pipeline" or "augmentation") for logging.

    Returns:
        True if all sources exist, False otherwise.
    """
    missing_folders = []

    for component_config in component_configs:
        component_name = component_config.get("name", "unknown")
        sources = component_config.get("sources", [])

        for source_dict in sources:
            try:
                source = SourceConfig(**source_dict)
                folder_abs_path = config.get_absolute_folder_path(
                    raw_dataset_name, source.folder
                )

                if not os.path.isdir(folder_abs_path):
                    missing_folders.append(
                        {
                            "component": component_name,
                            "folder": source.folder,
                            "path": folder_abs_path,
                        }
                    )
            except Exception as e:
                print(
                    f"Warning: Invalid source config for {component_type} '{component_name}': {e}"
                )

    if missing_folders:
        print(f"Error: Missing source folders for {component_type}s:")
        for missing in missing_folders:
            print(
                f"  - {component_type.title()} '{missing['component']}' references missing folder '{missing['folder']}' at {missing['path']}"
            )
        return False

    return True


def find_files_for_component(
    config: DatasetConfig,
    raw_dataset_name: str,
    component_config_dict: Dict[str, Any],
    component_type: str = "component",
) -> List[str]:
    """
    Generic function to find files for pipelines or augmentations based on their sources.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        component_config_dict: Dictionary containing component configuration.
        component_type: Type of component ("pipeline" or "augmentation") for logging.

    Returns:
        List of absolute file paths for the component to process.
    """
    component_name = component_config_dict["name"]

    # Sources-based approach
    if "sources" in component_config_dict and component_config_dict["sources"]:
        all_files = []
        for source_dict in component_config_dict["sources"]:
            # Convert dict to SourceConfig for type safety
            source = SourceConfig(**source_dict)
            files = find_files_in_source(config, raw_dataset_name, source)
            all_files.extend(files)
            if files:
                folder_info = f"folder '{source.folder}'"
                if source.file_regex:
                    folder_info += f" with regex '{source.file_regex}'"
                print(
                    f"  Found {len(files)} file(s) in {folder_info} for {component_type} '{component_name}'"
                )
        return all_files

    else:
        raise ValueError(
            f"{component_type.title()} '{component_name}' must specify sources. No default behavior is allowed."
        )


def get_augmented_text_strings(
    config: DatasetConfig,
    raw_dataset_name: str,
    build_name: str,
    augmentation_name: str,
) -> List[str]:
    """
    Returns a list of all augmented text strings for a specific augmentation.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build.
        augmentation_name: The name of the augmentation.

    Returns:
        List of text strings from the augmentation files.
    """
    # Import augmentation classes to access their extract methods
    from .augment.qa import SimpleQAAugmentation
    from .augment.conversation import BaseConversationAugmentation
    from .augment.contrast_pairs import ContrastPairAugmentation

    raw_data_dir = config.get_absolute_raw_data_path(raw_dataset_name)

    # Look for augmentation files in different folders
    augmentation_folders = [
        "augmentations/prompt_response_pairs",
        "augmentations/chats",
        "augmentations/contrast_pairs",
    ]

    all_text_strings = []

    for folder in augmentation_folders:
        folder_path = os.path.join(raw_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Find all .jsonl files in the folder
        pattern = os.path.join(folder_path, "*.jsonl")
        jsonl_files = glob.glob(pattern)

        for file_path in jsonl_files:
            # Check if this file contains the augmentation we're looking for
            # by reading a sample line to check augmentation_name
            try:
                import json

                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if data.get("augmentation_name") == augmentation_name:
                                # This file contains our augmentation, extract all text
                                if "prompt_response_pairs" in folder:
                                    text_strings = SimpleQAAugmentation.extract_text_strings_from_file(
                                        file_path
                                    )
                                elif "chats" in folder:
                                    text_strings = BaseConversationAugmentation.extract_text_strings_from_file(
                                        file_path
                                    )
                                elif "contrast_pairs" in folder:
                                    text_strings = ContrastPairAugmentation.extract_text_strings_from_file(
                                        file_path
                                    )
                                else:
                                    text_strings = []

                                all_text_strings.extend(text_strings)
                                break  # Found matching augmentation in this file
                        except json.JSONDecodeError:
                            continue
                        break  # Only check first valid line to determine file type
            except (IOError, UnicodeDecodeError):
                continue

    return all_text_strings


def get_corpus_text_strings(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
) -> List[str]:
    """
    Returns a list of all text strings from the corpus folder.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build (used to determine which corpus to use).

    Returns:
        List of text strings from all files in the corpus folder.
    """
    raw_data_dir = config.get_absolute_raw_data_path(raw_dataset_name)
    corpus_folder = os.path.join(raw_data_dir, "corpus")

    if not os.path.isdir(corpus_folder):
        print(f"Warning: Corpus folder not found at {corpus_folder}")
        return []

    all_text_strings = []

    # Find all files in the corpus folder recursively
    pattern = os.path.join(corpus_folder, "**/*")
    found_files = glob.glob(pattern, recursive=True)
    corpus_files = [f for f in found_files if os.path.isfile(f)]

    for file_path in corpus_files:
        content = read_file_contents(file_path)
        if content:
            all_text_strings.append(content)

    return all_text_strings


def get_corpus_text_strings_with_paths(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
) -> List[Tuple[str, str]]:
    """
    Returns a list of tuples containing (text_string, file_path) from the corpus folder.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build (used to determine which corpus to use).

    Returns:
        List of tuples (text_string, file_path) from all files in the corpus folder.
    """
    raw_data_dir = config.get_absolute_raw_data_path(raw_dataset_name)
    corpus_folder = os.path.join(raw_data_dir, "corpus")

    if not os.path.isdir(corpus_folder):
        print(f"Warning: Corpus folder not found at {corpus_folder}")
        return []

    text_with_paths = []

    # Find all files in the corpus folder recursively
    pattern = os.path.join(corpus_folder, "**/*")
    found_files = glob.glob(pattern, recursive=True)
    corpus_files = [f for f in found_files if os.path.isfile(f)]

    for file_path in corpus_files:
        content = read_file_contents(file_path)
        if content:
            # Use relative path from corpus folder for cleaner display
            relative_path = os.path.relpath(file_path, corpus_folder)
            text_with_paths.append((content, relative_path))

    return text_with_paths


def get_all_augmented_text_strings(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
) -> List[str]:
    """
    Returns a list of all augmented text strings from all augmentation files.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build.

    Returns:
        List of all text strings from all augmentation files.
    """
    # Import augmentation classes to access their extract methods
    from .augment.qa import SimpleQAAugmentation
    from .augment.conversation import BaseConversationAugmentation
    from .augment.contrast_pairs import ContrastPairAugmentation

    raw_data_dir = config.get_absolute_raw_data_path(raw_dataset_name)

    # Look for augmentation files in different folders
    augmentation_folders = [
        "augmentations/prompt_response_pairs",
        "augmentations/chats",
        "augmentations/contrast_pairs",
    ]

    all_text_strings = []

    for folder in augmentation_folders:
        folder_path = os.path.join(raw_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Find all .jsonl files in the folder
        pattern = os.path.join(folder_path, "*.jsonl")
        jsonl_files = glob.glob(pattern)

        for file_path in jsonl_files:
            # Extract text from each file based on folder type
            if "prompt_response_pairs" in folder:
                text_strings = SimpleQAAugmentation.extract_text_strings_from_file(
                    file_path
                )
            elif "chats" in folder:
                text_strings = (
                    BaseConversationAugmentation.extract_text_strings_from_file(
                        file_path
                    )
                )
            elif "contrast_pairs" in folder:
                text_strings = ContrastPairAugmentation.extract_text_strings_from_file(
                    file_path
                )
            else:
                text_strings = []

            all_text_strings.extend(text_strings)

    return all_text_strings


def get_all_augmented_text_strings_with_paths(
    config: DatasetConfig, raw_dataset_name: str, build_name: str
) -> List[Tuple[str, str]]:
    """
    Returns a list of tuples containing (text_string, file_path) from all augmentation files.

    Args:
        config: The DatasetConfig object.
        raw_dataset_name: The name of the raw dataset.
        build_name: The name of the build.

    Returns:
        List of tuples (text_string, file_path) from all augmentation files.
    """
    # Import augmentation classes to access their extract methods
    from .augment.qa import SimpleQAAugmentation
    from .augment.conversation import BaseConversationAugmentation
    from .augment.contrast_pairs import ContrastPairAugmentation

    raw_data_dir = config.get_absolute_raw_data_path(raw_dataset_name)

    # Look for augmentation files in different folders
    augmentation_folders = [
        "augmentations/prompt_response_pairs",
        "augmentations/chats",
        "augmentations/contrast_pairs",
    ]

    text_with_paths = []

    for folder in augmentation_folders:
        folder_path = os.path.join(raw_data_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Find all .jsonl files in the folder
        pattern = os.path.join(folder_path, "*.jsonl")
        jsonl_files = glob.glob(pattern)

        for file_path in jsonl_files:
            # Extract text from each file based on folder type
            if "prompt_response_pairs" in folder:
                text_strings = SimpleQAAugmentation.extract_text_strings_from_file(
                    file_path
                )
            elif "chats" in folder:
                text_strings = (
                    BaseConversationAugmentation.extract_text_strings_from_file(
                        file_path
                    )
                )
            elif "contrast_pairs" in folder:
                text_strings = ContrastPairAugmentation.extract_text_strings_from_file(
                    file_path
                )
            else:
                text_strings = []

            # Use relative path from augmentations folder for cleaner display
            relative_path = os.path.relpath(
                file_path, os.path.join(raw_data_dir, "augmentations")
            )

            # Add each text string with the file path
            for text_string in text_strings:
                text_with_paths.append((text_string, relative_path))

    return text_with_paths
