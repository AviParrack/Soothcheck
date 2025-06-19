# This file defines the DatasetConfig class, which is used to specify
# parameters for dataset loading, processing, and splitting.
# The config now supports multiple "builds" from a single raw dataset:
# - Augmentations operate on raw data (raw → raw) and are dataset-specific
# - Builds define how to process raw data into different processed datasets (raw → processed)
# - Each build specifies its own processing parameters for maximum flexibility
#
# Class: DatasetConfig
# Inherits from: BaseConfig
# Key Fields:
#  - builds (Dict[str, BuildConfig]): Defines different ways to build processed datasets
#                                     from the raw data (e.g., "sft", "dpo")
# Methods:
#  - get_raw_data_path(self, raw_dataset_name: str) -> str: Path to raw data for given dataset name.
#  - get_processed_data_path(self, raw_dataset_name: str, build_name: str) -> str: Path to processed data.
#  - get_absolute_raw_data_path(self, raw_dataset_name: str) -> str: Absolute path to raw data.
#  - get_absolute_processed_data_path(self, raw_dataset_name: str, build_name: str) -> str: Absolute path to processed data.
#  - load(cls, config_name: str) -> "DatasetConfig": (classmethod)
#    Loads a DatasetConfig by its name (e.g., 'general_config' for 'general_config.json').

from pydantic import Field, BaseModel
from typing import (
    List,
    Optional,
    Literal,
    Dict,
    Any,
)
import os

from .base_config import BaseConfig


# Model for specifying a data source (folder and optional file regex)
class SourceConfig(BaseModel):
    folder: str = Field(
        ...,
        description="Name of the subfolder within the raw data directory (e.g., 'corpus', 'augmented', 'contrast_pairs').",
    )
    file_regex: Optional[str] = Field(
        default=None,
        description="Optional regex pattern to match specific files within the folder. If not specified, all files in the folder will be processed regardless of extension.",
    )


# Model for individual pipeline configuration within a build
class PipelineConfigItem(BaseModel):
    name: str = Field(
        ..., description="Name of the data pipeline (e.g., 'basic', 'continues')."
    )
    sources: List[SourceConfig] = Field(
        default=[],
        description="List of data sources (folders and optional file patterns) for this pipeline to process.",
    )
    # Arbitrary other fields specific to the pipeline (e.g., min_grab_length)
    # will be captured by Pydantic's extra='allow' behavior.

    class Config:
        extra = (
            "allow"  # For Pydantic v1. For v2, it's model_config = {"extra": "allow"}
        )
        # Pydantic v2: model_config = {"extra": "allow"}


# Model for individual augmentation configuration within DatasetConfig
class AugmentationConfigItem(BaseModel):
    name: str = Field(..., description="Name of the data augmentation (e.g., 'qa').")
    sources: List[SourceConfig] = Field(
        default=[],
        description="List of data sources (folders and optional file patterns) for this augmentation to process.",
    )
    # Arbitrary other fields specific to the augmentation
    # will be captured by Pydantic's extra='allow' behavior.

    class Config:
        extra = (
            "allow"  # For Pydantic v1. For v2, it's model_config = {"extra": "allow"}
        )
        # Pydantic v2: model_config = {"extra": "allow"}


# Model for a build configuration (e.g., "sft", "dpo")
class BuildConfig(BaseModel):
    output_suffix: str = Field(
        ...,
        description="Suffix to append to the raw dataset name for the processed output (e.g., '_sft', '_dpo').",
    )
    validation_split_fraction: float = Field(
        ...,
        description="Fraction of data to use for validation (e.g., 0.2 for 20%). Each build must specify its own validation split.",
        gt=0,
        lt=1,
    )

    # Processing settings - each build can specify its own transformation parameters
    tokenizer_name_or_path: str = Field(
        ...,
        description="Hugging Face tokenizer name or path for this build. Different builds can use different tokenizers.",
    )
    max_length: int = Field(
        default=2048,
        description="Maximum sequence length for tokenization and chunking for this build.",
    )
    enable_chunking: bool = Field(
        default=False,
        description="Whether to enable document chunking for long documents in this build.",
    )
    chunk_at_markdown_headers: bool = Field(
        default=True,
        description="If chunking is enabled, whether to chunk at markdown headers (natural boundaries) for this build.",
    )
    chunk_overlap_ratio: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Ratio of overlap between chunks (0.0 = no overlap, 0.15 = 15% overlap). Only applies when chunking is enabled.",
    )
    prompt_format: Literal["chat", "text"] = Field(
        default="chat",
        description="Format to prepare the data in for this build. 'chat' will produce a 'messages' column with role/content dicts. 'text' will produce a 'text' column with raw string content.",
    )

    data_pipelines: List[PipelineConfigItem] = Field(
        default=[],
        description="List of data pipelines to apply for this build. Each item specifies the pipeline name, data sources, and any pipeline-specific parameters.",
    )

    class Config:
        extra = (
            "allow"  # For Pydantic v1. For v2, it's model_config = {"extra": "allow"}
        )


class DatasetConfig(BaseConfig):
    # Global settings that apply to the overall process
    split_seed: int = Field(
        default=42,
        description="Random seed for reproducible train/validation split across all builds.",
    )

    # Dataset-specific augmentations (raw → raw)
    data_augmentations: List[AugmentationConfigItem] = Field(
        default=[],
        description="List of data augmentations to apply to the raw dataset. These operate on raw data and are dataset-specific, not build-specific.",
    )

    # Build definitions (raw → processed)
    builds: Dict[str, BuildConfig] = Field(
        default={},
        description="Dictionary of build configurations. Each build defines how to process the raw dataset into a specific processed dataset (e.g., 'sft', 'dpo').",
    )

    def __post_init__(self):
        """Warn about deprecated fields."""
        if self.file_extensions is not None:
            print(
                "WARNING: file_extensions is deprecated. Use sources with folder and file_regex instead. This field is being ignored."
            )

    def get_raw_data_path(self, raw_dataset_name: str) -> str:
        """Returns the relative path to the raw data directory for the given raw dataset name."""
        return os.path.join("datasets", "raw", raw_dataset_name)

    def get_corpus_path(self, raw_dataset_name: str) -> str:
        """Returns the relative path to the corpus subfolder within the raw data directory."""
        return os.path.join(self.get_raw_data_path(raw_dataset_name), "corpus")

    def get_augmented_path(self, raw_dataset_name: str) -> str:
        """Returns the relative path to the augmented subfolder within the raw data directory."""
        return os.path.join(self.get_raw_data_path(raw_dataset_name), "augmented")

    def get_contrast_pairs_path(self, raw_dataset_name: str) -> str:
        """Returns the relative path to the contrast_pairs subfolder within the raw data directory."""
        return os.path.join(self.get_raw_data_path(raw_dataset_name), "contrast_pairs")

    def get_processed_data_path(self, raw_dataset_name: str, build_name: str) -> str:
        """Returns the relative path to the processed data directory for the given raw dataset and build."""
        build_config = self.builds[build_name]
        output_name = f"{raw_dataset_name}{build_config.output_suffix}"
        return os.path.join("datasets", "processed", output_name)

    def get_absolute_raw_data_path(self, raw_dataset_name: str) -> str:
        """Returns the absolute path to the raw data directory."""
        return os.path.join(os.getcwd(), self.get_raw_data_path(raw_dataset_name))

    def get_absolute_corpus_path(self, raw_dataset_name: str) -> str:
        """Returns the absolute path to the corpus subfolder."""
        return os.path.join(os.getcwd(), self.get_corpus_path(raw_dataset_name))

    def get_absolute_augmented_path(self, raw_dataset_name: str) -> str:
        """Returns the absolute path to the augmented subfolder."""
        return os.path.join(os.getcwd(), self.get_augmented_path(raw_dataset_name))

    def get_absolute_contrast_pairs_path(self, raw_dataset_name: str) -> str:
        """Returns the absolute path to the contrast_pairs subfolder."""
        return os.path.join(os.getcwd(), self.get_contrast_pairs_path(raw_dataset_name))

    def get_absolute_processed_data_path(
        self, raw_dataset_name: str, build_name: str
    ) -> str:
        """Returns the absolute path to the processed data directory."""
        return os.path.join(
            os.getcwd(), self.get_processed_data_path(raw_dataset_name, build_name)
        )

    def get_absolute_folder_path(self, raw_dataset_name: str, folder: str) -> str:
        """Returns the absolute path to a specific subfolder within the raw data directory."""
        return os.path.join(self.get_absolute_raw_data_path(raw_dataset_name), folder)

    def get_build_config(self, build_name: str) -> BuildConfig:
        """Returns the build configuration for the given build name."""
        if build_name not in self.builds:
            raise ValueError(
                f"Build '{build_name}' not found in configuration. Available builds: {list(self.builds.keys())}"
            )
        return self.builds[build_name]

    def get_effective_validation_split_fraction(self, build_name: str) -> float:
        """Returns the validation split fraction for a given build."""
        return self.get_build_config(build_name).validation_split_fraction

    def get_effective_prompt_format(self, build_name: str) -> str:
        """Returns the prompt format for a given build."""
        return self.get_build_config(build_name).prompt_format

    def get_effective_tokenizer_name_or_path(self, build_name: str) -> str:
        """Returns the tokenizer name or path for a given build."""
        return self.get_build_config(build_name).tokenizer_name_or_path

    def get_effective_max_length(self, build_name: str) -> int:
        """Returns the max length for a given build."""
        return self.get_build_config(build_name).max_length

    def get_effective_enable_chunking(self, build_name: str) -> bool:
        """Returns whether chunking is enabled for a given build."""
        return self.get_build_config(build_name).enable_chunking

    def get_effective_chunk_at_markdown_headers(self, build_name: str) -> bool:
        """Returns whether to chunk at markdown headers for a given build."""
        return self.get_build_config(build_name).chunk_at_markdown_headers

    def get_effective_chunk_overlap_ratio(self, build_name: str) -> float:
        """Returns the chunk overlap ratio for a given build."""
        return self.get_build_config(build_name).chunk_overlap_ratio

    @classmethod
    def load(cls, config_name: str) -> "DatasetConfig":
        """Loads a DatasetConfig by its name (e.g., 'general_config' for 'general_config.json')."""
        return super().load_from_name(config_name, "data")
