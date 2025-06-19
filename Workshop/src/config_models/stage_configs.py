# This file defines stage-specific configuration classes for multi-stage training pipelines.
# Each stage represents a distinct training phase (e.g., SFT, DPO) with its own hyperparameters,
# dataset requirements, and trainer configuration.
#
# Classes:
# - BaseStageConfig: Common configuration shared across all training stages
# - SFTStageConfig: Configuration specific to Supervised Fine-Tuning
# - DPOStageConfig: Configuration specific to Direct Preference Optimization
#
# Each stage config specifies:
# - Training hyperparameters (learning rate, batch size, etc.)
# - Dataset name (processed dataset to use for this stage) - optional, filled at runtime
# - Output configuration (where to save this stage's results)
# - Stage-specific parameters

from pydantic import Field, validator
from typing import Optional, List, Union, Literal
import os

from .base_config import BaseConfig


class BaseStageConfig(BaseConfig):
    """Base configuration class for all training stages."""

    stage_name: str = Field(
        ...,
        description="Name of this training stage (e.g., 'sft', 'dpo'). Used for directory naming and identification.",
    )

    processed_dataset_name: Optional[str] = Field(
        default=None,
        description="Name of the processed dataset to use for this stage (e.g., 'adam_sft', 'adam_dpo'). If not specified, will be set from command-line arguments.",
    )

    starting_model: Optional[str] = Field(
        default=None,
        description="Starting model for this stage. Either a HuggingFace model name (e.g., 'google/gemma-2-2b') or a local path to a saved model (e.g., 'experiments/adam/my_pipeline/stage_sft'). Set programmatically based on model config or previous stage output.",
    )

    # Core training parameters
    num_train_epochs: float = Field(
        default=1.0, description="Total number of training epochs to perform."
    )
    per_device_train_batch_size: int = Field(
        default=4, description="Batch size per GPU/TPU core/CPU for training."
    )
    per_device_eval_batch_size: int = Field(
        default=4, description="Batch size per GPU/TPU core/CPU for evaluation."
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    learning_rate: float = Field(
        default=5e-5, description="The initial learning rate for AdamW optimizer."
    )
    weight_decay: float = Field(
        default=0.0, description="Weight decay if we apply some."
    )
    lr_scheduler_type: str = Field(
        default="cosine",
        description="The scheduler type to use. (e.g., 'linear', 'cosine', 'constant', 'constant_with_warmup')",
    )
    warmup_ratio: float = Field(
        default=0.03,
        description="Ratio of total training steps used for a linear warmup from 0 to learning_rate.",
    )

    # Gradient clipping
    max_grad_norm: Optional[float] = Field(
        default=None,
        description="Maximum gradient norm for gradient clipping. If None, no gradient clipping is applied. Recommended values: 0.5-2.0 for stability.",
    )

    # Logging, Saving, Evaluation
    logging_strategy: str = Field(
        default="steps", description="Logging strategy ('steps', 'epoch')"
    )
    logging_steps: int = Field(
        default=10,
        description="Log every X updates steps. Used if logging_strategy='steps'.",
    )
    evaluation_strategy: str = Field(
        default="no",
        description="Evaluation strategy ('no', 'steps', 'epoch').",
    )
    eval_steps: Optional[int] = Field(
        default=None,
        description="Run an evaluation every X steps. Required if evaluation_strategy='steps'.",
    )
    save_strategy: str = Field(
        default="epoch", description="Save checkpoint strategy ('steps', 'epoch', 'no')"
    )
    save_steps: Optional[int] = Field(
        default=None,
        description="Save checkpoint every X updates steps. Required if save_strategy='steps'.",
    )
    save_total_limit: Optional[int] = Field(
        default=1,
        description="Limit the total amount of checkpoints. Deletes the older checkpoints in the output_dir.",
    )

    # Precision
    fp16: bool = Field(
        default=False,
        description="Whether to use 16-bit (mixed) precision training.",
    )
    bf16: bool = Field(
        default=True,
        description="Whether to use bfloat16 precision training.",
    )

    # Miscellaneous
    max_length: int = Field(
        default=2048,
        description="Maximum sequence length for tokenization and training.",
    )
    max_steps: Optional[int] = Field(
        default=None,
        description="If set, overrides num_train_epochs. Total number of training steps to perform.",
    )
    gradient_checkpointing: bool = Field(
        default=True, description="Enable gradient checkpointing to save memory."
    )

    # Data Loader specific arguments
    dataloader_num_workers: int = Field(
        default=0,
        description="Number of subprocesses to use for data loading.",
    )
    dataloader_pin_memory: bool = Field(
        default=True, description="Whether or not to pin memory for data loading."
    )
    dataloader_persistent_workers: bool = Field(
        default=False,
        description="If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.",
    )
    dataloader_prefetch_factor: Optional[int] = Field(
        default=None,
        description="Number of batches loaded in advance by each worker.",
    )


class SFTStageConfig(BaseStageConfig):
    """Configuration for Supervised Fine-Tuning stage."""

    stage_name: str = Field(default="sft", description="Stage name for SFT.")

    packing: bool = Field(
        default=False,
        description="Whether to use sequence packing for training efficiency. Automatically detects and uses custom packing for pre-tokenized data.",
    )

    packing_efficiency_threshold: float = Field(
        default=0.8,
        description="Minimum packing efficiency (ratio of used tokens to max_length) to create a packed sequence. Lower values pack more aggressively.",
    )

    pack_single_sequences: bool = Field(
        default=False,
        description="Whether to pack sequences even when only one sequence fits in max_length. Useful for consistent batch structure.",
    )

    block_cross_document_attention: bool = Field(
        default=False,
        description="When packing is enabled, whether to block attention between different documents in the same packed sequence. If True, creates custom attention masks to prevent cross-document attention. If False (default), allows standard cross-attention between all tokens.",
    )


class DPOStageConfig(BaseStageConfig):
    """Configuration for Direct Preference Optimization stage."""

    stage_name: str = Field(default="dpo", description="Stage name for DPO.")

    # DPO-specific parameters
    beta: float = Field(
        default=0.1,
        description="Temperature parameter for DPO loss. Higher values make the policy more deterministic.",
    )

    loss_type: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = Field(
        default="sigmoid",
        description="Type of loss function to use for DPO training.",
    )

    label_smoothing: float = Field(
        default=0.0,
        description="Label smoothing factor for DPO loss.",
    )

    reference_free: bool = Field(
        default=False,
        description="Whether to use reference-free DPO (no separate reference model).",
    )
