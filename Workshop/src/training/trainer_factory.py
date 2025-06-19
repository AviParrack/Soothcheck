# This file provides a factory for creating different types of trainers
# based on the training stage type (SFT, DPO, etc.).
#
# Classes:
# - ExplicitWandbCallback: Custom callback to ensure proper WandB metric formatting
#
# Functions:
# - create_trainer: Main factory function that returns the appropriate trainer
# - _create_sft_trainer: Creates an SFTTrainer instance
# - _create_dpo_trainer: Creates a DPOTrainer instance

from typing import Optional, Any, List
from pathlib import Path

from transformers import PreTrainedModel, PreTrainedTokenizer, TrainerCallback
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
import logging

from ..config_models.stage_configs import (
    BaseStageConfig,
    SFTStageConfig,
    DPOStageConfig,
)
from ..utils.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class ExplicitWandbCallback(TrainerCallback):
    """Custom callback to ensure key metrics are logged to wandb and detect NaN values."""

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Ensure key metrics are properly formatted in logs for wandb."""
        if logs is None:
            return

        # Check for NaN values in critical metrics
        import math

        critical_metrics = ["loss", "eval_loss", "grad_norm"]
        for metric in critical_metrics:
            if metric in logs and (
                math.isnan(logs[metric]) or math.isinf(logs[metric])
            ):
                logger.error(
                    f"NaN/Inf detected in {metric} at step {state.global_step}: {logs[metric]}"
                )
                logger.error(f"Full logs: {logs}")
                # Log warning to wandb if available
                if "wandb" in args.report_to:
                    logs["nan_detected"] = 1.0
                    logs["nan_metric"] = metric
                    logs["nan_step"] = state.global_step

        if "wandb" not in args.report_to:
            return

        # Instead of logging separately to wandb (which causes step conflicts),
        # we just ensure the metrics are properly formatted in the logs dict
        # that the trainer will use for its built-in wandb integration.

        # The built-in trainer logging will handle the actual wandb.log() call
        # with the correct step management. We just make sure the metrics
        # are available in the expected format.

        # Add prefixed versions of key metrics for better organization in wandb
        if "loss" in logs and "train/loss" not in logs:
            logs["train/loss"] = logs["loss"]
        if "grad_norm" in logs and "train/grad_norm" not in logs:
            logs["train/grad_norm"] = logs["grad_norm"]
        if "learning_rate" in logs and "train/learning_rate" not in logs:
            logs["train/learning_rate"] = logs["learning_rate"]
        if "epoch" in logs and "train/epoch" not in logs:
            logs["train/epoch"] = logs["epoch"]

        # Evaluation metrics
        if "eval_loss" in logs and "eval/loss" not in logs:
            logs["eval/loss"] = logs["eval_loss"]

        # Custom metrics with prefixes
        for key, value in list(
            logs.items()
        ):  # Use list() to avoid dict size change during iteration
            if key.startswith("eval_") and key not in ["eval_loss"]:
                prefixed_key = f"eval/{key[5:]}"
                if prefixed_key not in logs:
                    logs[prefixed_key] = value
            elif key in ["mean_token_accuracy", "num_tokens"]:
                prefixed_key = f"train/{key}"
                if prefixed_key not in logs:
                    logs[prefixed_key] = value


def create_trainer(
    stage_config: BaseStageConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    peft_config: Optional[LoraConfig],
    output_dir: Path,
    run_name: str,
    report_to: Optional[List[str]] = None,
    **kwargs,
) -> Any:
    """
    Factory function to create the appropriate trainer based on stage configuration.

    Args:
        stage_config: Configuration for the training stage
        model: The model to train
        tokenizer: The tokenizer to use
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        peft_config: PEFT configuration (optional)
        output_dir: Output directory for this stage
        run_name: Name for this run
        report_to: List of integrations to report to (e.g., ["wandb"])
        **kwargs: Additional arguments passed to specific trainer factories

    Returns:
        Trainer instance (SFTTrainer, DPOTrainer, etc.)
    """
    if report_to is None:
        report_to = []

    if isinstance(stage_config, SFTStageConfig):
        return _create_sft_trainer(
            stage_config=stage_config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            output_dir=output_dir,
            run_name=run_name,
            report_to=report_to,
            **kwargs,
        )
    elif isinstance(stage_config, DPOStageConfig):
        return _create_dpo_trainer(
            stage_config=stage_config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            peft_config=peft_config,
            output_dir=output_dir,
            run_name=run_name,
            report_to=report_to,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown stage config type: {type(stage_config)}")


def _create_sft_trainer(
    stage_config: SFTStageConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    peft_config: Optional[LoraConfig],
    output_dir: Path,
    run_name: str,
    report_to: List[str],
    **kwargs,
) -> SFTTrainer:
    """Create an SFTTrainer instance."""

    # Get model config from kwargs for template application
    model_config = kwargs.get("model_config", {})

    # Check if dataset needs chat template application
    if "messages" in train_dataset.column_names:
        logger.info("Applying chat templates to training dataset")

        # Apply chat templates to datasets
        def format_dataset_for_training(dataset):
            """Apply chat templates to a dataset using the new unified function."""

            def formatting_func(examples):
                all_input_ids = []
                all_attention_mask = []
                all_labels = []

                for messages in examples["messages"]:
                    result = TemplateManager.format_for_model(
                        data=messages,
                        tokenizer=tokenizer,
                        model_type="chat",
                        purpose="training",
                        max_length=stage_config.max_length,
                        truncation=True if stage_config.max_length else False,
                        padding=False,
                    )
                    all_input_ids.append(result["input_ids"])
                    all_attention_mask.append(result["attention_mask"])
                    all_labels.append(result["labels"])

                return {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_mask,
                    "labels": all_labels,
                }

            return dataset.map(
                formatting_func,
                batched=True,
                num_proc=4,
                remove_columns=dataset.column_names,
                desc="Applying chat template",
            )

        train_dataset = format_dataset_for_training(train_dataset)

        if eval_dataset and "messages" in eval_dataset.column_names:
            logger.info("Applying chat templates to evaluation dataset")
            eval_dataset = format_dataset_for_training(eval_dataset)

    # Convert stage config to SFTConfig
    sft_config_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,  # Each stage overwrites its own directory
        "num_train_epochs": stage_config.num_train_epochs,
        "max_steps": (
            stage_config.max_steps
            if stage_config.max_steps and stage_config.max_steps > 0
            else -1
        ),
        "per_device_train_batch_size": stage_config.per_device_train_batch_size,
        "per_device_eval_batch_size": stage_config.per_device_eval_batch_size,
        "gradient_accumulation_steps": stage_config.gradient_accumulation_steps,
        "learning_rate": stage_config.learning_rate,
        "weight_decay": stage_config.weight_decay,
        "lr_scheduler_type": stage_config.lr_scheduler_type,
        "warmup_ratio": stage_config.warmup_ratio,
        "logging_dir": str(output_dir / "logs"),
        "logging_strategy": stage_config.logging_strategy,
        "logging_steps": stage_config.logging_steps,
        "eval_strategy": (
            stage_config.evaluation_strategy if eval_dataset is not None else "no"
        ),
        "eval_steps": (
            stage_config.eval_steps
            if stage_config.evaluation_strategy == "steps" and eval_dataset is not None
            else None
        ),
        "save_strategy": stage_config.save_strategy,
        "save_steps": (
            stage_config.save_steps if stage_config.save_strategy == "steps" else None
        ),
        "save_total_limit": stage_config.save_total_limit,
        "fp16": stage_config.fp16,
        "bf16": stage_config.bf16,
        "run_name": run_name,
        "remove_unused_columns": False,  # Important: keep this False when using pre-tokenized data
        "max_length": stage_config.max_length,
        "gradient_checkpointing": stage_config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if stage_config.gradient_checkpointing else None
        ),
        "packing": stage_config.packing,
        "dataloader_num_workers": stage_config.dataloader_num_workers,
        "dataloader_pin_memory": stage_config.dataloader_pin_memory,
        "dataloader_persistent_workers": stage_config.dataloader_persistent_workers,
        "dataloader_prefetch_factor": stage_config.dataloader_prefetch_factor,
        # Use the provided report_to for this stage
        "report_to": report_to,
    }

    # Add gradient clipping if specified
    if stage_config.max_grad_norm is not None:
        sft_config_kwargs["max_grad_norm"] = stage_config.max_grad_norm

    # For pre-tokenized data (after chat template application), we don't need dataset_text_field
    # SFTTrainer will automatically detect the input_ids column

    sft_config = SFTConfig(**sft_config_kwargs)

    # Create custom data collator for pre-tokenized data
    data_collator = PackedSequenceCollator(
        tokenizer=tokenizer,
        max_length=stage_config.max_length,
        pad_token_id=tokenizer.pad_token_id,
        label_pad_token_id=-100,
        packing_efficiency_threshold=stage_config.packing_efficiency_threshold,
        pack_single_sequences=stage_config.pack_single_sequences,
        block_cross_document_attention=stage_config.block_cross_document_attention,
    )
    
    # Disable TRL's built-in packing since we're handling it with our collator
    use_trl_packing = False
    
    logger.info(f"Custom packing configured: efficiency_threshold={stage_config.packing_efficiency_threshold}, "
               f"pack_single_sequences={stage_config.pack_single_sequences}, "
               f"block_cross_document_attention={stage_config.block_cross_document_attention}")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if sft_config.eval_strategy != "no" else None,
        peft_config=peft_config,
        args=sft_config,
        processing_class=tokenizer,
        data_collator=data_collator,
        use_trl_packing=use_trl_packing,
    )

    # Add custom wandb callback if wandb is enabled
    if "wandb" in report_to:
        trainer.add_callback(ExplicitWandbCallback())

    return trainer


def _create_dpo_trainer(
    stage_config: DPOStageConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    peft_config: Optional[LoraConfig],
    output_dir: Path,
    run_name: str,
    report_to: List[str],
    **kwargs,
) -> DPOTrainer:
    """Create a DPOTrainer instance."""

    # Get model config from kwargs for template application
    model_config = kwargs.get("model_config", {})

    # For DPO, we need to handle the prompt formatting first before renaming columns
    # Check if dataset uses chat format (has 'prompt' field with messages)
    if train_dataset and len(train_dataset) > 0:
        first_item = train_dataset[0]
        if isinstance(first_item.get("prompt"), list) and len(first_item["prompt"]) > 0:
            # This is a chat-format DPO dataset
            logger.info("Detected chat-format DPO dataset, applying chat templates")

            # Apply chat template to prompts
            def format_dpo_prompts(examples):
                formatted_prompts = []
                for prompt_messages in examples["prompt"]:
                    if isinstance(prompt_messages, list):
                        # Apply chat template without generation prompt for DPO
                        formatted_prompt_tokens = TemplateManager.format_for_model(
                            data=prompt_messages,
                            tokenizer=tokenizer,
                            model_type="chat",
                            purpose="training",
                        )
                        # For DPO, we need the formatted text, not tokens
                        # So we decode back to text (DPO trainer handles tokenization)
                        formatted_prompt = tokenizer.decode(
                            formatted_prompt_tokens["input_ids"],
                            skip_special_tokens=False,
                        )
                        formatted_prompts.append(formatted_prompt)
                    else:
                        # Already a string, use as-is
                        formatted_prompts.append(prompt_messages)

                # Update the examples with formatted prompts
                examples["prompt"] = formatted_prompts
                return examples

            # Apply formatting
            train_dataset = train_dataset.map(
                format_dpo_prompts,
                batched=True,
                num_proc=4,
                desc="Applying chat templates to DPO prompts",
            )

            if eval_dataset and len(eval_dataset) > 0:
                first_eval_item = eval_dataset[0]
                if isinstance(first_eval_item.get("prompt"), list):
                    logger.info("Applying chat templates to evaluation DPO dataset")
                    eval_dataset = eval_dataset.map(
                        format_dpo_prompts,
                        batched=True,
                        num_proc=4,
                        desc="Applying chat templates to DPO eval prompts",
                    )

    # Rename columns to match TRL DPOTrainer expectations
    # Our datasets use 'response_accepted'/'response_rejected', but TRL expects 'chosen'/'rejected'
    def rename_dpo_columns(dataset: Dataset) -> Dataset:
        if dataset is None:
            return None

        column_mapping = {}
        if "response_accepted" in dataset.column_names:
            column_mapping["response_accepted"] = "chosen"
        if "response_rejected" in dataset.column_names:
            column_mapping["response_rejected"] = "rejected"

        if column_mapping:
            return dataset.rename_columns(column_mapping)
        return dataset

    train_dataset = rename_dpo_columns(train_dataset)
    eval_dataset = rename_dpo_columns(eval_dataset)

    # Convert stage config to DPOConfig
    dpo_config_kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,  # Each stage overwrites its own directory
        "num_train_epochs": stage_config.num_train_epochs,
        "max_steps": (
            stage_config.max_steps
            if stage_config.max_steps and stage_config.max_steps > 0
            else -1
        ),
        "per_device_train_batch_size": stage_config.per_device_train_batch_size,
        "per_device_eval_batch_size": stage_config.per_device_eval_batch_size,
        "gradient_accumulation_steps": stage_config.gradient_accumulation_steps,
        "learning_rate": stage_config.learning_rate,
        "weight_decay": stage_config.weight_decay,
        "lr_scheduler_type": stage_config.lr_scheduler_type,
        "warmup_ratio": stage_config.warmup_ratio,
        "logging_dir": str(output_dir / "logs"),
        "logging_strategy": stage_config.logging_strategy,
        "logging_steps": stage_config.logging_steps,
        "eval_strategy": (
            stage_config.evaluation_strategy if eval_dataset is not None else "no"
        ),
        "eval_steps": (
            stage_config.eval_steps
            if stage_config.evaluation_strategy == "steps" and eval_dataset is not None
            else None
        ),
        "save_strategy": stage_config.save_strategy,
        "save_steps": (
            stage_config.save_steps if stage_config.save_strategy == "steps" else None
        ),
        "save_total_limit": stage_config.save_total_limit,
        "fp16": stage_config.fp16,
        "bf16": stage_config.bf16,
        "run_name": run_name,
        "remove_unused_columns": True,
        "max_length": stage_config.max_length,
        "gradient_checkpointing": stage_config.gradient_checkpointing,
        "gradient_checkpointing_kwargs": (
            {"use_reentrant": False} if stage_config.gradient_checkpointing else None
        ),
        "dataloader_num_workers": stage_config.dataloader_num_workers,
        "dataloader_pin_memory": stage_config.dataloader_pin_memory,
        "dataloader_persistent_workers": stage_config.dataloader_persistent_workers,
        "dataloader_prefetch_factor": stage_config.dataloader_prefetch_factor,
        # DPO-specific parameters
        "beta": stage_config.beta,
        "loss_type": stage_config.loss_type,
        "label_smoothing": stage_config.label_smoothing,
        "reference_free": stage_config.reference_free,
        # Use the provided report_to for this stage
        "report_to": report_to,
    }

    # Add gradient clipping if specified
    if stage_config.max_grad_norm is not None:
        dpo_config_kwargs["max_grad_norm"] = stage_config.max_grad_norm

    dpo_config = DPOConfig(**dpo_config_kwargs)

    # For DPO, we need a reference model if not using reference_free
    #
    # Current behavior when ref_model=None with PEFT:
    # - TRL uses the base model WITHOUT adapters as reference
    # - This compares: base model (ref) vs base model + SFT adapters (train)
    #
    # Ideal behavior (not yet implemented):
    # - Reference model: base model + SFT PEFT adapters (frozen)
    # - Training model: base model + SFT PEFT adapters (trainable)
    # - Both share the same base model weights
    #
    # TODO: To implement the ideal behavior, we would need to:
    # 1. Load a second copy of the model with SFT adapters
    # 2. Disable adapters for the reference model or make them non-trainable
    # 3. Pass this as ref_model to DPOTrainer
    #
    # For now, we use TRL's default behavior (base model as reference)
    if stage_config.reference_free:
        ref_model = None
    else:
        # TRL will use the base model without adapters as reference
        ref_model = None

    # Enable input gradients for DPO training (fixes tensor type issues)
    model.enable_input_require_grads()

    # Custom preprocessing function to ensure tensors are properly typed
    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocessing function to ensure proper tensor types.
        This is a workaround for the tensor type issue with Gemma models.
        """
        if hasattr(logits, "logits"):
            logits = logits.logits
        return logits

    # Create a custom DPOTrainer class that ensures input_ids are long tensors
    from torch import Tensor
    import torch

    class GemmaDPOTrainer(DPOTrainer):
        def get_batch_loss_metrics(self, model, batch, train_eval="train"):
            """Override to ensure input_ids are long tensors."""
            # Convert any float tensors to long tensors in the batch
            for key in batch:
                if isinstance(batch[key], Tensor) and batch[key].dtype in [
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ]:
                    # Check if this looks like input_ids (integer values despite float type)
                    if key in [
                        "input_ids",
                        "prompt_input_ids",
                        "chosen_input_ids",
                        "rejected_input_ids",
                    ]:
                        batch[key] = batch[key].long()

            return super().get_batch_loss_metrics(model, batch, train_eval)

    trainer = GemmaDPOTrainer(
        model=model,
        ref_model=ref_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if dpo_config.eval_strategy != "no" else None,
        peft_config=peft_config,
        args=dpo_config,
        processing_class=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        # Let TRL use its default DataCollatorForPreference
    )

    # Add custom wandb callback if wandb is enabled
    if "wandb" in report_to:
        trainer.add_callback(ExplicitWandbCallback())

    return trainer
