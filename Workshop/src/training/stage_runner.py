# This file contains the logic for running individual training stages.
# It handles model loading (either from HF or previous stage), dataset loading,
# trainer creation, and training execution.
#
# Classes:
# - StageRunner: Executes a single training stage
#
# Functions:
# - _inspect_dataloader_batches: Helper function for batch inspection (preserved from original train.py)

import os
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_from_disk, Dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from ..config_models import ModelConfig, PeftConfig as PydanticPeftConfig
from ..config_models.stage_configs import BaseStageConfig
from ..utils.utils import get_torch_dtype
from .trainer_factory import create_trainer
from ..training.training_validator import (
    inspect_training_batch,
    validate_loss_calculation,
)

try:
    import wandb
except ImportError:
    wandb = None


class StageRunner:
    """Executes a single training stage."""

    def __init__(
        self,
        stage_config: BaseStageConfig,
        model_config: ModelConfig,
        peft_config: Optional[PydanticPeftConfig],
        base_model_hf_id: str,
        peft_adapter_path: Optional[str],
        output_dir: Path,
        run_name: str,
        report_to: List[str],
        wandb_project: Optional[str] = None,
        pipeline_info: Optional[Dict[str, Any]] = None,
    ):
        self.stage_config = stage_config
        self.model_config = model_config
        self.peft_config = peft_config
        self.base_model_hf_id = base_model_hf_id
        self.peft_adapter_path = peft_adapter_path
        self.output_dir = output_dir
        self.run_name = run_name
        self.report_to = report_to
        self.wandb_project = wandb_project
        self.pipeline_info = pipeline_info or {}

        # Track wandb initialization and training results
        self.wandb_initialized = False
        self.training_results: Optional[Dict[str, Any]] = None

    def run(self, inspect_batches: int = 0) -> None:
        """Run this training stage."""
        print(f"Running stage: {self.stage_config.stage_name}")
        print(f"Base model HF ID: {self.base_model_hf_id}")
        if self.peft_adapter_path:
            print(f"PEFT adapter path: {self.peft_adapter_path}")
        print(f"Dataset: {self.stage_config.processed_dataset_name}")
        print(f"Output directory: {self.output_dir}")

        try:
            # Initialize W&B for this specific stage
            self._initialize_wandb()

            # Load dataset
            train_dataset, eval_dataset = self._load_datasets()
            if train_dataset is None:
                raise RuntimeError(
                    f"Failed to load dataset: {self.stage_config.processed_dataset_name}"
                )

            # Setup tokenizer
            tokenizer = self._setup_tokenizer()

            # Setup model
            model, peft_config = self._setup_model()

            # Create trainer (now with proper report_to for this stage)
            trainer = create_trainer(
                stage_config=self.stage_config,
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                peft_config=peft_config,
                output_dir=self.output_dir,
                run_name=self.run_name,
                report_to=self.report_to,  # Pass through the actual report_to for this stage
                model_config=self.model_config,  # Pass model config for template application
            )

            print(f"{self.stage_config.stage_name.upper()}Trainer initialized.")

            # Handle batch inspection
            if inspect_batches > 0:
                print(f"Inspecting {inspect_batches} batches...")
                _inspect_dataloader_batches(
                    trainer, tokenizer, inspect_batches, self.report_to
                )
                print("Batch inspection complete.")
                return

            # Run training
            print("Starting training...")
            train_result = trainer.train()
            print("Training completed.")

            # Capture training results
            self.training_results = {
                "final_loss": train_result.metrics.get("train_loss"),
                "total_steps": train_result.metrics.get("train_steps", 0),
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples_per_second": train_result.metrics.get(
                    "train_samples_per_second"
                ),
                "train_steps_per_second": train_result.metrics.get(
                    "train_steps_per_second"
                ),
            }

            # Save model and tokenizer
            print(f"Saving model to: {self.output_dir}")
            trainer.save_model()
            tokenizer.save_pretrained(self.output_dir)

            # Save metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
            print("Training metrics and state saved.")

        except Exception as e:
            print(f"An error occurred during training: {e}")

            # Finish wandb with error code if initialized
            if self.wandb_initialized and wandb and wandb.run:
                wandb.finish(exit_code=1)

            raise
        finally:
            # Clean up wandb run
            if self.wandb_initialized and wandb and wandb.run:
                wandb.finish()

    def get_training_results(self) -> Optional[Dict[str, Any]]:
        """Get the training results from the last run."""
        return self.training_results

    def _initialize_wandb(self) -> None:
        """Initialize wandb for this specific stage."""
        if "wandb" not in self.report_to or not wandb:
            return

        if not self.wandb_project:
            print(
                "Warning: wandb requested but no project specified, skipping wandb initialization"
            )
            return

        try:
            # Create detailed config for this stage
            wandb_config = {
                "stage_name": self.stage_config.stage_name,
                "stage_type": type(self.stage_config).__name__,
                "processed_dataset_name": self.stage_config.processed_dataset_name,
                "base_model_hf_id": self.base_model_hf_id,
                "peft_adapter_path": self.peft_adapter_path,
                "model_config": self.model_config.model_dump(mode="json"),
                "stage_config": self.stage_config.model_dump(mode="json"),
            }

            # Add pipeline info if available
            if self.pipeline_info:
                wandb_config.update(
                    {
                        "pipeline_name": self.pipeline_info.get("pipeline_name"),
                        "dataset_name": self.pipeline_info.get("dataset_name"),
                        "stage_index": self.pipeline_info.get("stage_index"),
                        "total_stages": self.pipeline_info.get("total_stages"),
                    }
                )

            # Add PEFT config if available
            if self.peft_config:
                wandb_config["peft_config"] = self.peft_config.model_dump(mode="json")

            # Create tags for better organization
            tags = [self.stage_config.stage_name]
            if self.pipeline_info.get("pipeline_name"):
                tags.append(self.pipeline_info["pipeline_name"])
            if self.pipeline_info.get("dataset_name"):
                tags.append(self.pipeline_info["dataset_name"])

            # Add experiment suite tags if running as part of a suite
            if self.pipeline_info.get("suite_name"):
                tags.append(f"suite:{self.pipeline_info['suite_name']}")
                wandb_config["suite_name"] = self.pipeline_info["suite_name"]
            if self.pipeline_info.get("experiment_id"):
                wandb_config["experiment_id"] = self.pipeline_info["experiment_id"]

            # Add model and config tags for better filtering
            model_name = self.base_model_hf_id.split("/")[-1]
            tags.append(f"model:{model_name}")
            if self.peft_config:
                tags.append(f"peft:{self.peft_config.peft_type.lower()}")

            wandb.init(
                project=self.wandb_project,
                name=self.run_name,
                config=wandb_config,
                tags=tags,
                group=self.pipeline_info.get(
                    "pipeline_name"
                ),  # Group runs from same pipeline
            )

            self.wandb_initialized = True
            print(
                f"Weights & Biases initialized for stage: {self.stage_config.stage_name}"
            )

        except Exception as e:
            print(
                f"Could not initialize W&B for stage {self.stage_config.stage_name}: {e}. Continuing without W&B."
            )
            # Remove wandb from report_to for this stage
            self.report_to = [r for r in self.report_to if r != "wandb"]

    def _load_datasets(self) -> tuple[Optional[Dataset], Optional[Dataset]]:
        """Load the processed dataset for this stage."""
        processed_dataset_path = os.path.join(
            "datasets", "processed", self.stage_config.processed_dataset_name
        )
        print(f"Loading processed dataset from: {processed_dataset_path}")

        if not Path(processed_dataset_path).exists():
            print(f"ERROR: Processed dataset not found at {processed_dataset_path}")
            print(
                f"Please run prepare_dataset.py first to create: {self.stage_config.processed_dataset_name}"
            )
            return None, None

        dataset = load_from_disk(processed_dataset_path)
        print(f"Dataset loaded. Splits: {list(dataset.keys())}")

        train_dataset = dataset.get("train")
        eval_dataset = dataset.get(
            "test"
        )  # prepare_dataset.py saves validation as 'test'

        if not train_dataset:
            print("ERROR: 'train' split not found in the processed dataset.")
            return None, None

        if self.stage_config.evaluation_strategy != "no" and not eval_dataset:
            print(
                f"WARNING: Evaluation strategy is '{self.stage_config.evaluation_strategy}' but no 'test' (validation) split found."
            )

        return train_dataset, eval_dataset

    def _setup_tokenizer(self) -> AutoTokenizer:
        """Load and configure the tokenizer."""
        print(f"Loading tokenizer for: {self.model_config.model_name_or_path}")

        tokenizer_kwargs = {
            "trust_remote_code": self.model_config.trust_remote_code,
            "use_fast": True,
            "add_bos_token": False,  # Prevent trainer from adding a second BOS
        }

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_name_or_path,
            **tokenizer_kwargs,
        )

        if tokenizer.pad_token is None:
            print("Tokenizer missing pad_token, attempting to set to eos_token.")
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                print(f"Set pad_token to eos_token: '{tokenizer.eos_token}'")
            else:
                print("Warning: Tokenizer has no eos_token, pad_token remains None.")

        tokenizer.padding_side = "right"
        print("Tokenizer loaded and configured.")
        return tokenizer

    def _setup_model(self) -> tuple[AutoModelForCausalLM, Optional[LoraConfig]]:
        """Load the model using resolved model information."""
        torch_dtype = get_torch_dtype(self.model_config.torch_dtype)
        print(f"Using torch_dtype for model: {torch_dtype}")

        quantization_config_obj = None
        if self.model_config.quantization_config:
            print(
                f"Setting up quantization config: {self.model_config.quantization_config}"
            )
            bnb_config_params = self.model_config.quantization_config.copy()

            # Convert compute dtype string to actual torch.dtype
            dtype_keys = ["bnb_4bit_compute_dtype", "bnb_8bit_compute_dtype"]
            for key in dtype_keys:
                if key in bnb_config_params and isinstance(bnb_config_params[key], str):
                    try:
                        bnb_config_params[key] = get_torch_dtype(bnb_config_params[key])
                        print(
                            f"Converted {key} to torch.dtype: {bnb_config_params[key]}"
                        )
                    except ValueError as e:
                        print(f"Warning: Could not convert {key}: {e}")

            quantization_config_obj = BitsAndBytesConfig(**bnb_config_params)

        # Load base model or PEFT adapter
        if self.peft_adapter_path:
            print(f"Loading PEFT adapter from: {self.peft_adapter_path}")
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(self.peft_adapter_path)

            # Load base model
            print(f"Loading base model: {peft_config.base_model_name_or_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config_obj,
                attn_implementation=self.model_config.attn_implementation,
            )

            # Apply PEFT adapter
            print("Applying PEFT adapter...")
            model = PeftModel.from_pretrained(base_model, self.peft_adapter_path)
            print("PEFT adapter loaded successfully.")
            
            # Debug: Check what type of model we have
            print(f"🔍 Model type: {type(model)}")
            print(f"🔍 Model class name: {model.__class__.__name__}")
            print(f"🔍 Has peft_config: {hasattr(model, 'peft_config')}")
            if hasattr(model, 'peft_config'):
                print(f"🔍 PEFT config keys: {list(model.peft_config.keys()) if model.peft_config else 'None'}")
            
            # Check for adapter-related parameters before training mode setup
            adapter_param_names = []
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    adapter_param_names.append(name)
            print(f"🔍 Found {len(adapter_param_names)} adapter parameters before training setup")
            if len(adapter_param_names) > 0:
                print("🔍 First few adapter parameters:", adapter_param_names[:3])
            
            # CRITICAL: Ensure the model is in training mode and adapters are trainable
            print("🔧 Setting model to training mode and enabling adapter training...")
            model.train()  # Set model to training mode
            
            # For PeftModel, adapters are already loaded and active
            # Check if this is a PeftModel and if adapters are active
            if hasattr(model, 'peft_config') and model.peft_config:
                print("✓ PeftModel detected with active adapters")
                adapter_names = list(model.peft_config.keys())
                print(f"✓ Active adapters: {adapter_names}")
            else:
                print("⚠️ Warning: Expected PeftModel but peft_config not found")
            
            # Make sure adapter parameters are trainable
            adapter_params_enabled = 0
            for name, param in model.named_parameters():
                if "lora" in name.lower() or "adapter" in name.lower():
                    if not param.requires_grad:
                        param.requires_grad = True
                        adapter_params_enabled += 1
                        print(f"✓ Enabled gradients for: {name}")
            
            if adapter_params_enabled > 0:
                print(f"✓ Enabled gradients for {adapter_params_enabled} adapter parameters")
            
            # Some PEFT models need explicit training mode
            if hasattr(model, 'set_training_mode'):
                model.set_training_mode(True)
                print("✓ Training mode explicitly set")
            
            # Diagnostic: Check if adapter parameters are trainable
            print("\n🔍 PEFT Adapter Training Diagnostics:")
            trainable_params = 0
            total_params = 0
            peft_param_count = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    if "lora" in name.lower() or "adapter" in name.lower():
                        peft_param_count += param.numel()
                        print(f"  ✓ Trainable PEFT param: {name} ({param.numel():,} params)")
            
            print(f"  📊 Total params: {total_params:,}")
            print(f"  📊 Trainable params: {trainable_params:,}")
            print(f"  📊 PEFT params: {peft_param_count:,}")
            print(f"  📊 Percentage trainable: {100 * trainable_params / total_params:.2f}%")
            
            if trainable_params == 0:
                print("  ❌ ERROR: No trainable parameters found! This will cause training errors.")
                print("  🔧 Attempting to fix by enabling all adapter parameters...")
                
                # Emergency fix: find and enable all adapter parameters
                adapter_params_found = 0
                for name, param in model.named_parameters():
                    if any(keyword in name.lower() for keyword in ["lora", "adapter", "peft"]):
                        param.requires_grad = True
                        adapter_params_found += param.numel()
                        print(f"    🔧 Force-enabled: {name}")
                
                if adapter_params_found > 0:
                    print(f"  ✅ Force-enabled {adapter_params_found:,} adapter parameters")
                else:
                    print("  ❌ ERROR: No adapter parameters found in model!")
                    
            elif peft_param_count == 0:
                print("  ❌ ERROR: No PEFT parameters found! Model may not be loaded correctly.")
            else:
                print(f"  ✅ Found {trainable_params:,} trainable parameters ({peft_param_count:,} PEFT) - ready for training")
        else:
            print(f"Loading base model: {self.base_model_hf_id}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model_hf_id,
                trust_remote_code=self.model_config.trust_remote_code,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config_obj,
                attn_implementation=self.model_config.attn_implementation,
            )
            print("Base model loaded.")

        # Setup PEFT configuration for training (if provided)
        actual_peft_config: Optional[LoraConfig] = None
        if self.peft_adapter_path:
            # If we're loading a PEFT adapter, extract its config for the trainer
            print("✓ Extracting PEFT configuration from loaded adapter")
            print(f"🔍 Model has peft_config: {hasattr(model, 'peft_config')}")
            print(f"🔍 peft_config exists: {model.peft_config if hasattr(model, 'peft_config') else 'N/A'}")
            
            # Get the PEFT config from the loaded model
            if hasattr(model, 'peft_config') and model.peft_config:
                # Extract the first adapter's config (typically there's only one)
                adapter_name = list(model.peft_config.keys())[0]
                loaded_peft_config = model.peft_config[adapter_name]
                print(f"🔍 Loaded adapter '{adapter_name}' config: {loaded_peft_config}")
                
                # Convert to LoraConfig for the trainer
                actual_peft_config = LoraConfig(
                    r=loaded_peft_config.r,
                    lora_alpha=loaded_peft_config.lora_alpha,
                    lora_dropout=loaded_peft_config.lora_dropout,
                    target_modules=loaded_peft_config.target_modules,
                    bias=loaded_peft_config.bias,
                    task_type=loaded_peft_config.task_type,
                )
                print(f"✅ Extracted PEFT config: r={loaded_peft_config.r}, alpha={loaded_peft_config.lora_alpha}, target_modules={loaded_peft_config.target_modules}")
            else:
                print("❌ Warning: Could not extract PEFT config from loaded adapter")
                print(f"   hasattr(model, 'peft_config'): {hasattr(model, 'peft_config')}")
                if hasattr(model, 'peft_config'):
                    print(f"   model.peft_config: {model.peft_config}")
                actual_peft_config = None
            
            # Diagnostic: Check if adapter parameters are trainable
            print("\n🔍 PEFT Adapter Training Diagnostics:")
            trainable_params = 0
            total_params = 0
            peft_param_count = 0
            for name, param in model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    if "lora" in name.lower() or "adapter" in name.lower():
                        peft_param_count += param.numel()
                        print(f"  ✓ Trainable PEFT param: {name} ({param.numel():,} params)")
            
            print(f"  📊 Total params: {total_params:,}")
            print(f"  📊 Trainable params: {trainable_params:,}")
            print(f"  📊 PEFT params: {peft_param_count:,}")
            print(f"  📊 Percentage trainable: {100 * trainable_params / total_params:.2f}%")
            
            if trainable_params == 0:
                print("  ❌ ERROR: No trainable parameters found! This will cause training errors.")
            elif peft_param_count == 0:
                print("  ❌ ERROR: No PEFT parameters found! Model may not be loaded correctly.")
            else:
                print(f"  ✅ Found {trainable_params:,} trainable parameters ({peft_param_count:,} PEFT) - ready for training")
                
        elif self.peft_config:
            # Only apply new PEFT config if we're starting from a base model
            if self.peft_config.peft_type == "LORA":
                task_type_enum = getattr(
                    TaskType, self.peft_config.task_type.upper(), TaskType.CAUSAL_LM
                )
                actual_peft_config = LoraConfig(
                    r=self.peft_config.r,
                    lora_alpha=self.peft_config.lora_alpha,
                    lora_dropout=self.peft_config.lora_dropout,
                    target_modules=(
                        self.peft_config.target_modules
                        if isinstance(self.peft_config.target_modules, list)
                        else (
                            [self.peft_config.target_modules]
                            if self.peft_config.target_modules
                            else None
                        )
                    ),
                    bias=self.peft_config.bias,
                    task_type=task_type_enum,
                )
                print(f"PEFT LoRA config created: {actual_peft_config}")
            else:
                print(
                    f"WARNING: PEFT type '{self.peft_config.peft_type}' not supported"
                )

        return model, actual_peft_config


def _inspect_dataloader_batches(
    trainer,
    tokenizer: AutoTokenizer,
    num_batches_to_inspect: int,
    train_config_report_to: list[str],
):
    """Helper function to inspect and print batches from the trainer's dataloader."""
    if num_batches_to_inspect <= 0:
        return

    print(f"\n--- Inspecting first {num_batches_to_inspect} training batch(es) ---")

    # Use the new training validator for comprehensive inspection
    try:
        results = inspect_training_batch(trainer, tokenizer, num_samples=2)

        # Additional validation: test loss calculation on first batch
        if results:
            print(f"\n🧮 Testing Loss Calculation on First Sample")
            train_dataloader = trainer.get_train_dataloader()
            batch = next(iter(train_dataloader))

            # Move batch to model's device
            device = next(trainer.model.parameters()).device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            loss_results = validate_loss_calculation(trainer.model, batch, sample_idx=0)

            if loss_results["loss_calculation_correct"]:
                print("✅ Loss calculation validation passed!")
            else:
                print("❌ Loss calculation validation failed!")

    except Exception as e:
        print(f"⚠️  New validation failed ({e}), falling back to legacy inspection...")

        # Fallback to original inspection method
        train_dataloader = trainer.get_train_dataloader()

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
            print(
                f"Tokenizer pad_token_id is None, using eos_token_id ({pad_token_id}) for decoding labels."
            )

        for i, batch in enumerate(train_dataloader):
            if i >= num_batches_to_inspect:
                break
            print(f"\n--- Batch {i+1} ---")

            input_ids = batch["input_ids"].cpu()
            labels = batch.get("labels")
            if labels is not None:
                labels = labels.cpu()

            print(f"Input_ids shape: {input_ids.shape}")
            if labels is not None:
                print(f"Labels shape: {labels.shape}")
            else:
                print("Labels: Not found in batch")

            for j in range(min(2, input_ids.shape[0])):  # Show max 2 samples per batch
                print(f"\n  Sample {j+1} in Batch {i+1}:")

                current_input_ids = input_ids[j]
                print(
                    f"    Input_ids ({len(current_input_ids)} tokens): {current_input_ids.tolist()}"
                )
                decoded_inputs = tokenizer.decode(
                    current_input_ids, skip_special_tokens=False
                )
                print(f"    Decoded Input_ids: '{decoded_inputs}'")

                if labels is not None:
                    current_labels = labels[j]
                    actual_label_ids = [
                        label_id
                        for label_id in current_labels.tolist()
                        if label_id != -100
                    ]

                    print(
                        f"    Labels ({len(current_labels)} tokens, {len(actual_label_ids)} non-ignored): {current_labels.tolist()}"
                    )
                    if actual_label_ids:
                        clean_label_ids_for_decode = [
                            lid for lid in actual_label_ids if lid != pad_token_id
                        ]
                        if clean_label_ids_for_decode:
                            decoded_labels = tokenizer.decode(
                                clean_label_ids_for_decode, skip_special_tokens=False
                            )
                            print(
                                f"    Decoded Labels (non -100/non-padded): '{decoded_labels}'"
                            )
                        else:
                            print(
                                "    Decoded Labels (non -100/non-padded): '' (all tokens were -100 or padding)"
                            )
                    else:
                        print(
                            "    Decoded Labels (non -100/non-padded): '' (all tokens were -100)"
                        )
                else:
                    print("    Labels: Not available for this sample.")

    print("\n--- Data inspection finished. ---")
