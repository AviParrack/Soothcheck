from abc import ABC
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from typing import Optional, Dict, List, Tuple, Literal
from tqdm.auto import tqdm
import math

from probity.collection.activation_store import ActivationStore

# Probes now expect non-standardized data for forward pass and store unscaled directions
from probity.probes import (
    BaseProbe,
    DirectionalProbe,
    MultiClassLogisticProbe,
    LogisticProbe,
    LinearProbe,
)


@dataclass
class BaseTrainerConfig:
    """Enhanced base configuration shared by all trainers."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Optional[str] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-5  # For LR scheduling
    weight_decay: float = 0.01
    num_epochs: int = 10
    show_progress: bool = True
    optimizer_type: Literal["Adam", "SGD", "AdamW"] = "Adam"
    handle_class_imbalance: bool = True
    standardize_activations: bool = False  # Option to standardize *during* training


class BaseProbeTrainer(ABC):
    """Enhanced abstract base class for all probe trainers. Handles standardization during training."""

    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        # Store standardization stats if standardization is enabled
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        print(f"[DEBUG BaseProbeTrainer.__init__] Trainer initialized with device: {config.device}")
        print(f"[DEBUG BaseProbeTrainer.__init__] standardize_activations: {config.standardize_activations}")

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer, start_lr: float, end_lr: float, num_steps: int
    ) -> optim.lr_scheduler.LRScheduler:
        """Create exponential learning rate scheduler."""
        if start_lr <= 0 or end_lr <= 0:
            # Handle cases where LR might be zero or negative, default to constant LR
            print("Warning: start_lr or end_lr <= 0, using constant LR scheduler.")
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Ensure num_steps is positive
        if num_steps <= 0:
            print("Warning: num_steps <= 0 for LR scheduler, using constant LR.")
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        # Avoid log(0) or division by zero
        if start_lr == end_lr:
            gamma = 1.0
        else:
            # Ensure end_lr / start_lr is positive
            ratio = end_lr / start_lr
            if ratio <= 0:
                print(
                    f"Warning: Invalid learning rate range ({start_lr} -> {end_lr}). "
                    f"Using constant LR."
                )
                return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            gamma = math.exp(math.log(ratio) / num_steps)

        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _calculate_pos_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Calculate positive weights for handling class imbalance.

        Handles both single-output (y shape: [N, 1]) and
        multi-output (y shape: [N, C]) cases.
        """
        if y.dim() == 1:
            y = y.unsqueeze(1)

        # Calculate weights for each output dimension
        num_pos = y.sum(dim=0)
        num_neg = len(y) - num_pos
        weights = num_neg / (num_pos + 1e-8)  # Add epsilon to prevent division by zero

        print(f"[DEBUG _calculate_pos_weights] input dtype: {y.dtype}, output dtype: {weights.dtype}")
        return weights

    def _calculate_class_weights(
        self, y: torch.Tensor, num_classes: int
    ) -> Optional[torch.Tensor]:
        """Calculate class weights for multi-class CrossEntropyLoss."""
        print(f"[DEBUG _calculate_class_weights] input dtype: {y.dtype}, num_classes: {num_classes}")
        
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)  # Convert [N, 1] to [N]
        if y.dim() != 1:
            print("Warning: Cannot calculate class weights for non-1D target tensor.")
            return None

        # Check dtype before attempting conversion or calculations
        if y.dtype != torch.long:
            print(
                f"Warning: Cannot calculate class weights for non-Long target tensor "
                f"(dtype: {y.dtype})."
            )
            return None
            # Note: We no longer attempt conversion for non-long types.
            # If conversion is desired, the calling code should handle it.

        counts = torch.bincount(y, minlength=num_classes)
        # Avoid division by zero for classes with zero samples
        total_samples = counts.sum()
        if total_samples == 0:
            return None  # No samples to calculate weights from

        # Calculate weights: (total_samples / (num_classes * count_for_class))
        # This gives higher weight to less frequent classes.
        weights = total_samples / (num_classes * (counts + 1e-8))
        
        print(f"[DEBUG _calculate_class_weights] output weights dtype: {weights.dtype}")
        return weights

    def _create_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Create optimizer based on config."""
        print(f"[DEBUG _create_optimizer] Creating {self.config.optimizer_type} optimizer")
        
        optimizer_class = getattr(optim, self.config.optimizer_type, None)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        # Filter out parameters that do not require gradients
        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
        
        # Check parameter dtypes
        param_dtypes = set(p.dtype for p in model.parameters() if p.requires_grad)
        print(f"[DEBUG _create_optimizer] Parameter dtypes: {param_dtypes}")

        if self.config.optimizer_type in ["Adam", "AdamW"]:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "SGD":
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                # Add momentum? momentum=0.9 might be a good default
            )
        else:
            # Should be caught by getattr check, but belts and braces
            raise ValueError(
                f"Unsupported optimizer type: {self.config.optimizer_type}"
            )

        return optimizer

    def prepare_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training, optionally applying standardization."""
        X_expected, y_expected = activation_store.get_probe_data(position_key)
        
        print(f"[DEBUG prepare_data] Raw X dtype from activation store: {X_expected.dtype}")
        print(f"[DEBUG prepare_data] Raw y dtype from activation store: {y_expected.dtype}")
        print(f"[DEBUG prepare_data] X shape: {X_expected.shape}, y shape: {y_expected.shape}")
        
        X_train = X_expected  # Default to original if no standardization
    
        # Apply standardization only if configured
        if self.config.standardize_activations:
            print(f"[DEBUG prepare_data] Applying standardization")
            # Compute statistics if not already done (e.g., first call)
            if self.feature_mean is None or self.feature_std is None:
                self.feature_mean = X_expected.mean(dim=0, keepdim=True)
                self.feature_std = X_expected.std(dim=0, keepdim=True) + 1e-8
                
                print(f"[DEBUG prepare_data] Feature mean dtype: {self.feature_mean.dtype}")
                print(f"[DEBUG prepare_data] Feature std dtype: {self.feature_std.dtype}")
    
                # Move statistics to the correct device
                if self.config.device:
                    target_device = torch.device(self.config.device)
                    self.feature_mean = self.feature_mean.to(target_device)
                    self.feature_std = self.feature_std.to(target_device)

            # Apply standardization to create X_train
            if self.feature_mean is not None and self.feature_std is not None:
                # Ensure X_orig is on the same device as stats before operation
                if hasattr(self.feature_mean, "device"):
                    X_orig_dev = X_expected.to(self.feature_mean.device)
                    X_train = (X_orig_dev - self.feature_mean) / self.feature_std
                else:
                    # Fallback if stats don't have device info (shouldn't happen)
                    X_train = (X_expected - self.feature_mean) / self.feature_std
                
                print(f"[DEBUG prepare_data] After standardization - X_train dtype: {X_train.dtype}")

        return X_train, y_expected, X_expected  # Return both training and original activations


@dataclass
class SupervisedTrainerConfig(BaseTrainerConfig):
    """Enhanced config for supervised training methods."""

    train_ratio: float = 0.8
    patience: int = 5
    min_delta: float = 1e-4


class SupervisedProbeTrainer(BaseProbeTrainer):
    """Enhanced trainer for supervised probes with progress tracking and LR scheduling."""

    def __init__(self, config: SupervisedTrainerConfig):
        super().__init__(config)
        # Ensure self.config has the more specific type
        if not isinstance(config, SupervisedTrainerConfig):
            raise TypeError("SupervisedProbeTrainer requires a SupervisedTrainerConfig")
        self.config: SupervisedTrainerConfig = config
        print(f"[DEBUG SupervisedProbeTrainer.__init__] Initialized with train_ratio: {config.train_ratio}")

    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val splits with DataLoader creation.
        DataLoaders yield batches of (X_train, y, X_orig).
        """
        X_train_all, y_all, X_orig_all = self.prepare_data(
            activation_store, position_key
        )

        # Split data
        n_total = len(X_orig_all)
        n_train = int(n_total * self.config.train_ratio)
        print(f"[DEBUG prepare_supervised_data] Total examples: {n_total}, Training examples: {n_train}")
        
        # Ensure validation set is not empty
        if n_train == n_total:
            n_train = max(
                0, n_total - 1
            )  # Keep at least one sample for validation if possible
            if n_train == 0 and n_total > 0:
                print(
                    "Warning: Only one data point available. Using it for training and validation."
                )
                n_train = 1
            elif n_total > 0:
                print(
                    "Warning: train_ratio resulted in no validation data. "
                    "Adjusting to keep one sample for validation."
                )

        # Ensure train set is not empty if n_total > 0
        if n_train == 0 and n_total > 0:
            print(
                "Warning: train_ratio resulted in no training data. Using all data for training."
            )
            n_train = n_total

        # Generate random permutation
        indices = torch.randperm(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        # Handle edge case: If either split is empty (can happen if n_total is very small)
        if len(train_indices) == 0 and n_total > 0:
            train_indices = indices  # Use all data for training
            val_indices = indices  # Use all data for validation too (less ideal)
            print(
                "Warning: No training samples after split. Using all data for training and validation."
            )
        elif len(val_indices) == 0 and n_total > 0:
            val_indices = (
                train_indices  # Use training data for validation if val split is empty
            )
            print(
                "Warning: No validation samples after split. "
                "Using training data for validation."
            )

        X_train_split, X_val_split = (
            X_train_all[train_indices],
            X_train_all[val_indices],
        )
        X_orig_train_split, X_orig_val_split = (
            X_orig_all[train_indices],
            X_orig_all[val_indices],
        )
        y_train_split, y_val_split = y_all[train_indices], y_all[val_indices]

        # Create dataloaders
        # Handle both single and multi-dimensional labels
        y_train_split = (
            y_train_split if y_train_split.dim() > 1 else y_train_split.unsqueeze(1)
        )
        y_val_split = y_val_split if y_val_split.dim() > 1 else y_val_split.unsqueeze(1)

        print(f"[DEBUG prepare_supervised_data] Final X_train_split dtype: {X_train_split.dtype}")
        print(f"[DEBUG prepare_supervised_data] Final y_train_split dtype: {y_train_split.dtype}")
        
        # Note: We don't move tensors to device here because DataLoader will
        # create copies that could waste memory. Instead, we move tensors to
        # device in training loop just before using them.
        train_dataset = TensorDataset(X_train_split, y_train_split, X_orig_train_split)
        val_dataset = TensorDataset(X_val_split, y_val_split, X_orig_val_split)

        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        return train_loader, val_loader

    def train_epoch(
        self,
        model: torch.nn.Module,  # Use base Module type here, checked in train
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        is_multi_class: bool = False,  # Flag for multi-class loss
    ) -> float:
        """Run one epoch of training with progress tracking. Uses X_train for training."""
        model.train()
        total_loss = 0

        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not self.config.show_progress,
            leave=False,
        )

        for batch_idx, (batch_x_train, batch_y, _) in enumerate(batch_pbar):  # Ignore X_orig during training pass
            if batch_idx == 0:  # Only print for first batch
                print(f"[DEBUG train_epoch] First batch - x dtype: {batch_x_train.dtype}, y dtype: {batch_y.dtype}")
                print(f"[DEBUG train_epoch] Model dtype: {next(model.parameters()).dtype}")
                print(f"[DEBUG train_epoch] Loss function: {type(loss_fn).__name__}")
            
            optimizer.zero_grad()
            batch_x_train = batch_x_train.to(self.config.device)
            batch_y = batch_y.to(self.config.device)
            
            if batch_idx == 0:
                print(f"[DEBUG train_epoch] After moving to device - x dtype: {batch_x_train.dtype}")

            # --- Adjust target shape/type for loss ---
            model_dtype = next(model.parameters()).dtype
            if is_multi_class:
                # CrossEntropyLoss expects Long targets of shape [N]
                if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                    batch_y = batch_y.squeeze(1)
                batch_y = batch_y.long()  # Ensure Long type
                if batch_idx == 0:
                    print(f"[DEBUG train_epoch] Using Long targets for multi-class: {batch_y.dtype}")
            else:  # Other losses expect same dtype as model parameters
                batch_y = batch_y.to(dtype=model_dtype)  # Ensure same dtype as model
                if batch_idx == 0:
                    print(f"[DEBUG train_epoch] Converted targets to model dtype: {batch_y.dtype}")
            # -----------------------------------------

            # Model forward pass uses the potentially standardized data
            outputs = model(batch_x_train)
            if batch_idx == 0:
                print(f"[DEBUG train_epoch] Forward pass output dtype: {outputs.dtype}")
                print(f"[DEBUG train_epoch] Target for loss dtype: {batch_y.dtype}")

            try:
                loss = loss_fn(outputs, batch_y)
                if batch_idx == 0:
                    print(f"[DEBUG train_epoch] Loss computation dtype: {loss.dtype}")
                
                loss.backward()
                if batch_idx == 0:
                    # Check parameter gradients after backward
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            print(f"[DEBUG train_epoch] Gradient for {name} - dtype: {param.grad.dtype}")
                        else:
                            print(f"[DEBUG train_epoch] No gradient for {name}")
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
            except Exception as e:
                print(f"[ERROR train_epoch] Exception during training: {str(e)}")
                # Print detailed tensor information to debug dtype issues
                print(f"[ERROR train_epoch] outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                print(f"[ERROR train_epoch] batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
                print(f"[ERROR train_epoch] loss_fn type: {type(loss_fn).__name__}")
                raise  # Re-raise the exception

        return total_loss / len(train_loader)

    def train(
        self,
        model: BaseProbe,  # Expect a BaseProbe instance
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train model, potentially unscale direction if standardization was used."""
        # Ensure model is a BaseProbe instance
        if not isinstance(model, BaseProbe):
            raise TypeError(
                "SupervisedProbeTrainer expects model to be an instance of BaseProbe"
            )

        # Ensure model is on the correct device
        target_device = torch.device(self.config.device)
        model.to(target_device)

        # Determine if this is a multi-class probe
        is_multi_class = isinstance(model, MultiClassLogisticProbe)
        print(f"[DEBUG train] Model class: {model.__class__.__name__}, is_multi_class: {is_multi_class}")
        print(f"[DEBUG train] Model dtype: {next(model.parameters()).dtype}")
        
        # Standardization stats are managed by the trainer, not transferred

        optimizer = self._create_optimizer(model)
        scheduler = self._get_lr_scheduler(
            optimizer,
            self.config.learning_rate,
            self.config.end_learning_rate,
            self.config.num_epochs,
        )

        # --- Set up loss function ---
        loss_fn: nn.Module
        all_train_y: Optional[torch.Tensor] = None
        if self.config.handle_class_imbalance:
            # Calculate all labels only once if needed for any weight calculation
            all_train_y = torch.cat([y for _, y, _ in train_loader])
            print(f"[DEBUG train] Combined labels for class weights - shape: {all_train_y.shape}, dtype: {all_train_y.dtype}")

        model_dtype = next(model.parameters()).dtype
        print(f"[DEBUG train] Setting up loss function for dtype: {model_dtype}")

        if isinstance(model, MultiClassLogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                num_classes = model.config.output_size
                class_weights = self._calculate_class_weights(all_train_y, num_classes)
                if class_weights is not None:
                    weights_arg = class_weights.to(target_device).to(dtype=model_dtype)
                    print(f"[DEBUG train] Class weights set to dtype: {weights_arg.dtype}")
            # Pass `class_weights` keyword arg
            loss_fn = model.get_loss_fn(class_weights=weights_arg)

        elif isinstance(model, LogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device).to(dtype=model_dtype)
                    print(f"[DEBUG train] Pos weights set to dtype: {weights_arg.dtype}")
            # Pass `pos_weight` keyword arg
            loss_fn = model.get_loss_fn(pos_weight=weights_arg)

        elif isinstance(model, LinearProbe):
            # LinearProbe loss (MSE, L1, Cosine) doesn't use these weights
            loss_fn = model.get_loss_fn()
            # Ensure loss function is of the correct dtype
            loss_fn = loss_fn.to(dtype=model_dtype)
            print(f"[DEBUG train] Linear loss type: {model.config.loss_type}")
            if self.config.handle_class_imbalance:
                print(
                    f"Warning: Class imbalance handling enabled, but may not be effective "
                    f"for LinearProbe with loss type '{model.config.loss_type}'."
                )
        else:
            # Fallback for other BaseProbe types that might be supervised
            probe_type_name = type(model).__name__
            print(
                f"Warning: Unknown supervised probe type '{probe_type_name}'. "
                f"Attempting default BCEWithLogitsLoss. Ensure this is appropriate."
            )
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device).to(dtype=model_dtype)
            # Assume BCE loss for unknown types
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights_arg).to(dtype=model_dtype)
        
        # Ensure loss function is on the correct device
        loss_fn = loss_fn.to(device=target_device)
        print(f"[DEBUG train] Loss function: {type(loss_fn).__name__}")
        # --- End Loss Function Setup ---

        # Training history
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            disable=not self.config.show_progress,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in epoch_pbar:
            # Train epoch uses X_train
            train_loss = self.train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                epoch,
                self.config.num_epochs,
                is_multi_class=is_multi_class,  # Pass flag
            )
            history["train_loss"].append(train_loss)

            # Validation uses X_orig
            if val_loader is not None:
                val_loss = self.validate(
                    model, val_loader, loss_fn, is_multi_class=is_multi_class
                )
                history["val_loss"].append(val_loss)

                # Early stopping check
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state? Optional
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "Val Loss": f"{val_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
            else:
                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            history["learning_rate"].append(scheduler.get_last_lr()[0])
            scheduler.step()

        # --- Post-Training Direction Unscaling ---
        if self.config.standardize_activations and self.feature_std is not None:
            print("Unscaling probe direction...")
            with torch.no_grad():
                # Get the direction learned on standardized data
                learned_direction = model._get_raw_direction_representation()
                print(f"[DEBUG train] Learned direction dtype before unscaling: {learned_direction.dtype}")

                # Unscale the direction
                # Ensure std dev matches direction dims for division
                std_dev = self.feature_std.squeeze().to(learned_direction.device)
                print(f"[DEBUG train] Feature std dtype for unscaling: {std_dev.dtype}")

                # Handle potential shape mismatches (e.g., [1, dim] vs [dim])
                if (
                    learned_direction.dim() == 2
                    and learned_direction.shape[0] == 1
                    and std_dev.dim() == 1
                ):
                    # Common case for Linear/Logistic: weights are [1, dim], std_dev is [dim]
                    unscaled_direction = learned_direction / std_dev.unsqueeze(0)
                    print(f"[DEBUG train] Unscaled case 1: shape {unscaled_direction.shape}")
                elif learned_direction.shape == std_dev.shape:
                    unscaled_direction = learned_direction / std_dev
                    print(f"[DEBUG train] Unscaled case 2: shape {unscaled_direction.shape}")
                elif (
                    learned_direction.dim() == 1
                    and std_dev.dim() == 1
                    and learned_direction.shape[0] == std_dev.shape[0]
                ):
                    # Case for single vector directions (like maybe directional probes before squeeze?)
                    unscaled_direction = learned_direction / std_dev
                    print(f"[DEBUG train] Unscaled case 3: shape {unscaled_direction.shape}")
                else:
                    print(
                        f"Warning: Shape mismatch during final unscaling. Direction: {learned_direction.shape}, StdDev: {std_dev.shape}. Skipping unscaling."
                    )
                    unscaled_direction = (
                        learned_direction  # Keep original if shapes mismatch
                    )
                    print(f"[DEBUG train] Using original direction due to shape mismatch")

                print(f"[DEBUG train] Unscaled direction dtype: {unscaled_direction.dtype}")
                
                # Update the probe's internal representation with the unscaled direction
                model._set_raw_direction_representation(unscaled_direction)

        return history

    def validate(
        self,
        model: torch.nn.Module,  # Use base Module type here
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        is_multi_class: bool = False,  # Flag for multi-class loss
    ) -> float:
        """Run validation with progress tracking. Uses X_orig for validation."""
        model.eval()
        total_loss = 0
        model_dtype = next(model.parameters()).dtype

        with torch.no_grad():
            for batch_idx, (_, batch_y, batch_x_orig) in enumerate(val_loader):  # Use X_orig for validation
                if batch_idx == 0:
                    print(f"[DEBUG validate] Validation batch - orig x dtype: {batch_x_orig.dtype}, y dtype: {batch_y.dtype}")
                
                batch_x_orig = batch_x_orig.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                # --- Adjust target shape/type for loss ---
                if is_multi_class:
                    # CrossEntropyLoss expects Long targets of shape [N]
                    if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.long()  # Ensure Long type
                    if batch_idx == 0:
                        print(f"[DEBUG validate] Using Long targets for multi-class: {batch_y.dtype}")
                else:
                    # Ensure target has same dtype as model for consistent loss calculation
                    batch_y = batch_y.to(dtype=model_dtype)
                    if batch_idx == 0:
                        print(f"[DEBUG validate] Converted targets to model dtype: {batch_y.dtype}")
                # -----------------------------------------

                # Model forward pass uses original (non-standardized) data
                # Assumes the probe direction has been unscaled if needed after training
                outputs = model(batch_x_orig)
                if batch_idx == 0:
                    print(f"[DEBUG validate] Validation output dtype: {outputs.dtype}")
                
                try:
                    loss = loss_fn(outputs, batch_y)
                    total_loss += loss.item()
                except Exception as e:
                    print(f"[ERROR validate] Exception during validation: {str(e)}")
                    print(f"[ERROR validate] outputs shape: {outputs.shape}, dtype: {outputs.dtype}")
                    print(f"[ERROR validate] batch_y shape: {batch_y.shape}, dtype: {batch_y.dtype}")
                    print(f"[ERROR validate] loss_fn type: {type(loss_fn).__name__}")
                    raise  # Re-raise the exception

        return total_loss / len(val_loader)


@dataclass
class DirectionalTrainerConfig(BaseTrainerConfig):
    """Configuration for training direction-finding probes."""

    pass  # Uses base config settings


class DirectionalProbeTrainer(BaseProbeTrainer):
    """Trainer for probes that find directions through direct computation (KMeans, PCA, MeanDiff)."""

    def __init__(self, config: DirectionalTrainerConfig):
        super().__init__(config)
        # Ensure self.config has the more specific type
        if not isinstance(config, DirectionalTrainerConfig):
            raise TypeError(
                "DirectionalProbeTrainer requires a DirectionalTrainerConfig"
            )
        self.config: DirectionalTrainerConfig = config
        print(f"[DEBUG DirectionalProbeTrainer.__init__] Initialized")

    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for directional probe computation.
        Returns two identical DataLoaders yielding (X_train, y, X_orig).
        """
        X_train_all, y_all, X_orig_all = self.prepare_data(
            activation_store, position_key
        )

        # Create a dataset with all data
        # Handle single/multi-dimensional labels
        y_all = y_all if y_all.dim() > 1 else y_all.unsqueeze(1)
        dataset = TensorDataset(X_train_all, y_all, X_orig_all)  # Include X_orig

        # Create loader for all data
        # Shuffle False might be better if order matters, but usually doesn't for these methods
        # Use full dataset length as batch size for single fit step
        batch_size_fit = len(dataset)
        if batch_size_fit == 0:
            print("Warning: Dataset is empty for DirectionalProbeTrainer")
            # Return empty loaders to avoid errors
            return DataLoader([]), DataLoader([])

        all_data_loader = DataLoader(
            dataset,
            batch_size=batch_size_fit,
            shuffle=False,  # No need to shuffle if using full batch
        )

        print(f"[DEBUG prepare_supervised_data] Created dataloader with batch size: {batch_size_fit}")
        print(f"[DEBUG prepare_supervised_data] X dtype: {X_train_all.dtype}, y dtype: {y_all.dtype}")

        # Return the same loader twice to maintain the trainer interface
        return all_data_loader, all_data_loader

    def train(
        self,
        model: DirectionalProbe,  # Expect DirectionalProbe instance
        train_loader: DataLoader,
        val_loader: Optional[
            DataLoader
        ] = None,  # val_loader is ignored but kept for API consistency
    ) -> Dict[str, List[float]]:
        """Train method for directional probes.
        1. Accumulate all data (X_train, y, X_orig)
        2. Call probe's fit method with X_train, which returns the initial direction
        3. Unscale the initial direction if standardization was used
        4. Set the probe's final direction buffer
        5. Compute metrics using X_orig and the final probe state
        """
        # Ensure model is a DirectionalProbe instance
        if not isinstance(model, DirectionalProbe):
            raise TypeError(
                "DirectionalProbeTrainer expects model to be an instance of DirectionalProbe"
            )

        print(f"[DEBUG DirectionalProbeTrainer.train] Model class: {model.__class__.__name__}")
        print(f"[DEBUG DirectionalProbeTrainer.train] Model dtype: {model.dtype}")

        # Ensure model is on the correct device
        target_device = torch.device(self.config.device)
        model.to(target_device)

        # Standardization stats are managed by trainer

        # Accumulate all data (loader should have batch_size=len(dataset))
        try:
            x_train_tensor, y_train_tensor, x_orig_tensor = next(iter(train_loader))
            print(f"[DEBUG DirectionalProbeTrainer.train] Loaded data - x_train dtype: {x_train_tensor.dtype}, y dtype: {y_train_tensor.dtype}")
        except StopIteration:
            print("Warning: Training loader is empty.")
            return {"train_loss": [], "val_loss": []}  # Return empty history

        x_train_tensor = x_train_tensor.to(target_device)
        y_train_tensor = y_train_tensor.to(target_device)
        x_orig_tensor = x_orig_tensor.to(target_device)  # Keep original data too
        print(f"[DEBUG DirectionalProbeTrainer.train] Data moved to device {target_device}")

        # Fit the probe using potentially standardized data (X_train)
        # The fit method now returns the initial direction (potentially scaled)
        print(f"[DEBUG DirectionalProbeTrainer.train] Calling probe.fit() method")
        initial_direction = model.fit(x_train_tensor, y_train_tensor)

        print(f"[DEBUG DirectionalProbeTrainer.train] After fit - initial_direction dtype: {initial_direction.dtype}")
        print(f"[DEBUG DirectionalProbeTrainer.train] After fit - initial_direction shape: {initial_direction.shape}")

        # --- Unscale Direction ---
        final_direction = initial_direction  # Default if no standardization
        if self.config.standardize_activations and self.feature_std is not None:
            print("[DEBUG DirectionalProbeTrainer.train] Unscaling directional probe direction...")
            print(f"[DEBUG DirectionalProbeTrainer.train] feature_std dtype: {self.feature_std.dtype}")
            std_dev = self.feature_std.squeeze().to(initial_direction.device)
            
            # Assume initial_direction is [dim]
            if initial_direction.shape == std_dev.shape:
                final_direction = initial_direction / std_dev
                print(f"[DEBUG DirectionalProbeTrainer.train] Unscaled with case 1 (same shape)")
            # Add case for [1, dim] direction and [dim] std_dev
            elif (
                initial_direction.dim() == 2
                and initial_direction.shape[0] == 1
                and std_dev.dim() == 1
            ):
                final_direction = initial_direction / std_dev.unsqueeze(0)
                print(f"[DEBUG DirectionalProbeTrainer.train] Unscaled with case 2 (unsqueezed std)")
            else:
                print(
                    f"Warning: Shape mismatch during directional probe unscaling. "
                    f"Direction: {initial_direction.shape}, StdDev: {std_dev.shape}. "
                    f"Skipping unscaling."
                )
                print(f"[DEBUG DirectionalProbeTrainer.train] Keeping original direction due to shape mismatch")

            print(f"[DEBUG DirectionalProbeTrainer.train] Final direction dtype after unscaling: {final_direction.dtype}")
            print(f"[DEBUG DirectionalProbeTrainer.train] Final direction shape after unscaling: {final_direction.shape}")

        # Set the final, unscaled direction in the probe's buffer
        print(f"[DEBUG DirectionalProbeTrainer.train] Setting final direction in probe")
        model._set_raw_direction_representation(final_direction)

        # --- Compute Metrics ---
        # Metrics should be computed using the *original* data and the *final* probe state
        history: Dict[str, List[float]] = {
            "train_loss": [],
            # Validation loss is same as train loss since we use all data
            # and val_loader is ignored
            "val_loss": [],
        }

        # Calculate loss using original data and final probe state
        with torch.no_grad():
            print(f"[DEBUG DirectionalProbeTrainer.train] Computing metrics with original data")
            # Probe forward uses the final (unscaled) direction set above
            preds = model(x_orig_tensor)
            print(f"[DEBUG DirectionalProbeTrainer.train] Prediction dtype: {preds.dtype}")
            
            # Ensure y_train_tensor matches expected shape for loss
            # Get model's dtype for loss calculation
            model_dtype = model.dtype  # Use model.dtype directly as DirectionalProbe has this attribute
            y_target = y_train_tensor.to(dtype=model_dtype)
            print(f"[DEBUG DirectionalProbeTrainer.train] Target dtype for loss: {y_target.dtype}")
            
            if preds.dim() == 1 and y_target.dim() == 2 and y_target.shape[1] == 1:
                y_target = y_target.squeeze(1)
                print(f"[DEBUG DirectionalProbeTrainer.train] Squeezed y_target shape: {y_target.shape}")
            elif preds.shape != y_target.shape:
                # Attempt to align shapes if possible (e.g., [N] vs [N, 1])
                try:
                    y_target = y_target.view_as(preds)
                    print(f"[DEBUG DirectionalProbeTrainer.train] Reshaped y_target to match preds: {y_target.shape}")
                except RuntimeError:
                    print(
                        f"Warning: Could not align prediction ({preds.shape}) and "
                        f"target ({y_target.shape}) shapes for loss calculation."
                    )
                    # Fallback or skip loss calculation
                    history["train_loss"].append(float("nan"))
                    history["val_loss"].append(float("nan"))
                    return history

            # Use BCEWithLogitsLoss as it's common for binary classification probes
            # Use original y labels
            loss_item = float("nan")  # Default value if calculation fails
            try:
                # Use the appropriate loss based on the probe type
                if isinstance(model, MultiClassLogisticProbe):
                    print(f"[DEBUG DirectionalProbeTrainer.train] Using CrossEntropyLoss for multi-class")
                    loss_fn = nn.CrossEntropyLoss().to(device=target_device, dtype=model_dtype)
                    y_target = y_target.long().squeeze()  # Ensure long and [N] shape
                else:
                    print(f"[DEBUG DirectionalProbeTrainer.train] Using BCEWithLogitsLoss for binary")
                    loss_fn = nn.BCEWithLogitsLoss().to(device=target_device, dtype=model_dtype)
                    # y_target is already float
                
                print(f"[DEBUG DirectionalProbeTrainer.train] Loss function: {type(loss_fn).__name__}")
                print(f"[DEBUG DirectionalProbeTrainer.train] Final shapes - preds: {preds.shape}, y_target: {y_target.shape}")
                print(f"[DEBUG DirectionalProbeTrainer.train] Final dtypes - preds: {preds.dtype}, y_target: {y_target.dtype}")

                loss = loss_fn(preds, y_target)
                loss_item = loss.item()
                print(f"[DEBUG DirectionalProbeTrainer.train] Loss calculated: {loss_item}")
            except Exception as e:
                print(f"[ERROR DirectionalProbeTrainer.train] Error during loss calculation: {e}")
                # Provide detailed information about the tensor shapes and types
                print(f"[ERROR DirectionalProbeTrainer.train] preds shape: {preds.shape}, dtype: {preds.dtype}")
                print(f"[ERROR DirectionalProbeTrainer.train] y_target shape: {y_target.shape}, dtype: {y_target.dtype}")
                # Still set loss_item to NaN so history recording continues

        history["train_loss"].append(loss_item)
        history["val_loss"].append(loss_item)  # Use same loss for val

        return history