from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from typing import Optional, Dict, Any, List, Tuple, Type, Literal
from tqdm.auto import tqdm
import math

from probity.collection.activation_store import ActivationStore
from probity.probes.linear_probe import LinearProbe, LinearProbeConfig

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


class BaseProbeTrainer(ABC):
    """Enhanced abstract base class for all probe trainers."""
    
    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        
    def _get_lr_scheduler(
        self, 
        optimizer: optim.Optimizer,
        start_lr: float,
        end_lr: float,
        num_steps: int
    ) -> optim.lr_scheduler.LRScheduler:
        """Create exponential learning rate scheduler."""
        gamma = math.exp(math.log(end_lr / start_lr) / num_steps)
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
        
        return weights
        
    def _create_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer_type == "Adam":
            return optim.Adam(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "AdamW":
            return optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "SGD":
            return optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def prepare_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> torch.Tensor:
        """Prepare data for training with normalization."""
        X, _ = activation_store.get_probe_data(position_key)
        
        if self.feature_mean is None:
            self.feature_mean = X.mean(dim=0, keepdim=True)
            self.feature_std = X.std(dim=0, keepdim=True) + 1e-8

        X = (X - self.feature_mean) / self.feature_std
        return X


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
        self.config = config
        
    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val splits with DataLoader creation."""
        X = self.prepare_data(activation_store, position_key)
        _, y = activation_store.get_probe_data(position_key)
        
        # Convert labels to float
        y = y.float()  # Add this line
        
        # Split data
        n_train = int(len(X) * self.config.train_ratio)
        indices = torch.randperm(len(X))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Create dataloaders
        # Handle both single and multi-dimensional labels
        y_train = y_train if y_train.dim() > 1 else y_train.unsqueeze(1)
        y_val = y_val if y_val.dim() > 1 else y_val.unsqueeze(1)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
    
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size
        )
        
        return train_loader, val_loader

    def train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epoch: int,
        num_epochs: int,
    ) -> float:
        """Run one epoch of training with progress tracking."""
        model.train()
        total_loss = 0
        
        # Create progress bar for batches
        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not self.config.show_progress,
            leave=False
        )
        
        for batch_x, batch_y in batch_pbar:
            optimizer.zero_grad()
            batch_x = batch_x.to(self.config.device)
            batch_y = batch_y.to(self.config.device)
            
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
            
        return total_loss / len(train_loader)

    def train(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train model with full features including LR scheduling and progress tracking."""
        optimizer = self._create_optimizer(model)
        scheduler = self._get_lr_scheduler(
            optimizer,
            self.config.learning_rate,
            self.config.end_learning_rate,
            self.config.num_epochs
        )
        
        # Set up loss function with class imbalance handling if needed
        if self.config.handle_class_imbalance:
            pos_weight = self._calculate_pos_weights(
                torch.cat([y for _, y in train_loader])
            ).to(self.config.device)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        # Create progress bar for epochs
        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            disable=not self.config.show_progress
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in epoch_pbar:
            # Train epoch
            train_loss = self.train_epoch(
                model, train_loader, optimizer, loss_fn, epoch, self.config.num_epochs
            )
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(model, val_loader, loss_fn)
                history["val_loss"].append(val_loss)
                
                # Early stopping check
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
                
                epoch_pbar.set_postfix({
                    "Train Loss": f"{train_loss:.6f}",
                    "Val Loss": f"{val_loss:.6f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            else:
                epoch_pbar.set_postfix({
                    "Train Loss": f"{train_loss:.6f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            history["learning_rate"].append(scheduler.get_last_lr()[0])
            scheduler.step()
        
        return history

    def validate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module
    ) -> float:
        """Run validation with progress tracking."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device)
                
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)