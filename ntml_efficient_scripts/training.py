"""
PyTorch training loop for NTML binary token classification.

Implements efficient training of binary classification probes on assistant tokens.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .config import NTMLBinaryTrainingConfig

logger = logging.getLogger(__name__)


class BinaryTokenProbe(nn.Module):
    """Simple binary classification probe for token-level prediction."""
    
    def __init__(self, input_size: int, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, 1),
        )
        
        # Initialize weights
        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns logits."""
        return self.classifier(x).squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities using sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions."""
        probs = self.predict_proba(x)
        return (probs > threshold).long()


class NTMLBinaryTrainer:
    """Trainer for binary token classification on NTML data."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision training
        
        # Metrics tracking
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auroc": [],
            "learning_rate": [],
        }
        
        # Setup mixed precision training if using fp16/bf16
        if config.dtype in ["float16", "bfloat16"]:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Enabled mixed precision training")
    
    def prepare_data(self, activations: torch.Tensor, labels: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        
        logger.info(f"Preparing data loaders from {len(activations)} tokens")
        
        # Split into train/val
        num_samples = len(activations)
        num_train = int(num_samples * self.config.train_ratio)
        
        # Shuffle indices
        indices = torch.randperm(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_X = activations[train_indices]
        train_y = labels[train_indices]
        val_X = activations[val_indices]
        val_y = labels[val_indices]
        
        logger.info(f"Train samples: {len(train_X)}, Val samples: {len(val_X)}")
        logger.info(f"Train label distribution: {torch.bincount(train_y).tolist()}")
        logger.info(f"Val label distribution: {torch.bincount(val_y).tolist()}")
        
        # Create datasets
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.config.device == "cuda" else False
        )
        
        return train_loader, val_loader
    
    def setup_model(self, input_size: int, train_loader: DataLoader):
        """Initialize model, optimizer, and scheduler."""
        
        # Create model
        self.model = BinaryTokenProbe(input_size).to(self.device)
        logger.info(f"Created probe with input size: {input_size}")
        
        # Setup optimizer
        if self.config.optimizer_type == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_type}")
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if self.config.scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps - warmup_steps
            )
        elif self.config.scheduler_type == "linear":
            self.scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
            )
        else:  # constant
            self.scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)
        
        logger.info(f"Setup {self.config.optimizer_type} optimizer and {self.config.scheduler_type} scheduler")
    
    def calculate_class_weights(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """Calculate class weights for handling imbalance."""
        
        if not self.config.handle_class_imbalance:
            return None
        
        # Count classes
        counts = torch.bincount(labels)
        total = len(labels)
        
        # Calculate weights inversely proportional to class frequency
        weights = total / (2.0 * counts)
        
        logger.info(f"Class counts: {counts.tolist()}")
        logger.info(f"Class weights: {weights.tolist()}")
        
        return weights.to(self.device)
    
    def train_epoch(self, train_loader: DataLoader, class_weights: Optional[torch.Tensor]) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        
        # Setup loss function
        if class_weights is not None:
            # Convert to pos_weight for BCEWithLogitsLoss
            pos_weight = class_weights[1] / class_weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", disable=not self.config.verbose)
        
        for batch_idx, (batch_X, batch_y) in enumerate(progress_bar):
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.float().to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                
                self.optimizer.step()
            
            # Update scheduler
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.long().cpu().numpy())
            
            # Log progress
            if batch_idx % self.config.log_every == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{current_lr:.2e}"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        num_batches = 0
        
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(self.device, non_blocking=True)
            batch_y = batch_y.float().to(self.device, non_blocking=True)
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(batch_X)
                    loss = criterion(logits, batch_y)
            else:
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Get predictions and probabilities
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.long().cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        # Calculate AUROC if we have both classes
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0  # Only one class present
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
        }
    
    def train(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Main training loop."""
        
        logger.info("Starting NTML binary token training")
        logger.info(f"Training configuration: {self.config.num_epochs} epochs, batch size {self.config.batch_size}")
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(activations, labels)
        
        # Setup model
        self.setup_model(activations.shape[-1], train_loader)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(labels)
        
        # Training loop
        best_val_f1 = 0.0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, class_weights)
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "auroc": 0.0}
            
            # Update history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1"])
            self.training_history["val_auroc"].append(val_metrics["auroc"])
            self.training_history["learning_rate"].append(train_metrics["learning_rate"])
            
            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = self.model.state_dict().copy()
            
            # Log metrics
            if self.config.verbose:
                logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                if epoch % self.config.eval_every == 0:
                    logger.info(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
                              f"Val F1: {val_metrics['f1']:.4f}, Val AUROC: {val_metrics['auroc']:.4f}")
            
            # Save checkpoint
            if self.config.save_checkpoints and (epoch + 1) % self.config.checkpoint_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model with validation F1: {best_val_f1:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.1f} seconds")
        
        # Final validation
        final_val_metrics = self.validate(val_loader)
        
        return {
            "training_history": self.training_history,
            "final_metrics": final_val_metrics,
            "best_val_f1": best_val_f1,
            "training_time": total_time,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
    
    def save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "config": self.config.to_dict(),
        }, checkpoint_path)
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def save_model(self, output_path: str, metadata: Dict[str, Any]):
        """Save the trained model and metadata."""
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_size": self.model.input_size,
                "model_type": "BinaryTokenProbe",
            },
            "training_config": self.config.to_dict(),
            "metadata": metadata,
        }
        
        torch.save(model_data, output_path)
        logger.info(f"Saved trained probe: {output_path}")
        
        # Save training history as JSON
        history_path = output_path.with_suffix(".json")
        with open(history_path, "w") as f:
            json.dump({
                "training_history": self.training_history,
                "metadata": metadata,
                "config": self.config.to_dict(),
            }, f, indent=2)
        
        logger.info(f"Saved training history: {history_path}")
    
    @classmethod
    def load_model(cls, model_path: str, device: str = "cpu") -> Tuple["BinaryTokenProbe", Dict]:
        """Load a trained model."""
        
        model_data = torch.load(model_path, map_location=device)
        
        # Create model
        model_config = model_data["model_config"]
        model = BinaryTokenProbe(model_config["input_size"])
        model.load_state_dict(model_data["model_state_dict"])
        model.to(device)
        model.eval()
        
        return model, model_data["metadata"]