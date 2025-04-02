from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn
import json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal, Generic, TypeVar, List, Union
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Forward reference for ProbeVector to avoid circular imports
ProbeVector = None

@dataclass
class ProbeConfig:
    """Base configuration for probes with metadata."""
    # Core configuration
    input_size: int
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Metadata fields (from former ProbeVector)
    model_name: str = "unknown_model"
    hook_point: str = "unknown_hook"
    hook_layer: int = 0
    hook_head_index: Optional[int] = None
    name: str = "unnamed_probe"
    
    # Dataset information
    dataset_path: Optional[str] = None
    prepend_bos: bool = True
    context_size: int = 128
    
    # Technical settings
    dtype: str = "float32"
    
    # Additional metadata
    additional_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LinearProbeConfig(ProbeConfig):
    """Configuration for linear probe."""
    loss_type: Literal["mse", "cosine", "l1"] = "mse"
    normalize_weights: bool = True
    bias: bool = False
    output_size: int = 1  # Number of output dimensions

@dataclass
class LogisticProbeConfig(ProbeConfig):
    """Configuration for logistic regression probe."""
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1  # Number of output dimensions

@dataclass
class KMeansProbeConfig(ProbeConfig):
    """Configuration for K-means clustering probe."""
    n_clusters: int = 2
    n_init: int = 10
    normalize_weights: bool = True
    random_state: int = 42

@dataclass
class PCAProbeConfig(ProbeConfig):
    """Configuration for PCA-based probe."""
    n_components: int = 1
    normalize_weights: bool = True

@dataclass
class MeanDiffProbeConfig(ProbeConfig):
    """Configuration for mean difference probe."""
    normalize_weights: bool = True

T = TypeVar('T', bound=ProbeConfig)


class BaseProbe(ABC, nn.Module, Generic[T]):
    """Abstract base class for probes with vector functionality."""
    
    def __init__(self, config: T):
        super().__init__()
        self.config = config
        self.dtype = torch.float32  # Add default dtype
        self.name = config.name or "unnamed_probe"  # Use config name or default
        
    @abstractmethod
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        pass
    
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot product between activations and the probe direction."""
        direction = self.get_direction()
        return torch.einsum("...d,d->...", acts, direction)
    
    def _apply_standardization(self, x: torch.Tensor) -> torch.Tensor:
        """Apply standardization if feature statistics are available.
        
        Returns the input unchanged if feature statistics aren't available.
        """
        if (hasattr(self, 'feature_mean') and 
            isinstance(self.feature_mean, torch.Tensor) and 
            self.feature_mean is not None):
            return (x - self.feature_mean) / self.feature_std
        return x
    
    def save(self, path: str) -> None:
        """Save probe state, config, and direction in a single file."""
        # Gather standardization buffers
        metadata = {
            'is_standardized': hasattr(self, 'feature_mean') and self.feature_mean is not None
        }
        
        if metadata['is_standardized']:
            metadata['feature_mean'] = self.feature_mean.cpu().numpy().tolist()
            metadata['feature_std'] = self.feature_std.cpu().numpy().tolist()
            
        # Add bias if present
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            metadata['has_bias'] = True
            metadata['bias'] = self.linear.bias.data.cpu().numpy().tolist()
        else:
            metadata['has_bias'] = False
            
        # Update config with metadata
        self.config.additional_info = {**self.config.additional_info, **metadata}
        
        # Save full state
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'direction': self.get_direction(normalized=False).cpu(),
            'probe_type': self.__class__.__name__
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BaseProbe':
        """Load probe from saved state.
        
        The loaded probe is automatically set to evaluation mode (probe.eval())
        to prevent recalculating feature statistics during inference.
        """
        if path.endswith(".json"):
            # Use load_json if it's a JSON file
            return cls.load_json(path)
            
        # Load data
        data = torch.load(path, weights_only=False)
        
        # Create probe with the saved config
        probe = cls(data['config'])
        
        # First, restore standardization buffers if they exist in additional_info
        # This needs to happen BEFORE loading the state dict to ensure the buffers exist
        additional_info = getattr(data['config'], 'additional_info', {})
        if additional_info.get('is_standardized', False):
            if 'feature_mean' in additional_info:
                mean_data = torch.tensor(additional_info['feature_mean'])
                probe.register_buffer("feature_mean", mean_data)
                
            if 'feature_std' in additional_info:
                std_data = torch.tensor(additional_info['feature_std'])
                probe.register_buffer("feature_std", std_data)
        
        # Now load state dict if available
        if 'state_dict' in data:
            # Create a new state dict that doesn't include feature_mean and feature_std
            # if they were already set from additional_info
            if hasattr(probe, 'feature_mean') and probe.feature_mean is not None:
                state_dict = {k: v for k, v in data['state_dict'].items() 
                             if not k.endswith('feature_mean') and not k.endswith('feature_std')}
                probe.load_state_dict(state_dict, strict=False)
            else:
                probe.load_state_dict(data['state_dict'])
        # Otherwise set direction directly
        elif 'direction' in data:
            direction = data['direction']
            if hasattr(probe, "linear"):
                with torch.no_grad():
                    probe.linear.weight.data = direction.unsqueeze(0)
            else:
                probe.direction = direction
                
        # Restore bias if it exists and wasn't in state_dict
        if additional_info.get('has_bias', False) and 'bias' in additional_info:
            if hasattr(probe, "linear") and hasattr(probe.linear, "bias"):
                with torch.no_grad():
                    probe.linear.bias.data = torch.tensor(additional_info['bias'])
        
        # Set the probe to evaluation mode
        probe.eval()
        
        return probe

    def save_json(self, path: str) -> None:
        """Save probe direction and metadata as JSON.
        
        The saved direction is always normalized for consistency.
        Standardization information is preserved in the metadata.
        """
        # ensure the path ends in .json
        if not path.endswith(".json"):
            path += ".json"

        # ensure folder exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the raw linear weights, not the direction
        # This ensures compatibility with the .pt format
        if hasattr(self, "linear") and hasattr(self.linear, "weight"):
            raw_weight = self.linear.weight.data.detach().clone().cpu()
            if raw_weight.dim() > 1 and raw_weight.size(0) == 1:
                # Convert the weight from shape [1, dim] to [dim] for consistency
                raw_weight = raw_weight.squeeze(0)
        else:
            # Handle non-linear probes (like KMeans, PCA, etc.)
            raw_weight = self.get_direction(normalized=False).detach().clone().cpu()
        
        # Prepare standardization info for metadata
        metadata = {
            "model_name": self.config.model_name,
            "hook_point": self.config.hook_point,
            "hook_layer": self.config.hook_layer,
            "hook_head_index": self.config.hook_head_index,
            "vector_name": self.name,
            "vector_dimension": raw_weight.shape[0],
            "probe_type": self.__class__.__name__,
            "dataset_path": self.config.dataset_path,
            "prepend_bos": self.config.prepend_bos,
            "context_size": self.config.context_size,
            "dtype": self.config.dtype,
            "device": self.config.device,
        }
        
        # Add standardization info
        metadata['is_standardized'] = hasattr(self, 'feature_mean') and self.feature_mean is not None
        if metadata['is_standardized']:
            metadata['feature_mean'] = self.feature_mean.cpu().numpy().tolist()
            metadata['feature_std'] = self.feature_std.cpu().numpy().tolist()
            
        # Add bias if present
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            metadata['has_bias'] = True
            metadata['bias'] = self.linear.bias.data.cpu().numpy().tolist()
        else:
            metadata['has_bias'] = False
            
        # Save required information for get_direction consistency
        additional_info = getattr(self.config, 'additional_info', {})
        # Mark as not normalized/unscaled to use the raw weights as they are
        metadata['is_normalized'] = False
        metadata['is_unscaled'] = False
        
        # Save as JSON
        save_data = {
            "vector": raw_weight.numpy().tolist(),
            "metadata": metadata
        }
        
        with open(path, "w") as f:
            json.dump(save_data, f)
            
    @classmethod
    def load_json(cls, path: str, config: Optional[T] = None) -> 'BaseProbe':
        """Load probe from JSON file.
        
        Args:
            path: Path to the JSON file
            config: Optional config to use, otherwise will create one from metadata
            
        Returns:
            Loaded probe instance
        """
        with open(path, "r") as f:
            data = json.load(f)
            
        # Extract data
        vector = torch.tensor(data["vector"])
        metadata = data["metadata"]
        
        # Create config if not provided
        if config is None:
            dim = vector.shape[0]
            
            # Create appropriate config based on class
            if cls.__name__ == "LinearProbe":
                from probity.probes.linear_probe import LinearProbeConfig
                config = LinearProbeConfig(input_size=dim)
            elif cls.__name__ == "LogisticProbe":
                from probity.probes.linear_probe import LogisticProbeConfig
                config = LogisticProbeConfig(input_size=dim)
            elif cls.__name__ == "KMeansProbe":
                from probity.probes.linear_probe import KMeansProbeConfig
                config = KMeansProbeConfig(input_size=dim)
            elif cls.__name__ == "PCAProbe":
                from probity.probes.linear_probe import PCAProbeConfig
                config = PCAProbeConfig(input_size=dim)
            elif cls.__name__ == "MeanDifferenceProbe":
                from probity.probes.linear_probe import MeanDiffProbeConfig
                config = MeanDiffProbeConfig(input_size=dim)
            else:
                from probity.probes.linear_probe import ProbeConfig
                config = ProbeConfig(input_size=dim)
            
            # Update config with metadata
            for key in ["model_name", "hook_point", "hook_layer", "hook_head_index", 
                       "name", "dataset_path", "prepend_bos", "context_size",
                       "dtype", "device"]:
                if key in metadata:
                    setattr(config, key, metadata.get(key))
                    
            # Also update additional_info with is_normalized and is_unscaled flags
            if 'is_normalized' in metadata:
                config.additional_info['is_normalized'] = metadata['is_normalized']
            if 'is_unscaled' in metadata:
                config.additional_info['is_unscaled'] = metadata['is_unscaled']
        
        # Create the probe
        probe = cls(config)
        
        # Now set standardization buffers if they exist
        if metadata.get('is_standardized', False):
            if 'feature_mean' in metadata:
                mean_data = torch.tensor(metadata['feature_mean'])
                probe.register_buffer("feature_mean", mean_data)
                
            if 'feature_std' in metadata:
                std_data = torch.tensor(metadata['feature_std'])
                probe.register_buffer("feature_std", std_data)
        
        # Set the vector direction - For linear probes, unsqueeze to [1, dim] if needed
        if hasattr(probe, "linear"):
            with torch.no_grad():
                if vector.dim() == 1:  # [dim] to [1, dim]
                    probe.linear.weight.data = vector.unsqueeze(0)
                else:
                    probe.linear.weight.data = vector
                
                # Restore bias if it exists
                if metadata.get('has_bias', False) and 'bias' in metadata:
                    bias_data = torch.tensor(metadata['bias'])
                    probe.linear.bias.data = bias_data
        else:
            probe.direction = vector
        
        # Set to evaluation mode
        probe.eval()
        
        return probe


class ProbeSet:
    """A collection of probes."""
    
    def __init__(
        self,
        probes: List[BaseProbe],
    ):
        self.probes = probes
        
        # Validate that all probes have compatible dimensions
        dims = [p.get_direction().shape[0] for p in probes]
        if len(set(dims)) > 1:
            raise ValueError(f"All probes must have the same input dimension, got {dims}")
        
        # Extract common metadata for convenience
        if probes:
            self.model_name = probes[0].config.model_name
            self.hook_point = probes[0].config.hook_point
            self.hook_layer = probes[0].config.hook_layer
        
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot products with all probes.
        
        Args:
            acts: Activations to project, shape [..., d_model]
            
        Returns:
            Projected values, shape [..., n_vectors]
        """
        # Stack all vectors into a matrix
        weight_matrix = torch.stack([p.get_direction() for p in self.probes])
        
        # Project all at once
        return torch.einsum("...d,nd->...n", acts, weight_matrix)
    
    def __getitem__(self, idx) -> BaseProbe:
        """Get a probe by index."""
        return self.probes[idx]
    
    def __len__(self) -> int:
        """Get number of probes."""
        return len(self.probes)
        
    def save(self, directory: str) -> None:
        """Save all probes to a directory.
        
        Args:
            directory: Directory to save the probes
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save index file with common metadata
        index = {
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "hook_layer": self.hook_layer,
            "probes": []
        }
        
        # Save each probe
        for i, probe in enumerate(self.probes):
            filename = f"probe_{i}_{probe.name}.pt"
            filepath = os.path.join(directory, filename)
            probe.save(filepath)
            
            # Add to index
            index["probes"].append({
                "name": probe.name,
                "file": filename,
                "probe_type": probe.__class__.__name__
            })
            
        # Save index
        with open(os.path.join(directory, "index.json"), "w") as f:
            json.dump(index, f)
            
    @classmethod
    def load(cls, directory: str) -> "ProbeSet":
        """Load a ProbeSet from a directory.
        
        Args:
            directory: Directory containing the probes
            
        Returns:
            ProbeSet instance
        """
        # Load index
        with open(os.path.join(directory, "index.json")) as f:
            index = json.load(f)
            
        # Load each probe
        probes = []
        for entry in index["probes"]:
            filepath = os.path.join(directory, entry["file"])
            
            # Determine the probe class
            probe_type = entry.get("probe_type", "LinearProbe")
            if probe_type == "LinearProbe":
                from probity.probes.linear_probe import LinearProbe
                probe = LinearProbe.load(filepath)
            elif probe_type == "LogisticProbe":
                from probity.probes.linear_probe import LogisticProbe
                probe = LogisticProbe.load(filepath)
            elif probe_type == "KMeansProbe":
                from probity.probes.linear_probe import KMeansProbe
                probe = KMeansProbe.load(filepath)
            elif probe_type == "PCAProbe":
                from probity.probes.linear_probe import PCAProbe
                probe = PCAProbe.load(filepath)
            elif probe_type == "MeanDifferenceProbe":
                from probity.probes.linear_probe import MeanDifferenceProbe
                probe = MeanDifferenceProbe.load(filepath)
            else:
                # Default to base class with runtime error
                raise ValueError(f"Unknown probe type: {probe_type}")
                
            probes.append(probe)
            
        return cls(probes)


class LinearProbe(BaseProbe[LinearProbeConfig]):
    """Simple linear probe that learns one or more directions in activation space.
    
    This probe implements pure linear projection without any activation functions.
    Different loss functions can be used depending on the task:
    - MSE loss: For general regression tasks
    - Cosine loss: For learning directions that match target vectors
    - L1 loss: For robust regression with less sensitivity to outliers
    """
    
    def __init__(self, config: LinearProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        # Initialize weights 
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        if config.bias:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with standardization.
        
        Uses pre-computed standardization statistics from the trainer.
        """
        # Apply standardization from pre-computed stats
        x = self._apply_standardization(x)
        return self.linear(x)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        direction = self.linear.weight.data.clone()
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        # Only apply unscaling if we have standardization buffers and it wasn't done already
        if not already_unscaled and isinstance(self.feature_std, torch.Tensor):
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        # Normalize if requested and needed
        if normalized and self.config.normalize_weights and not already_normalized:
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                direction = direction / (torch.norm(direction) + 1e-8)
        elif normalized and already_normalized:
            # If the weight is already normalized and normalization is requested,
            # ensure the direction has unit norm
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                if not torch.allclose(norms, torch.ones_like(norms)):
                    direction = direction / (norms + 1e-8)
            else:
                norm = torch.norm(direction)
                if not torch.allclose(norm, torch.tensor(1.0, device=direction.device)):
                    direction = direction / (norm + 1e-8)
                
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction

    def get_loss_fn(self) -> nn.Module:
        """Enhanced loss function selection with better numerical stability."""
        if self.config.loss_type == "mse":
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Center and scale targets to [-1, 1] range for better stability
                target = 2 * target - 1
                mse_loss = nn.MSELoss()(pred, target)
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return mse_loss + l2_reg
                
        elif self.config.loss_type == "hinge":
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Convert targets to {-1, 1}
                target = 2 * target - 1
                # Hinge loss
                hinge_loss = torch.mean(torch.relu(1 - pred * target))
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return hinge_loss + l2_reg
                
        elif self.config.loss_type == "cosine":
            cosine_loss = nn.CosineEmbeddingLoss()
            def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                # Convert targets to {-1, 1}
                target = 2 * target - 1
                loss = cosine_loss(pred, target, target)
                l2_lambda = 0.01
                l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
                return loss + l2_reg
                
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
            
        return loss_fn
        

class LogisticProbe(BaseProbe[LogisticProbeConfig]):
    """Logistic regression probe that learns directions using cross-entropy loss."""
    
    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
        # Initialize weights using sklearn-like initialization
        nn.init.zeros_(self.linear.weight)
        if config.bias:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with standardization.
        
        Uses pre-computed standardization statistics from the trainer.
        """
        # Apply standardization from pre-computed stats
        x = self._apply_standardization(x)
        return self.linear(x)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        direction = self.linear.weight.data.clone()
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        # Unscale only if we have standardization buffers and it wasn't done already
        if not already_unscaled and isinstance(self.feature_std, torch.Tensor):
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        # Normalize if requested and needed
        if normalized and self.config.normalize_weights and not already_normalized:
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                direction = direction / (torch.norm(direction) + 1e-8)
        elif normalized and already_normalized:
            # If the weight is already normalized and normalization is requested,
            # ensure the direction has unit norm
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                if not torch.allclose(norms, torch.ones_like(norms)):
                    direction = direction / (norms + 1e-8)
            else:
                norm = torch.norm(direction)
                if not torch.allclose(norm, torch.tensor(1.0, device=direction.device)):
                    direction = direction / (norm + 1e-8)
                
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction
    
    def get_loss_fn(self) -> nn.Module:
        """Get binary cross entropy loss with L2 regularization."""
        def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            # Binary cross entropy loss
            bce_loss = nn.BCEWithLogitsLoss()(pred, target)
            # L2 regularization (matching sklearn's default C=1.0)
            l2_lambda = 0.01  # C=1.0 in sklearn corresponds to reg_lambda=0.01
            l2_reg = l2_lambda * torch.norm(self.linear.weight)**2
            return bce_loss + l2_reg
            
        return loss_fn


class KMeansProbe(BaseProbe[KMeansProbeConfig]):
    """K-means clustering based probe that finds directions through centroids."""
    
    def __init__(self, config: KMeansProbeConfig):
        super().__init__(config)
        self.kmeans = KMeans(
            n_clusters=config.n_clusters,
            n_init=config.n_init,
            random_state=config.random_state
        )
        self.direction: Optional[torch.Tensor] = None
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit K-means and compute direction from centroids."""
        # Convert to numpy for sklearn
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Fit K-means
        self.kmeans.fit(x_np)
        centroids = self.kmeans.cluster_centers_
        
        # Determine positive and negative centroids based on cluster assignments
        labels = self.kmeans.labels_
        cluster_labels = np.zeros(self.config.n_clusters)
        for i in range(self.config.n_clusters):
            mask = labels == i
            if mask.any():
                cluster_labels[i] = np.mean(y_np[mask])
        
        pos_centroid = centroids[np.argmax(cluster_labels)]
        neg_centroid = centroids[np.argmin(cluster_labels)]
        
        # Direction is from negative to positive centroid
        direction = torch.tensor(
            pos_centroid - neg_centroid, 
            device=self.config.device,
            dtype=self.dtype
        )
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
        
        # Apply standardization from pre-computed stats if available
        x = self._apply_standardization(x)
        
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
            
        direction = self.direction.clone()
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        # Unscale only if we have standardization buffers and it wasn't done already
        if not already_unscaled and isinstance(self.feature_std, torch.Tensor):
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        # Normalize if requested and needed
        if normalized and self.config.normalize_weights and not already_normalized:
            direction = direction / (torch.norm(direction) + 1e-8)
        elif normalized and already_normalized:
            # If the weight is already normalized and normalization is requested,
            # ensure the direction has unit norm
            norm = torch.norm(direction)
            if not torch.allclose(norm, torch.tensor(1.0, device=direction.device)):
                direction = direction / (norm + 1e-8)
                
        return direction


class PCAProbe(BaseProbe[PCAProbeConfig]):
    """PCA-based probe that finds directions through principal components."""
    
    def __init__(self, config: PCAProbeConfig):
        super().__init__(config)
        self.pca = PCA(n_components=config.n_components)
        self.direction: Optional[torch.Tensor] = None
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit PCA and determine direction sign based on correlation with labels."""
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Fit PCA
        self.pca.fit(x_np)
        components = self.pca.components_
        
        # Project data onto components
        projections = np.dot(x_np, components.T)
        
        # Determine sign based on correlation with labels
        correlations = np.array([np.corrcoef(proj, y_np)[0,1] for proj in projections.T])
        signs = np.sign(correlations)
        
        # Apply signs to components
        components = components * signs[:, np.newaxis]
        
        # Get primary direction (first component)
        direction = torch.tensor(
            components[0], 
            device=self.config.device,
            dtype=self.dtype
        )
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
            
        # Apply standardization from pre-computed stats if available
        x = self._apply_standardization(x)
            
        # Ensure consistent dtype
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
            
        direction = self.direction.clone()
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        # Unscale only if we have standardization buffers and it wasn't done already
        if not already_unscaled and isinstance(self.feature_std, torch.Tensor):
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        # Normalize if requested and needed
        if normalized and self.config.normalize_weights and not already_normalized:
            direction = direction / (torch.norm(direction) + 1e-8)
        elif normalized and already_normalized:
            # If the weight is already normalized and normalization is requested,
            # ensure the direction has unit norm
            norm = torch.norm(direction)
            if not torch.allclose(norm, torch.tensor(1.0, device=direction.device)):
                direction = direction / (norm + 1e-8)
                
        return direction


class MeanDifferenceProbe(BaseProbe[MeanDiffProbeConfig]):
    """Probe that finds direction through mean difference between classes."""
    
    def __init__(self, config: MeanDiffProbeConfig):
        super().__init__(config)
        self.direction: Optional[torch.Tensor] = None
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Compute direction as difference between class means."""
        # Ensure input tensors are float32
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype)
        
        # Calculate means for positive and negative classes
        pos_mask = y == 1
        neg_mask = y == 0
        
        pos_mean = x[pos_mask].mean(dim=0)
        neg_mean = x[neg_mask].mean(dim=0)
        
        # Direction from negative to positive
        direction = pos_mean - neg_mean
        
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        self.direction = direction
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        if self.direction is None:
            raise RuntimeError("Must call fit() before forward()")
            
        # Apply standardization from pre-computed stats if available
        x = self._apply_standardization(x)
            
        x = x.to(dtype=torch.float32)
        self.direction = self.direction.to(dtype=torch.float32)
        return torch.matmul(x, self.direction)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction with proper rescaling.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        if self.direction is None:
            raise RuntimeError("Must call fit() before get_direction()")
            
        direction = self.direction.clone()
        
        # Check if we need to apply unscaling (not needed if already unscaled)
        additional_info = getattr(self.config, 'additional_info', {})
        already_unscaled = additional_info.get('is_unscaled', False)
        already_normalized = additional_info.get('is_normalized', False)
        
        # Unscale only if we have standardization buffers and it wasn't done already
        if not already_unscaled and isinstance(self.feature_std, torch.Tensor):
            # Unscale the coefficients to match standardized training
            direction = direction / self.feature_std.squeeze()
            
        # Normalize if requested and needed
        if normalized and self.config.normalize_weights and not already_normalized:
            direction = direction / (torch.norm(direction) + 1e-8)
        elif normalized and already_normalized:
            # If the weight is already normalized and normalization is requested,
            # ensure the direction has unit norm
            norm = torch.norm(direction)
            if not torch.allclose(norm, torch.tensor(1.0, device=direction.device)):
                direction = direction / (norm + 1e-8)
                
        return direction
    
# alternative implementations of logistic probe for testing
@dataclass
class LogisticProbeConfigBase(ProbeConfig):
    """Base config shared by both implementations."""
    standardize: bool = True
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1

@dataclass
class SklearnLogisticProbeConfig(LogisticProbeConfigBase):
    """Config for sklearn-based probe."""
    max_iter: int = 100
    random_state: int = 42
    

class SklearnLogisticProbe(BaseProbe[SklearnLogisticProbeConfig]):
    """Logistic regression probe using scikit-learn, matching paper implementation."""
    
    def __init__(self, config: SklearnLogisticProbeConfig):
        super().__init__(config)
        self.scaler = StandardScaler() if config.standardize else None
        self.model = LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
            fit_intercept=config.bias
        )
        
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the probe using sklearn's LogisticRegression."""
        # Convert to numpy
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Standardize if requested
        if self.scaler is not None:
            x_np = self.scaler.fit_transform(x_np)
            
        # Fit logistic regression
        self.model.fit(x_np, y_np)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input onto the learned direction."""
        x_np = x.cpu().numpy()
        if self.scaler is not None:
            x_np = self.scaler.transform(x_np)
        
        # Get logits
        logits = self.model.decision_function(x_np)
        return torch.tensor(logits, device=x.device)

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction, matching paper's implementation.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length
                      (this won't modify stored weights, just the returned vector)
        
        Returns:
            The (optionally normalized) direction vector
        """
        # Get coefficients and intercept
        coef = self.model.coef_[0]  # Shape: (input_size,)
        
        if self.scaler is not None:
            # Unscale the coefficients as done in paper
            coef = coef / self.scaler.scale_
            
        # Convert to tensor and normalize if requested
        direction = torch.tensor(coef, device=self.config.device)
        if self.config.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        if normalized:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        return direction
