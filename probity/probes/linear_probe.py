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
class MultiClassLogisticProbeConfig(ProbeConfig):
    """Configuration for multi-class logistic regression probe."""
    output_size: int = 2  # Must be specified, > 1
    normalize_weights: bool = True
    bias: bool = True

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
    """Abstract base class for probes. Probes store directions in the original activation space."""
    
    def __init__(self, config: T):
        super().__init__()
        self.config = config
        self.dtype = torch.float32 if config.dtype == "float32" else torch.float16
        self.name = config.name or "unnamed_probe"  # Use config name or default
        # Standardization is handled by the trainer, not stored in the probe.

    @abstractmethod
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction in the original activation space.
        
        Args:
            normalized: Whether to normalize the direction vector to unit length.
                      The probe's internal configuration (`normalize_weights`)
                      also influences this. Normalization occurs only if
                      `normalized` is True AND `config.normalize_weights` is True.
        
        Returns:
            The processed (optionally normalized) direction vector
            representing the probe in the original activation space.
        """
        pass

    @abstractmethod
    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw internal representation (weights/vector) before normalization."""
        pass

    @abstractmethod
    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw internal representation (weights/vector) from a (potentially adjusted) vector."""
        pass
    
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot product between activations and the probe direction."""
        # Ensure direction is normalized for consistent projection magnitude
        direction = self.get_direction(normalized=True)
        return torch.einsum("...d,d->...", acts, direction)

    def save(self, path: str) -> None:
        """Save probe state and config in a single .pt file."""
        save_dir = os.path.dirname(path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # Clear previous standardization info if present in older saves
        self.config.additional_info.pop('is_standardized', None)
        self.config.additional_info.pop('feature_mean', None)
        self.config.additional_info.pop('feature_std', None)

        # Save bias info if relevant (e.g., for LinearProbe, LogisticProbe)
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            self.config.additional_info['has_bias'] = True
            # Bias is saved in state_dict
        else:
             self.config.additional_info['has_bias'] = False

        # Ensure config reflects runtime normalization choice if needed for reconstruction
        if hasattr(self.config, 'normalize_weights'):
            self.config.additional_info['normalize_weights'] = self.config.normalize_weights
        if hasattr(self.config, 'bias'):
            self.config.additional_info['bias'] = self.config.bias

        # Save full state
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
            'probe_type': self.__class__.__name__
        }, path)

    @classmethod
    def load(cls, path: str) -> 'BaseProbe':
        """Load probe from saved state (.pt file)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved probe file found at {path}")
            
        if path.endswith(".json"):
            # Delegate to load_json if it's a JSON file
            return cls.load_json(path)
            
        # Load data for .pt file
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = torch.load(path, map_location=map_location, weights_only=False)
        
        saved_config = data['config']
        
        # Create probe instance using the loaded config
        probe = cls(saved_config)
        
        # Load the state dict
        probe.load_state_dict(data['state_dict'])
        
        # Standardization buffers are no longer loaded here

        # Move probe to the device specified in its config
        probe.to(probe.config.device)
        
        # Set the probe to evaluation mode
        probe.eval()
        
        return probe

    def save_json(self, path: str) -> None:
        """Save probe's internal direction and metadata as JSON."""
        if not path.endswith(".json"):
            path += ".json"

        save_dir = os.path.dirname(path)
        if save_dir:
             os.makedirs(save_dir, exist_ok=True)
        
        # Get the internal representation (now always in original activation space)
        vector = self._get_raw_direction_representation().detach().clone().cpu()
        
        # Prepare metadata
        metadata = {
            "model_name": self.config.model_name,
            "hook_point": self.config.hook_point,
            "hook_layer": self.config.hook_layer,
            "hook_head_index": self.config.hook_head_index,
            "name": self.name,
            "vector_dimension": vector.shape[-1], # Use last dim for robustness
            "probe_type": self.__class__.__name__,
            "dataset_path": self.config.dataset_path,
            "prepend_bos": self.config.prepend_bos,
            "context_size": self.config.context_size,
            "dtype": self.config.dtype,
            "device": self.config.device, # Device used during training/creation
            # Standardization info is no longer saved
        }
        
        # Add bias info if relevant
        if hasattr(self, "linear") and hasattr(self.linear, "bias") and self.linear.bias is not None:
            metadata['has_bias'] = True
            metadata['bias'] = self.linear.bias.data.detach().clone().cpu().numpy().tolist()
        else:
            metadata['has_bias'] = False
            
        # Save relevant config flags needed for reconstruction
        if hasattr(self.config, 'normalize_weights'):
             metadata['normalize_weights'] = self.config.normalize_weights
        if hasattr(self.config, 'bias'): # For linear/logistic
             metadata['bias_config'] = self.config.bias
        # Add other necessary config flags for specific probe types if needed

        # Save as JSON
        save_data = {
            "vector": vector.numpy().tolist(), # Renamed from raw_vector
            "metadata": metadata
        }
        
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)
            
    @classmethod
    def load_json(cls, path: str, device: Optional[str] = None) -> 'BaseProbe':
        """Load probe from JSON file.
        
        Args:
            path: Path to the JSON file
            device: Optional device override. If None, uses device from metadata or default.
            
        Returns:
            Loaded probe instance
        """
        if not os.path.exists(path):
             raise FileNotFoundError(f"No saved probe JSON file found at {path}")
             
        with open(path, "r") as f:
            data = json.load(f)
            
        # Extract data
        vector = torch.tensor(data["vector"]) # Renamed from raw_vector
        metadata = data["metadata"]
        
        # Determine device
        target_device = device or metadata.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # Create appropriate config
        dim = metadata["vector_dimension"]
        probe_type = metadata.get("probe_type", cls.__name__) # Use metadata type, fallback to cls
        
        # Dynamically get config class based on probe_type string
        config_cls_name = f"{probe_type}Config"
        try:
            # Find config class within the current module's scope
            config_cls = globals()[config_cls_name]
        except KeyError:
            # Fallback or error if config class not found
            print(f"Warning: Config class {config_cls_name} not found. Using base ProbeConfig.")
            config_cls = ProbeConfig

        # Instantiate config
        config = config_cls(input_size=dim)

        # Update config with metadata fields
        common_fields = [
            "model_name", "hook_point", "hook_layer", "hook_head_index", "name",
            "dataset_path", "prepend_bos", "context_size", "dtype", "device"
        ]
        for key in common_fields:
            if key in metadata:
                setattr(config, key, metadata.get(key))
        # Override device if explicitly provided
        config.device = target_device

        # Update probe-specific config fields from metadata
        if hasattr(config, 'normalize_weights') and 'normalize_weights' in metadata:
            config.normalize_weights = metadata['normalize_weights']
        if hasattr(config, 'bias') and 'bias_config' in metadata:
            config.bias = metadata['bias_config']
        # Add other specific fields as needed (e.g., n_clusters for KMeans)
        if probe_type == "KMeansProbe" and hasattr(config, 'n_clusters') and 'n_clusters' in metadata:
             config.n_clusters = metadata.get('n_clusters', 2) # Example: Get n_clusters if saved
        
        # Create the probe instance with the configured settings
        probe = cls(config)
        probe.to(target_device) # Move to target device early
        
        # Standardization metadata is no longer loaded

        # Set the vector representation
        probe._set_raw_direction_representation(vector.to(target_device))
                
        # Restore bias if it exists (for Linear/Logistic)
        if metadata.get('has_bias', False) and 'bias' in metadata:
             if hasattr(probe, "linear") and hasattr(probe.linear, "bias"):
                 with torch.no_grad():
                     bias_data = torch.tensor(metadata['bias'], dtype=probe.dtype).to(target_device)
                     # Ensure bias parameter exists before assigning
                     if probe.linear.bias is not None:
                          probe.linear.bias.copy_(bias_data)
                     else:
                          # This case shouldn't happen if config.bias was True, but handle defensively
                          print("Warning: Bias metadata found, but probe.linear.bias is None.")
        
        # Set to evaluation mode
        probe.eval()
        
        return probe
    

class LinearProbe(BaseProbe[LinearProbeConfig]):
    """Simple linear probe for regression or finding directions.
    
    Learns a linear transformation Wx + b. The direction is derived from W.
    Operates on original activation space.
    """
    
    def __init__(self, config: LinearProbeConfig):
        super().__init__(config)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        
        # Initialize weights (optional, can use default PyTorch init)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='linear')
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input x is expected in the original activation space."""
        # Standardization is handled by the trainer externally if needed
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights."""
        if self.linear.weight.shape != vector.shape:
             # Reshape if necessary (e.g., loaded [dim] but need [1, dim])
             if self.linear.weight.dim() == 2 and self.linear.weight.shape[0] == 1 and vector.dim() == 1:
                 vector = vector.unsqueeze(0)
             elif self.linear.weight.dim() == 1 and vector.dim() == 2 and vector.shape[0] == 1:
                 vector = vector.squeeze(0)
             else:
                  raise ValueError(f"Shape mismatch loading vector. Probe weight: {self.linear.weight.shape}, Loaded vector: {vector.shape}")
        with torch.no_grad():
             self.linear.weight.copy_(vector)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction, applying normalization."""
        # Start with raw weights (already in original activation space)
        direction = self._get_raw_direction_representation().clone()
        
        # Unscaling is no longer needed here

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            if self.config.output_size > 1:
                # Normalize each output direction independently
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                # Normalize the single direction vector
                norm = torch.norm(direction)
                direction = direction / (norm + 1e-8)
                
        # Squeeze if single output dimension for convenience
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction

    # get_loss_fn remains specific to LinearProbe, not moved to base
    def get_loss_fn(self) -> nn.Module:
        """Selects loss function based on config."""
        if self.config.loss_type == "mse":
            return nn.MSELoss()
        elif self.config.loss_type == "cosine":
            # CosineEmbeddingLoss expects targets y = 1 or -1
            # Input: (x1, x2, y) -> computes loss based on y * cos(x1, x2)
            # Here, pred is x1, target direction (implicit) is x2, label is y
            # We might need a wrapper if target vectors aren't directly available
             print("Warning: Cosine loss in LinearProbe assumes target vectors are handled externally.")
             return nn.CosineEmbeddingLoss()
        elif self.config.loss_type == "l1":
            return nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        

class LogisticProbe(BaseProbe[LogisticProbeConfig]):
    """Logistic regression probe implemented using nn.Linear. Operates on original activation space."""
    
    def __init__(self, config: LogisticProbeConfig):
        super().__init__(config)
        # Logistic regression is essentially a linear layer followed by sigmoid (handled by loss)
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)
        
        # Initialize weights (zeros often work well for logistic init)
        nn.init.zeros_(self.linear.weight)
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits. Input x is expected in the original activation space."""
        # Standardization is handled by the trainer externally if needed
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights."""
        if self.linear.weight.shape != vector.shape:
            if self.linear.weight.dim() == 2 and self.linear.weight.shape[0] == 1 and vector.dim() == 1:
                 vector = vector.unsqueeze(0)
            elif self.linear.weight.dim() == 1 and vector.dim() == 2 and vector.shape[0] == 1:
                 vector = vector.squeeze(0)
            else:
                  raise ValueError(f"Shape mismatch loading vector. Probe weight: {self.linear.weight.shape}, Loaded vector: {vector.shape}")
        with torch.no_grad():
             self.linear.weight.copy_(vector)
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction, applying normalization."""
        # Start with raw weights (already in original activation space)
        direction = self._get_raw_direction_representation().clone()
        
        # Unscaling is no longer needed here

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            if self.config.output_size > 1:
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                norm = torch.norm(direction)
                direction = direction / (norm + 1e-8)
                
        # Squeeze if single output dimension
        if self.config.output_size == 1:
            direction = direction.squeeze(0)
            
        return direction
    
    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        """Get binary cross entropy loss with logits.
        
        Args:
            pos_weight: Optional weight for positive class (for class imbalance).
        """
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


class MultiClassLogisticProbe(BaseProbe[MultiClassLogisticProbeConfig]):
    """Multi-class logistic regression probe (Softmax Regression)."""

    def __init__(self, config: MultiClassLogisticProbeConfig):
        super().__init__(config)
        if config.output_size <= 1:
            raise ValueError("MultiClassLogisticProbe requires output_size > 1.")
        self.linear = nn.Linear(config.input_size, config.output_size, bias=config.bias)

        # Initialize weights
        nn.init.zeros_(self.linear.weight)
        if config.bias and self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits for each class."""
        return self.linear(x)

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the raw linear layer weights (weight matrix)."""
        return self.linear.weight.data

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the raw linear layer weights (weight matrix)."""
        if self.linear.weight.shape != vector.shape:
            raise ValueError(
                f"Shape mismatch loading vector. Probe weight: "
                f"{self.linear.weight.shape}, Loaded vector: {vector.shape}"
            )
        with torch.no_grad():
            self.linear.weight.copy_(vector)

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe directions (weight matrix), applying normalization per class."""
        # Start with raw weights
        directions = self._get_raw_direction_representation().clone()

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            # Normalize each class direction (row) independently
            norms = torch.norm(directions, dim=1, keepdim=True)
            directions = directions / (norms + 1e-8)

        return directions # Shape [output_size, input_size]

    def get_loss_fn(self, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """Get cross entropy loss.

        Args:
            class_weights: Optional weights for each class (for class imbalance).
                           Shape [output_size].
        """
        return nn.CrossEntropyLoss(weight=class_weights)


# --- Directional Probes (KMeans, PCA, MeanDiff) ---
# These probes compute their direction directly rather than through gradient descent.

class DirectionalProbe(BaseProbe[T]):
     """Base class for probes computing direction directly (KMeans, PCA, MeanDiff).
     Stores the final direction in the original activation space.
     """

     def __init__(self, config: T):
         super().__init__(config)
         # Stores the final direction (in original activation space)
         self.register_buffer('direction_vector', None, persistent=True)

     @abstractmethod
     def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
         """Fit the probe (e.g., run KMeans/PCA) and compute the initial direction.
         The input x may be standardized by the trainer.

         Returns:
             The computed direction tensor *before* potential unscaling by the trainer.
         """
         pass

     def _get_raw_direction_representation(self) -> torch.Tensor:
         """Return the computed final internal direction (in original activation space)."""
         if self.direction_vector is None:
             # Changed from RuntimeError to returning a dummy tensor or raising specific error
             # if direction hasn't been set yet (e.g., after init but before training)
             # For now, assume it will be set by the trainer after fit.
             raise AttributeError("Direction vector has not been computed or set.")
         return self.direction_vector

     def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
         """Set the final internal direction (in original activation space)."""
         if self.direction_vector is not None and self.direction_vector.shape != vector.shape:
             raise ValueError(f"Shape mismatch loading vector. Probe direction: {self.direction_vector.shape}, Loaded vector: {vector.shape}")
         # Use setattr to assign to the buffer
         setattr(self, 'direction_vector', vector)

     def forward(self, x: torch.Tensor) -> torch.Tensor:
         """Project input onto the final (normalized) direction. Input x is in original activation space."""
         # Standardization is handled by the trainer externally if needed

         # Get the final interpretable direction (normalized by default)
         direction = self.get_direction(normalized=True)

         # Ensure consistent dtypes for matmul
         x = x.to(dtype=self.dtype)
         direction = direction.to(dtype=self.dtype)

         # Project onto the direction
         # Handle potential dimension mismatch (e.g., x: [B, D], direction: [D])
         if direction.dim() == 1 and x.dim() >= 2:
             return torch.matmul(x, direction)
         elif direction.dim() == x.dim(): # Should not happen for typical probes
             return torch.matmul(x, direction) # Or adjust einsum if needed
         else:
             # Fallback or error for unexpected shapes
             # Assuming standard case: project batch onto single vector
             return torch.matmul(x, direction)


     def get_direction(self, normalized: bool = True) -> torch.Tensor:
         """Get the computed direction (already in original activation space), applying normalization."""
         direction = self._get_raw_direction_representation().clone()

         # Unscaling is no longer needed here

         # Normalize if requested and configured
         should_normalize = normalized and self.config.normalize_weights
         if should_normalize:
             norm = torch.norm(direction)
             direction = direction / (norm + 1e-8)

         return direction


class KMeansProbe(DirectionalProbe[KMeansProbeConfig]):
    """K-means clustering based probe."""
    
    def __init__(self, config: KMeansProbeConfig):
        super().__init__(config)
        # Sklearn model stored internally, not part of state_dict
        self.kmeans_model: Optional[KMeans] = None
        
    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fit K-means and compute direction from centroids.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        if y is None:
             raise ValueError("KMeansProbe requires labels (y) to determine centroid polarity.")

        # K-means expects float32
        x_np = x.cpu().numpy().astype(np.float32)
        y_np = y.cpu().numpy()

        self.kmeans_model = KMeans(
            n_clusters=self.config.n_clusters,
            n_init=self.config.n_init,
            random_state=self.config.random_state,
            init='k-means++' # Specify init strategy
        )

        # Fit K-means
        cluster_assignments = self.kmeans_model.fit_predict(x_np)
        centroids = self.kmeans_model.cluster_centers_ # Shape: [n_clusters, dim]

        # Determine positive and negative centroids based on label correlation
        # Ensure y_np is 1D
        if y_np.ndim > 1:
             y_np = y_np.squeeze()

        # Handle case where a cluster might be empty (highly unlikely with k-means++)
        cluster_labels_mean = np.zeros(self.config.n_clusters)
        for i in range(self.config.n_clusters):
            mask = cluster_assignments == i
            if np.any(mask):
                # Calculate mean label only for assigned points
                 cluster_labels_mean[i] = np.mean(y_np[mask])
            else:
                 # Handle empty cluster - assign neutral value or based on nearest centroid?
                 # Assigning NaN to avoid influencing argmax/argmin, or handle later
                 cluster_labels_mean[i] = np.nan # Or handle differently if needed


        # Check for NaN means (empty clusters) before argmax/argmin
        if np.isnan(cluster_labels_mean).any():
             print("Warning: One or more K-means clusters were empty.")
             # Fallback logic might be needed here if empty clusters are possible/problematic
             # For now, proceed assuming at least two non-empty clusters for diff

        # Find centroids most correlated with positive (1) and negative (0) labels
        # Use nanargmax/nanargmin to handle potential empty clusters
        pos_centroid_idx = np.nanargmax(cluster_labels_mean)
        neg_centroid_idx = np.nanargmin(cluster_labels_mean)

        # Check if the same cluster was chosen for both (e.g., only one cluster or all means equal)
        if pos_centroid_idx == neg_centroid_idx:
             print("Warning: Could not distinguish positive/negative K-means centroids based on labels.")
             # Fallback: Use first two centroids? Or raise error?
             if self.config.n_clusters >= 2:
                  pos_centroid_idx, neg_centroid_idx = 0, 1
             else:
                  raise ValueError("Cannot compute difference with only one K-means cluster.")

        pos_centroid = centroids[pos_centroid_idx]
        neg_centroid = centroids[neg_centroid_idx]

        # Direction is from negative to positive centroid
        # This initial direction is potentially in the standardized space
        initial_direction_np = pos_centroid - neg_centroid

        # Return the initial direction tensor (trainer will handle unscaling and setting)
        initial_direction_tensor = torch.tensor(
            initial_direction_np,
            device=self.config.device,
            dtype=self.dtype
        )
        # Do NOT set the buffer here; trainer does it after unscaling
        # setattr(self, 'direction_vector', initial_direction_tensor) # REMOVED
        return initial_direction_tensor


class PCAProbe(DirectionalProbe[PCAProbeConfig]):
    """PCA-based probe."""
    
    def __init__(self, config: PCAProbeConfig):
        super().__init__(config)
        self.pca_model: Optional[PCA] = None # Store sklearn model if needed later
        
    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Fit PCA and determine direction sign based on correlation with labels.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        # PCA works best with float32 or float64
        x_np = x.cpu().numpy().astype(np.float32)
        
        self.pca_model = PCA(n_components=self.config.n_components)
        
        # Fit PCA
        self.pca_model.fit(x_np)
        # Components are rows in pca_model.components_
        # Shape: [n_components, dim]
        components = self.pca_model.components_ 
        
        # Get the first principal component
        pc1 = components[0] # Shape: [dim]

        # Determine sign based on correlation with labels if provided
        if y is not None:
            y_np = y.cpu().numpy()
            if y_np.ndim > 1:
                y_np = y_np.squeeze()
                
            # Project potentially standardized data onto the first component
            projections = np.dot(x_np, pc1.T) # Shape: [batch]
            
            # Calculate correlation between projections and labels
            # Ensure labels are numeric for correlation
            correlation = np.corrcoef(projections, y_np.astype(float))[0, 1]
            
            # Flip component sign if correlation is negative
            sign = np.sign(correlation) if not np.isnan(correlation) else 1.0
            pc1 = pc1 * sign
        
        # Return the initial direction (first PC, potentially sign-corrected)
        initial_direction_tensor = torch.tensor(
            pc1,
            device=self.config.device,
            dtype=self.dtype
        )
        # setattr(self, 'direction_vector', initial_direction_tensor) # REMOVED
        return initial_direction_tensor


class MeanDifferenceProbe(DirectionalProbe[MeanDiffProbeConfig]):
    """Probe finding direction through mean difference between classes."""
    
    def fit(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute direction as difference between class means.
        Input x may be standardized by the trainer.
        Returns the computed direction tensor *before* potential unscaling.
        """
        if y is None:
             raise ValueError("MeanDifferenceProbe requires labels (y).")
             
        # Ensure consistent dtypes
        x = x.to(dtype=self.dtype)
        y = y.to(dtype=self.dtype) # Labels might be float or long, ensure consistency
        
        # Calculate means for positive (1) and negative (0) classes
        # Ensure y is boolean or integer {0, 1} for masking
        pos_mask = (y == 1).squeeze()
        neg_mask = (y == 0).squeeze()
        
        if not torch.any(pos_mask) or not torch.any(neg_mask):
             raise ValueError("MeanDifferenceProbe requires data for both classes (0 and 1).")
             
        pos_mean = x[pos_mask].mean(dim=0)
        neg_mean = x[neg_mask].mean(dim=0)
        
        # Direction from negative to positive mean
        # This initial direction is potentially in the standardized space
        initial_direction_tensor = pos_mean - neg_mean
        
        # Return the initial direction
        # setattr(self, 'direction_vector', initial_direction_tensor) # REMOVED
        return initial_direction_tensor


# SklearnLogisticProbe remains separate for now.
# Its internal standardization is independent of the trainer's standardization option.

@dataclass
class LogisticProbeConfigBase(ProbeConfig):
    """Base config shared by sklearn implementations."""
    standardize: bool = True # Internal standardization for sklearn
    normalize_weights: bool = True
    bias: bool = True
    output_size: int = 1 # Usually 1 for logistic

@dataclass
class SklearnLogisticProbeConfig(LogisticProbeConfigBase):
    """Config for sklearn-based probe."""
    max_iter: int = 100
    random_state: int = 42
    solver: str = 'lbfgs' # Example of adding solver


class SklearnLogisticProbe(BaseProbe[SklearnLogisticProbeConfig]):
    """Logistic regression probe using scikit-learn. Handles its own standardization internally."""
    
    def __init__(self, config: SklearnLogisticProbeConfig):
        super().__init__(config)
        # Store scaler and model internally
        self.scaler: Optional[StandardScaler] = StandardScaler() if config.standardize else None
        self.model: LogisticRegression = LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
            fit_intercept=config.bias,
            solver=config.solver
        )
        # Store the final, unscaled coefficients and intercept as tensors
        self.register_buffer('unscaled_coef_', None, persistent=True)
        self.register_buffer('intercept_', None, persistent=True)
        # Removed raw_coef_, raw_intercept_ which were numpy arrays

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Fit the probe using sklearn's LogisticRegression.
        Stores unscaled coefficients internally.
        Input x is expected to be in the original activation space for this fit method.
        """
        x_np = x.cpu().numpy().astype(np.float32)
        y_np = y.cpu().numpy()
        if y_np.ndim > 1:
             y_np = y_np.squeeze() # Ensure y is 1D

        # Apply internal standardization if requested
        if self.scaler is not None:
            x_np_scaled = self.scaler.fit_transform(x_np)
        else:
            x_np_scaled = x_np

        # Fit logistic regression on potentially scaled data
        self.model.fit(x_np_scaled, y_np)

        # Get coefficients (potentially scaled) and intercept
        coef_ = self.model.coef_.squeeze() # Shape [dim]
        intercept_ = self.model.intercept_ # Shape [1] or [n_classes]

        # Unscale coefficients if internal standardization was used
        if self.scaler is not None:
            coef_unscaled = coef_ / (self.scaler.scale_ + 1e-8) # Add epsilon
        else:
            coef_unscaled = coef_

        # Store unscaled coefficients and intercept as tensors (buffers)
        setattr(self, 'unscaled_coef_', torch.tensor(coef_unscaled, dtype=self.dtype))
        setattr(self, 'intercept_', torch.tensor(intercept_, dtype=self.dtype))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits using the learned coefficients. Input x is in original activation space."""
        if self.unscaled_coef_ is None:
            raise RuntimeError("SklearnLogisticProbe must be fitted before calling forward.")

        # Forward pass uses the stored unscaled coefficients and intercept
        x = x.to(dtype=self.dtype)
        coef = self.unscaled_coef_.to(dtype=self.dtype)
        intercept = self.intercept_.to(dtype=self.dtype) if self.intercept_ is not None else None

        # Calculate logits: (x @ coef^T) + intercept
        logits = torch.matmul(x, coef)
        if intercept is not None:
            logits += intercept

        return logits


    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the stored unscaled coefficients."""
        if self.unscaled_coef_ is None:
            raise RuntimeError("SklearnLogisticProbe must be fitted first.")
        return self.unscaled_coef_

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the unscaled coefficients. Used primarily for loading."""
        # vector should already be unscaled when loading
        if self.unscaled_coef_ is not None and self.unscaled_coef_.shape != vector.shape:
            # Reshape if necessary (e.g. [1, dim] vs [dim])
            if self.unscaled_coef_.dim() == 1 and vector.dim() == 2 and vector.shape[0] == 1:
                vector = vector.squeeze(0)
            elif self.unscaled_coef_.dim() == 2 and self.unscaled_coef_.shape[0] == 1 and vector.dim() == 1:
                vector = vector.unsqueeze(0) # Should match LogisticRegression coef_ shape [1, dim] usually
            else:
                raise ValueError(f"Shape mismatch loading vector. Probe coef: {self.unscaled_coef_.shape}, Loaded vector: {vector.shape}")

        setattr(self, 'unscaled_coef_', vector)
        # Note: Intercept loading needs to be handled separately if saving/loading JSON directly
        # For .pt saves, intercept buffer is handled by state_dict


    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the probe direction (unscaled coefficients), applying normalization."""
        if self.unscaled_coef_ is None:
            raise RuntimeError("SklearnLogisticProbe must be fitted first.")

        # Direction is already unscaled
        direction = self._get_raw_direction_representation().clone()

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            norm = torch.norm(direction)
            direction = direction / (norm + 1e-8)

        return direction


class ProbeSet:
    """A collection of probes."""
    
    def __init__(
        self,
        probes: List[BaseProbe],
    ):
        self.probes = probes
        
        # Validate compatibility (optional, could check more than dim)
        if probes:
            first_probe = probes[0]
            self.input_dim = first_probe.config.input_size
            self.model_name = first_probe.config.model_name
            self.hook_point = first_probe.config.hook_point
            self.hook_layer = first_probe.config.hook_layer
            
            for p in probes[1:]:
                if p.config.input_size != self.input_dim:
                    raise ValueError(
                        f"All probes in a set must have the same input dimension. "
                        f"Expected {self.input_dim}, got {p.config.input_size} for probe '{p.name}'."
                    )
                # Could add checks for model_name, hook_point etc. if strict consistency is needed
        else:
            self.input_dim = None
            self.model_name = None
            self.hook_point = None
            self.hook_layer = None
        
    def encode(self, acts: torch.Tensor) -> torch.Tensor:
        """Compute dot products with all probes' normalized directions.
        
        Args:
            acts: Activations to project, shape [..., d_model]
            
        Returns:
            Projected values, shape [..., n_probes]
        """
        if not self.probes:
            return torch.empty(acts.shape[:-1] + (0,), device=acts.device)
            
        # Get normalized directions for all probes
        weight_matrix = torch.stack(
            [p.get_direction(normalized=True) for p in self.probes]
        ).to(acts.device) # Shape [n_probes, d_model]
        
        # Project all at once
        return torch.einsum("...d,nd->...n", acts, weight_matrix)
    
    def __getitem__(self, idx) -> BaseProbe:
        """Get a probe by index."""
        return self.probes[idx]
    
    def __len__(self) -> int:
        """Get number of probes."""
        return len(self.probes)
        
    def save(self, directory: str, use_json: bool = False) -> None:
        """Save all probes to a directory.
        
        Args:
            directory: Directory to save the probes.
            use_json: If True, save probes in JSON format, otherwise use .pt.
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save index file with common metadata
        index = {
            "model_name": self.model_name,
            "hook_point": self.hook_point,
            "hook_layer": self.hook_layer,
            "format": "json" if use_json else "pt",
            "probes": []
        }
        
        # Save each probe
        for i, probe in enumerate(self.probes):
            # Sanitize probe name for filename
            safe_name = "".join(c if c.isalnum() else "_" for c in probe.name)
            filename = f"probe_{i}_{safe_name}.{'json' if use_json else 'pt'}"
            filepath = os.path.join(directory, filename)
            
            if use_json:
                probe.save_json(filepath)
            else:
                probe.save(filepath)
            
            # Add to index
            index["probes"].append({
                "name": probe.name,
                "file": filename,
                "probe_type": probe.__class__.__name__
            })
            
        # Save index
        with open(os.path.join(directory, "index.json"), "w") as f:
            json.dump(index, f, indent=2)
            
    @classmethod
    def load(cls, directory: str, device: Optional[str] = None) -> "ProbeSet":
        """Load a ProbeSet from a directory.
        
        Args:
            directory: Directory containing the probes and index.json.
            device: Optional device override for loading probes.
            
        Returns:
            ProbeSet instance
        """
        index_path = os.path.join(directory, "index.json")
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"Index file not found in directory: {directory}")
             
        # Load index
        with open(index_path) as f:
            index = json.load(f)
            
        # Load each probe
        probes = []
        save_format = index.get("format", "pt") # Default to .pt if format not specified

        for entry in index["probes"]:
            filepath = os.path.join(directory, entry["file"])
            probe_type_name = entry.get("probe_type")
            
            if not probe_type_name:
                 raise ValueError(f"Probe type missing for entry: {entry}")

            # Dynamically get the probe class
            try:
                 probe_cls = globals()[probe_type_name]
            except KeyError:
                 raise ValueError(f"Unknown probe type '{probe_type_name}' encountered during loading.")
                 
            if not issubclass(probe_cls, BaseProbe):
                 raise TypeError(f"Class '{probe_type_name}' is not a valid probe type.")

            # Load the individual probe using its class's load method
            if save_format == "json":
                probe = probe_cls.load_json(filepath, device=device)
            else:
                probe = probe_cls.load(filepath) # .load handles map_location
                # Ensure loaded probe is on the correct device if override specified
                if device and probe.config.device != device:
                     target_device = torch.device(device)
                     probe.to(target_device)
                     probe.config.device = str(target_device) # Update config string

            probes.append(probe)
            
        return cls(probes)