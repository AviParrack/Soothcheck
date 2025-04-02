import torch
from typing import List, Optional, Union, Any
from transformer_lens import HookedTransformer

from probity.probes.linear_probe import BaseProbe


class ProbeInference:
    """
    Primary interface for running trained probes on new text.
    
    This class handles BaseProbe instances in a consistent way.
    It manages the full pipeline of tokenization, model forward pass, activation extraction,
    and probe application with proper handling of standardization and transformations.
    """
    
    def __init__(
        self,
        model_name: str,
        hook_point: str,
        probe: BaseProbe,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.hook_point = hook_point
        
        # Set up probe
        self.probe = probe.to(device)
        self.probe.eval()
        self.probe_type = "base_probe"
        self.probe_class = probe.__class__.__name__
        
        # Setup model
        self.model = HookedTransformer.from_pretrained_no_processing(model_name)
        self.model.to(device)
        
    def get_activations(
        self, 
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get model activations for text input.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of activations (batch_size, seq_len, hidden_size)
        """
        # Ensure text is a list
        if isinstance(text, str):
            text = [text]
            
        # Tokenize
        tokens = self.model.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True
        )
        
        # Get activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens["input_ids"].to(self.device),
                names_filter=[self.hook_point],
                return_cache_object=True
            )
            
        return cache[self.hook_point]
    
    def __call__(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """
        Deprecated: Please use get_direction_activations() for raw activations or 
        get_probabilities() for properly transformed outputs.
        
        Get probe outputs for text.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of probe outputs (batch_size, seq_len, [n_vectors for VectorSet])
        """
        import warnings
        warnings.warn(
            "Direct calling is deprecated. Use get_direction_activations() for "
            "raw activations or get_probabilities() for properly transformed outputs.",
            DeprecationWarning, 
            stacklevel=2
        )
        return self.get_direction_activations(text)
    
    def get_direction_activations(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get activations along the probe direction.
        
        This projects the activations onto the learned probe direction by taking
        the dot product between the activations and the direction vector.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of direction activations (batch_size, seq_len)
        """
        # Get activations
        activations = self.get_activations(text)
        batch_size, seq_len, hidden_size = activations.shape
        
        # Reshape for batch calculation
        flat_activations = activations.view(-1, hidden_size)
        
        # Let the probe handle standardization (don't reimplement here)
        if hasattr(self.probe, '_apply_standardization'):
            flat_activations = self.probe._apply_standardization(flat_activations)
        
        # Get the probe direction (always normalized for consistency)
        direction = self.probe.get_direction(normalized=True)
        
        # Calculate dot product with direction
        with torch.no_grad():
            # For multi-dimensional directions (rarely used)
            if direction.dim() > 1:
                outputs = torch.matmul(flat_activations, direction.t())
            else:
                # For single vector directions (common case)
                outputs = torch.matmul(flat_activations, direction)
        
        # Reshape back
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-dimensional output
            return outputs.view(batch_size, seq_len, -1).cpu()
        else:
            # Single dimension output
            return outputs.view(batch_size, seq_len).cpu()
        

    def get_probe_outputs(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get outputs using the probe's forward method.
        
        This uses the probe's specific forward implementation which may include
        additional transformations beyond a simple dot product. The probe's forward
        method now handles standardization internally.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of probe outputs (batch_size, seq_len)
        """
        # Get activations
        activations = self.get_activations(text)
        batch_size, seq_len, hidden_size = activations.shape
        
        # Reshape for probe
        flat_activations = activations.view(-1, hidden_size)
        
        # Run probe - the probe's forward method handles all standardization internally
        with torch.no_grad():
            outputs = self.probe(flat_activations)
            
        # Reshape back
        if outputs.dim() > 1 and outputs.size(1) > 1:
            # Multi-dimensional output
            return outputs.view(batch_size, seq_len, -1).cpu()
        else:
            # Single dimension output
            return outputs.view(batch_size, seq_len).cpu()
    

    def get_probabilities(
        self,
        text: Union[str, List[str]]
    ) -> torch.Tensor:
        """Get probabilities from the probe.
        
        For logistic probes, applies sigmoid to the probe outputs.
        For other probes, returns outputs from get_probe_outputs.
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Tensor of probabilities (batch_size, seq_len)
        """
        outputs = self.get_probe_outputs(text)
        
        # Apply sigmoid for logistic probes
        logistic_probe_types = ["LogisticProbe", "SklearnLogisticProbe"]
        if self.probe_class in logistic_probe_types:
            return torch.sigmoid(outputs)
        else:
            return outputs
    
    @classmethod
    def from_saved_probe(
        cls,
        model_name: str,
        hook_point: str,
        probe_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        probe_class: Optional[Any] = None
    ) -> 'ProbeInference':
        """Create inference instance from saved probe.
        
        Args:
            model_name: Name of the model
            hook_point: Hook point to use
            probe_path: Path to the saved probe
            device: Device to run on
            probe_class: Optional probe class for custom probe types
            
        Returns:
            ProbeInference instance
        """
        # Load the probe based on the file extension
        if probe_path.endswith('.json'):
            # JSON format
            if probe_class is None:
                from probity.probes.linear_probe import LogisticProbe
                probe_class = LogisticProbe
            probe = probe_class.load_json(probe_path)
        else:
            # PyTorch format
            if probe_class is None:
                from probity.probes.linear_probe import LogisticProbe
                probe_class = LogisticProbe
            probe = probe_class.load(probe_path)
            
        return cls(model_name, hook_point, probe, device=device)