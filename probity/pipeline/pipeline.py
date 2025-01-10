from dataclasses import dataclass
from typing import List, Optional, Dict, Type, Generic, TypeVar
import torch
from pathlib import Path

from probity.collection.collectors import TransformerLensCollector, TransformerLensConfig
from probity.collection.activation_store import ActivationStore
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes.linear_probe import BaseProbe, ProbeConfig
from probity.training.trainer import BaseProbeTrainer, BaseTrainerConfig

C = TypeVar('C', bound=BaseTrainerConfig)
P = TypeVar('P', bound=ProbeConfig)

@dataclass
class ProbePipelineConfig:
    """Configuration for probe pipeline."""
    dataset: TokenizedProbingDataset
    probe_cls: Type[BaseProbe]
    probe_config: ProbeConfig
    trainer_cls: Type[BaseProbeTrainer]
    trainer_config: BaseTrainerConfig
    position_key: str
    cache_dir: Optional[str] = None
    model_name: Optional[str] = None
    hook_points: Optional[List[str]] = None
    activation_batch_size: int = 32

class ProbePipeline(Generic[C, P]):
    """Pipeline for end-to-end probe training including activation collection."""
    
    def __init__(self, config: ProbePipelineConfig):
        self.config = config
        self.activation_stores: Optional[Dict[str, ActivationStore]] = None
        self.probe: Optional[BaseProbe] = None
        
    def _collect_activations(self) -> Dict[str, ActivationStore]:
        """Collect activations if needed."""
        if not self.config.model_name or not self.config.hook_points:
            raise ValueError("Model name and hook points required for activation collection")
            
        self.collector = TransformerLensCollector(
            TransformerLensConfig(
                model_name=self.config.model_name,
                hook_points=self.config.hook_points,
                batch_size=self.config.activation_batch_size
            )
        )

        return self.collector.collect(self.config.dataset)
    
    def _load_or_collect_activations(self) -> Dict[str, ActivationStore]:
        """Load activations from cache if available and valid, otherwise collect them."""
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            if cache_path.exists():
                stores = {}
                cache_valid = True
                
                # Try to load and validate each hook point
                for hook_point in self.config.hook_points or []:
                    store_path = cache_path / hook_point.replace(".", "_")
                    if store_path.exists():
                        try:
                            store = ActivationStore.load(str(store_path))
                            if self._validate_cache_compatibility(store, hook_point):
                                stores[hook_point] = store
                            else:
                                print(f"Cache for {hook_point} is incompatible with current configuration")
                                cache_valid = False
                                break
                        except Exception as e:
                            print(f"Error loading cache for {hook_point}: {e}")
                            cache_valid = False
                            break
                            
                if cache_valid and stores:
                    return stores
                else:
                    # If cache is invalid, clear it
                    import shutil
                    shutil.rmtree(cache_path)
                    print("Cleared invalid cache directory")
                    
        # If we get here, need to collect activations
        stores = self._collect_activations()
        
        # Cache if directory specified
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Save configuration hash with the cache
            for hook_point, store in stores.items():
                store_path = cache_path / hook_point.replace(".", "_")
                store.save(str(store_path))
                
        return stores
    
    def _validate_cache_compatibility(self, store: ActivationStore, hook_point: str) -> bool:
        """Validate if cached activation store matches current configuration.
        
        Args:
            store: Loaded activation store
            hook_point: Hook point being validated
            
        Returns:
            bool: Whether cache is compatible
        """
        # Check dataset compatibility
        if len(store.dataset.examples) != len(self.config.dataset.examples):
            return False
        
        # Check if tokenization configs match
        if not hasattr(store.dataset, 'tokenization_config') or \
           not hasattr(self.config.dataset, 'tokenization_config'):
            return False
        
        store_config = store.dataset.tokenization_config
        current_config = self.config.dataset.tokenization_config
        
        if store_config.tokenizer_name != current_config.tokenizer_name:
            return False
        
        # Check if position types match
        if store.dataset.position_types != self.config.dataset.position_types:
            return False
        
        return True
    
    def run(self, hook_point: Optional[str] = None) -> BaseProbe:
        """Run the full pipeline.
        
        Args:
            hook_point: Which activation layer to use. If not specified,
                      uses first available hook point.
        """
        # Get activations
        if self.activation_stores is None:
            self.activation_stores = self._load_or_collect_activations()
            
        if not hook_point:
            hook_point = list(self.activation_stores.keys())[0]
            
        if hook_point not in self.activation_stores:
            raise ValueError(f"Hook point {hook_point} not found in activation stores")
            
        # Initialize probe
        self.probe = self.config.probe_cls(self.config.probe_config)
        
        # Initialize trainer
        trainer = self.config.trainer_cls(self.config.trainer_config)
        
        # Prepare data and train
        train_loader, val_loader = trainer.prepare_supervised_data(
            self.activation_stores[hook_point],
            self.config.position_key
        )
        
        history = trainer.train(self.probe, train_loader, val_loader)
        
        return self.probe, history

    def save(self, path: str) -> None:
        """Save pipeline state."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save probe if exists
        if self.probe:
            self.probe.save(str(save_path / "probe.pt"))
            
        # Save config
        torch.save(self.config, str(save_path / "config.pt"))
        
    @classmethod
    def load(cls, path: str) -> 'ProbePipeline':
        """Load pipeline from saved state."""
        load_path = Path(path)
        
        # Load config
        config = torch.load(str(load_path / "config.pt"))
        
        # Create pipeline
        pipeline = cls(config)
        
        # Load probe if exists
        probe_path = load_path / "probe.pt"
        if probe_path.exists():
            pipeline.probe = BaseProbe.load(str(probe_path))
            
        return pipeline
    
    def _get_cache_key(self) -> str:
        """Generate a unique key for the current configuration."""
        import hashlib
        import json
        
        # Collect relevant configuration details
        config_dict = {
            "model_name": self.config.model_name,
            "hook_points": self.config.hook_points,
            "position_key": self.config.position_key,
            "tokenizer_name": self.config.dataset.tokenization_config.tokenizer_name,
            "dataset_size": len(self.config.dataset.examples),
            "position_types": list(self.config.dataset.position_types)
        }
        
        # Create deterministic string representation
        config_str = json.dumps(config_dict, sort_keys=True)
        
        # Generate hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _get_cache_path(self) -> Path:
        """Get cache path including configuration hash."""
        if not self.config.cache_dir:
            raise ValueError("Cache directory not specified")
        
        base_path = Path(self.config.cache_dir)
        config_hash = self._get_cache_key()
        return base_path / config_hash