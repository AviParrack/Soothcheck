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
        """Load activations from cache if available, otherwise collect them."""
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            if cache_path.exists():
                stores = {}
                for hook_point in self.config.hook_points or []:
                    store_path = cache_path / hook_point.replace(".", "_")
                    if store_path.exists():
                        stores[hook_point] = ActivationStore.load(str(store_path))
                if stores:
                    return stores
                    
        # If we get here, need to collect activations
        stores = self._collect_activations()
        
        # Cache if directory specified
        if self.config.cache_dir:
            cache_path = Path(self.config.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            for hook_point, store in stores.items():
                store_path = cache_path / hook_point.replace(".", "_")
                store.save(str(store_path))
                
        return stores
    
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