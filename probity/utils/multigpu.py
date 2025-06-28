from dataclasses import dataclass
from typing import List, Literal, Optional
import torch

@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU/distributed training."""
    enabled: bool = False
    backend: Literal["DataParallel", "DistributedDataParallel"] = "DataParallel"
    device_ids: Optional[List[int]] = None  # If None, use all available
    main_device: int = 0  # Default device for model/optimizer
    world_size: int = 1  # For DDP
    rank: int = 0  # For DDP
    dist_url: str = "env://"  # For DDP


def wrap_model_for_multigpu(model, config: MultiGPUConfig):
    if not config.enabled:
        return model
    if config.backend == "DataParallel":
        return torch.nn.DataParallel(model, device_ids=config.device_ids)
    elif config.backend == "DistributedDataParallel":
        # User must call torch.distributed.init_process_group externally
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.main_device] if config.device_ids is None else config.device_ids
        )
    else:
        raise ValueError(f"Unknown multi-GPU backend: {config.backend}") 