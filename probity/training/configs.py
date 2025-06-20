import torch
from typing import Dict
from transformer_lens import HookedTransformer
from tqdm import tqdm

from probity.probes import (
    LogisticProbe, LogisticProbeConfig,
    PCAProbe, PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe, KMeansProbeConfig,
    LinearProbe, LinearProbeConfig
)
from probity.training.trainer import (
    SupervisedProbeTrainer, SupervisedTrainerConfig,
    DirectionalProbeTrainer, DirectionalTrainerConfig
)


def get_probe_config(probe_type: str, hidden_size: int, model_name: str, 
                    hook_point: str, layer: int, dtype: torch.dtype) -> Dict:
    """Get probe configuration based on type"""
    # Convert torch dtype to string for configs
    if dtype == torch.bfloat16:
        dtype_str = 'bfloat16'
    elif dtype == torch.float16:
        dtype_str = 'float16'
    else:
        dtype_str = 'float32'

    configs = {
        'logistic': LogisticProbeConfig(
            input_size=hidden_size,
            normalize_weights=True,
            bias=True,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"lie_truth_{probe_type}_layer_{layer}",
            dtype=dtype_str
        ),
        'linear': LinearProbeConfig(
            input_size=hidden_size,
            normalize_weights=True,
            bias=False,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"lie_truth_{probe_type}_layer_{layer}",
            dtype=dtype_str
        ),
        'pca': PCAProbeConfig(
            input_size=hidden_size,
            n_components=1,
            normalize_weights=True,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"lie_truth_{probe_type}_layer_{layer}",
            dtype=dtype_str
        ),
        'meandiff': MeanDiffProbeConfig(
            input_size=hidden_size,
            normalize_weights=True,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"lie_truth_{probe_type}_layer_{layer}",
            dtype=dtype_str
        ),
        'kmeans': KMeansProbeConfig(
            input_size=hidden_size,
            n_clusters=2,
            normalize_weights=True,
            model_name=model_name,
            hook_point=hook_point,
            hook_layer=layer,
            name=f"lie_truth_{probe_type}_layer_{layer}",
            dtype=dtype_str
        )
    }
    return configs.get(probe_type)


def get_probe_class(probe_type: str):
    """Get probe class based on type"""
    classes = {
        'logistic': LogisticProbe,
        'linear': LinearProbe,
        'pca': PCAProbe,
        'meandiff': MeanDifferenceProbe,
        'kmeans': KMeansProbe
    }
    return classes.get(probe_type)


def get_trainer_config(probe_type: str, device: str, batch_size: int) -> Dict:
    """Get trainer configuration based on probe type"""
    if probe_type in ['logistic', 'linear']:
        return SupervisedTrainerConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            num_epochs=10,
            weight_decay=0.01,
            train_ratio=0.8,
            handle_class_imbalance=True,
            show_progress=True,
            device=device,
            standardize_activations=True
        )
    else:
        return DirectionalTrainerConfig(
            batch_size=batch_size,
            device=device,
            standardize_activations=True
        )


def get_trainer_class(probe_type: str):
    """Get trainer class based on probe type"""
    if probe_type in ['logistic', 'linear']:
        return SupervisedProbeTrainer
    else:
        return DirectionalProbeTrainer
