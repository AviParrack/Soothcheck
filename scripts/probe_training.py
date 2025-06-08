import argparse
import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm

from probity.collection.activation_store import ActivationStore
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

from utils import load_lie_truth_dataset, get_model_dtype


def parse_args():
    parser = argparse.ArgumentParser(description='Train probes efficiently')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_dataset_dir', type=str, required=True)
    parser.add_argument('--probe_types', nargs='+', 
                       choices=['logistic', 'linear', 'pca', 'meandiff', 'kmeans'],
                       default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--layers', nargs='+', default=['all'])
    parser.add_argument('--probe_save_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


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

    print(f"[DEBUG get_probe_config] Creating config for {probe_type} with dtype_str: {dtype_str}")
    print(f"[DEBUG get_probe_config] Input torch dtype: {dtype}")
    
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


def collect_activations_with_model(model: HookedTransformer, dataset, layers: List[int], 
                                  cache_dir: str, batch_size: int, device: str, dtype: torch.dtype) -> Dict[str, ActivationStore]:
    """Collect activations using the already loaded model"""
    # Check cache first
    cache_path = Path(cache_dir) / f"{model.cfg.model_name.replace('/', '_')}_activations.pt"
    
    if cache_path.exists():
        print(f"Loading cached activations from {cache_path}")
        try:
            stores = torch.load(cache_path, map_location=device)
            # Ensure cached activations have correct dtype
            for store in stores.values():
                if store.raw_activations.dtype != dtype:
                    store.raw_activations = store.raw_activations.to(dtype)
            return stores
        except Exception as e:
            print(f"Failed to load cache: {e}. Recollecting...")
    
    # Manually collect activations for all layers
    hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    all_activations = {hook: [] for hook in hook_points}
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(dataset.examples), batch_size):
            batch_end = min(batch_start + batch_size, len(dataset.examples))
            batch_indices = list(range(batch_start, batch_end))
            
            # Get batch tensors
            batch = dataset.get_batch_tensors(batch_indices)
            input_ids = batch["input_ids"].to(device)
            
            # Run model with caching
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_points,
                return_cache_object=True,
                stop_at_layer=max(layers) + 1  # Stop early to save computation
            )
            
            # Store activations for each hook point (keep them in original dtype)
            for hook in hook_points:
                all_activations[hook].append(cache[hook].cpu())
    
    # Create ActivationStore objects
    activation_stores = {}
    for hook, activations in all_activations.items():
        layer = int(hook.split(".")[1])
        
        # Stack all activations
        raw_activations = torch.cat(activations, dim=0)
        
        activation_stores[hook] = ActivationStore(
            raw_activations=raw_activations,
            hook_point=hook,
            example_indices=torch.arange(len(dataset.examples)),
            sequence_lengths=torch.tensor(dataset.get_token_lengths()),
            hidden_size=raw_activations.shape[-1],
            dataset=dataset,
            labels=torch.tensor([ex.label for ex in dataset.examples]),
            label_texts=[ex.label_text for ex in dataset.examples],
        )
    
    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(activation_stores, cache_path)
    print(f"Saved activations to {cache_path}")
    
    return activation_stores


def main():
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.train_dataset_dir}")
    dataset = load_lie_truth_dataset(args.train_dataset_dir)
    print(f"Dataset size: {len(dataset.examples)}")
    
    # Load model once using from_pretrained_no_processing
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    
    try:
        # Use from_pretrained_no_processing as requested
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name, 
            device=args.device,
            dtype=model_dtype
        )
    except Exception as e:
        # Fallback for unsupported models
        print(f"Error with from_pretrained_no_processing: {e}")
        print("Attempting alternative loading method...")
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
            device_map=args.device
        )
        model = HookedTransformer.from_pretrained(
            args.model_name,
            hf_model=hf_model,
            device=args.device,
            dtype=model_dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    
    hidden_size = model.cfg.d_model
    
    # Determine layers
    if 'all' in args.layers:
        layers = list(range(model.cfg.n_layers))
    else:
        layers = [int(l) for l in args.layers]
    
    print(f"Training on layers: {layers}")
    print(f"Model dtype: {model_dtype}")
    
    # Collect activations using the loaded model
    print("Collecting activations...")
    activation_stores = collect_activations_with_model(
        model, dataset, layers, args.cache_dir, args.batch_size, args.device, model_dtype
    )
    
    # Free model memory after collecting activations
    del model
    torch.cuda.empty_cache()
    
    # Train probes
    results = {}
    
    for layer in tqdm(layers, desc="Layers"):
        hook_point = f"blocks.{layer}.hook_resid_pre"
        activation_store = activation_stores[hook_point]
        
        layer_results = {}
        
        for probe_type in args.probe_types:
            print(f"\nTraining {probe_type} probe on layer {layer}")
            
            # Get configurations with correct dtype
            probe_config = get_probe_config(
                probe_type, hidden_size, args.model_name, 
                hook_point, layer, model_dtype
            )
            probe_cls = get_probe_class(probe_type)
            trainer_config = get_trainer_config(probe_type, args.device, args.batch_size)
            trainer_cls = get_trainer_class(probe_type)
            
            # Initialize probe and trainer
            probe = probe_cls(probe_config).to(args.device)


            print(f"[DEBUG main] Created {probe_type} probe")
            print(f"[DEBUG main] Probe dtype attribute: {probe.dtype if hasattr(probe, 'dtype') else 'No dtype attr'}")
            if hasattr(probe, 'linear'):
                print(f"[DEBUG main] Probe linear weight dtype: {probe.linear.weight.dtype}")
            if hasattr(probe, 'direction_vector'):
                print(f"[DEBUG main] Probe direction_vector dtype: {probe.direction_vector.dtype}")
            
            trainer = trainer_cls(trainer_config)
            
            # Prepare data
            train_loader, val_loader = trainer.prepare_supervised_data(
                activation_store, "LIE_SPAN"
            )
            
            # Train
            history = trainer.train(probe, train_loader, val_loader)
            
            # Save probe immediately after training
            save_dir = Path(args.probe_save_dir) / probe_type
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"layer_{layer}_probe.json"
            probe.save_json(str(save_path))
            print(f"Saved {probe_type} probe for layer {layer} to {save_path}")
            
            layer_results[probe_type] = {
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None
            }
        
        results[layer] = layer_results
    
    # Save training summary
    summary_path = Path(args.probe_save_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTraining complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()