import argparse
import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm
import hashlib

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


def get_dataset_hash(dataset) -> str:
    """Generate a hash for the dataset to enable smart caching"""
    # Create a hash based on dataset content
    content_str = ""
    for ex in dataset.examples[:10]:  # Sample first 10 examples for efficiency
        content_str += f"{ex.text[:100]}_{ex.label}_"  # First 100 chars + label
    
    content_str += f"_size_{len(dataset.examples)}"
    return hashlib.md5(content_str.encode()).hexdigest()[:16]


def smart_cache_activations(model: HookedTransformer, dataset, layers: List[int], 
                           cache_dir: str, batch_size: int, device: str, 
                           dtype: torch.dtype, force_recache: bool = False) -> Dict[str, ActivationStore]:
    """Smart caching that checks dataset compatibility and model compatibility"""
    
    # Create cache path with dataset and model info
    model_name_clean = model.cfg.model_name.replace('/', '_').replace('-', '_')
    dataset_hash = get_dataset_hash(dataset)
    
    cache_base = Path(cache_dir) / f"{model_name_clean}_{dataset_hash}"
    cache_metadata_path = cache_base / "cache_metadata.json"
    
    # Check if cache exists and is valid
    if not force_recache and cache_metadata_path.exists():
        try:
            with open(cache_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify cache is compatible
            if (metadata.get("model_name") == model.cfg.model_name and
                metadata.get("dataset_size") == len(dataset.examples) and
                metadata.get("dtype") == str(dtype) and
                set(metadata.get("layers", [])) >= set(layers)):
                
                print(f"Loading compatible cached activations from {cache_base}")
                
                # Load cached activation stores
                stores = {}
                for layer in layers:
                    store_path = cache_base / f"layer_{layer}.pt"
                    if store_path.exists():
                        stores[f"blocks.{layer}.hook_resid_pre"] = torch.load(store_path, map_location=device)
                        # Ensure correct dtype
                        store = stores[f"blocks.{layer}.hook_resid_pre"]
                        if store.raw_activations.dtype != dtype:
                            store.raw_activations = store.raw_activations.to(dtype)
                
                if len(stores) == len(layers):
                    print(f"Successfully loaded {len(stores)} cached activation stores")
                    return stores
                else:
                    print(f"Cache incomplete: found {len(stores)}/{len(layers)} layers")
        
        except Exception as e:
            print(f"Failed to load cache: {e}. Recollecting...")
    
    # Collect activations
    print(f"Collecting activations for {len(layers)} layers...")
    
    hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    all_activations = {hook: [] for hook in hook_points}
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset.examples), batch_size), 
                               desc="Collecting activations"):
            batch_end = min(batch_start + batch_size, len(dataset.examples))
            batch_indices = list(range(batch_start, batch_end))
            
            # Get batch tensors
            batch = dataset.get_batch_tensors(batch_indices)
            input_ids = batch["input_ids"].to(device)
            
            # Run model with caching for all layers at once
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_points,
                return_cache_object=True,
                stop_at_layer=max(layers) + 1
            )
            
            # Store activations for each hook point
            for hook in hook_points:
                all_activations[hook].append(cache[hook].cpu())
    
    # Create ActivationStore objects
    activation_stores = {}
    cache_base.mkdir(parents=True, exist_ok=True)
    
    for hook, activations in all_activations.items():
        layer = int(hook.split(".")[1])
        
        # Stack all activations
        raw_activations = torch.cat(activations, dim=0)
        
        store = ActivationStore(
            raw_activations=raw_activations,
            hook_point=hook,
            example_indices=torch.arange(len(dataset.examples)),
            sequence_lengths=torch.tensor(dataset.get_token_lengths()),
            hidden_size=raw_activations.shape[-1],
            dataset=dataset,
            labels=torch.tensor([ex.label for ex in dataset.examples]),
            label_texts=[ex.label_text for ex in dataset.examples],
        )
        
        activation_stores[hook] = store
        
        # Save individual layer cache
        layer_cache_path = cache_base / f"layer_{layer}.pt"
        torch.save(store, layer_cache_path)
    
    # Save cache metadata
    metadata = {
        "model_name": model.cfg.model_name,
        "dataset_size": len(dataset.examples),
        "dtype": str(dtype),
        "layers": layers,
        "cache_version": "1.0"
    }
    
    with open(cache_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved activations cache to {cache_base}")
    
    return activation_stores


def train_all_probes_for_layer(layer: int, activation_store: ActivationStore, 
                              probe_types: List[str], args, 
                              model_name: str, hidden_size: int, 
                              device: str, dtype: torch.dtype) -> Dict[str, Dict]:
    """Train all probe types for a single layer efficiently"""
    
    hook_point = f"blocks.{layer}.hook_resid_pre"
    layer_results = {}
    
    for probe_type in probe_types:
        print(f"Training {probe_type} probe on layer {layer}")
        
        # Get configurations
        probe_config = get_probe_config(
            probe_type, hidden_size, model_name, 
            hook_point, layer, dtype
        )
        probe_cls = get_probe_class(probe_type)
        trainer_config = get_trainer_config(probe_type, device, args.batch_size)
        trainer_cls = get_trainer_class(probe_type)
        
        # Initialize probe and trainer
        probe = probe_cls(probe_config).to(device)
        trainer = trainer_cls(trainer_config)
        
        # Prepare data once per layer (shared across probe types)
        train_loader, val_loader = trainer.prepare_supervised_data(
            activation_store, "LIE_SPAN"
        )
        
        # Train
        history = trainer.train(probe, train_loader, val_loader)
        
        # Save probe immediately
        save_dir = Path(args.probe_save_dir) / probe_type
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"layer_{layer}_probe.json"
        probe.save_json(str(save_path))
        
        layer_results[probe_type] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
            'save_path': str(save_path)
        }
        
        print(f"Saved {probe_type} probe for layer {layer} to {save_path}")
        
        # Clear probe from memory
        del probe
        torch.cuda.empty_cache()
    
    return layer_results


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
    parser.add_argument('--activation_batch_size', type=int, default=16, 
                       help='Batch size for activation collection (separate from training)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_recache', action='store_true', 
                       help='Force recollection of activations even if cache exists')
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


def main():
    args = parse_args()
    
    # Load dataset
    print(f"Loading dataset from {args.train_dataset_dir}")
    dataset = load_lie_truth_dataset(args.train_dataset_dir)
    print(f"Dataset size: {len(dataset.examples)}")
    
    # Load model once
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name, 
            device=args.device,
            dtype=model_dtype
        )
    except Exception as e:
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
    
    # Collect activations for all layers at once using smart caching
    print("Collecting/loading activations...")
    activation_stores = smart_cache_activations(
        model, dataset, layers, args.cache_dir, 
        args.activation_batch_size, args.device, model_dtype, 
        args.force_recache
    )
    
    # Free model memory after collecting activations
    del model
    torch.cuda.empty_cache()
    
    # Train probes efficiently - one layer at a time, all probe types per layer
    results = {}
    
    for layer in tqdm(layers, desc="Training layers"):
        hook_point = f"blocks.{layer}.hook_resid_pre"
        activation_store = activation_stores[hook_point]
        
        # Train all probe types for this layer
        layer_results = train_all_probes_for_layer(
            layer, activation_store, args.probe_types, args,
            args.model_name, hidden_size, args.device, model_dtype
        )
        
        results[layer] = layer_results
        
        # Optional: Clear activation store to save memory if processing many layers
        if len(layers) > 16:  # Only clear for large numbers of layers
            del activation_stores[hook_point]
            torch.cuda.empty_cache()
    
    # Save training summary
    summary_path = Path(args.probe_save_dir) / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete. Summary saved to {summary_path}")
    
    # Print summary statistics
    print("\nTraining Summary:")
    for layer, layer_results in results.items():
        print(f"\nLayer {layer}:")
        for probe_type, probe_results in layer_results.items():
            final_loss = probe_results['final_train_loss']
            print(f"  {probe_type}: Final loss = {final_loss:.6f}")


if __name__ == "__main__":
    main()