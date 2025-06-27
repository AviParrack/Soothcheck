#!/usr/bin/env python3
"""
Multi-Layer NTML Probe Training & Optimization

Train NTML statement-level probes across all layers of GPT-2, automatically
find the best performing layers, and provide comprehensive training monitoring.

Features:
- Trains logistic probes across all GPT-2 layers (0-11)
- Captures detailed training histories (loss curves, validation metrics)
- Automatically identifies best performing layers
- Generates comprehensive evaluation reports
- Optimized for laptop-friendly GPT-2 training

Usage:
    python train_ntml_all_layers.py --dataset 2T1L_500samples
    python train_ntml_all_layers.py --dataset 5T5L_500samples --num_epochs 15
    python train_ntml_all_layers.py --jsonl_path data/custom.jsonl --layers 6,7,8,9,10,11
"""

import argparse
import sys
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np

def find_repo_root():
    """Find the repository root directory."""
    current = Path.cwd()
    
    # Look for key indicators
    primary_indicators = [
        "Statement-level-probe-implementation-plan.md",
        "data/NTML-datasets",
        "probity_extensions"
    ]
    
    for path in [current] + list(current.parents):
        if all((path / indicator).exists() for indicator in primary_indicators):
            return path
    
    return current

def setup_paths():
    """Setup Python paths for imports."""
    repo_root = find_repo_root()
    probity_dir = repo_root / "probity"
    
    if not probity_dir.exists():
        raise FileNotFoundError(f"Probity directory not found at {probity_dir}")
    
    sys.path.insert(0, str(repo_root))
    return repo_root, probity_dir

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NTML probes across multiple layers with optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., '2T1L_500samples')"
    )
    dataset_group.add_argument(
        "--jsonl_path",
        type=str,
        help="Full path to NTML JSONL dataset file"
    )
    
    # Model and layer configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name (default: gpt2)"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to train on: 'all', or comma-separated list (e.g., '6,7,8,9')"
    )
    
    # Training configuration
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)"
    )
    
    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for probes and results"
    )
    parser.add_argument(
        "--results_name",
        type=str,
        default=None,
        help="Name for results files (default: auto-generated)"
    )
    
    # Analysis options
    parser.add_argument(
        "--eval_after_training",
        action="store_true",
        help="Run evaluation on all trained probes"
    )
    parser.add_argument(
        "--plot_training_curves",
        action="store_true",
        help="Generate training curve plots"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_and_prepare_dataset(dataset_path: Path, repo_root: Path):
    """Load and prepare NTML dataset for training."""
    print(f"\n📥 Loading NTML dataset: {dataset_path.name}")
    
    # Import probity components
    from transformers import AutoTokenizer
    from probity.datasets.tokenized import TokenizedProbingDataset
    from probity_extensions.conversational import ConversationalProbingDataset
    
    # Load NTML dataset
    conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(dataset_path))
    stmt_dataset = conv_dataset.get_statement_dataset()
    
    print(f"✅ Loaded {len(conv_dataset.examples)} conversations → {len(stmt_dataset.examples)} statements")
    
    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=stmt_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=True
    )
    
    print(f"✅ Tokenized dataset ready: {len(tokenized_dataset.examples)} examples")
    return tokenized_dataset

def train_probe_on_layer(
    layer: int,
    tokenized_dataset,
    model_name: str,
    training_config: dict,
    output_dir: Path,
    model=None,
    verbose: bool = False
) -> Tuple[object, Dict]:
    """Train a single probe on a specific layer."""
    
    from probity.probes.logistic import LogisticProbe, LogisticProbeConfig
    from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
    from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
    
    hook_point = f"blocks.{layer}.hook_resid_pre"
    probe_name = f"ntml_multilayer_layer_{layer}"
    
    if verbose:
        print(f"   🎯 Layer {layer}: {hook_point}")
    
    # Determine input size based on model
    if "llama" in model_name.lower():
        if "70b" in model_name.lower():
            input_size = 8192  # Llama 3.1 70B hidden size
        else:
            input_size = 4096  # Llama 3.1 8B hidden size
    else:
        input_size = 768  # GPT-2 hidden size
    
    # Configure probe
    probe_config = LogisticProbeConfig(
        input_size=input_size,
        normalize_weights=True,
        bias=False,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer,
        name=probe_name
    )
    
    # Configure trainer
    trainer_config = SupervisedTrainerConfig(
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        num_epochs=training_config["num_epochs"],
        patience=training_config["patience"],
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=verbose
    )
    
    # Configure pipeline
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="target",
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir=str(output_dir.parent / "cache" / "ntml_multilayer_cache")
    )
    
    # Train probe - pass the pre-loaded model if available
    pipeline = ProbePipeline(pipeline_config)
    if model is not None:
        # Use the pre-loaded model instead of loading a new one
        probe, history = pipeline.run_with_model(model)
    else:
        probe, history = pipeline.run()
    
    # Save probe in probity format
    probe_type_dir = output_dir / "logistic"
    probe_type_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_type_dir / f"layer_{layer}_probe.json"
    probe.save_json(str(probe_path))
    
    # Calculate final metrics
    final_metrics = {
        "layer": layer,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float('inf'),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float('inf'),
        "min_val_loss": min(history["val_loss"]) if history["val_loss"] else float('inf'),
        "epochs_trained": len(history["train_loss"]),
        "probe_path": str(probe_path)
    }
    
    if verbose:
        print(f"   ✅ Layer {layer}: Final val loss = {final_metrics['final_val_loss']:.4f}")
    
    return probe, history, final_metrics

def train_probe_on_layer_with_activations(
    layer: int,
    activation_store,
    model_name: str,
    training_config: dict,
    output_dir: Path,
    verbose: bool = False
) -> Tuple[object, Dict]:
    """Train a single probe on a specific layer using pre-collected activations."""
    
    from probity.probes.logistic import LogisticProbe, LogisticProbeConfig
    from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
    
    hook_point = f"blocks.{layer}.hook_resid_pre"
    probe_name = f"ntml_multilayer_layer_{layer}"
    
    if verbose:
        print(f"   🎯 Layer {layer}: {hook_point}")
    
    # Determine input size based on model
    if "llama" in model_name.lower():
        if "70b" in model_name.lower():
            input_size = 8192  # Llama 3.1 70B hidden size
        else:
            input_size = 4096  # Llama 3.1 8B hidden size
    else:
        input_size = 768  # GPT-2 hidden size
    
    # Configure probe
    probe_config = LogisticProbeConfig(
        input_size=input_size,
        normalize_weights=True,
        bias=False,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=layer,
        name=probe_name
    )
    
    # Configure trainer
    trainer_config = SupervisedTrainerConfig(
        batch_size=training_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        num_epochs=training_config["num_epochs"],
        patience=training_config["patience"],
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=verbose
    )
    
    # Initialize probe
    probe = LogisticProbe(probe_config)
    probe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize trainer
    trainer = SupervisedProbeTrainer(trainer_config)
    
    # Prepare data and train using the pre-collected activations
    train_loader, val_loader = trainer.prepare_supervised_data(
        activation_store, "target"
    )
    
    history = trainer.train(probe, train_loader, val_loader)
    
    # Save probe in probity format
    probe_type_dir = output_dir / "logistic"
    probe_type_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_type_dir / f"layer_{layer}_probe.json"
    probe.save_json(str(probe_path))
    
    # Calculate final metrics
    final_metrics = {
        "layer": layer,
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else float('inf'),
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else float('inf'),
        "min_val_loss": min(history["val_loss"]) if history["val_loss"] else float('inf'),
        "epochs_trained": len(history["train_loss"]),
        "probe_path": str(probe_path)
    }
    
    if verbose:
        print(f"   ✅ Layer {layer}: Final val loss = {final_metrics['final_val_loss']:.4f}")
    
    return probe, history, final_metrics

def analyze_training_results(
    all_histories: Dict[int, Dict],
    all_metrics: Dict[int, Dict],
    results_name: str,
    output_dir: Path,
    plot_curves: bool = True
) -> Dict:
    """Analyze training results across all layers."""
    
    print(f"\n📊 Analyzing Training Results")
    print("=" * 50)
    
    # Find best layers by different metrics
    best_final_val = min(all_metrics.keys(), key=lambda k: all_metrics[k]["final_val_loss"])
    best_min_val = min(all_metrics.keys(), key=lambda k: all_metrics[k]["min_val_loss"])
    
    # Create summary dataframe
    df_data = []
    for layer in sorted(all_metrics.keys()):
        metrics = all_metrics[layer]
        df_data.append({
            "Layer": layer,
            "Final_Train_Loss": metrics["final_train_loss"],
            "Final_Val_Loss": metrics["final_val_loss"],
            "Min_Val_Loss": metrics["min_val_loss"],
            "Epochs_Trained": metrics["epochs_trained"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Save results
    results_dir = output_dir / "analysis"
    results_dir.mkdir(exist_ok=True)
    
    # Save CSV
    csv_path = results_dir / f"{results_name}_layer_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    # Save detailed results
    detailed_results = {
        "best_layers": {
            "best_final_val_loss": {
                "layer": best_final_val,
                "loss": all_metrics[best_final_val]["final_val_loss"]
            },
            "best_min_val_loss": {
                "layer": best_min_val,
                "loss": all_metrics[best_min_val]["min_val_loss"]
            }
        },
        "layer_metrics": all_metrics,
        "summary_stats": {
            "mean_final_val_loss": df["Final_Val_Loss"].mean(),
            "std_final_val_loss": df["Final_Val_Loss"].std(),
            "best_performing_layers": df.nsmallest(3, "Final_Val_Loss")["Layer"].tolist()
        }
    }
    
    json_path = results_dir / f"{results_name}_detailed_results.json"
    with open(json_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Print summary
    print(f"📈 Best Layer (Final Val Loss): Layer {best_final_val} ({all_metrics[best_final_val]['final_val_loss']:.4f})")
    print(f"📈 Best Layer (Min Val Loss): Layer {best_min_val} ({all_metrics[best_min_val]['min_val_loss']:.4f})")
    print(f"📊 Top 3 Layers: {detailed_results['summary_stats']['best_performing_layers']}")
    print(f"💾 Results saved to: {results_dir}")
    
    # Plot training curves
    if plot_curves and all_histories:
        plot_training_curves(all_histories, all_metrics, results_name, results_dir)
    
    return detailed_results

def plot_training_curves(
    all_histories: Dict[int, Dict],
    all_metrics: Dict[int, Dict],
    results_name: str,
    results_dir: Path
):
    """Generate training curve plots."""
    
    print(f"📈 Generating training curves...")
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: All training curves
    for layer in sorted(all_histories.keys()):
        history = all_histories[layer]
        if history["train_loss"] and history["val_loss"]:
            epochs = range(1, len(history["train_loss"]) + 1)
            ax1.plot(epochs, history["train_loss"], label=f"Layer {layer}", alpha=0.7)
    
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Curves by Layer")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: All validation curves
    for layer in sorted(all_histories.keys()):
        history = all_histories[layer]
        if history["val_loss"]:
            epochs = range(1, len(history["val_loss"]) + 1)
            ax2.plot(epochs, history["val_loss"], label=f"Layer {layer}", alpha=0.7)
    
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss Curves by Layer")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final validation loss by layer
    layers = sorted(all_metrics.keys())
    final_val_losses = [all_metrics[layer]["final_val_loss"] for layer in layers]
    
    bars = ax3.bar(layers, final_val_losses, alpha=0.7)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Final Validation Loss")
    ax3.set_title("Final Validation Loss by Layer")
    ax3.grid(True, alpha=0.3)
    
    # Highlight best layer
    best_layer = min(layers, key=lambda k: all_metrics[k]["final_val_loss"])
    best_idx = layers.index(best_layer)
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)
    
    # Plot 4: Training efficiency (epochs to convergence)
    epochs_trained = [all_metrics[layer]["epochs_trained"] for layer in layers]
    
    ax4.bar(layers, epochs_trained, alpha=0.7, color='green')
    ax4.set_xlabel("Layer")
    ax4.set_ylabel("Epochs Trained")
    ax4.set_title("Training Efficiency (Epochs to Convergence)")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = results_dir / f"{results_name}_training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {plot_path}")
    
    plt.close()

def run_evaluation_on_all_probes(
    probe_dir: Path,
    dataset_path: Path,
    model_name: str,
    results_name: str
):
    """Run probity's evaluation system on all trained probes."""
    
    print(f"\n🧪 Running Evaluation on All Probes")
    print("=" * 50)
    
    eval_results_dir = probe_dir.parent / f"{results_name}_evaluation"
    
    # Use probity's evaluation script
    import subprocess
    import sys
    
    eval_command = [
        sys.executable,
        "scripts/probe_eval.py",
        "--model_name", model_name,
        "--eval_dataset_dir", str(dataset_path),
        "--probe_dir", str(probe_dir),
        "--results_save_dir", str(eval_results_dir),
        "--batch_size", "4"
    ]
    
    print(f"Running: {' '.join(eval_command)}")
    
    try:
        result = subprocess.run(eval_command, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"✅ Evaluation completed successfully")
            print(f"📊 Results saved to: {eval_results_dir}")
            return eval_results_dir
        else:
            print(f"❌ Evaluation failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print(f"❌ Evaluation timed out after 10 minutes")
        return None
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        return None

def main():
    """Main training function with memory-efficient layer-by-layer training."""
    args = parse_args()
    repo_root, probity_dir = setup_paths()
    
    print("🚀 Multi-Layer NTML Probe Training")
    print(f"📁 Dataset: {args.dataset or args.jsonl_path}")
    print(f"🤖 Model: {args.model_name}")
    print(f"🎯 Layers: {args.layers}")
    print(f"📊 Training config: {args.num_epochs} epochs, batch_size={args.batch_size}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = probity_dir / "trained_probes"
    
    output_dir.mkdir(exist_ok=True)
    print(f"💾 Output: {output_dir}")
    
    # Load dataset
    if args.dataset:
        dataset_path = probity_dir / "data" / "NTML-datasets" / f"{args.dataset}.jsonl"
    else:
        dataset_path = Path(args.jsonl_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    tokenized_dataset = load_and_prepare_dataset(dataset_path, repo_root)
    
    # Determine layers to train
    if args.layers == "all":
        if args.model_name == "gpt2":
            layers = list(range(12))
        elif "llama" in args.model_name.lower():
            layers = list(range(32))  # Llama 8B has 32 layers
        else:
            layers = list(range(12))  # Default
    else:
        layers = [int(l.strip()) for l in args.layers.split(",")]
    
    print(f"🎯 Training on layers: {layers}")
    
    # Training configuration
    training_config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience
    }
    
    # Memory-efficient sequential training
    print(f"\n🔧 Training layers sequentially (memory-efficient mode)...")
    
    all_histories = {}
    all_metrics = {}
    
    for layer in tqdm(layers, desc="Training layers"):
        print(f"\n🎯 Training layer {layer}...")
        
        try:
            # Train probe on this layer (loads model fresh each time)
            probe, history, metrics = train_probe_on_layer(
                layer=layer,
                tokenized_dataset=tokenized_dataset,
                model_name=args.model_name,
                training_config=training_config,
                output_dir=output_dir,
                verbose=args.verbose
            )
            
            all_histories[layer] = history
            all_metrics[layer] = metrics
            
            print(f"✅ Layer {layer} completed - Val Loss: {metrics.get('val_loss', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"❌ Error training layer {layer}: {e}")
            continue
    
    # Generate results name
    if args.results_name:
        results_name = args.results_name
    else:
        dataset_name = dataset_path.stem
        results_name = f"ntml_multilayer_{dataset_name}"
    
    # Analyze results
    print(f"\n📊 Analyzing training results...")
    analysis_results = analyze_training_results(
        all_histories=all_histories,
        all_metrics=all_metrics,
        results_name=results_name,
        output_dir=output_dir,
        plot_curves=args.plot_training_curves
    )
    
    # Run evaluation if requested
    if args.eval_after_training:
        print(f"\n🔍 Running evaluation on all trained probes...")
        run_evaluation_on_all_probes(
            probe_dir=output_dir,
            dataset_path=dataset_path,
            model_name=args.model_name,
            results_name=results_name
        )
    
    # Print summary
    print(f"\n🎉 Multi-Layer Training Complete!")
    print(f"📋 Summary:")
    print(f"   • Trained {len(all_metrics)} probes successfully")
    
    if all_metrics:
        best_layer = min(all_metrics.keys(), key=lambda k: all_metrics[k].get('val_loss', float('inf')))
        best_loss = all_metrics[best_layer].get('val_loss', 'N/A')
        print(f"   • Best layer: {best_layer}")
        print(f"   • Best validation loss: {best_loss}")
    
    print(f"   • Results saved to: {output_dir}")

if __name__ == "__main__":
    exit(main()) 