import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from probity.probes import BaseProbe
from utils import load_lie_truth_dataset, get_model_dtype

class MemoryEfficientProbeInference:
    """Memory efficient probe inference using shared model"""
    def __init__(self, model, hook_point, probe, device):
        self.model = model
        self.hook_point = hook_point
        self.probe = probe.to(device)
        self.device = device
        self.probe.eval()
    
    def get_activations_for_text(self, text: str) -> torch.Tensor:
        """Get activations for a single text"""
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        tokens = tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        # Use hook to get activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=[self.hook_point],
                return_cache_object=True
            )
            act = cache[self.hook_point]
            # Ensure activation dtype matches probe dtype
            if hasattr(self.probe, 'dtype'):
                act = act.to(self.probe.dtype)
            return act
    
    def get_scores(self, texts: List[str]) -> np.ndarray:
        """Get probe scores for texts"""
        scores = []
        
        for text in texts:
            # Get activations for this text
            act = self.get_activations_for_text(text)
            
            with torch.no_grad():
                # Take mean activation across sequence
                # Shape: [1, seq_len, hidden_dim] -> [1, hidden_dim]
                mean_act = act.mean(dim=1)  # Average over sequence dimension
                
                # Get probe output
                if hasattr(self.probe, 'forward'):
                    score = self.probe.forward(mean_act.to(self.device))
                else:
                    score = self.probe(mean_act.to(self.device))
                
                # Apply sigmoid for logistic probes
                if hasattr(score, 'sigmoid'):
                    score = torch.sigmoid(score)
                elif self.probe.__class__.__name__ == 'LogisticProbe':
                    score = torch.sigmoid(score)

                # Handle different output shapes
                if score.dim() > 0:
                    if score.dtype == torch.bfloat16:
                        score = score.to(torch.float32)
                    score_value = score.squeeze().cpu().numpy().item()
                else:
                    if score.dtype == torch.bfloat16:
                        score = score.to(torch.float32)
                    score_value = score.cpu().numpy().item()
                
                scores.append(score_value)
        
        return np.array(scores)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained probes')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_dataset_dir', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def evaluate_probe(inference: MemoryEfficientProbeInference, texts: List[str], 
                  labels: List[int]) -> Dict[str, float]:
    """Evaluate a single probe"""
    scores = inference.get_scores(texts)
    
    # For directional probes, normalize scores to 0-1
    if scores.min() < 0 or scores.max() > 1:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    # Binary predictions
    predictions = (scores > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'auroc': roc_auc_score(labels, scores)
    }
    
    return metrics, scores, predictions


def create_performance_plots(results: Dict, save_dir: Path):
    """Create performance visualization plots"""
    # Prepare data for plotting
    probe_types = list(next(iter(results.values())).keys())
    layers = sorted(results.keys())
    
    # Create AUROC heatmap
    plt.figure(figsize=(12, 8))
    auroc_matrix = np.zeros((len(probe_types), len(layers)))
    
    for i, probe_type in enumerate(probe_types):
        for j, layer in enumerate(layers):
            if probe_type in results[layer]:
                auroc_matrix[i, j] = results[layer][probe_type]['metrics']['auroc']
    
    sns.heatmap(auroc_matrix, 
                xticklabels=layers, 
                yticklabels=probe_types,
                annot=True, 
                fmt='.3f',
                cmap='viridis')
    plt.title('AUROC Scores Across Layers and Probe Types')
    plt.xlabel('Layer')
    plt.ylabel('Probe Type')
    plt.tight_layout()
    plt.savefig(save_dir / 'auroc_heatmap.png', dpi=300)
    plt.close()
    
    # Create performance comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for probe_type in probe_types:
            values = [results[layer][probe_type]['metrics'][metric] 
                     for layer in layers if probe_type in results[layer]]
            ax.plot(layers, values, marker='o', label=probe_type)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_comparison.png', dpi=300)
    plt.close()


def main():
    args = parse_args()
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset from {args.eval_dataset_dir}")
    dataset = load_lie_truth_dataset(args.eval_dataset_dir)
    
    # Extract texts and labels
    texts = [ex.text for ex in dataset.examples]
    labels = [ex.label for ex in dataset.examples]
    
    # Load model once
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=model_dtype,
    )
    
    # Find all probe files
    probe_dir = Path(args.probe_dir)
    results = {}
    
    # Iterate through probe types
    for probe_type_dir in probe_dir.iterdir():
        if not probe_type_dir.is_dir():
            continue
        
        probe_type = probe_type_dir.name
        print(f"\nEvaluating {probe_type} probes")
        
        # Find all layer probe files
        probe_files = sorted(probe_type_dir.glob("layer_*_probe.json"))
        
        for probe_file in tqdm(probe_files, desc=f"{probe_type} probes"):
            # Extract layer number
            layer = int(probe_file.stem.split('_')[1])
            
            # Load probe
            probe = BaseProbe.load_json(str(probe_file))
            hook_point = f"blocks.{layer}.hook_resid_pre"
            
            # Create inference object
            inference = MemoryEfficientProbeInference(
                model, hook_point, probe, args.device
            )
            
            # Evaluate
            metrics, scores, predictions = evaluate_probe(inference, texts, labels)
            
            # Store results
            if layer not in results:
                results[layer] = {}
            
            results[layer][probe_type] = {
                'metrics': metrics,
                'scores': scores.tolist(),
                'predictions': predictions.tolist()
            }
    
    # Save results
    results_dir = Path(args.results_save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results as JSON
    with open(results_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary CSV for each probe type
    for probe_type in set(pt for layer_results in results.values() for pt in layer_results.keys()):
        csv_data = []
        for layer in sorted(results.keys()):
            if probe_type in results[layer]:
                row = {'layer': layer}
                row.update(results[layer][probe_type]['metrics'])
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = results_dir / f'{probe_type}_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved {probe_type} results to {csv_path}")
    
    # Create visualization plots
    create_performance_plots(results, results_dir)
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    
    # Print best performing configurations
    print("\nBest performing configurations:")
    best_configs = []
    
    for probe_type in set(pt for layer_results in results.values() for pt in layer_results.keys()):
        best_layer = -1
        best_auroc = -1
        
        for layer in results.keys():
            if probe_type in results[layer]:
                auroc = results[layer][probe_type]['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
        
        best_configs.append({
            'probe_type': probe_type,
            'layer': best_layer,
            'auroc': best_auroc
        })
        print(f"{probe_type}: Layer {best_layer} (AUROC: {best_auroc:.4f})")
    
    # Save best configs
    with open(results_dir / 'best_configurations.json', 'w') as f:
        json.dump(best_configs, f, indent=2)


if __name__ == "__main__":
    main()