import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from probity.probes import BaseProbe
from probity.visualisation.token_highlight import generate_token_visualization, save_token_scores_csv
from utils import load_lie_truth_dataset, get_model_dtype

class MemoryEfficientProbeInference:
    """Memory efficient probe inference using shared model"""
    def __init__(self, model, hook_point, probe, device):
        self.model = model
        self.hook_point = hook_point
        self.probe = probe.to(device)
        self.device = device
        self.probe.eval()
    
    def get_activations_for_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get activations and tokens for a single text"""
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        tokens = tokenizer.encode(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=[self.hook_point],
                return_cache_object=True
            )
            act = cache[self.hook_point]
            if hasattr(self.probe, 'dtype'):
                act = act.to(self.probe.dtype)
            return act, tokens[0]
    
    def get_token_scores_raw(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get raw (unnormalized) token-level scores for texts"""
        token_details = []
        
        for text in texts:
            # Get activations and tokens
            activations, tokens = self.get_activations_for_text(text)
            token_texts = self.model.tokenizer.convert_ids_to_tokens(tokens)
            
            with torch.no_grad():
                token_scores = []
                for i in range(activations.shape[1]):
                    token_act = activations[:, i, :]
                    score = self.probe(token_act.to(self.device))
                    
                    # Apply sigmoid only to LogisticProbe
                    if self.probe.__class__.__name__ == 'LogisticProbe':
                        score = torch.sigmoid(score)
                    
                    token_scores.append(score.cpu().item())
            
            token_detail = {
                "text": text,
                "tokens": token_texts,
                "token_scores": token_scores,
                "mean_score": float(np.mean(token_scores)),
                "max_score": float(np.max(token_scores)),
                "min_score": float(np.min(token_scores))
            }
            token_details.append(token_detail)
        
        return token_details

    def get_scores_raw(self, texts: List[str]) -> np.ndarray:
        """Get raw (unnormalized) mean probe scores for texts"""
        scores = []
        
        for text in texts:
            act, _ = self.get_activations_for_text(text)
            
            with torch.no_grad():
                mean_act = act.mean(dim=1)
                score = self.probe(mean_act.to(self.device))
                
                # Apply sigmoid only to LogisticProbe
                if self.probe.__class__.__name__ == 'LogisticProbe':
                    score = torch.sigmoid(score)

                if score.dim() > 0:
                    score = score.to(torch.float32)
                    score_value = score.squeeze().cpu().numpy().item()
                else:
                    score = score.to(torch.float32)
                    score_value = score.cpu().numpy().item()
                
                scores.append(score_value)
        
        return np.array(scores)

def normalize_to_01(scores: List[float]) -> List[float]:
    """Normalize scores to [0, 1] range"""
    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score == min_score:
        return [0.5] * len(scores)  # Set to neutral value if all scores are identical
    
    # Scale to [0, 1] range
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained probes')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_dataset_dir', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def evaluate_probe(inference: MemoryEfficientProbeInference, texts: List[str], 
                  labels: List[int]) -> Dict[str, Any]:
    """Evaluate a single probe and get detailed token-level results"""
    all_samples = []
    all_token_scores = []  # Collect all raw token scores for global normalization
    
    # First pass: collect all raw scores
    for idx, (text, true_label) in enumerate(zip(texts, labels)):
        # Get activations and tokens
        act, tokens = inference.get_activations_for_text(text)
        token_texts = inference.model.tokenizer.convert_ids_to_tokens(tokens)
        
        # Get raw score for each token
        token_scores = []
        with torch.no_grad():
            for i in range(act.shape[1]):
                token_act = act[:, i, :]
                score = inference.probe(token_act.to(inference.device))
                
                # Apply sigmoid only to LogisticProbe
                if inference.probe.__class__.__name__ == 'LogisticProbe':
                    score = torch.sigmoid(score)
                
                token_scores.append(score.cpu().item())
        
        all_token_scores.extend(token_scores)
        all_samples.append({
            "idx": idx,
            "text": text,
            "true_label": true_label,
            "tokens": token_texts,
            "raw_token_scores": token_scores
        })
    
    # Global normalization to [0,1] for non-logistic probes
    if inference.probe.__class__.__name__ == 'LogisticProbe':
        # LogisticProbe already outputs [0,1] after sigmoid, no normalization needed
        for sample in all_samples:
            sample["token_scores"] = sample["raw_token_scores"]
    else:
        # Normalize all scores globally to [0,1]
        normalized_scores = normalize_to_01(all_token_scores)
        
        # Distribute normalized scores back to samples
        score_idx = 0
        for sample in all_samples:
            num_tokens = len(sample["raw_token_scores"])
            sample["token_scores"] = normalized_scores[score_idx:score_idx + num_tokens]
            score_idx += num_tokens
    
    # Second pass: calculate metrics with normalized scores
    final_samples = []
    all_mean_scores = []
    
    for sample in all_samples:
        mean_score = np.mean(sample["token_scores"])
        all_mean_scores.append(mean_score)
        
        sample_info = {
            "sample_id": sample["idx"],
            "text": sample["text"],
            "true_label": int(sample["true_label"]),  # Convert to native Python int
            "tokens": sample["tokens"],
            "token_scores": sample["token_scores"],
            "mean_score": float(mean_score),
            "predicted_label": int(1 if mean_score > 0.5 else 0),  # Convert to native Python int
            "is_correct": bool((mean_score > 0.5) == (sample["true_label"] == 1)),  # Convert to native Python bool
        }
        final_samples.append(sample_info)
    
    # Convert to numpy array for metrics
    mean_scores = np.array(all_mean_scores)
    predictions = (mean_scores > 0.5).astype(int)
    
    # Calculate metrics
    # High probe activation (>0.5) should indicate a lie (label=1)
    # Low probe activation (≤0.5) should indicate truth (label=0)
    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions)),  # Precision for detecting lies
        'recall': float(recall_score(labels, predictions)),  # Recall for detecting lies
        'f1': float(f1_score(labels, predictions)),
        'auroc': float(roc_auc_score(labels, mean_scores))
    }
    
    # Add interpretation of metrics
    metrics_interpretation = {
        "accuracy": "Fraction of correctly classified truth/lie statements",
        "precision": "Of statements classified as lies, fraction that were actually lies",
        "recall": "Of actual lies, fraction that were detected",
        "f1": "Harmonic mean of precision and recall for lie detection",
        "auroc": "Area under ROC curve (1.0 = perfect separation of truths and lies)"
    }
    
    # Prepare token_details for visualization (matching expected format)
    token_details = []
    for sample in final_samples:
        # Clean LLaMA tokens for visualization
        clean_tokens = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in sample["tokens"]]
        
        token_detail = {
            "text": sample["text"],
            "label": int(sample["true_label"]),  # Convert to native Python int
            "tokens": clean_tokens,
            "token_scores": sample["token_scores"],
            "mean_score": sample["mean_score"],
            "max_score": float(np.max(sample["token_scores"])),
            "min_score": float(np.min(sample["token_scores"]))
        }
        token_details.append(token_detail)
    
    return {
        'metrics': metrics,
        'metrics_interpretation': metrics_interpretation,
        'all_samples': final_samples,  # Complete per-token scores for every sample
        'token_details': token_details,  # For visualization
        'mean_scores': mean_scores.tolist(),
        'predictions': predictions.tolist()
    }

def create_probe_type_plots(probe_results: Dict, save_dir: Path, probe_type: str):
    """Create plots for a specific probe type."""
    layers = sorted(probe_results.keys())
    
    # Plot metrics across layers
    plt.figure(figsize=(10, 6))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc']
    for metric in metrics:
        values = [probe_results[layer]['metrics'][metric] for layer in layers]
        plt.plot(layers, values, marker='o', label=metric.upper())
    
    plt.xlabel('Layer')
    plt.ylabel('Score')
    plt.title(f'{probe_type} Metrics Across Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'metrics_by_layer.png', dpi=300)
    plt.close()
    
    # AUROC curves for all layers of this probe type
    plt.figure(figsize=(10, 8))
    for layer in layers:
        if layer in probe_results:
            labels = [sample['true_label'] for sample in probe_results[layer]['all_samples']]
            scores = [sample['mean_score'] for sample in probe_results[layer]['all_samples']]
            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = probe_results[layer]['metrics']['auroc']
            plt.plot(fpr, tpr, label=f'Layer {layer} (AUROC: {auroc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{probe_type} ROC Curves - All Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves_all_layers.png', dpi=300)
    plt.close()
    
    # Score distribution plots for best layer
    best_layer = max(layers, key=lambda l: probe_results[l]['metrics']['auroc'])
    best_results = probe_results[best_layer]
    
    # Get scores and labels
    truth_scores = [sample['mean_score'] for sample in best_results['all_samples'] if sample['true_label'] == 0]
    lie_scores = [sample['mean_score'] for sample in best_results['all_samples'] if sample['true_label'] == 1]
    all_scores = [sample['mean_score'] for sample in best_results['all_samples']]
    all_labels = [sample['true_label'] for sample in best_results['all_samples']]
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Create distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram with 0.5 threshold
    ax = axes[0, 0]
    ax.hist(truth_scores, bins=30, alpha=0.7, label='Truth', color='blue', density=True)
    ax.hist(lie_scores, bins=30, alpha=0.7, label='Lie', color='red', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax.set_xlabel('Mean Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{probe_type} Layer {best_layer} - Score Distribution (0.5 Threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram with optimal threshold
    ax = axes[0, 1]
    ax.hist(truth_scores, bins=30, alpha=0.7, label='Truth', color='blue', density=True)
    ax.hist(lie_scores, bins=30, alpha=0.7, label='Lie', color='red', density=True)
    ax.axvline(x=optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax.set_xlabel('Mean Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{probe_type} Layer {best_layer} - Score Distribution (Optimal Threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Violin plot with 0.5 threshold
    ax = axes[1, 0]
    violin_data = [truth_scores, lie_scores]
    parts = ax.violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)
    ax.axhline(y=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Truth', 'Lie'])
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{probe_type} Layer {best_layer} - Score Distribution (0.5 Threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Violin plot with optimal threshold
    ax = axes[1, 1]
    parts = ax.violinplot(violin_data, positions=[0, 1], showmeans=True, showmedians=True)
    ax.axhline(y=optimal_threshold, color='green', linestyle='--', label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Truth', 'Lie'])
    ax.set_ylabel('Mean Score')
    ax.set_title(f'{probe_type} Layer {best_layer} - Score Distribution (Optimal Threshold)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'score_distributions.png', dpi=300)
    plt.close()

def create_performance_plots(results: Dict, save_dir: Path):
    """Create performance visualization plots"""
    probe_types = list(next(iter(results.values())).keys())
    layers = sorted(results.keys())
    
    # Create AUROC heatmap with larger dimensions
    plt.figure(figsize=(max(16, len(layers) * 0.8), max(10, len(probe_types) * 1.2)))
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
                cmap='viridis',
                annot_kws={'size': max(8, min(12, 100 // len(layers)))})  # Adjust font size based on number of layers
    plt.title('AUROC Scores Across Layers and Probe Types', fontsize=16)
    plt.xlabel('Layer', fontsize=14)
    plt.ylabel('Probe Type', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_dir / 'auroc_heatmap.png', dpi=300)
    plt.close()
    
    # Create performance comparison plots
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
    
    # Create ROC curves for best layer of each probe type
    plt.figure(figsize=(10, 8))
    
    for probe_type in probe_types:
        # Find best layer for this probe type
        best_auroc = -1
        best_layer = -1
        best_results = None
        
        for layer in layers:
            if probe_type in results[layer]:
                auroc = results[layer][probe_type]['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
                    best_results = results[layer][probe_type]
        
        if best_results is not None:
            # Get labels and scores
            labels = [sample['true_label'] for sample in best_results['all_samples']]
            scores = [sample['mean_score'] for sample in best_results['all_samples']]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(labels, scores)
            plt.plot(fpr, tpr, label=f'{probe_type} Layer {best_layer} (AUROC: {best_auroc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Best Layer for Each Probe Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curves_best_layers.png', dpi=300)
    plt.close()
    
    # Create overall score distribution comparison
    fig, axes = plt.subplots(len(probe_types), 2, figsize=(15, 6 * len(probe_types)))
    if len(probe_types) == 1:
        axes = axes.reshape(1, -1)
    
    for i, probe_type in enumerate(probe_types):
        # Find best layer for this probe type
        best_auroc = -1
        best_layer = -1
        best_results = None
        
        for layer in layers:
            if probe_type in results[layer]:
                auroc = results[layer][probe_type]['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
                    best_results = results[layer][probe_type]
        
        if best_results is not None:
            # Get scores by label
            truth_scores = [sample['mean_score'] for sample in best_results['all_samples'] if sample['true_label'] == 0]
            lie_scores = [sample['mean_score'] for sample in best_results['all_samples'] if sample['true_label'] == 1]
            all_scores = [sample['mean_score'] for sample in best_results['all_samples']]
            all_labels = [sample['true_label'] for sample in best_results['all_samples']]
            
            # Find optimal threshold
            fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            # Histogram with 0.5 threshold
            ax = axes[i, 0]
            ax.hist(truth_scores, bins=30, alpha=0.7, label='Truth', color='blue', density=True)
            ax.hist(lie_scores, bins=30, alpha=0.7, label='Lie', color='red', density=True)
            ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
            ax.set_xlabel('Mean Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{probe_type} Layer {best_layer} - 0.5 Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Histogram with optimal threshold
            ax = axes[i, 1]
            ax.hist(truth_scores, bins=30, alpha=0.7, label='Truth', color='blue', density=True)
            ax.hist(lie_scores, bins=30, alpha=0.7, label='Lie', color='red', density=True)
            ax.axvline(x=optimal_threshold, color='green', linestyle='--', 
                      label=f'Optimal Threshold ({optimal_threshold:.3f})')
            ax.set_xlabel('Mean Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{probe_type} Layer {best_layer} - Optimal Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'overall_score_distributions.png', dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset from {args.eval_dataset_dir}")
    dataset = load_lie_truth_dataset(args.eval_dataset_dir)
    
    # Extract texts and labels
    texts = [ex.text for ex in dataset.examples]
    labels = [ex.label for ex in dataset.examples]
    
    # Create results directory
    results_dir = Path(args.results_save_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name,
        device=args.device,
        dtype=model_dtype,
    )
    
    results = {}
    probe_dir = Path(args.probe_dir)
    
    # Iterate through probe types
    for probe_type_dir in probe_dir.iterdir():
        if not probe_type_dir.is_dir():
            continue
        
        probe_type = probe_type_dir.name
        print(f"\nEvaluating {probe_type} probes")
        
        probe_type_results = {}
        probe_type_csv_data = []
        
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
            evaluation_results = evaluate_probe(inference, texts, labels)
            
            # Store results
            if layer not in results:
                results[layer] = {}
            
            results[layer][probe_type] = evaluation_results
            probe_type_results[layer] = evaluation_results
            
            # Add to CSV data
            row = {'layer': layer}
            row.update(evaluation_results['metrics'])
            probe_type_csv_data.append(row)
            
            # Save individual layer results with complete sample information
            layer_dir = results_dir / probe_type / f"layer_{layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)
            
            # Save full results including all samples
            with open(layer_dir / 'full_results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            
            # Save just the sample-level data for easier analysis
            with open(layer_dir / 'sample_token_scores.json', 'w') as f:
                json.dump(evaluation_results['all_samples'], f, indent=2)
            
            # Save detailed results
            with open(layer_dir / 'results.json', 'w') as f:
                json.dump(evaluation_results, f, indent=2)

            
            # Generate visualization
            generate_token_visualization(evaluation_results['token_details'], 
                                      layer_dir / 'token_visualization.html')
            
            # Save token scores as CSV
            save_token_scores_csv(evaluation_results['token_details'], 
                                layer_dir / 'token_scores.csv')
        
        # Save probe type results
        probe_type_dir = results_dir / probe_type
        probe_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all layers results for this probe type
        with open(probe_type_dir / 'all_layers_results.json', 'w') as f:
            json.dump(probe_type_results, f, indent=2)
        
        # Save CSV for this probe type
        df = pd.DataFrame(probe_type_csv_data)
        df.to_csv(probe_type_dir / 'metrics.csv', index=False)
        print(f"Saved {probe_type} results to {probe_type_dir}")
        
        # Create plots for this probe type
        create_probe_type_plots(probe_type_results, probe_type_dir, probe_type)
    
    # Save final combined results
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create overall visualization plots
    create_performance_plots(results, results_dir)
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    
    # Save best configurations
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
    
    with open(results_dir / 'best_configurations.json', 'w') as f:
        json.dump(best_configs, f, indent=2)

if __name__ == "__main__":
    main()