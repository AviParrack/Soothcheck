import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from probity.probes import BaseProbe
from probity.visualisation.token_highlight import generate_token_visualization, save_token_scores_csv
from utils import load_lie_truth_dataset, get_model_dtype


class OptimizedBatchProbeEvaluator:
    """Optimized evaluator that runs model once and applies all probes"""
    
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name
        
        # Load model once
        print(f"Loading model {model_name}")
        model_dtype = get_model_dtype(model_name)
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=model_dtype,
        )
        self.model.eval()
        
        # Cache for activations
        self._activation_cache = {}
        
    def get_batch_activations(self, texts: List[str], layers: List[int], 
                            batch_size: int = 8) -> Dict[int, torch.Tensor]:
        """Get activations for all texts and layers efficiently"""
        
        # Create cache key
        cache_key = (tuple(sorted(texts)), tuple(sorted(layers)))
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        
        hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
        
        # Tokenize all texts at once
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        # First, tokenize all texts to find the maximum length
        print("Tokenizing all texts to determine max length...")
        all_text_tokens = []
        max_length = 0
        
        for text in texts:
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            seq_len = tokens["input_ids"].shape[1]
            max_length = max(max_length, seq_len)
            all_text_tokens.append(tokens["input_ids"][0])
        
        # Cap max_length for memory efficiency
        max_length = min(max_length, 512)
        print(f"Using max_length: {max_length}")
        
        # Process in batches with consistent padding
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize batch with fixed max_length for consistency
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding="max_length",  # Use max_length padding
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Run model with caching
                _, cache = self.model.run_with_cache(
                    tokens["input_ids"],
                    names_filter=hook_points,
                    return_cache_object=True,
                    stop_at_layer=max(layers) + 1
                )
                
                # Store activations for each layer
                for layer in layers:
                    hook_point = f"blocks.{layer}.hook_resid_pre"
                    all_activations[layer].append(cache[hook_point].cpu())
                
                # Store tokens for later use (with original lengths, not padded)
                for j, text in enumerate(batch_texts):
                    text_tokens = tokens["input_ids"][j]
                    # Find actual length (before padding)
                    if tokenizer.pad_token_id is not None:
                        actual_length = (text_tokens != tokenizer.pad_token_id).sum().item()
                    else:
                        actual_length = len(text_tokens)
                    
                    actual_tokens = text_tokens[:actual_length]
                    token_texts = tokenizer.convert_ids_to_tokens(actual_tokens)
                    all_tokens.append(token_texts)
        
        # Concatenate all batches (now they should have consistent dimensions)
        final_activations = {}
        for layer in layers:
            final_activations[layer] = torch.cat(all_activations[layer], dim=0)
        
        # Cache results
        result = {
            'activations': final_activations,
            'tokens_by_text': all_tokens
        }
        self._activation_cache[cache_key] = result
        
        return result
    
    def evaluate_all_probes(self, texts: List[str], labels: List[int], 
                          probe_configs: Dict[Tuple[int, str], BaseProbe]) -> Dict[Tuple[int, str], Dict]:
        """Evaluate all probes efficiently using cached activations"""
        
        # Extract unique layers from probe configs
        layers = list(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations for all required layers at once
        print("Getting activations for all layers...")
        activation_data = self.get_batch_activations(texts, layers)
        activations = activation_data['activations']
        tokens_by_text = activation_data['tokens_by_text']
        
        results = {}
        
        # Group probes by layer for efficient processing
        probes_by_layer = {}
        for (layer, probe_type), probe in probe_configs.items():
            if layer not in probes_by_layer:
                probes_by_layer[layer] = {}
            probes_by_layer[layer][probe_type] = probe
        
        # Process each layer
        for layer in tqdm(layers, desc="Processing layers"):
            layer_activations = activations[layer]  # Shape: [num_texts, seq_len, hidden_size]
            layer_probes = probes_by_layer[layer]
            
            # Calculate mean activations once for this layer
            mean_activations = layer_activations.mean(dim=1)  # Shape: [num_texts, hidden_size]
            
            # Apply all probes for this layer
            for probe_type, probe in layer_probes.items():
                print(f"Evaluating {probe_type} probe on layer {layer}")
                
                # Get probe scores efficiently
                probe_results = self._evaluate_single_probe_batch(
                    probe, layer_activations, mean_activations, 
                    texts, labels, tokens_by_text
                )
                
                results[(layer, probe_type)] = probe_results
        
        return results
    
    def _evaluate_single_probe_batch(self, probe: BaseProbe, 
                                   layer_activations: torch.Tensor,
                                   mean_activations: torch.Tensor,
                                   texts: List[str], labels: List[int],
                                   tokens_by_text: List[List[str]]) -> Dict:
        """Evaluate a single probe on batch data"""
        
        probe = probe.to(self.device)
        probe.eval()
        
        # Get token-level scores for all texts at once
        all_token_scores = []
        all_samples = []
        
        with torch.no_grad():
            # Process each text individually to handle variable lengths properly
            for i, (text, true_label) in enumerate(zip(texts, labels)):
                # Get tokens and actual length for this text
                tokens = tokens_by_text[i]
                actual_length = len(tokens)
                
                # Get activations for this text (up to actual length)
                text_activations = layer_activations[i, :actual_length, :].to(self.device)
                
                # Apply probe to all tokens for this text
                token_scores = probe(text_activations)
                
                # Apply sigmoid only to LogisticProbe
                if probe.__class__.__name__ == 'LogisticProbe':
                    token_scores = torch.sigmoid(token_scores)
                
                # Convert to list
                token_scores_list = token_scores.cpu().squeeze().tolist()
                
                # Handle single token case
                if isinstance(token_scores_list, float):
                    token_scores_list = [token_scores_list]
                
                # Ensure we have the right number of scores
                if len(token_scores_list) != actual_length:
                    print(f"Warning: Score length mismatch for text {i}: {len(token_scores_list)} vs {actual_length}")
                    # Pad or truncate as needed
                    if len(token_scores_list) < actual_length:
                        token_scores_list.extend([0.0] * (actual_length - len(token_scores_list)))
                    else:
                        token_scores_list = token_scores_list[:actual_length]
                
                all_token_scores.extend(token_scores_list)
                
                sample_info = {
                    "idx": i,
                    "text": text,
                    "true_label": true_label,
                    "tokens": tokens,
                    "raw_token_scores": token_scores_list
                }
                all_samples.append(sample_info)
        
        # Global normalization for non-logistic probes
        if probe.__class__.__name__ == 'LogisticProbe':
            # LogisticProbe already outputs [0,1] after sigmoid
            for sample in all_samples:
                sample["token_scores"] = sample["raw_token_scores"]
        else:
            # Normalize all scores globally to [0,1]
            normalized_scores = self._normalize_to_01(all_token_scores)
            
            # Distribute normalized scores back to samples
            score_idx = 0
            for sample in all_samples:
                num_tokens = len(sample["raw_token_scores"])
                sample["token_scores"] = normalized_scores[score_idx:score_idx + num_tokens]
                score_idx += num_tokens
        
        # Calculate final metrics
        final_samples = []
        all_mean_scores = []
        
        for sample in all_samples:
            mean_score = np.mean(sample["token_scores"])
            all_mean_scores.append(mean_score)
            
            sample_info = {
                "sample_id": sample["idx"],
                "text": sample["text"],
                "true_label": int(sample["true_label"]),
                "tokens": sample["tokens"],
                "token_scores": sample["token_scores"],
                "mean_score": float(mean_score),
                "predicted_label": int(1 if mean_score > 0.5 else 0),
                "is_correct": bool((mean_score > 0.5) == (sample["true_label"] == 1)),
            }
            final_samples.append(sample_info)
        
        # Calculate metrics
        mean_scores = np.array(all_mean_scores)
        predictions = (mean_scores > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions)),
            'recall': float(recall_score(labels, predictions)),
            'f1': float(f1_score(labels, predictions)),
            'auroc': float(roc_auc_score(labels, mean_scores))
        }
        
        # Prepare token_details for visualization
        token_details = []
        for sample in final_samples:
            # Clean tokens for visualization
            clean_tokens = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in sample["tokens"]]
            
            token_detail = {
                "text": sample["text"],
                "label": int(sample["true_label"]),
                "tokens": clean_tokens,
                "token_scores": sample["token_scores"],
                "mean_score": sample["mean_score"],
                "max_score": float(np.max(sample["token_scores"])),
                "min_score": float(np.min(sample["token_scores"]))
            }
            token_details.append(token_detail)
        
        return {
            'metrics': metrics,
            'metrics_interpretation': {
                "accuracy": "Fraction of correctly classified truth/lie statements",
                "precision": "Of statements classified as lies, fraction that were actually lies",
                "recall": "Of actual lies, fraction that were detected",
                "f1": "Harmonic mean of precision and recall for lie detection",
                "auroc": "Area under ROC curve (1.0 = perfect separation of truths and lies)"
            },
            'all_samples': final_samples,
            'token_details': token_details,
            'mean_scores': mean_scores.tolist(),
            'predictions': predictions.tolist()
        }
    
    def _normalize_to_01(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()


def create_probe_type_plots(probe_results: Dict, save_dir: Path, probe_type: str):
    """Create plots for a specific probe type."""
    layers = sorted(probe_results.keys())

    if not layers:
        print(f"No layers found for {probe_type}, skipping plots...")
        return
    
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
    # Reorganize results by layer and probe type
    layer_results = {}
    for (layer, probe_type), result in results.items():
        if layer not in layer_results:
            layer_results[layer] = {}
        layer_results[layer][probe_type] = result
    
    probe_types = list(set(probe_type for _, probe_type in results.keys()))
    layers = sorted(set(layer for layer, _ in results.keys()))

    if not layers:
        print("No layers found, skipping plots...")
        return
    
    # Create AUROC heatmap
    plt.figure(figsize=(max(16, len(layers) * 0.8), max(10, len(probe_types) * 1.2)))
    auroc_matrix = np.zeros((len(probe_types), len(layers)))
    
    for i, probe_type in enumerate(probe_types):
        for j, layer in enumerate(layers):
            if layer in layer_results and probe_type in layer_results[layer]:
                auroc_matrix[i, j] = layer_results[layer][probe_type]['metrics']['auroc']
    
    sns.heatmap(auroc_matrix, 
                xticklabels=layers, 
                yticklabels=probe_types,
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                annot_kws={'size': max(8, min(12, 100 // len(layers)))})
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
            values = [layer_results[layer][probe_type]['metrics'][metric] 
                     for layer in layers if probe_type in layer_results[layer]]
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
            if layer in layer_results and probe_type in layer_results[layer]:
                auroc = layer_results[layer][probe_type]['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
                    best_results = layer_results[layer][probe_type]
        
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
            if layer in layer_results and probe_type in layer_results[layer]:
                auroc = layer_results[layer][probe_type]['metrics']['auroc']
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_layer = layer
                    best_results = layer_results[layer][probe_type]
        
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


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained probes')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--eval_dataset_dir', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for activation collection')
    return parser.parse_args()


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
    
    # Create optimized evaluator
    evaluator = OptimizedBatchProbeEvaluator(args.model_name, args.device)
    
    # Load all probes
    probe_configs = {}
    probe_dir = Path(args.probe_dir)
    
    print("Loading all probes...")
    for probe_type_dir in probe_dir.iterdir():
        if not probe_type_dir.is_dir():
            continue

        if probe_type_dir.name.startswith('.') or probe_type_dir.name in ['__pycache__', '.ipynb_checkpoints']:
            continue
        
        probe_type = probe_type_dir.name
        probe_files = sorted(probe_type_dir.glob("layer_*_probe.json"))

        if not probe_files:
            print(f"No probe files found in {probe_type_dir}, skipping...")
            continue
        
        for probe_file in probe_files:
            # Extract layer number
            layer = int(probe_file.stem.split('_')[1])
            
            # Load probe
            probe = BaseProbe.load_json(str(probe_file))
            probe_configs[(layer, probe_type)] = probe
    
    print(f"Loaded {len(probe_configs)} probes")
    
    # Evaluate all probes efficiently
    print("Running optimized evaluation...")
    all_results = evaluator.evaluate_all_probes(texts, labels, probe_configs)
    
    # Reorganize results and save
    print("Saving results...")
    results_by_type = {}
    
    for (layer, probe_type), result in all_results.items():
        if probe_type not in results_by_type:
            results_by_type[probe_type] = {}
        results_by_type[probe_type][layer] = result
        
        # Save individual layer results
        layer_dir = results_dir / probe_type / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full results
        with open(layer_dir / 'full_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # Generate visualization
        generate_token_visualization(result['token_details'], 
                                  layer_dir / 'token_visualization.html')
        
        # Save token scores as CSV
        save_token_scores_csv(result['token_details'], 
                            layer_dir / 'token_scores.csv')
    
    # Save probe type summaries and create plots
    for probe_type, probe_results in results_by_type.items():
        probe_type_dir = results_dir / probe_type
        probe_type_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all layers results
        with open(probe_type_dir / 'all_layers_results.json', 'w') as f:
            json.dump(probe_results, f, indent=2)
        
        # Save CSV
        csv_data = []
        for layer, result in probe_results.items():
            row = {'layer': layer}
            row.update(result['metrics'])
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(probe_type_dir / 'metrics.csv', index=False)
        
        # Create plots
        create_probe_type_plots(probe_results, probe_type_dir, probe_type)
    
    # Save final combined results
    json_results = {f"{layer}_{probe_type}": result for (layer, probe_type), result in all_results.items()}
    with open(results_dir / 'all_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Create overall visualization plots
    create_performance_plots(all_results, results_dir)
    print(f"\nEvaluation complete. Results saved to {results_dir}")
    
    # Save best configurations
    best_configs = []
    probe_types = set(probe_type for _, probe_type in all_results.keys())
    
    for probe_type in probe_types:
        best_layer = -1
        best_auroc = -1
        
        for (layer, pt), result in all_results.items():
            if pt == probe_type:
                auroc = result['metrics']['auroc']
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