import numpy as np
from pathlib import Path
from typing import Dict

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


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
