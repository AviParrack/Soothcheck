#!/usr/bin/env python3
"""
Comprehensive evaluation script for NTML-trained probes.
Evaluates probe performance and generates visualizations.
"""

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict, Tuple, Any

# Add probity to path
import sys
sys.path.append('.')

from probity_extensions import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes.logistic import LogisticProbe
from probity.collection.activation_store import ActivationStore


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NTML-trained probe')
    parser.add_argument('--dataset', type=str, default='2T1L_20samples', 
                       help='Dataset name (default: 2T1L_20samples)')
    parser.add_argument('--probe_path', type=str, 
                       default='trained_probes/ntml_2T1L_20samples_blocks_6_hook_resid_pre.pt',
                       help='Path to trained probe')
    parser.add_argument('--model_name', type=str, default='gpt2', 
                       help='Model name (default: gpt2)')
    parser.add_argument('--hook_point', type=str, default='blocks.6.hook_resid_pre',
                       help='Hook point used during training')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results and plots')
    parser.add_argument('--test_split', type=float, default=0.2,
                       help='Fraction of data to use for testing (default: 0.2)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length (default: 512)')
    return parser.parse_args()


def load_probe_and_model(probe_path: str, model_name: str, device: str) -> Tuple[LogisticProbe, HookedTransformer]:
    """Load the trained probe and model."""
    print(f"Loading probe from: {probe_path}")
    probe = LogisticProbe.load(probe_path)
    probe.to(device)
    
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    
    return probe, model


def create_test_dataset(dataset_name: str, test_split: float, max_length: int) -> TokenizedProbingDataset:
    """Create test dataset from the original NTML dataset."""
    print(f"Loading dataset: {dataset_name}")
    
    # Load conversational dataset
    dataset_path = f"data/NTML-datasets/{dataset_name}.jsonl"
    conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(dataset_path)
    
    # Convert to statement-level dataset
    stmt_dataset = conv_dataset.get_statement_dataset()
    print(f"Created statement dataset with {len(stmt_dataset.examples)} examples")
    
    # Tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=stmt_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True
    )
    
    # Split dataset (use same split as training for consistency)
    total_examples = len(tokenized_dataset.examples)
    train_size = int(total_examples * (1 - test_split))
    
    # Create test dataset with remaining examples
    test_examples = tokenized_dataset.examples[train_size:]
    test_dataset = TokenizedProbingDataset(
        examples=test_examples,
        tokenization_config=tokenized_dataset.tokenization_config,
        position_types=tokenized_dataset.position_types,
        task_type=tokenized_dataset.task_type,
        label_mapping=tokenized_dataset.label_mapping,
        dataset_attributes=tokenized_dataset.dataset_attributes
    )
    
    print(f"Test dataset: {len(test_dataset.examples)} examples")
    return test_dataset


def evaluate_probe(probe: LogisticProbe, model: HookedTransformer, 
                  dataset: TokenizedProbingDataset, hook_point: str,
                  device: str) -> Dict[str, Any]:
    """Evaluate the probe on the test dataset."""
    print(f"Evaluating probe on {len(dataset.examples)} examples...")
    
    predictions = []
    probabilities = []
    true_labels = []
    statement_texts = []
    
    with torch.no_grad():
        for i, example in enumerate(dataset.examples):
            # Get tokens and target position
            tokens = torch.tensor(example.tokens).unsqueeze(0).to(device)
            target_pos = example.token_positions.positions['target']
            
            # Get model activations
            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            activations = cache[hook_point][0, target_pos, :].unsqueeze(0)  # Shape: (1, hidden_size)
            
            # Get probe prediction
            logits = probe(activations)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            true_labels.append(example.label)
            
            # Get statement text if available
            if hasattr(example, 'attributes') and 'statement_text' in example.attributes:
                statement_texts.append(example.attributes['statement_text'])
            else:
                statement_texts.append(f"Statement {i+1}")
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    auroc = roc_auc_score(true_labels, probabilities)
    
    # Get confusion matrix and classification report
    cm = confusion_matrix(true_labels, predictions)
    class_report = classification_report(true_labels, predictions, 
                                       target_names=['False', 'True'], 
                                       output_dict=True)
    
    return {
        'predictions': predictions,
        'probabilities': probabilities,
        'true_labels': true_labels,
        'statement_texts': statement_texts,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }


def create_visualizations(results: Dict[str, Any], output_dir: Path, 
                         model_name: str, hook_point: str):
    """Create comprehensive visualizations of the results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    true_labels = results['true_labels']
    predictions = results['predictions']
    probabilities = results['probabilities']
    metrics = results['metrics']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Metrics Summary Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities)
    axes[0, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {metrics["auroc"]:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Score Distribution
    true_scores = [p for p, t in zip(probabilities, true_labels) if t == 1]
    false_scores = [p for p, t in zip(probabilities, true_labels) if t == 0]
    
    axes[0, 1].hist(false_scores, bins=20, alpha=0.7, label='False Statements', color='red', density=True)
    axes[0, 1].hist(true_scores, bins=20, alpha=0.7, label='True Statements', color='blue', density=True)
    axes[0, 1].axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    axes[0, 1].set_xlabel('Probability Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False', 'True'], yticklabels=['False', 'True'],
                ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Metrics Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1'], metrics['auroc']]
    
    bars = axes[1, 1].bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].set_ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Probe Evaluation Results\nModel: {model_name}, Hook: {hook_point}', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'evaluation_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Statement Analysis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot of predictions
    correct_preds = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p == t]
    incorrect_preds = [i for i, (p, t) in enumerate(zip(predictions, true_labels)) if p != t]
    
    # Plot correct predictions
    if correct_preds:
        ax.scatter([probabilities[i] for i in correct_preds], 
                  [true_labels[i] for i in correct_preds],
                  color='green', alpha=0.7, s=60, label='Correct Predictions')
    
    # Plot incorrect predictions
    if incorrect_preds:
        ax.scatter([probabilities[i] for i in incorrect_preds], 
                  [true_labels[i] for i in incorrect_preds],
                  color='red', alpha=0.7, s=60, label='Incorrect Predictions', marker='x')
    
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Decision Threshold')
    ax.set_xlabel('Probability Score')
    ax.set_ylabel('True Label')
    ax.set_title('Statement-Level Predictions')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['False', 'True'])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'statement_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def save_detailed_results(results: Dict[str, Any], output_dir: Path, 
                         dataset_name: str, model_name: str, hook_point: str):
    """Save detailed results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'dataset': dataset_name,
        'model': model_name,
        'hook_point': hook_point,
        'num_examples': len(results['true_labels']),
        'metrics': results['metrics'],
        'confusion_matrix': results['confusion_matrix']
    }
    
    with open(output_dir / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    detailed_results = []
    for i, (pred, prob, true_label, stmt_text) in enumerate(zip(
        results['predictions'], results['probabilities'], 
        results['true_labels'], results['statement_texts']
    )):
        detailed_results.append({
            'example_id': i,
            'statement_text': stmt_text,
            'true_label': true_label,
            'predicted_label': pred,
            'probability': prob,
            'correct': pred == true_label,
            'confidence': abs(prob - 0.5) * 2  # Distance from 0.5, scaled to 0-1
        })
    
    # Save as JSON
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(detailed_results)
    df.to_csv(output_dir / 'detailed_results.csv', index=False)
    
    print(f"Detailed results saved to: {output_dir}")
    return detailed_results


def print_evaluation_summary(results: Dict[str, Any], detailed_results: List[Dict]):
    """Print a comprehensive evaluation summary."""
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print("🎯 PROBE EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   • Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   • Precision: {metrics['precision']:.3f}")
    print(f"   • Recall:    {metrics['recall']:.3f}")
    print(f"   • F1 Score:  {metrics['f1']:.3f}")
    print(f"   • AUROC:     {metrics['auroc']:.3f}")
    
    print(f"\n🔍 CONFUSION MATRIX:")
    cm = np.array(results['confusion_matrix'])
    print(f"              Predicted")
    print(f"              False  True")
    print(f"   Actual False  {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"          True   {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # Analysis of correct vs incorrect predictions
    correct_preds = [r for r in detailed_results if r['correct']]
    incorrect_preds = [r for r in detailed_results if not r['correct']]
    
    print(f"\n📈 PREDICTION ANALYSIS:")
    print(f"   • Total Examples: {len(detailed_results)}")
    print(f"   • Correct Predictions: {len(correct_preds)} ({len(correct_preds)/len(detailed_results)*100:.1f}%)")
    print(f"   • Incorrect Predictions: {len(incorrect_preds)} ({len(incorrect_preds)/len(detailed_results)*100:.1f}%)")
    
    if correct_preds:
        avg_correct_conf = np.mean([r['confidence'] for r in correct_preds])
        print(f"   • Average Confidence (Correct): {avg_correct_conf:.3f}")
    
    if incorrect_preds:
        avg_incorrect_conf = np.mean([r['confidence'] for r in incorrect_preds])
        print(f"   • Average Confidence (Incorrect): {avg_incorrect_conf:.3f}")
    
    # Show some examples
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    
    # Show highest confidence correct predictions
    correct_sorted = sorted(correct_preds, key=lambda x: x['confidence'], reverse=True)
    if correct_sorted:
        print(f"\n   ✅ Most Confident CORRECT Predictions:")
        for i, result in enumerate(correct_sorted[:3]):
            label_str = "TRUE" if result['true_label'] else "FALSE"
            print(f"      {i+1}. \"{result['statement_text'][:60]}...\"")
            print(f"         → {label_str} (prob: {result['probability']:.3f}, conf: {result['confidence']:.3f})")
    
    # Show incorrect predictions
    if incorrect_preds:
        print(f"\n   ❌ INCORRECT Predictions:")
        for i, result in enumerate(incorrect_preds[:3]):
            true_label_str = "TRUE" if result['true_label'] else "FALSE"
            pred_label_str = "TRUE" if result['predicted_label'] else "FALSE"
            print(f"      {i+1}. \"{result['statement_text'][:60]}...\"")
            print(f"         → Predicted: {pred_label_str}, Actual: {true_label_str} (prob: {result['probability']:.3f})")
    
    print("\n" + "="*60)


def main():
    args = parse_args()
    
    print("🚀 NTML Probe Evaluation")
    print(f"📄 Dataset: {args.dataset}")
    print(f"🧠 Model: {args.model_name}")
    print(f"🎯 Hook Point: {args.hook_point}")
    print(f"🔍 Probe: {args.probe_path}")
    
    try:
        # Load probe and model
        probe, model = load_probe_and_model(args.probe_path, args.model_name, args.device)
        
        # Create test dataset
        test_dataset = create_test_dataset(args.dataset, args.test_split, args.max_length)
        
        # Evaluate probe
        results = evaluate_probe(probe, model, test_dataset, args.hook_point, args.device)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        
        # Save detailed results
        detailed_results = save_detailed_results(
            results, output_dir, args.dataset, args.model_name, args.hook_point
        )
        
        # Create visualizations
        create_visualizations(results, output_dir, args.model_name, args.hook_point)
        
        # Print summary
        print_evaluation_summary(results, detailed_results)
        
        print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 