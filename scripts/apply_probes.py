#!/usr/bin/env python3
"""
Script to apply trained probes to B2W data and save augmented results.
"""

import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import datetime

from probity.probes import BaseProbe, LogisticProbe, LogisticProbeConfig
from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import get_model_dtype
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_jsonl(file_path: str, num_samples: Optional[int] = None) -> List[Dict]:
    """Load JSONL file into list of dictionaries.
    
    Args:
        file_path: Path to JSONL file
        num_samples: If provided, only load this many samples from the start
    """
    data = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def extract_conversations(data: List[Dict]) -> List[str]:
    """Extract conversations from B2W data format."""
    conversations = []
    for item in data:
        # Get messages from the main conversation branch
        conv = ""
        messages = item.get('conversations', {}).get('main', {}).get('messages', [])
        for msg in messages:
            if msg['role'] != 'system':  # Skip system messages as they're not part of the conversation
                conv += f"{msg['role']}: {msg['content']}\n"
        conversations.append(conv.strip())
    return conversations

def load_probe(probe_path: str, device: str) -> BaseProbe:
    """Load probe from file."""
    if probe_path.endswith('.json'):
        probe = BaseProbe.load_json(probe_path)
    else:
        # Load PT file (NTML format)
        state = torch.load(probe_path, map_location=device)
        # NTML probes are always LogisticProbe
        config = LogisticProbeConfig(
            input_size=state['model_config']['input_size'],
            device=device,
            model_name=state['training_config'].get('model_name', 'unknown'),
            hook_point=state['training_config'].get('hook_point', 'unknown'),
            hook_layer=state['training_config'].get('hook_layer', 0),
            normalize_weights=state['model_config'].get('normalize_weights', True),
            bias=state['model_config'].get('bias', True)
        )
        probe = LogisticProbe(config=config)
        probe.load_state_dict(state['model_state_dict'])
    
    probe.to(device)
    return probe

def apply_probe(
    conversations: List[str],
    model_name: str,
    probe_path: str,
    layer: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 8
) -> List[List[float]]:
    """Apply probe to conversations and return scores."""
    print(f"\nInitializing probe from {probe_path}")
    probe = load_probe(probe_path, device)
    
    print(f"Initializing evaluator with model {model_name}")
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=model_name,
        device=device
    )
    
    print(f"\nGetting activations for {len(conversations)} conversations")
    print(f"Using batch size: {batch_size}")
    activation_data = evaluator.get_batch_activations(
        texts=conversations,
        layers=[layer],
        batch_size=batch_size
    )
    
    print("\nApplying probe to activations")
    layer_activations = activation_data['activations'][layer]
    
    probe_scores = []
    with torch.no_grad():
        for i, text in enumerate(conversations):
            print(f"\nProcessing conversation {i+1}/{len(conversations)}")
            print(f"Text length: {len(text)} chars")
            
            # Get tokens and actual length for this text
            tokens = activation_data['tokens_by_text'][i]
            actual_length = len(tokens)
            print(f"Token count: {actual_length}")
            print(f"First 50 chars: {text[:50]}...")
            
            # Get activations for this text (up to actual length)
            text_activations = layer_activations[i, :actual_length, :].to(device)
            print(f"Activation shape: {text_activations.shape}")
            
            # Apply probe to all tokens for this text
            token_scores = probe(text_activations)
            
            # Apply sigmoid for LogisticProbe
            if probe.__class__.__name__ == 'LogisticProbe':
                token_scores = torch.sigmoid(token_scores)
            
            # Convert to list
            token_scores_list = token_scores.cpu().squeeze().tolist()
            
            # Handle single token case
            if isinstance(token_scores_list, float):
                token_scores_list = [token_scores_list]
            
            # Calculate and print mean score
            mean_score = sum(token_scores_list) / len(token_scores_list)
            print(f"Mean probe score: {mean_score:.4f}")
            print(f"Prediction: {'deceptive' if mean_score > 0.5 else 'honest'}")
            
            # Add the full list of token scores
            probe_scores.append(token_scores_list)
            
            # Print separator
            print("-" * 80)
    
    print("\nFinished processing all conversations")
    return probe_scores

def get_output_path(input_path: str, output_dir: str, num_samples: Optional[int] = None) -> str:
    """Generate output path based on input dataset name."""
    input_file = Path(input_path).stem
    # Extract dataset name from the first item in the file
    with open(input_path, 'r') as f:
        first_item = json.loads(f.readline())
        dataset_name = first_item.get('dataset', input_file)
    
    # Add sample count to filename if specified
    if num_samples is not None:
        return str(Path(output_dir) / f"scored_{dataset_name}_{num_samples}samples.jsonl")
    return str(Path(output_dir) / f"scored_{dataset_name}.jsonl")

def convert_label_to_binary(label: str) -> int:
    """Convert text label to binary value.
    
    Args:
        label: Text label from the dataset
        
    Returns:
        1 for deceptive/lie labels, 0 for honest/truth labels, -1 for skip/unknown
    """
    label = str(label).lower().strip()
    if label in ['deceptive', 'lie', 'lies', 'false']:
        return 1
    elif label in ['honest', 'truth', 'true']:
        return 0
    else:
        return -1  # Skip/unknown label

def compute_critical_scores(all_samples, data):
    """Compute mean scores for critical tokens only."""
    critical_scores = []
    for sample, item in zip(all_samples, data):
        # Get critical indices from the main conversation
        critical_indices = item.get('conversations', {}).get('main', {}).get('critical_indices', [])
        if not critical_indices:
            continue
            
        # Get scores for critical tokens only
        critical_token_scores = [sample['token_scores'][i] for i in critical_indices if i < len(sample['token_scores'])]
        if critical_token_scores:
            critical_scores.append({
                'mean_score': float(sum(critical_token_scores) / len(critical_token_scores)),
                'token_scores': critical_token_scores,
                'indices': critical_indices,
                'critical_claim': '',  # These fields don't exist in new format
                'user_query': ''  # These fields don't exist in new format
            })
        
    return critical_scores

def main():
    parser = argparse.ArgumentParser(description="Apply probes to B2W data")
    parser.add_argument("--input_file", type=str, required=True,
                      help="Path to input B2W JSONL file")
    parser.add_argument("--output_dir", type=str, default="data/b2w-scores/raw",
                      help="Directory to save augmented JSONL file (default: data/b2w-scores/raw)")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model to use (e.g., 'Qwen/Qwen2.5-0.5B')")
    parser.add_argument("--probe_path", type=str, required=True,
                      help="Path to saved probe")
    parser.add_argument("--layer", type=int, required=True,
                      help="Layer number to extract activations from")
    parser.add_argument("--batch_size", type=int, default=8,
                      help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run on (cuda/cpu)")
    parser.add_argument("--num_samples", type=int, default=None,
                      help="Number of samples to process (default: all)")
    parser.add_argument("--threshold", type=float, default=None,
                      help="Classification threshold (default: auto-computed from score distribution)")
    parser.add_argument("--debug", action="store_true",
                      help="Enable detailed debug output")
    parser.add_argument("--critical_only", action="store_true",
                      help="Only use critical tokens for final metrics")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output path
    output_file = get_output_path(args.input_file, args.output_dir, args.num_samples)
    
    # Load data
    print(f"Loading data from {args.input_file}")
    if args.num_samples:
        print(f"Processing first {args.num_samples} samples")
    data = load_jsonl(args.input_file, args.num_samples)
    print(f"Loaded {len(data)} samples")
    
    # Extract conversations and convert labels
    print("\n=== Label Processing ===")
    conversations = extract_conversations(data)
    
    # Print raw labels before conversion
    print("\nRaw labels before conversion:")
    for i, item in enumerate(data):
        label = item.get('metadata', {}).get('label', 'skip')
        print(f"Sample {i}: {label}")
    
    labels = [convert_label_to_binary(item.get('metadata', {}).get('label', 'skip')) for item in data]
    
    # Print converted labels
    print("\nConverted binary labels:")
    for i, (label, orig) in enumerate(zip(labels, data)):
        print(f"Sample {i}: {orig.get('metadata', {}).get('label', 'skip')} -> {label}")
    
    # Filter out skipped samples
    valid_indices = [i for i, label in enumerate(labels) if label != -1]
    if len(valid_indices) < len(labels):
        print(f"\nFiltered out {len(labels) - len(valid_indices)} samples with skip/unknown labels")
        print("Removed indices:", [i for i in range(len(labels)) if i not in valid_indices])
        conversations = [conversations[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        data = [data[i] for i in valid_indices]
    
    print(f"\nProcessing {len(conversations)} valid samples")
    print(f"Binary label distribution:")
    print(f"  Deceptive (1): {sum(labels)}")
    print(f"  Honest (0): {len(labels) - sum(labels)}")
    print(f"  Ratio deceptive/honest: {sum(labels)/(len(labels) - sum(labels)):.2f}")
    
    # Load probe
    print(f"\nInitializing probe from {args.probe_path}")
    probe = load_probe(args.probe_path, args.device)
    
    # Initialize evaluator
    print(f"Initializing evaluator with model {args.model_name}")
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=args.model_name,
        device=args.device
    )
    
    # Evaluate probe
    print(f"\nEvaluating probe from layer {args.layer}")
    probe_configs = {(args.layer, probe.__class__.__name__): probe}
    results = evaluator.evaluate_all_probes(
        texts=conversations,
        labels=labels,
        probe_configs=probe_configs
    )
    
    # Analyze results in detail
    print("\n=== Detailed Results Analysis ===")
    probe_key = (args.layer, probe.__class__.__name__)
    all_samples = results[probe_key]['all_samples']
    mean_scores = results[probe_key]['mean_scores']
    
    print("\nPer-sample analysis:")
    honest_scores = []
    deceptive_scores = []
    
    for i, (sample, label) in enumerate(zip(all_samples, labels)):
        score = sample['mean_score']
        if label == 1:
            deceptive_scores.append(score)
        else:
            honest_scores.append(score)
        
        print(f"\nSample {i}:")
        print(f"  True label: {label} ({'deceptive' if label == 1 else 'honest'})")
        print(f"  Mean score: {score:.4f}")
        print(f"  Text preview: {sample['text'][:100]}...")
        print(f"  Token count: {len(sample['tokens'])}")
        print(f"  Score range: {min(sample['token_scores']):.4f} - {max(sample['token_scores']):.4f}")
    
    print("\nScore distribution by class:")
    if honest_scores:
        print(f"  Honest (0) scores:")
        print(f"    Min: {min(honest_scores):.4f}")
        print(f"    Max: {max(honest_scores):.4f}")
        print(f"    Mean: {sum(honest_scores)/len(honest_scores):.4f}")
    if deceptive_scores:
        print(f"  Deceptive (1) scores:")
        print(f"    Min: {min(deceptive_scores):.4f}")
        print(f"    Max: {max(deceptive_scores):.4f}")
        print(f"    Mean: {sum(deceptive_scores)/len(deceptive_scores):.4f}")
    
    print("\nOverall score distribution:")
    print(f"  Min score: {min(mean_scores):.4f}")
    print(f"  Max score: {max(mean_scores):.4f}")
    print(f"  Mean score: {sum(mean_scores)/len(mean_scores):.4f}")
    
    # After evaluating probe and before computing metrics:
    print("\n=== Critical Token Analysis ===")
    critical_results = compute_critical_scores(all_samples, data)
    
    if critical_results:
        print(f"\nFound critical tokens in {len(critical_results)} samples")
        critical_mean_scores = [r['mean_score'] for r in critical_results]
        
        print("\nCritical token score distribution:")
        print(f"  Min score: {min(critical_mean_scores):.4f}")
        print(f"  Max score: {max(critical_mean_scores):.4f}")
        print(f"  Mean score: {sum(critical_mean_scores)/len(critical_mean_scores):.4f}")
        
        # Detailed per-sample critical analysis
        print("\nPer-sample critical token analysis:")
        for i, result in enumerate(critical_results):
            print(f"\nSample {i}:")
            print(f"  Critical token count: {len(result['indices'])}")
            print(f"  Critical tokens: {' '.join(result['tokens'])}")
            print(f"  Mean critical score: {result['mean_score']:.4f}")
            print(f"  Score range: {min(result['token_scores']):.4f} - {max(result['token_scores']):.4f}")
        
        # If using critical tokens only, update mean_scores
        if args.critical_only:
            print("\nUsing critical tokens only for final metrics")
            mean_scores = critical_mean_scores
    else:
        print("No critical token information found in the dataset")
    
    # Calculate optimal threshold if not provided
    if args.threshold is None:
        # Sort scores and find gap between classes
        sorted_scores = sorted((score, label) for score, label in zip(mean_scores, labels))
        max_gap = 0
        best_threshold = 0.5
        
        print("\nGap analysis for threshold:")
        for i in range(len(sorted_scores) - 1):
            gap = sorted_scores[i + 1][0] - sorted_scores[i][0]
            print(f"  Gap between {sorted_scores[i][0]:.4f} ({sorted_scores[i][1]}) and {sorted_scores[i+1][0]:.4f} ({sorted_scores[i+1][1]}): {gap:.4f}")
            if gap > max_gap:
                max_gap = gap
                best_threshold = (sorted_scores[i][0] + sorted_scores[i + 1][0]) / 2
                print(f"    New best threshold: {best_threshold:.4f}")
        
        threshold = best_threshold
        print(f"\nAuto-computed threshold: {threshold:.4f}")
    else:
        threshold = args.threshold
        print(f"\nUsing provided threshold: {threshold:.4f}")
    
    # Recalculate metrics with new threshold
    predictions = [1 if score > threshold else 0 for score in mean_scores]
    
    print("\nConfusion matrix:")
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)
    print(f"  True Positives (deceptive correctly identified): {tp}")
    print(f"  False Positives (honest misclassified as deceptive): {fp}")
    print(f"  False Negatives (deceptive misclassified as honest): {fn}")
    print(f"  True Negatives (honest correctly identified): {tn}")
    
    metrics = {
        'accuracy': float(accuracy_score(labels, predictions)),
        'precision': float(precision_score(labels, predictions)),
        'recall': float(recall_score(labels, predictions)),
        'f1': float(f1_score(labels, predictions)),
        'auroc': float(roc_auc_score(labels, mean_scores))
    }
    
    print("\nProbe Performance Metrics:")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Add critical AUROC if available
    if critical_results:
        # Get labels for samples with critical tokens
        critical_sample_indices = [i for i, item in enumerate(data) if 'critical_analysis' in item and 
                                isinstance(item['critical_analysis'], dict) and 
                                item['critical_analysis'].get('critical_token_indices')]
        critical_labels = [labels[i] for i in critical_sample_indices]
        
        critical_metrics = {
            'accuracy': float(accuracy_score(critical_labels, [1 if s > threshold else 0 for s in critical_mean_scores])),
            'precision': float(precision_score(critical_labels, [1 if s > threshold else 0 for s in critical_mean_scores])),
            'recall': float(recall_score(critical_labels, [1 if s > threshold else 0 for s in critical_mean_scores])),
            'f1': float(f1_score(critical_labels, [1 if s > threshold else 0 for s in critical_mean_scores])),
            'auroc': float(roc_auc_score(critical_labels, critical_mean_scores))
        }
        
        print("\nCritical Tokens Only - Probe Performance Metrics:")
        print(f"AUROC: {critical_metrics['auroc']:.4f}")
        print(f"Accuracy: {critical_metrics['accuracy']:.4f}")
        print(f"F1 Score: {critical_metrics['f1']:.4f}")
        print(f"Precision: {critical_metrics['precision']:.4f}")
        print(f"Recall: {critical_metrics['recall']:.4f}")
    
    # Add scores to data
    print("\nAdding scores to data")
    probe_name = Path(args.probe_path).stem
    for item, score_list in zip(data, all_samples):
        # Store scores in the main conversation branch
        if 'conversations' not in item:
            item['conversations'] = {}
        if 'main' not in item['conversations']:
            item['conversations']['main'] = {}
            
        item['conversations']['main']['token_scores'] = score_list['token_scores']
        item['conversations']['main']['probe_name'] = probe_name
        item['conversations']['main']['layer'] = args.layer
        item['conversations']['main']['mean_score'] = float(score_list['mean_score'])
        item['conversations']['main']['predicted_label'] = int(score_list['mean_score'] > threshold)
        
        # Add critical token scores if available
        critical_indices = item.get('conversations', {}).get('main', {}).get('critical_indices', [])
        if critical_indices:
            critical_token_scores = [score_list['token_scores'][i] for i in critical_indices if i < len(score_list['token_scores'])]
            if critical_token_scores:
                item['conversations']['main']['critical_token_scores'] = critical_token_scores
                item['conversations']['main']['critical_mean_score'] = float(sum(critical_token_scores) / len(critical_token_scores))
                item['conversations']['main']['critical_predicted_label'] = int(item['conversations']['main']['critical_mean_score'] > threshold)
    
    # Save augmented data
    print(f"\nSaving results to {output_file}")
    save_jsonl(data, output_file)

if __name__ == "__main__":
    main() 