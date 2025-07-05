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

def extract_conversations(data: List[Dict]) -> List[Dict[str, str]]:
    """Extract all conversations from B2W data format.
    
    Args:
        data: List of data items
        
    Returns:
        List of dictionaries mapping conversation branch names to their text content
    """
    all_conversations = []
    for item in data:
        # Get all conversation branches
        conversations = item.get('conversations', {})
        conv_dict = {}
        
        # Process each conversation branch
        for branch_name, branch_data in conversations.items():
            conv = ""
            messages = branch_data.get('messages', [])
            for msg in messages:
                # Include all messages, including system messages
                conv += f"{msg['role']}: {msg['content']}\n"
            conv_dict[branch_name] = conv.strip()
        
        all_conversations.append(conv_dict)
    return all_conversations

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
    # Simplified version - just return empty list since we're not using this
    return []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="Model name")
    parser.add_argument("--probe_path", type=str, required=True, help="Path to probe file")
    parser.add_argument("--layer", type=int, required=True, help="Layer number")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--context_length", type=int, default=8192, 
                       help="Context length to use. For Llama-3.3-70B, max is 128000. Larger values use more memory.")
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    all_conversations = extract_conversations(data)
    
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
        all_conversations = [all_conversations[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        data = [data[i] for i in valid_indices]
    
    print(f"\nProcessing {len(all_conversations)} valid samples")
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
        device=args.device if torch.cuda.is_available() else "cpu",
        context_length=args.context_length
    )
    
    # Process each conversation branch
    print("\n=== Processing Conversation Branches ===")
    for branch_name in set().union(*[conv.keys() for conv in all_conversations]):
        print(f"\nProcessing branch: {branch_name}")
        
        # Extract conversations for this branch
        branch_conversations = [conv.get(branch_name, "") for conv in all_conversations]
        
        # Skip empty conversations
        if not any(branch_conversations):
            print(f"No conversations found for branch {branch_name}, skipping...")
            continue
        
        # Evaluate probe
        print(f"\nEvaluating probe from layer {args.layer}")
        probe_configs = {(args.layer, probe.__class__.__name__): probe}
        results = evaluator.evaluate_all_probes(
            texts=branch_conversations,
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
            if not sample['text'].strip():  # Skip empty conversations
                continue
                
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
        
        # Add scores to data
        print(f"\nAdding scores for branch {branch_name} to data")
        probe_name = Path(args.probe_path).stem
        for item, score_list in zip(data, all_samples):
            if not score_list['text'].strip():  # Skip empty conversations
                continue
                
            # Initialize conversations dict if needed
            if 'conversations' not in item:
                item['conversations'] = {}
            if branch_name not in item['conversations']:
                item['conversations'][branch_name] = {}
            
            # Add probe to available_probes
            if 'available_probes' not in item['conversations'][branch_name]:
                item['conversations'][branch_name]['available_probes'] = []
            if probe_name not in item['conversations'][branch_name]['available_probes']:
                item['conversations'][branch_name]['available_probes'].append(probe_name)
            
            # Initialize probe_scores if needed
            if 'probe_scores' not in item['conversations'][branch_name]:
                item['conversations'][branch_name]['probe_scores'] = {}
            
            # Store scores directly as a list like pairs_probe and rp_probe
            item['conversations'][branch_name]['probe_scores'][probe_name] = score_list['token_scores']
            
            # Store token list in a separate field for later alignment
            if 'token_lists' not in item['conversations'][branch_name]:
                item['conversations'][branch_name]['token_lists'] = {}
            item['conversations'][branch_name]['token_lists'][probe_name] = score_list['tokens']
            
            # Validate the stored data
            print(f"\nValidating stored data for sample {item.get('id', 'unknown')}:")
            print(f"  Branch: {branch_name}")
            print(f"  Probe: {probe_name}")
            print(f"  Token count: {len(score_list['token_scores'])}")
            print(f"  Available probes: {item['conversations'][branch_name]['available_probes']}")
            print(f"  First few tokens: {score_list['tokens'][:5]}")
            print(f"  Last few tokens: {score_list['tokens'][-5:]}")
    
    # Save augmented data
    print(f"\nSaving results to {output_file}")
    save_jsonl(data, output_file)

if __name__ == "__main__":
    main() 