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

from probity.probes import BaseProbe, LogisticProbe, LogisticProbeConfig
from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import get_model_dtype

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
        # Combine messages into a single string, preserving roles
        conv = ""
        for msg in item['messages']:
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
            if i % 10 == 0:
                print(f"Processing conversation {i+1}/{len(conversations)}")
            
            # Get tokens and actual length for this text
            tokens = activation_data['tokens_by_text'][i]
            actual_length = len(tokens)
            
            # Get activations for this text (up to actual length)
            text_activations = layer_activations[i, :actual_length, :].to(device)
            
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
            
            # Add the full list of token scores
            probe_scores.append(token_scores_list)
    
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
    
    # Extract conversations and labels
    print("Extracting conversations")
    conversations = extract_conversations(data)
    labels = [item.get('label', 0) for item in data]  # Default to 0 if no label
    
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
    
    # Print metrics
    probe_key = (args.layer, probe.__class__.__name__)
    metrics = results[probe_key]['metrics']
    print("\nProbe Performance Metrics:")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Get token scores from results
    scores = []
    for sample in results[probe_key]['all_samples']:
        scores.append(sample['token_scores'])
    
    # Add scores to data
    print("\nAdding scores to data")
    probe_name = Path(args.probe_path).stem
    for item, score_list in zip(data, scores):
        item['token_scores'] = score_list
        item['probe_name'] = probe_name
        item['layer'] = args.layer
    
    # Save augmented data
    print(f"\nSaving results to {output_file}")
    save_jsonl(data, output_file)

if __name__ == "__main__":
    main() 