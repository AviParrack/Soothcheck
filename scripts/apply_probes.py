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
import os
import traceback

from probity.probes import BaseProbe, LogisticProbe, LogisticProbeConfig
from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import get_model_dtype
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Set memory optimization env var
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

def load_dataset(file_path: str) -> List[Dict]:
    """Load and validate dataset"""
    data = load_jsonl(file_path)
    
    # Filter valid samples
    valid_samples = []
    for sample in data:
        if isinstance(sample, dict) and "text" in sample and "label" in sample:
            if isinstance(sample["text"], str) and isinstance(sample["label"], (int, bool)):
                valid_samples.append({
                    "text": sample["text"],
                    "label": int(sample["label"])
                })
    
    print(f"\nLoaded {len(valid_samples)} valid samples from {file_path}")
    return valid_samples

def load_probes(probe_paths: List[str]) -> Dict[tuple, BaseProbe]:
    """Load probes from paths"""
    probes = {}
    for path in probe_paths:
        print(f"\nInitializing probe from {path}")
        probe = LogisticProbe.load(path)
        layer = int(path.split("layer_")[-1].split(".")[0])
        probes[(layer, "logistic")] = probe
    return probes

def save_results(results: Dict, output_dir: Path, dataset_name: str):
    """Save evaluation results"""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results for each probe
    for (layer, probe_type), probe_results in results.items():
        probe_dir = output_dir / f"{dataset_name}/layer_{layer}"
        probe_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = probe_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(probe_results['metrics'], f, indent=2)
        
        # Save detailed results
        details_file = probe_dir / "detailed_results.json"
        with open(details_file, 'w') as f:
            json.dump({
                'token_details': probe_results['token_details'],
                'all_samples': probe_results['all_samples']
            }, f, indent=2)
        
        print(f"\nResults for layer {layer} {probe_type} probe saved to {probe_dir}")
        print("Metrics:", probe_results['metrics'])

def process_dataset(file_path: str, evaluator: OptimizedBatchProbeEvaluator, 
                   probe_configs: Dict, output_dir: Path):
    """Process a single dataset file"""
    # Get dataset name from file path
    dataset_name = Path(file_path).stem
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Load and prepare dataset
    dataset = load_dataset(file_path)
    if not dataset:
        print(f"No valid samples found in {file_path}")
        return
    
    # Get texts and labels
    texts = [sample["text"] for sample in dataset]
    labels = [sample["label"] for sample in dataset]
    
    print(f"\nProcessing {len(texts)} valid samples")
    print("Binary label distribution:")
    print(f"  Deceptive (1): {sum(labels)}")
    print(f"  Honest (0): {len(labels) - sum(labels)}")
    print(f"  Ratio deceptive/honest: {sum(labels)/(len(labels) - sum(labels)):.2f}\n")
    
    # Clear any existing cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run evaluation with memory monitoring
    try:
        results = evaluator.evaluate_all_probes(
            texts=texts,
            labels=labels,
            probe_configs=probe_configs,
        )
        
        # Save results
        save_results(results, output_dir, dataset_name)
        
    except Exception as e:
        print(f"\nError processing {dataset_name}: {str(e)}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Apply probes to multiple datasets")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct",
                      help="Name of the model to use")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on (cuda or cpu)")
    parser.add_argument("--probe_path", type=str, required=True,
                      help="Path to probe file")
    parser.add_argument("--layer", type=int, required=True,
                      help="Layer number for the probe")
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size for processing (default: 1)")
    parser.add_argument("--output_dir", type=str, default="results/probe_evaluations",
                      help="Directory to save results")
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=args.model_name,
        device=args.device if torch.cuda.is_available() else "cpu"
    )
    
    # Set batch size
    evaluator.batch_size = args.batch_size
    
    # Load probes
    probe_configs = load_probes([args.probe_path])
    
    # Setup output directory with timestamp
    output_dir = Path(args.output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / timestamp
    
    # Process all jsonl files in raw directory
    raw_dir = Path("data/b2w-data/raw")
    for file_path in raw_dir.glob("*.jsonl"):
        try:
            process_dataset(str(file_path), evaluator, probe_configs, output_dir)
        except Exception as e:
            print(f"\nFailed to process {file_path}: {str(e)}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main() 