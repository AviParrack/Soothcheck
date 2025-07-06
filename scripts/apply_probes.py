#!/usr/bin/env python3
"""
Script to apply trained probes to B2W data and save augmented results.
"""

import argparse
import json
import torch
import gc
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional
import datetime
import re

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

def extract_conversations(data: List[Dict]) -> List[Dict[str, List[Dict]]]:
    """Extract all conversations from B2W data format.
    
    Args:
        data: List of data items
        
    Returns:
        List of dictionaries mapping conversation branch names to their message lists
    """
    print("DEBUG: extract_conversations() called")
    print(f"DEBUG: Processing {len(data)} data items")
    
    all_conversations = []
    for item_idx, item in enumerate(data):
        print(f"\nDEBUG: Processing item {item_idx}")
        print(f"DEBUG: Item keys: {list(item.keys())}")
        
        # Show metadata
        metadata = item.get('metadata', {})
        print(f"DEBUG: Metadata: {metadata}")
        
        # Get all conversation branches
        conversations = item.get('conversations', {})
        print(f"DEBUG: Found {len(conversations)} conversation branches: {list(conversations.keys())}")
        
        conv_dict = {}
        
        # Process each conversation branch
        for branch_name, branch_data in conversations.items():
            print(f"\nDEBUG: Processing branch '{branch_name}'")
            print(f"DEBUG: Branch data keys: {list(branch_data.keys())}")
            
            # Check if this branch has messages
            messages = branch_data.get('messages', [])
            print(f"DEBUG: Found {len(messages)} messages in branch '{branch_name}'")
            
            # Clean message content and store original message structure
            cleaned_messages = []
            for msg_idx, msg in enumerate(messages):
                print(f"DEBUG: Message {msg_idx}: role='{msg['role']}', content_len={len(msg['content'])}")
                
                # Clean malformed characters from message content
                content = msg['content']
                
                # Fix the ACTUAL issue - Ċ characters should be newlines
                content = content.replace('Ċ', '\n')
                
                # Also handle the display corruption we were chasing
                content = content.replace('ÄĬ', '\n')
                content = content.replace('Ä', '\n')   
                content = content.replace('Ĭ', '\n')
                
                # Clean up multiple consecutive newlines - but preserve specific patterns
                # Don't collapse all multiple newlines, some might be intentional
                content = re.sub(r'\n{4,}', '\n\n\n', content)  # Max 3 consecutive newlines
                
                # CRITICAL FIX: Remove chat template prefix from system messages
                # The dataset already contains this prefix, but chat template adds it again
                if msg['role'] == 'system':
                    # Remove the automatic prefix that chat template will add
                    prefix_patterns = [
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n",
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
                    ]
                    
                    original_length = len(content)
                    for pattern in prefix_patterns:
                        if content.startswith(pattern):
                            content = content[len(pattern):]
                            print(f"DEBUG: Removed duplicate prefix from system message ({original_length} -> {len(content)} chars)")
                            break
                    
                    # Also check if it appears elsewhere in the content
                    if len(content) == original_length:  # No removal happened
                        for pattern in prefix_patterns:
                            if pattern in content:
                                content = content.replace(pattern, "", 1)
                                print(f"DEBUG: Removed duplicate prefix found within system message ({original_length} -> {len(content)} chars)")
                                break
                
                cleaned_messages.append({
                    "role": msg['role'],
                    "content": content
                })
            
            conv_dict[branch_name] = cleaned_messages
            print(f"DEBUG: Branch '{branch_name}' has {len(cleaned_messages)} cleaned messages")
        
        all_conversations.append(conv_dict)
    
    print(f"\nDEBUG: extract_conversations() completed - processed {len(all_conversations)} conversations")
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

def load_jsonl_streaming(file_path: str, num_samples: Optional[int] = None):
    """Generator that yields JSONL items one at a time for memory efficiency.
    
    Args:
        file_path: Path to JSONL file
        num_samples: If provided, only yield this many samples from the start
    """
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            yield json.loads(line)

def save_sample_result(output_file: str, sample_data: Dict, mode: str = 'a'):
    """Save a single sample result to JSONL file.
    
    Args:
        output_file: Path to output file
        sample_data: Sample data to save
        mode: File mode ('w' for first sample, 'a' for append)
    """
    with open(output_file, mode) as f:
        f.write(json.dumps(sample_data) + '\n')
        f.flush()  # Ensure immediate write to disk

def process_sample_streaming(
    sample_data: Dict,
    sample_idx: int,
    probe,
    evaluator,
    layer: int,
    output_file: str
) -> bool:
    """Process a single sample with immediate saving and memory cleanup.
    
    Returns:
        bool: True if successful, False if should skip
    """
    import gc
    import torch
    
    print(f"\n=== Processing Sample {sample_idx} ===")
    
    try:
        # Extract conversations from this single sample
        conversations = sample_data.get('conversations', {})
        if not conversations:
            print(f"Sample {sample_idx}: No conversations found, skipping")
            return False
        
        # Convert label
        label = convert_label_to_binary(sample_data.get('metadata', {}).get('label', 'skip'))
        if label == -1:
            print(f"Sample {sample_idx}: Skip/unknown label, skipping")
            return False
        
        print(f"Sample {sample_idx}: Label = {label} ({'deceptive' if label == 1 else 'honest'})")
        
        # Process each conversation branch
        for branch_name, branch_data in conversations.items():
            print(f"\nProcessing branch: {branch_name}")
            
            # Get messages for this branch
            messages = branch_data.get('messages', [])
            if not messages:
                print(f"Branch {branch_name}: No messages, skipping")
                continue
            
            # Clean messages (same as before)
            cleaned_messages = []
            for msg in messages:
                content = msg['content']
                
                # Character cleaning
                content = content.replace('Ċ', '\n')
                content = content.replace('ÄĬ', '\n')
                content = content.replace('Ä', '\n')   
                content = content.replace('Ĭ', '\n')
                content = re.sub(r'\n{4,}', '\n\n\n', content)
                
                # Remove duplicate prefix from system messages
                if msg['role'] == 'system':
                    prefix_patterns = [
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n",
                        "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024",
                    ]
                    
                    original_length = len(content)
                    for pattern in prefix_patterns:
                        if content.startswith(pattern):
                            content = content[len(pattern):]
                            print(f"DEBUG: Removed duplicate prefix from system message ({original_length} -> {len(content)} chars)")
                            break
                    
                    if len(content) == original_length:
                        for pattern in prefix_patterns:
                            if pattern in content:
                                content = content.replace(pattern, "", 1)
                                print(f"DEBUG: Removed duplicate prefix found within system message ({original_length} -> {len(content)} chars)")
                                break
                
                cleaned_messages.append({
                    "role": msg['role'],
                    "content": content
                })
            
            # Process this single conversation with ultra-conservative memory
            print(f"Getting activations for 1 conversation...")
            
            # Clear GPU memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Get activations for this single conversation
            activation_data = evaluator.get_batch_activations(
                messages_list=[cleaned_messages],  # Single conversation
                layers=[layer],
                batch_size=1
            )
            
            # Process probe immediately
            probe_configs = {(layer, probe.__class__.__name__): probe}
            results = evaluator.evaluate_all_probes(
                messages_list=[cleaned_messages],
                labels=[label],
                probe_configs=probe_configs
            )
            
            # Extract results
            probe_key = (layer, probe.__class__.__name__)
            sample_result = results[probe_key]['all_samples'][0]
            
            # Add probe scores to the sample data
            probe_name = f"layer_{layer}_{probe.__class__.__name__}"
            
            # Initialize structures if needed
            if 'conversations' not in sample_data:
                sample_data['conversations'] = {}
            if branch_name not in sample_data['conversations']:
                sample_data['conversations'][branch_name] = {}
            
            # Add probe results
            if 'available_probes' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['available_probes'] = []
            if probe_name not in sample_data['conversations'][branch_name]['available_probes']:
                sample_data['conversations'][branch_name]['available_probes'].append(probe_name)
            
            if 'probe_scores' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['probe_scores'] = {}
            sample_data['conversations'][branch_name]['probe_scores'][probe_name] = sample_result['token_scores']
            
            if 'token_lists' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['token_lists'] = {}
            sample_data['conversations'][branch_name]['token_lists'][probe_name] = sample_result['tokens']
            
            print(f"Branch {branch_name}: Added {len(sample_result['token_scores'])} token scores")
            print(f"Mean score: {sample_result['mean_score']:.4f}")
            
            # Aggressive cleanup after each branch
            del activation_data, results, sample_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save this sample immediately
        mode = 'w' if sample_idx == 0 else 'a'
        save_sample_result(output_file, sample_data, mode)
        print(f"Sample {sample_idx}: Saved to {output_file}")
        
        # Final cleanup
        del sample_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"ERROR processing sample {sample_idx}: {e}")
        # Emergency cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False

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
    parser.add_argument("--batch_size", type=int, default=1,
                      help="Batch size (fixed at 1 for streaming)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run on (cuda/cpu)")
    parser.add_argument("--num_samples", type=int, default=None,
                      help="Number of samples to process (default: all)")
    parser.add_argument("--resume_from", type=int, default=0,
                      help="Resume processing from this sample index (default: 0)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate output path
    output_file = get_output_path(args.input_file, args.output_dir, args.num_samples)
    
    print(f"=== STREAMING PROBE PROCESSING ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Model: {args.model_name}")
    print(f"Probe: {args.probe_path}")
    print(f"Layer: {args.layer}")
    print(f"Max samples: {args.num_samples or 'all'}")
    print(f"Resume from: {args.resume_from}")
    
    # Load probe
    print(f"\nInitializing probe from {args.probe_path}")
    probe = load_probe(args.probe_path, args.device)
    
    # Initialize evaluator
    print(f"Initializing evaluator with model {args.model_name}")
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=args.model_name,
        device=args.device
    )
    
    # Process samples one by one using streaming
    print(f"\n=== STARTING STREAMING PROCESSING ===")
    
    processed_count = 0
    successful_count = 0
    
    # Count total samples for progress tracking
    total_samples = 0
    with open(args.input_file, 'r') as f:
        for _ in f:
            total_samples += 1
    
    if args.num_samples:
        total_samples = min(total_samples, args.num_samples)
    
    print(f"Total samples to process: {total_samples}")
    print(f"Starting from sample: {args.resume_from}")
    
    # Stream process each sample
    for sample_idx, sample_data in enumerate(load_jsonl_streaming(args.input_file, args.num_samples)):
        # Skip samples if resuming
        if sample_idx < args.resume_from:
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING SAMPLE {sample_idx + 1}/{total_samples}")
        print(f"{'='*60}")
        
        # Process this sample
        success = process_sample_streaming(
            sample_data=sample_data,
            sample_idx=sample_idx,
            probe=probe,
            evaluator=evaluator,
            layer=args.layer,
            output_file=output_file
        )
        
        processed_count += 1
        if success:
            successful_count += 1
        
        # Progress update
        print(f"\nProgress: {processed_count}/{total_samples - args.resume_from} processed, {successful_count} successful")
        
        # Memory status
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Emergency break if we're getting close to memory limit
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 60 * 1024**3:  # 60GB threshold
            print("WARNING: High GPU memory usage detected, forcing cleanup...")
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    print(f"\n{'='*60}")
    print(f"STREAMING PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {processed_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {processed_count - successful_count}")
    print(f"Results saved to: {output_file}")
    
    if successful_count > 0:
        print(f"\n✅ Successfully processed {successful_count} samples!")
        print(f"📁 Output file: {output_file}")
    else:
        print(f"\n❌ No samples were successfully processed.")
    
    # Final cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

if __name__ == "__main__":
    main() 