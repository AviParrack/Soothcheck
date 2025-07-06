import argparse
import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import gc
import glob

from probity.evaluation.batch_evaluator_tok import TokenBasedBatchEvaluator
from probity.probes.base import BaseProbe

def load_probe(probe_path: str, device: str) -> BaseProbe:
    """Load probe from path."""
    if probe_path.endswith('.json'):
        probe = BaseProbe.load_json(probe_path)
    else:
        # Load PT file (NTML format)
        state = torch.load(probe_path, map_location=device)
        # NTML probes are always LogisticProbe
        from probity.probes import LogisticProbe, LogisticProbeConfig
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

def load_jsonl_streaming(file_path: str, num_samples: Optional[int] = None):
    """Load JSONL file one line at a time."""
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if num_samples is not None and count >= num_samples:
                break
            yield json.loads(line)
            count += 1

def convert_label_to_binary(label: str) -> int:
    """Convert string label to binary (0/1) or -1 for skip."""
    if label.lower() in ['deceptive', 'lie', 'lying', 'dishonest']:
        return 1
    elif label.lower() in ['honest', 'truth', 'truthful']:
        return 0
    else:
        return -1  # Skip unknown labels

def save_sample_result(output_file: str, sample_data: Dict, mode: str = 'a'):
    """Save sample result to file."""
    with open(output_file, mode) as f:
        f.write(json.dumps(sample_data) + '\n')
        f.flush()  # Ensure immediate write to disk

def extract_token_lists(sample_data: Dict, tokenizer=None) -> Dict[str, List[int]]:
    """Extract token lists from sample data.
    
    Based on the JSON structure, tokens are in the 'tokens' field of each conversation branch,
    not in 'token_lists' which contains probe-specific token lists.
    """
    token_lists = {}
    for branch_name, branch_data in sample_data.get('conversations', {}).items():
        # First try to get tokens from existing token_lists (if already processed)
        if 'token_lists' in branch_data and branch_data['token_lists']:
            # Get first available token list
            first_probe = next(iter(branch_data['token_lists'].values()))
            token_lists[branch_name] = first_probe
        # Otherwise get from the 'tokens' field (original tokenization)
        elif 'tokens' in branch_data:
            tokens = branch_data['tokens']
            if isinstance(tokens[0], str):
                # Convert token strings to token IDs using tokenizer
                if tokenizer is None:
                    print(f"Warning: Found string tokens in {branch_name} but no tokenizer provided")
                    continue
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                token_lists[branch_name] = token_ids
            else:
                # Assume they're already token IDs
                token_lists[branch_name] = tokens
        else:
            print(f"Warning: No tokens found in branch {branch_name}")
            continue
    return token_lists

def process_sample_streaming(
    sample_data: Dict,
    sample_idx: int,
    probe,
    evaluator: TokenBasedBatchEvaluator,
    layer: int,
    output_file: str
) -> bool:
    """Process a single sample using streaming to avoid memory issues."""
    try:
        # Extract conversations from this single sample
        conversations = sample_data.get('conversations', {})
        if not conversations:
            print(f"Sample {sample_idx}: No conversations found, skipping")
            return False

        # Convert label using the same logic as original
        label = convert_label_to_binary(sample_data.get('metadata', {}).get('label', 'skip'))
        if label == -1:
            print(f"Sample {sample_idx}: Skip/unknown label, skipping")
            return False

        print(f"Sample {sample_idx}: Label = {label} ({'deceptive' if label == 1 else 'honest'})")

        token_lists = extract_token_lists(sample_data, evaluator.tokenizer)
        if not token_lists:
            print(f"No token lists found in sample {sample_idx}")
            return False
            
        # Compact output - just show branch count
        branch_count = len(token_lists)
        total_tokens = sum(len(tokens) for tokens in token_lists.values())
        print(f"Processing {branch_count} branches, {total_tokens} total tokens")
        
        for branch_name, tokens in token_lists.items():
            # Compact branch processing output
            print(f"  {branch_name}: {len(tokens)} tokens", end=" -> ")
            
            # Use the main label for all branches (same as original script)
            
            # Clear GPU memory before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            probe_configs = {(layer, probe.__class__.__name__): probe}
            results = evaluator.evaluate_all_probes_from_tokens(
                token_lists=[tokens],
                labels=[label],
                probe_configs=probe_configs,
                batch_size=1
            )
            
            probe_key = (layer, probe.__class__.__name__)
            sample_result = results[probe_key]['all_samples'][0]
            probe_name = f"layer_{layer}_{probe.__class__.__name__}"
            
            # Initialize structures if needed (same as original)
            if 'conversations' not in sample_data:
                sample_data['conversations'] = {}
            if branch_name not in sample_data['conversations']:
                sample_data['conversations'][branch_name] = {}
            
            # Add probe results (same format as original)
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
            
            # Add human-readable token strings for verification
            if 'token_strings' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['token_strings'] = {}
            # Convert token IDs back to strings for verification
            token_strings = evaluator.tokenizer.convert_ids_to_tokens(sample_result['tokens'])
            sample_data['conversations'][branch_name]['token_strings'][probe_name] = token_strings
            
            # Compact output - just show mean score and token verification
            mean_score = sample_result['mean_score']
            print(f"score={mean_score:.3f}, tokens=[{token_strings[0]}...{token_strings[-1]}]")
            
            # Aggressive cleanup after each branch (same as original)
            del results, sample_result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Save this sample immediately (same as original)
        mode = 'w' if sample_idx == 0 else 'a'
        save_sample_result(output_file, sample_data, mode)
        print(f"Sample {sample_idx}: Saved to {output_file}")
        
        # Final cleanup (same as original)
        del sample_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"ERROR processing sample {sample_idx}: {e}")
        # Emergency cleanup (same as original)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False

def process_file(
    input_file: str,
    output_file: str,
    probe,
    evaluator: TokenBasedBatchEvaluator,
    layer: int,
    num_samples: Optional[int] = None,
    resume_from: int = 0
) -> tuple[int, int]:
    """Process a single input file."""
    print(f"\n{'='*60}")
    print(f"PROCESSING FILE: {input_file}")
    print(f"{'='*60}")
    
    # Clear output file if not resuming
    if resume_from == 0 and Path(output_file).exists():
        Path(output_file).unlink()
    
    processed_count = 0
    successful_count = 0
    
    # Count total samples for this file
    total_samples = 0
    with open(input_file, 'r') as f:
        for _ in f:
            total_samples += 1
    
    if num_samples:
        total_samples = min(total_samples, num_samples)
    
    print(f"Total samples to process: {total_samples}")
    print(f"Starting from sample: {resume_from}")
    
    # Create progress bar for this file
    pbar = tqdm(total=total_samples - resume_from, 
                desc=f"Processing {Path(input_file).name}", 
                unit="samples",
                position=0,
                leave=True)
    
    # Process samples
    for sample_idx, sample_data in enumerate(load_jsonl_streaming(input_file, num_samples)):
        if sample_idx < resume_from:
            continue
            
        # Update progress bar description with current sample
        pbar.set_description(f"Processing {Path(input_file).name} - Sample {sample_idx + 1}")
        
        success = process_sample_streaming(
            sample_data=sample_data,
            sample_idx=sample_idx,
            probe=probe,
            evaluator=evaluator,
            layer=layer,
            output_file=output_file
        )
        
        processed_count += 1
        if success:
            successful_count += 1
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'Success': f"{successful_count}/{processed_count}",
            'GPU_GB': f"{torch.cuda.memory_allocated() / 1024**3:.1f}" if torch.cuda.is_available() else "N/A"
        })
        
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 60 * 1024**3:  # 60GB threshold
            print("\nWARNING: High GPU memory usage detected, forcing cleanup...")
            torch.cuda.empty_cache()
            gc.collect()
    
    pbar.close()
    return processed_count, successful_count

def parse_args():
    parser = argparse.ArgumentParser(description='Apply probes to conversations using pre-tokenized input')
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--probe_path', type=str, required=True,
                      help='Path to trained probe')
    parser.add_argument('--layer', type=int, required=True,
                      help='Layer to extract activations from')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--num_samples', type=int, default=None,
                      help='Number of samples to process per file (None for all)')
    parser.add_argument('--resume_from', type=int, default=0,
                      help='Sample index to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup input/output directories
    input_dir = Path('data/b2w-data/raw')
    output_dir = Path('data/b2w-scores/tok')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all input files
    input_files = list(input_dir.glob('*.jsonl'))
    if not input_files:
        print(f"No .jsonl files found in {input_dir}")
        return
        
    print(f"Found {len(input_files)} input files:")
    for f in input_files:
        print(f"  - {f.name}")
    
    # Load probe
    print(f"\nInitializing probe from {args.probe_path}")
    probe = load_probe(args.probe_path, args.device)
    
    # Initialize evaluator
    print(f"Initializing evaluator with model {args.model_name}")
    evaluator = TokenBasedBatchEvaluator(
        model_name=args.model_name,
        device=args.device
    )
    
    # Process each file
    total_processed = 0
    total_successful = 0
    
    # Create overall progress bar for all files
    overall_pbar = tqdm(total=len(input_files), 
                       desc="Processing files", 
                       unit="files",
                       position=1,
                       leave=True)
    
    for file_idx, input_file in enumerate(input_files):
        output_file = output_dir / f"{input_file.stem}_tok.jsonl"
        
        overall_pbar.set_description(f"File {file_idx + 1}/{len(input_files)}: {input_file.name}")
        
        processed, successful = process_file(
            input_file=str(input_file),
            output_file=str(output_file),
            probe=probe,
            evaluator=evaluator,
            layer=args.layer,
            num_samples=args.num_samples,
            resume_from=args.resume_from
        )
        
        total_processed += processed
        total_successful += successful
        
        overall_pbar.update(1)
        overall_pbar.set_postfix({
            'Total_Success': f"{total_successful}/{total_processed}",
            'Current_File': f"{successful}/{processed}"
        })
        
        print(f"\nCompleted {input_file.name}: {successful}/{processed} successful")
    
    overall_pbar.close()
    
    print(f"\n{'='*60}")
    print(f"ALL FILES PROCESSED")
    print(f"{'='*60}")
    print(f"Total files processed: {len(input_files)}")
    print(f"Total samples processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_processed - total_successful}")
    print(f"Success rate: {total_successful/total_processed*100:.1f}%")

if __name__ == "__main__":
    main() 