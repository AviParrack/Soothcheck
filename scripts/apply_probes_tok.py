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
    probe = BaseProbe.load_json(probe_path)
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

def extract_token_lists(sample_data: Dict) -> Dict[str, List[int]]:
    """Extract token lists from sample data."""
    token_lists = {}
    for branch_name, branch_data in sample_data.get('conversations', {}).items():
        if 'token_lists' in branch_data:
            first_probe = next(iter(branch_data['token_lists'].values()))
            token_lists[branch_name] = first_probe
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
        token_lists = extract_token_lists(sample_data)
        if not token_lists:
            print(f"No token lists found in sample {sample_idx}")
            return False
            
        print(f"Found {len(token_lists)} conversation branches")
        
        for branch_name, tokens in token_lists.items():
            print(f"\nProcessing branch: {branch_name}")
            print(f"Token count: {len(tokens)}")
            
            label = 1 if 'deception' in branch_name.lower() else 0
            print(f"Label: {label} ({'deceptive' if label else 'honest'})")
            
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
            
            if 'conversations' not in sample_data:
                sample_data['conversations'] = {}
            if branch_name not in sample_data['conversations']:
                sample_data['conversations'][branch_name] = {}
            
            if 'available_probes' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['available_probes'] = []
            if probe_name not in sample_data['conversations'][branch_name]['available_probes']:
                sample_data['conversations'][branch_name]['available_probes'].append(probe_name)
            
            if 'probe_scores' not in sample_data['conversations'][branch_name]:
                sample_data['conversations'][branch_name]['probe_scores'] = {}
            sample_data['conversations'][branch_name]['probe_scores'][probe_name] = sample_result['token_scores']
            
            print(f"Branch {branch_name}: Added {len(sample_result['token_scores'])} token scores")
            
        with open(output_file, 'a') as f:
            f.write(json.dumps(sample_data) + '\n')
            
        return True
        
    except Exception as e:
        print(f"Error processing sample {sample_idx}: {e}")
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
    
    # Process samples
    for sample_idx, sample_data in enumerate(load_jsonl_streaming(input_file, num_samples)):
        if sample_idx < resume_from:
            continue
            
        print(f"\n{'='*60}")
        print(f"PROCESSING SAMPLE {sample_idx + 1}/{total_samples}")
        print(f"{'='*60}")
        
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
        
        print(f"\nProgress: {processed_count}/{total_samples - resume_from} processed, {successful_count} successful")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 60 * 1024**3:  # 60GB threshold
            print("WARNING: High GPU memory usage detected, forcing cleanup...")
            torch.cuda.empty_cache()
            gc.collect()
    
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
    
    for input_file in input_files:
        output_file = output_dir / f"{input_file.stem}_tok.jsonl"
        
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
        
        print(f"\nCompleted {input_file.name}:")
        print(f"Processed: {processed}")
        print(f"Successful: {successful}")
        print(f"Failed: {processed - successful}")
        print(f"Output saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print(f"ALL FILES PROCESSED")
    print(f"{'='*60}")
    print(f"Total files processed: {len(input_files)}")
    print(f"Total samples processed: {total_processed}")
    print(f"Total successful: {total_successful}")
    print(f"Total failed: {total_processed - total_successful}")

if __name__ == "__main__":
    main() 