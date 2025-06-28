#!/usr/bin/env python3
"""
Simple evaluation script using probity's built-in evaluation system.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.probes import BaseProbe
from probity_extensions.conversational import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from transformers import AutoTokenizer

def eval_with_probity():
    """Evaluate the trained probe using probity's built-in system"""
    print("=== Evaluating with Probity's Built-in System ===\n")
    
    # 1. Load the trained probe
    print("1. Loading trained probe...")
    probe = BaseProbe.load_json("trained_probes/logistic/layer_7_probe.json")
    print(f"   Probe loaded from trained_probes/logistic/layer_7_probe.json")
    
    # 2. Load evaluation dataset
    print("\n2. Loading evaluation dataset...")
    ntml_dataset = ConversationalProbingDataset.from_jsonl(
        "data/NTML-datasets/2T1L_2samples.jsonl"
    )
    statement_dataset = ntml_dataset.to_statement_level()
    
    # 3. Tokenize
    print("\n3. Tokenizing dataset...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        statement_dataset, 
        tokenizer, 
        max_length=128
    )
    
    # 4. Extract texts and labels for evaluation
    print("\n4. Preparing data for evaluation...")
    texts = [ex.text for ex in tokenized_dataset]
    labels = [ex.label for ex in tokenized_dataset]
    
    print(f"   {len(texts)} examples ready for evaluation")
    
    # 5. Create evaluator and run evaluation
    print("\n5. Running probity evaluation...")
    evaluator = OptimizedBatchProbeEvaluator("gpt2", "cuda")
    
    # Configure probe for evaluation
    probe_configs = {(7, "logistic"): probe}  # Layer 7, logistic probe
    
    # Run evaluation
    results = evaluator.evaluate_all_probes(texts, labels, probe_configs)
    
    # 6. Display results
    print("\n6. Evaluation Results:")
    print("=" * 50)
    
    for (layer, probe_type), result in results.items():
        metrics = result['metrics']
        print(f"\nLayer {layer} - {probe_type.upper()} Probe:")
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1-Score:  {metrics['f1']:.3f}")
        print(f"  AUROC:     {metrics['auroc']:.3f}")
        
        # Show predictions
        print(f"\n  Predictions vs True Labels:")
        for i, sample in enumerate(result['all_samples'][:5]):  # Show first 5
            print(f"    Example {i+1}: Pred={sample['predicted_label']} (prob={sample['mean_score']:.3f}), True={sample['true_label']}")
        
        if len(result['all_samples']) > 5:
            print(f"    ... and {len(result['all_samples']) - 5} more examples")
    
    print("\n✅ Probity evaluation completed successfully!")
    print("\nThe evaluation used probity's optimized batch evaluator which:")
    print("- Runs the model once and caches activations")
    print("- Applies all probes efficiently")
    print("- Provides comprehensive metrics and token-level analysis")

if __name__ == "__main__":
    eval_with_probity() 