#!/usr/bin/env python3
"""
Quick evaluation script for the NTML probe we just trained.
"""

import torch
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probity_extensions.conversational import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes import LogisticProbe
from probity.utils.multigpu import MultiGPUConfig
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def quick_eval():
    """Quick evaluation of the trained NTML probe"""
    print("=== Quick NTML Probe Evaluation ===\n")
    
    # 1. Load the trained probe
    print("1. Loading trained probe...")
    probe = LogisticProbe.load("test_ntml_probe.pt")
    probe.to("cuda")
    print(f"   Probe loaded from test_ntml_probe.pt")
    
    # 2. Load the same dataset for evaluation
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
    
    # 4. Load model
    print("\n4. Loading GPT-2 model...")
    model = HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    
    # 5. Evaluate
    print("\n5. Running evaluation...")
    predictions = []
    true_labels = []
    probabilities = []
    
    with torch.no_grad():
        for i, example in enumerate(tokenized_dataset):
            # Get tokens and target position
            tokens = torch.tensor(example.tokens).unsqueeze(0).to("cuda")
            target_pos = example.token_positions.positions['statement_end']
            
            # Get model activations
            _, cache = model.run_with_cache(tokens, names_filter=["blocks.7.hook_resid_post"])
            activations = cache["blocks.7.hook_resid_post"][0, target_pos, :].unsqueeze(0)
            
            # Get probe prediction
            logits = probe(activations)
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0
            
            predictions.append(pred)
            probabilities.append(prob)
            true_labels.append(example.label)
            
            print(f"   Example {i+1}: Pred={pred} (prob={prob:.3f}), True={example.label}")
    
    # 6. Calculate metrics
    print("\n6. Calculating metrics...")
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"\nPredictions: {predictions}")
    print(f"True Labels: {true_labels}")
    print(f"Probabilities: {[f'{p:.3f}' for p in probabilities]}")
    
    # 7. Test probe direction
    print("\n7. Testing probe direction...")
    direction = probe.get_direction()
    print(f"   Direction norm: {torch.norm(direction):.4f}")
    print(f"   Direction shape: {direction.shape}")
    
    print("\n✅ Quick evaluation completed!")

if __name__ == "__main__":
    quick_eval() 