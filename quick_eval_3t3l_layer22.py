#!/usr/bin/env python3
"""
Quick evaluation script for the 3T3L Layer 22 probe.
Loads the saved probe and generates performance plots.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from probity.probes import LogisticProbe
from probity_extensions.conversational import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from transformers import AutoTokenizer

def quick_eval_3t3l_layer22():
    """Quick evaluation of the 3T3L Layer 22 probe."""
    print("=== Quick Evaluation: 3T3L Layer 22 Probe ===\n")
    
    # 1. Load the trained probe
    print("1. Loading trained probe...")
    probe = LogisticProbe.load("test_llama70b_3t3l_layer22_probe.pt")
    print(f"   Probe loaded: {probe.name}")
    print(f"   Hook point: {probe.hook_point}")
    print(f"   Input size: {probe.input_size}")
    
    # 2. Load the dataset for evaluation
    print("\n2. Loading dataset for evaluation...")
    ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(
        "data/NTML-datasets/3T3L_20samples.jsonl"
    )
    statement_dataset = ntml_dataset.get_statement_dataset()
    
    # Tokenize
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=statement_dataset,
        tokenizer=tokenizer,
        padding=True,
        max_length=4096,
        add_special_tokens=True,
    )
    print(f"   Dataset loaded: {len(tokenized_dataset.examples)} statements")
    
    # 3. Get probe predictions
    print("\n3. Generating predictions...")
    probe.eval()
    with torch.no_grad():
        # Get activations from the saved cache or recompute
        # For now, let's use the probe's internal state if available
        if hasattr(probe, 'training_history'):
            print("   Using training history for analysis...")
            history = probe.training_history
            
            # Plot training history
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['train_loss'], label='Train Loss', marker='o')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training History')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            if 'train_acc' in history:
                plt.plot(history['train_acc'], label='Train Accuracy', marker='o')
            if 'val_acc' in history:
                plt.plot(history['val_acc'], label='Validation Accuracy', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Accuracy History')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('3t3l_layer22_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"   Training history plot saved: 3t3l_layer22_training_history.png")
            
            # Print final metrics
            print(f"\n4. Final Training Metrics:")
            print(f"   Final Train Loss: {history['train_loss'][-1]:.4f}")
            if 'val_loss' in history:
                print(f"   Final Val Loss: {history['val_loss'][-1]:.4f}")
            if 'train_acc' in history:
                print(f"   Final Train Accuracy: {history['train_acc'][-1]:.4f}")
            if 'val_acc' in history:
                print(f"   Final Val Accuracy: {history['val_acc'][-1]:.4f}")
    
    # 4. Analyze probe direction
    print(f"\n5. Probe Analysis:")
    direction = probe.get_direction()
    print(f"   Direction norm: {torch.norm(direction):.4f}")
    print(f"   Direction shape: {direction.shape}")
    
    # Plot direction distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(direction.cpu().numpy(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Probe Direction Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(direction.cpu().numpy()[:100])  # First 100 dimensions
    plt.xlabel('Hidden Dimension')
    plt.ylabel('Weight Value')
    plt.title('Probe Direction (First 100 dims)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('3t3l_layer22_probe_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   Probe analysis plot saved: 3t3l_layer22_probe_analysis.png")
    
    print(f"\n✅ Quick evaluation completed!")
    print(f"   - Training history: 3t3l_layer22_training_history.png")
    print(f"   - Probe analysis: 3t3l_layer22_probe_analysis.png")

if __name__ == "__main__":
    quick_eval_3t3l_layer22() 