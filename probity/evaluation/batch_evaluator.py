import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from probity.probes import (
    BaseProbe,
    LogisticProbe, LogisticProbeConfig,
    PCAProbe, PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe, KMeansProbeConfig,
    LinearProbe, LinearProbeConfig
)

from probity.utils.dataset_loading import get_model_dtype


class OptimizedBatchProbeEvaluator:
    """Optimized evaluator that runs model once and applies all probes"""
    
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name
        
        # Load model once
        print(f"Loading model {model_name}")
        self.model_dtype = get_model_dtype(model_name)
        print(f"Using model dtype: {self.model_dtype}")
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=self.model_dtype,
        )
        self.model.eval()
        
        # Cache for activations
        self._activation_cache = {}
        
    def get_batch_activations(self, texts: List[str], layer_indices: List[int], batch_size: int = 4) -> Dict[int, torch.Tensor]:
        """Get activations for a batch of texts at specified layers."""
        print("\nGetting activations for all layers...")
        
        # Print tokenizer info
        print(f"Tokenizer pad_token_id: {self.model.tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token_id: {self.model.tokenizer.eos_token_id}")
        print(f"Tokenizer vocab size: {self.model.tokenizer.vocab_size}")
        
        # Debug single text tokenization
        print("\n=== DEBUG: Single text tokenization ===")
        sample_text = texts[0]
        print(f"Sample text length: {len(sample_text)}")
        print(f"Sample text end: ...{sample_text[-100:]}")
        tokens_no_special = self.model.tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
        tokens_with_special = self.model.tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)
        print(f"Without add_special_tokens: {tokens_no_special['input_ids'].shape}")
        print(f"With add_special_tokens: {tokens_with_special['input_ids'].shape}")
        
        # Print last 10 tokens for comparison
        last_10_no_special = self.model.tokenizer.convert_ids_to_tokens(tokens_no_special['input_ids'][0][-10:].tolist())
        last_10_with_special = self.model.tokenizer.convert_ids_to_tokens(tokens_with_special['input_ids'][0][-10:].tolist())
        print(f"Last 10 tokens (no special): {last_10_no_special}")
        print(f"Last 10 tokens (with special): {last_10_with_special}")
        print("=== END DEBUG ===\n")
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        all_activations = {layer_idx: [] for layer_idx in layer_indices}
        max_length = 0
        
        with tqdm(total=num_batches, desc="Processing batches") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Print token info for each text in batch
                for j, text in enumerate(batch_texts, 1):
                    tokens = self.model.tokenizer(text, return_tensors="pt")
                    num_tokens = tokens['input_ids'].shape[1]
                    print(f"Text {j}: {num_tokens} tokens")
                    print(f"  First 5: {self.model.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][:5].tolist())}")
                    print(f"  Last 5: {self.model.tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][-5:].tolist())}")
                
                # Get activations for batch
                batch_activations = self.model.get_activations(
                    batch_texts,
                    layer_indices,
                    add_special_tokens=True
                )
                
                # Update max sequence length
                current_max_length = max(act.shape[1] for act in batch_activations.values())
                max_length = max(max_length, current_max_length)
                
                # Store activations
                for layer_idx in layer_indices:
                    all_activations[layer_idx].append(batch_activations[layer_idx])
                
                pbar.update(1)
        
        print(f"Maximum sequence length across all batches: {max_length}")
        
        # Pad and concatenate activations
        final_activations = {}
        for layer_idx in layer_indices:
            padded_activations = []
            for act in all_activations[layer_idx]:
                pad_length = max_length - act.shape[1]
                if pad_length > 0:
                    padding = torch.zeros(act.shape[0], pad_length, act.shape[2], dtype=act.dtype, device=act.device)
                    padded_act = torch.cat([act, padding], dim=1)
                else:
                    padded_act = act
                padded_activations.append(padded_act)
            final_activations[layer_idx] = torch.cat(padded_activations, dim=0)
            print(f"Final activations for layer {layer_idx}: {final_activations[layer_idx].shape}, dtype: {final_activations[layer_idx].dtype}")
        
        return final_activations
    
    def evaluate_all_probes(self, texts: List[str], labels: List[int], 
                          probe_configs: Dict[Tuple[int, str], BaseProbe]) -> Dict[Tuple[int, str], Dict]:
        """Evaluate all probes efficiently using cached activations"""
        
        # Extract unique layers from probe configs
        layers = list(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations for all required layers at once
        print("Getting activations for all layers...")
        activation_data = self.get_batch_activations(texts, layers)
        activations = activation_data
        
        results = {}
        
        # Group probes by layer for efficient processing
        probes_by_layer = {}
        for (layer, probe_type), probe in probe_configs.items():
            if layer not in probes_by_layer:
                probes_by_layer[layer] = {}
            probes_by_layer[layer][probe_type] = probe
        
        # Process each layer
        for layer in tqdm(layers, desc="Processing layers"):
            layer_activations = activations[layer]  # Shape: [num_texts, seq_len, hidden_size]
            layer_probes = probes_by_layer[layer]
            
            # Calculate mean activations once for this layer
            mean_activations = layer_activations.mean(dim=1)  # Shape: [num_texts, hidden_size]
            
            # Apply all probes for this layer
            for probe_type, probe in layer_probes.items():
                print(f"Evaluating {probe_type} probe on layer {layer}")
                
                # Get probe scores efficiently
                probe_results = self._evaluate_single_probe_batch(
                    probe, layer, layer_activations
                )
                
                results[(layer, probe_type)] = probe_results
        
        return results

    
    
    def _evaluate_single_probe_batch(self, probe: BaseProbe, layer_idx: int, batch_activations: torch.Tensor) -> torch.Tensor:
        """Evaluate a single probe on a batch of activations."""
        print(f"Evaluating {type(probe).__name__} probe on layer {layer_idx}")
        print(f"Probe dtype: {probe.weight.dtype}, Activations dtype: {batch_activations.dtype}")
        
        scores = []
        with tqdm(total=batch_activations.shape[0], desc="Processing texts") as pbar:
            for i in range(batch_activations.shape[0]):
                text_activations = batch_activations[i]
                text_scores = probe(text_activations)
                scores.append(text_scores)
                pbar.update(1)
        
        return torch.stack(scores)
    
    def _normalize_to_01(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
