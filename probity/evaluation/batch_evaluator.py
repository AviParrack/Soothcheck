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
        
    def _find_chunk_boundaries(self, text: str, tokenizer, max_length: int = 2048) -> List[Tuple[int, int]]:
        """Find natural chunk boundaries in text that respect the max_length constraint.
        Returns list of (start_idx, end_idx) tuples for token indices."""
        
        # First tokenize the entire text without truncation to get full token count
        tokens = tokenizer(text, add_special_tokens=False, truncation=False)
        token_ids = tokens['input_ids']
        
        # If text fits in one chunk, return it whole
        if len(token_ids) <= max_length:
            return [(0, len(token_ids))]
        
        # Initialize boundaries
        chunk_boundaries = []
        current_start = 0
        
        while current_start < len(token_ids):
            # If remaining tokens fit in max_length, make them the last chunk
            if len(token_ids) - current_start <= max_length:
                chunk_boundaries.append((current_start, len(token_ids)))
                break
            
            # Look for a good splitting point within max_length
            # Start from max possible position and work backwards
            split_pos = min(current_start + max_length, len(token_ids))
            best_split = split_pos
            
            # Decode a window of tokens around the potential split point
            window_start = max(0, split_pos - 100)  # Look back up to 100 tokens
            window_end = min(len(token_ids), split_pos + 100)  # Look ahead up to 100 tokens
            window_tokens = token_ids[window_start:window_end]
            window_text = tokenizer.decode(window_tokens)
            
            # Find sentence boundaries in the window
            sentence_ends = []
            for i, char in enumerate(window_text):
                if char == '.' and (i == len(window_text) - 1 or window_text[i + 1] in [' ', '\n', '\t']):
                    # Convert window position to token position
                    token_pos = window_start + len(tokenizer.encode(window_text[:i+1], add_special_tokens=False))
                    if token_pos <= split_pos:  # Only consider boundaries before split_pos
                        sentence_ends.append(token_pos)
            
            # If we found sentence boundaries, use the last one before split_pos
            if sentence_ends:
                best_split = max(x for x in sentence_ends if x <= split_pos)
            
            # Add the chunk and continue
            chunk_boundaries.append((current_start, best_split))
            current_start = best_split
        
        return chunk_boundaries
        
    def get_batch_activations(self, texts: List[str], layers: List[int], 
                            batch_size: int = 1) -> Dict[int, torch.Tensor]:
        """Get activations for all texts and layers efficiently, handling long sequences
        by splitting at natural boundaries."""
        
        # Create cache key
        cache_key = (tuple(sorted(texts)), tuple(sorted(layers)))
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        
        hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
        
        # Get tokenizer
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        # Process in smaller batches with memory cleanup
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        # Start with minimal batch size
        actual_batch_size = 1
        
        with torch.no_grad():
            for text_idx, text in enumerate(texts):
                print(f"\nProcessing text {text_idx + 1}/{len(texts)}")
                
                # Find natural chunk boundaries
                chunk_boundaries = self._find_chunk_boundaries(text, tokenizer)
                print(f"Split into {len(chunk_boundaries)} chunks")
                
                # Process each chunk
                chunk_tokens = []
                chunk_activations = {layer: [] for layer in layers}
                
                # First tokenize the entire text to get full token sequence
                full_tokens = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                    add_special_tokens=False
                )
                
                for chunk_idx, (start_idx, end_idx) in enumerate(chunk_boundaries):
                    print(f"Processing chunk {chunk_idx + 1}/{len(chunk_boundaries)} (tokens {start_idx}-{end_idx})")
                    
                    # Aggressive memory cleanup before each chunk
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    
                    # Extract this chunk's tokens
                    chunk_input_ids = full_tokens["input_ids"][0][start_idx:end_idx].unsqueeze(0)
                    
                    try:
                        # Run model with caching
                        _, cache = self.model.run_with_cache(
                            chunk_input_ids.to(self.device),
                            names_filter=hook_points,
                            return_cache_object=True,
                            stop_at_layer=max(layers) + 1
                        )
                        
                        # Store activations for each layer - move to CPU immediately
                        for layer in layers:
                            hook_point = f"blocks.{layer}.hook_resid_pre"
                            layer_activations = cache[hook_point].cpu()
                            chunk_activations[layer].append(layer_activations)
                        
                        # Clear cache reference and force cleanup
                        del cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Store tokens for this chunk
                        token_texts = tokenizer.convert_ids_to_tokens(chunk_input_ids[0])
                        chunk_tokens.extend(token_texts)
                        
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            print(f"\nOOM with chunk {chunk_idx}")
                            raise
                        else:
                            raise
                    
                    # Clear chunk data
                    del chunk_input_ids
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                
                # Combine chunk activations for this text
                for layer in layers:
                    # Concatenate along sequence length dimension
                    combined_activations = torch.cat(chunk_activations[layer], dim=1)
                    all_activations[layer].append(combined_activations)
                
                # Store combined tokens
                all_tokens.append(chunk_tokens)
                
                if len(all_tokens) <= 3 or len(all_tokens) % 50 == 0:
                    print(f"Text {len(all_tokens)}: {len(chunk_tokens)} tokens")
                    print(f"  First 5: {chunk_tokens[:5]}")
                    print(f"  Last 5: {chunk_tokens[-5:]}")
                    print(f"  Total tokens: {len(chunk_tokens)}")
        
        # Process final activations
        max_seq_len = max(batch.shape[1] for layer in layers for batch in all_activations[layer])
        print(f"Maximum sequence length across all texts: {max_seq_len}")
        
        final_activations = {}
        for layer in layers:
            print(f"\nProcessing layer {layer} final activations...")
            
            # Initialize empty tensor on CPU to store all activations
            total_samples = len(texts)
            hidden_size = all_activations[layer][0].shape[2]
            final_tensor = torch.zeros(
                (total_samples, max_seq_len, hidden_size),
                dtype=all_activations[layer][0].dtype,
                device='cpu'
            )
            
            # Fill tensor text by text
            for i, text_activations in enumerate(all_activations[layer]):
                seq_len = text_activations.shape[1]
                final_tensor[i, :seq_len] = text_activations.squeeze(0)
            
            final_activations[layer] = final_tensor
            
            # Clear layer data
            del all_activations[layer]
            gc.collect()
        
        # Clear intermediate results
        del all_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        result = {
            'activations': final_activations,
            'tokens_by_text': all_tokens
        }
        self._activation_cache[cache_key] = result
        
        return result
    
    def evaluate_all_probes(self, texts: List[str], labels: List[int], 
                          probe_configs: Dict[Tuple[int, str], BaseProbe]) -> Dict[Tuple[int, str], Dict]:
        """Evaluate all probes efficiently using cached activations"""
        
        # Extract unique layers from probe configs
        layers = list(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations for all required layers at once
        print("Getting activations for all layers...")
        activation_data = self.get_batch_activations(texts, layers)
        activations = activation_data['activations']
        tokens_by_text = activation_data['tokens_by_text']
        
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
                    probe, layer_activations, mean_activations, 
                    texts, labels, tokens_by_text
                )
                
                results[(layer, probe_type)] = probe_results
        
        return results

    
    
    def _evaluate_single_probe_batch(self, probe: BaseProbe, 
                                   layer_activations: torch.Tensor,
                                   mean_activations: torch.Tensor,
                                   texts: List[str], labels: List[int],
                                   tokens_by_text: List[List[str]]) -> Dict:
        """Evaluate a single probe on batch data"""
        
        # Move probe to device and ensure it matches model dtype
        probe = probe.to(self.device)
        probe.eval()
        
        # Get probe's expected dtype
        probe_dtype = next(probe.parameters()).dtype if any(True for _ in probe.parameters()) else self.model_dtype
        
        print(f"Probe dtype: {probe_dtype}, Activations dtype: {layer_activations.dtype}")
        
        # Process in small chunks to manage memory
        chunk_size = 1  # Process one sample at a time
        all_token_scores = []
        all_samples = []
        
        with torch.no_grad():
            for i, (text, true_label) in tqdm(enumerate(zip(texts, labels)), total=len(texts), desc="Processing texts"):
                # Clear memory before each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                # Get tokens and actual length for this text
                tokens = tokens_by_text[i]
                actual_length = len(tokens)
                
                # Get activations for this text (up to actual length)
                text_activations = layer_activations[i, :actual_length].to(self.device)
                
                # Ensure dtype compatibility
                if text_activations.dtype != probe_dtype:
                    text_activations = text_activations.to(dtype=probe_dtype)
                
                # Apply probe to all tokens for this text
                token_scores = probe(text_activations)
                
                # Move results to CPU immediately
                token_scores = token_scores.cpu()
                
                # Apply sigmoid only to LogisticProbe
                if probe.__class__.__name__ == 'LogisticProbe':
                    token_scores = torch.sigmoid(token_scores)
                
                # Convert to list
                token_scores_list = token_scores.squeeze().tolist()
                
                # Handle single token case
                if isinstance(token_scores_list, float):
                    token_scores_list = [token_scores_list]
                
                # Ensure we have the right number of scores
                if len(token_scores_list) != actual_length:
                    print(f"Warning: Score length mismatch for text {i}: {len(token_scores_list)} vs {actual_length}")
                    if len(token_scores_list) < actual_length:
                        token_scores_list.extend([0.0] * (actual_length - len(token_scores_list)))
                    else:
                        token_scores_list = token_scores_list[:actual_length]
                
                all_token_scores.extend(token_scores_list)
                
                sample_info = {
                    "idx": i,
                    "text": text,
                    "true_label": true_label,
                    "tokens": tokens,
                    "raw_token_scores": token_scores_list
                }
                all_samples.append(sample_info)
                
                # Clear sample data
                del text_activations
                del token_scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Global normalization for non-logistic probes
        if probe.__class__.__name__ == 'LogisticProbe':
            # LogisticProbe already outputs [0,1] after sigmoid
            for sample in all_samples:
                sample["token_scores"] = sample["raw_token_scores"]
        else:
            # Normalize all scores globally to [0,1]
            normalized_scores = self._normalize_to_01(all_token_scores)
            
            # Distribute normalized scores back to samples
            score_idx = 0
            for sample in all_samples:
                num_tokens = len(sample["raw_token_scores"])
                sample["token_scores"] = normalized_scores[score_idx:score_idx + num_tokens]
                score_idx += num_tokens
        
        # Calculate final metrics
        final_samples = []
        all_mean_scores = []
        
        for sample in all_samples:
            mean_score = float(np.mean(sample["token_scores"]))
            all_mean_scores.append(mean_score)
            
            sample_info = {
                "sample_id": sample["idx"],
                "text": sample["text"],
                "true_label": int(sample["true_label"]),
                "tokens": sample["tokens"],
                "token_scores": sample["token_scores"],
                "mean_score": mean_score,
                "predicted_label": int(1 if mean_score > 0.5 else 0),
                "is_correct": bool((mean_score > 0.5) == (sample["true_label"] == 1)),
            }
            final_samples.append(sample_info)
        
        # Calculate metrics
        mean_scores = np.array(all_mean_scores)
        predictions = (mean_scores > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions)),
            'recall': float(recall_score(labels, predictions)),
            'f1': float(f1_score(labels, predictions)),
            'auroc': float(roc_auc_score(labels, mean_scores))
        }
        
        # Prepare token_details for visualization
        token_details = []
        for sample in final_samples:
            # Clean tokens for visualization
            clean_tokens = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in sample["tokens"]]
            
            token_detail = {
                "text": sample["text"],
                "label": int(sample["true_label"]),
                "tokens": clean_tokens,
                "token_scores": sample["token_scores"],
                "mean_score": sample["mean_score"],
                "max_score": float(np.max(sample["token_scores"])),
                "min_score": float(np.min(sample["token_scores"]))
            }
            token_details.append(token_detail)
        
        return {
            'metrics': metrics,
            'metrics_interpretation': {
                "accuracy": "Fraction of correctly classified truth/lie statements",
                "precision": "Of statements classified as lies, fraction that were actually lies",
                "recall": "Of actual lies, fraction that were detected",
                "f1": "Harmonic mean of precision and recall for lie detection",
                "auroc": "Area under ROC curve (1.0 = perfect separation of truths and lies)"
            },
            'all_samples': final_samples,
            'token_details': token_details,
            'mean_scores': mean_scores.tolist(),
            'predictions': predictions.tolist()
        }
    
    def _normalize_to_01(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
