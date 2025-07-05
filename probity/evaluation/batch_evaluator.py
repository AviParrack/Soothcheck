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
        
    def get_batch_activations(self, texts: List[str], layers: List[int], 
                            batch_size: int = 8) -> Dict[int, torch.Tensor]:
        """Get activations for all texts and layers efficiently"""
        
        # Create cache key
        cache_key = (tuple(sorted(texts)), tuple(sorted(layers)))
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        
        hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
        
        # Tokenize all texts at once
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # Debug first text tokenization
        if texts:
            print("\n=== DEBUG: Single text tokenization ===")
            sample_text = texts[0]
            print(f"Sample text length: {len(sample_text)}")
            tokens = tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
            print(f"Token count: {tokens['input_ids'].shape[1]}")
            print("=== END DEBUG ===\n")
        
        # Process in smaller batches with memory cleanup
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        # Reduce batch size for long sequences
        actual_batch_size = min(batch_size, 4)  # Cap at 4 for safety
        num_batches = (len(texts) + actual_batch_size - 1) // actual_batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), actual_batch_size), total=num_batches, desc="Processing batches"):
                batch_texts = texts[i:i + actual_batch_size]
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Tokenize batch
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=False,
                    add_special_tokens=False
                ).to(self.device)
                
                try:
                    # Run model with caching
                    _, cache = self.model.run_with_cache(
                        tokens["input_ids"],
                        names_filter=hook_points,
                        return_cache_object=True,
                        stop_at_layer=max(layers) + 1
                    )
                    
                    # Store activations for each layer - move to CPU immediately
                    for layer in layers:
                        hook_point = f"blocks.{layer}.hook_resid_pre"
                        # Get activations and move to CPU
                        layer_activations = cache[hook_point].cpu()
                        all_activations[layer].append(layer_activations)
                    
                    # Clear cache reference
                    del cache
                    
                    # Store tokens
                    for j, text in enumerate(batch_texts):
                        text_tokens = tokens["input_ids"][j]
                        if tokenizer.pad_token_id is not None:
                            pad_positions = torch.where(text_tokens == tokenizer.pad_token_id)[0]
                            if len(pad_positions) > 0:
                                first_pad_pos = pad_positions[0].item()
                                actual_tokens = text_tokens[:first_pad_pos]
                            else:
                                actual_tokens = text_tokens
                        else:
                            actual_tokens = text_tokens
                        
                        token_texts = tokenizer.convert_ids_to_tokens(actual_tokens)
                        all_tokens.append(token_texts)
                        
                        if len(all_tokens) <= 3 or len(all_tokens) % 50 == 0:
                            print(f"Text {len(all_tokens)}: {len(token_texts)} tokens")
                            print(f"  First 5: {token_texts[:5]}")
                            print(f"  Last 5: {token_texts[-5:]}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        # If OOM occurs, try with even smaller batch
                        print(f"\nOOM with batch size {actual_batch_size}, reducing...")
                        if actual_batch_size > 1:
                            actual_batch_size = max(1, actual_batch_size // 2)
                            num_batches = (len(texts) + actual_batch_size - 1) // actual_batch_size
                            # Clear memory and retry
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise  # If batch_size=1 still fails, we have a problem
                    else:
                        raise
                
                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Since batches may have different padding lengths, we need to pad all to the same length
        max_seq_len = max(batch.shape[1] for layer in layers for batch in all_activations[layer])
        print(f"Maximum sequence length across all batches: {max_seq_len}")
        
        # Process each layer separately to manage memory
        final_activations = {}
        for layer in layers:
            print(f"\nProcessing layer {layer} final activations...")
            padded_batches = []
            
            # Process each batch
            for batch_activations in all_activations[layer]:
                batch_size, seq_len, hidden_size = batch_activations.shape
                if seq_len < max_seq_len:
                    padding = torch.zeros(
                        batch_size, max_seq_len - seq_len, hidden_size, 
                        dtype=batch_activations.dtype, device='cpu'  # Keep on CPU
                    )
                    padded_batch = torch.cat([batch_activations, padding], dim=1)
                else:
                    padded_batch = batch_activations
                padded_batches.append(padded_batch)
            
            # Move to GPU and concatenate in chunks if needed
            chunk_size = 2  # Process 2 batches at a time
            final_chunks = []
            
            for chunk_start in range(0, len(padded_batches), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(padded_batches))
                chunk_batches = padded_batches[chunk_start:chunk_end]
                
                # Move chunk to GPU and concatenate
                gpu_chunk = torch.cat([batch.to(self.device) for batch in chunk_batches], dim=0)
                final_chunks.append(gpu_chunk)
                
                # Clear chunk batches
                del chunk_batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final concatenation of chunks
            final_activations[layer] = torch.cat(final_chunks, dim=0)
            
            # Clear intermediate results
            del padded_batches
            del final_chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Clear more memory
        del all_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
        
        # Get probe's expected dtype - check parameters first, then buffers
        probe_dtype = self.model_dtype  # Default to model dtype
        try:
            probe_dtype = next(probe.parameters()).dtype
        except StopIteration:
            # No parameters, check buffers
            try:
                probe_dtype = next(probe.buffers()).dtype
            except StopIteration:
                # No buffers either, use model dtype
                probe_dtype = self.model_dtype
        
        print(f"Probe dtype: {probe_dtype}, Activations dtype: {layer_activations.dtype}")
        
        probe.eval()
        
        # Get token-level scores for all texts at once
        all_token_scores = []
        all_samples = []
        
        with torch.no_grad():
            # Process each text individually to handle variable lengths properly
            for i, (text, true_label) in tqdm(enumerate(zip(texts, labels)), total=len(texts), desc="Processing texts"):
                # Get tokens and actual length for this text
                tokens = tokens_by_text[i]
                actual_length = len(tokens)
                
                # Get activations for this text (up to actual length)
                text_activations = layer_activations[i, :actual_length, :].to(self.device)
                
                # ENSURE DTYPE COMPATIBILITY
                if text_activations.dtype != probe_dtype:
                    text_activations = text_activations.to(dtype=probe_dtype)
                
                # Apply probe to all tokens for this text
                token_scores = probe(text_activations)
                
                # Apply sigmoid only to LogisticProbe
                if probe.__class__.__name__ == 'LogisticProbe':
                    token_scores = torch.sigmoid(token_scores)
                
                # Convert to list
                token_scores_list = token_scores.cpu().squeeze().tolist()
                
                # Handle single token case
                if isinstance(token_scores_list, float):
                    token_scores_list = [token_scores_list]
                
                # Ensure we have the right number of scores
                if len(token_scores_list) != actual_length:
                    print(f"Warning: Score length mismatch for text {i}: {len(token_scores_list)} vs {actual_length}")
                    # Pad or truncate as needed
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
            mean_score = np.mean(sample["token_scores"])
            all_mean_scores.append(mean_score)
            
            sample_info = {
                "sample_id": sample["idx"],
                "text": sample["text"],
                "true_label": int(sample["true_label"]),
                "tokens": sample["tokens"],
                "token_scores": sample["token_scores"],
                "mean_score": float(mean_score),
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
