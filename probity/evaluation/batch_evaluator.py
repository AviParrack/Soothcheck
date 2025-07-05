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
        
        # Check if this is Llama 3.3 70B and configure RoPE scaling
        if "Llama-3.3-70B" in model_name:
            print("Detected Llama 3.3 70B model")
            print("Note: This model has a large context window but we'll use chunking for safety")
            
            # Load model normally - we'll handle long sequences through chunking
            self.model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=self.model_dtype
            )
            
            # Set a conservative max sequence length for chunking
            self.max_seq_length = 2048  # Conservative limit to avoid position embedding errors
            
        else:
            # Load model normally for other models
            self.model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                device=device,
                dtype=self.model_dtype
            )
            
            # Set max sequence length based on model config
            self.max_seq_length = getattr(self.model.cfg, 'n_ctx', 2048)
        
        self.model.eval()
        
        # Check the model's actual context window
        print(f"Model context window: {self.model.cfg.n_ctx}")
        
        # Cache for activations
        self._activation_cache = {}
        
    def get_batch_activations(self, texts: List[str], layers: List[int], 
                            batch_size: int = 1) -> Dict[int, torch.Tensor]:
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
        
        # Process in smaller batches with memory cleanup
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        # Start with minimal batch size
        actual_batch_size = 1
        num_batches = (len(texts) + actual_batch_size - 1) // actual_batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), actual_batch_size), total=num_batches, desc="Processing batches"):
                # Aggressive memory cleanup before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
                
                batch_texts = texts[i:i + actual_batch_size]
                
                # Tokenize batch with conservative max length
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,  # Use our conservative max length
                    add_special_tokens=False
                ).to(self.device)
                
                # Check actual sequence lengths
                actual_lengths = []
                for j, text in enumerate(batch_texts):
                    actual_len = tokens["input_ids"][j].ne(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0).sum().item()
                    actual_lengths.append(actual_len)
                    if actual_len >= self.max_seq_length * 0.95:  # Close to limit
                        print(f"Warning: Text {j} was truncated to {actual_len} tokens (max: {self.max_seq_length})")
                        print(f"  Original text length: {len(text)} characters")
                        print(f"  Text preview: {text[:200]}...")
                
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
                    
                    # Clear cache reference and force cleanup
                    del cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
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
                        print(f"\nOOM with batch size {actual_batch_size}")
                        raise  # If batch_size=1 fails, we can't reduce further
                    else:
                        raise
                
                # Clear batch data
                del tokens
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Process final activations in smaller chunks
        max_seq_len = max(batch.shape[1] for layer in layers for batch in all_activations[layer])
        print(f"Maximum sequence length across all batches: {max_seq_len}")
        
        final_activations = {}
        for layer in layers:
            print(f"\nProcessing layer {layer} final activations...")
            
            # Initialize empty tensor on CPU to store all activations
            total_samples = sum(batch.shape[0] for batch in all_activations[layer])
            hidden_size = all_activations[layer][0].shape[2]
            final_tensor = torch.zeros(
                (total_samples, max_seq_len, hidden_size),
                dtype=all_activations[layer][0].dtype,
                device='cpu'
            )
            
            # Fill tensor batch by batch
            current_idx = 0
            for batch in tqdm(all_activations[layer], desc=f"Processing layer {layer} batches"):
                batch_size, seq_len, _ = batch.shape
                
                # Handle padding on CPU
                if seq_len < max_seq_len:
                    padding = torch.zeros(
                        batch_size, max_seq_len - seq_len, hidden_size,
                        dtype=batch.dtype, device='cpu'
                    )
                    padded_batch = torch.cat([batch, padding], dim=1)
                else:
                    padded_batch = batch
                
                # Copy to final tensor
                final_tensor[current_idx:current_idx + batch_size] = padded_batch
                current_idx += batch_size
                
                # Clear intermediate tensors
                del padded_batch
                gc.collect()
            
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

    def evaluate_conversations(self, conversations: List[str], position_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of conversations and return probe predictions.
        
        Args:
            conversations: List of conversation strings
            position_indices: List of position indices for each conversation
            
        Returns:
            List of dictionaries containing probe predictions and metadata
        """
        results = []
        
        for i, (conversation, position_idx) in enumerate(zip(conversations, position_indices)):
            try:
                # Tokenize the conversation
                tokens = self.tokenizer.encode(conversation, return_tensors="pt").to(self.device)
                
                # Check if sequence exceeds max length
                if tokens.shape[1] > self.max_seq_length:
                    print(f"Warning: Sequence length {tokens.shape[1]} exceeds max length {self.max_seq_length}")
                    print("Using chunking strategy to process long sequence...")
                    
                    # Use chunking strategy for long sequences
                    result = self._evaluate_long_sequence(conversation, position_idx, tokens)
                else:
                    # Process normally for shorter sequences
                    result = self._evaluate_normal_sequence(conversation, position_idx, tokens)
                
                results.append(result)
                
                # Memory cleanup
                del tokens
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing conversation {i}: {str(e)}")
                # Return a default result for failed conversations
                results.append({
                    'conversation_id': i,
                    'probe_prediction': 0.5,  # Neutral prediction
                    'error': str(e),
                    'sequence_length': len(conversation.split()) if conversation else 0
                })
                
        return results
    
    def _evaluate_normal_sequence(self, conversation: str, position_idx: int, tokens: torch.Tensor) -> Dict[str, Any]:
        """Evaluate a sequence that fits within the model's context window."""
        # Get model activations
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(tokens, names_filter=self.hook_name)
        
        # Extract activations at the specified position
        activations = cache[self.hook_name][0, position_idx, :]  # [d_model]
        
        # Apply probe
        probe_prediction = self.probe(activations.unsqueeze(0)).item()
        
        return {
            'conversation_id': hash(conversation) % 1000000,  # Simple ID
            'probe_prediction': probe_prediction,
            'sequence_length': tokens.shape[1],
            'position_index': position_idx,
            'processing_method': 'normal'
        }
    
    def _evaluate_long_sequence(self, conversation: str, position_idx: int, tokens: torch.Tensor) -> Dict[str, Any]:
        """Evaluate a long sequence using chunking strategy."""
        # Strategy: Take the last chunk that contains the position we're interested in
        seq_length = tokens.shape[1]
        
        # Calculate chunk boundaries
        if position_idx < self.max_seq_length:
            # Position is in the first chunk
            chunk_start = 0
            chunk_end = min(self.max_seq_length, seq_length)
            chunk_position_idx = position_idx
        else:
            # Position is beyond first chunk - take the last chunk that includes it
            chunk_end = min(position_idx + self.max_seq_length // 2, seq_length)
            chunk_start = max(0, chunk_end - self.max_seq_length)
            chunk_position_idx = position_idx - chunk_start
        
        # Extract the chunk
        chunk_tokens = tokens[:, chunk_start:chunk_end]
        
        print(f"Processing chunk: positions {chunk_start}-{chunk_end}, target position {chunk_position_idx}")
        
        # Process the chunk
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(chunk_tokens, names_filter=self.hook_name)
        
        # Extract activations at the specified position within the chunk
        activations = cache[self.hook_name][0, chunk_position_idx, :]  # [d_model]
        
        # Apply probe
        probe_prediction = self.probe(activations.unsqueeze(0)).item()
        
        return {
            'conversation_id': hash(conversation) % 1000000,  # Simple ID
            'probe_prediction': probe_prediction,
            'sequence_length': seq_length,
            'chunk_start': chunk_start,
            'chunk_end': chunk_end,
            'chunk_length': chunk_end - chunk_start,
            'position_index': position_idx,
            'chunk_position_index': chunk_position_idx,
            'processing_method': 'chunked'
        }
