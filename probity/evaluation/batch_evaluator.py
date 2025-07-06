import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from probity.probes import (
    BaseProbe,
    LogisticProbe, LogisticProbeConfig,
    PCAProbe, PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe, KMeansProbeConfig,
    LinearProbe, LinearProbeConfig
)


class OptimizedBatchProbeEvaluator:
    """Memory-optimized batch evaluator for probes with HuggingFace models"""
    
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name
        
        # Set model dtype for memory efficiency
        if torch.cuda.is_available() and 'cuda' in device:
            # Use bfloat16 for better memory efficiency on modern GPUs
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32
            
        print(f"Using model dtype: {self.model_dtype}")
        
        # Load HuggingFace model with proper RoPE scaling support
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # Configure RoPE scaling for Llama 3.3 models
        if "Llama-3.3-70B" in model_name:
            print("Configuring Llama 3.3 with extended context window using RoPE scaling...")
            
            config = AutoConfig.from_pretrained(model_name)
            print(f"Original max_position_embeddings: {config.max_position_embeddings}")
            
            # Configure RoPE scaling for extended context
            config.rope_scaling = {
                "type": "dynamic",
                "factor": 8.0  # This extends context window significantly
            }
            print(f"RoPE scaling factor: {config.rope_scaling['factor']}")
            print(f"Effective context window: ~{int(config.max_position_embeddings * config.rope_scaling['factor'])}")
            
            # Load model with modified config
            self.model = AutoModel.from_pretrained(
                model_name,
                config=config,
                torch_dtype=self.model_dtype,
                device_map=device,
                trust_remote_code=True
            )
        else:
            # Load normally for other models
            self.model = AutoModel.from_pretrained(
            model_name,
                torch_dtype=self.model_dtype,
                device_map=device,
                trust_remote_code=True
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()
        
        # Check the model's actual context window
        print(f"Model context window: {self.model.config.max_position_embeddings}")
        
        # Cache for activations
        self._activation_cache = {}
        
    def _parse_text_to_messages(self, text: str) -> List[Dict[str, str]]:
        """Simple parser for backward compatibility with text-based inputs"""
        # Very basic parsing - assumes format like "role: content\nrole: content"
        messages = []
        current_role = None
        current_content = []
        
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this is a role line
            if ':' in line and line.split(':', 1)[0].strip() in ['system', 'user', 'assistant']:
                # Save previous message
                if current_role and current_content:
                    messages.append({
                        "role": current_role,
                        "content": '\n'.join(current_content).strip()
                    })
                
                # Start new message
                parts = line.split(':', 1)
                current_role = parts[0].strip()
                current_content = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
            else:
                # Add to current content
                if current_role:
                    current_content.append(line)
        
        # Add final message
        if current_role and current_content:
            messages.append({
                "role": current_role,
                "content": '\n'.join(current_content).strip()
            })
        
        return messages
        
    def get_batch_activations(self, messages_list=None, texts=None, layers: List[int] = None, 
                            batch_size: int = 1) -> Dict[int, torch.Tensor]:
        """Get activations for all message lists/texts and layers efficiently using HuggingFace model
        
        Args:
            messages_list: List of message lists (new streaming approach)
            texts: List of text strings (old batch approach) 
            layers: List of layer indices
            batch_size: Batch size for processing
            
        Note: Provide either messages_list OR texts, not both
        """
        
        # Backward compatibility: handle both old and new approaches
        if texts is not None and messages_list is not None:
            raise ValueError("Provide either messages_list OR texts, not both")
        
        if texts is not None:
            # OLD APPROACH: Convert texts back to message format for processing
            print("DEBUG: Using backward compatibility mode with text inputs")
            messages_list = []
            for text in texts:
                # Parse text back to messages (simple parsing)
                messages = self._parse_text_to_messages(text)
                messages_list.append(messages)
        
        if messages_list is None:
            raise ValueError("Must provide either messages_list or texts")
        
        print("DEBUG: get_batch_activations called with message lists")
        print(f"DEBUG: Processing {len(messages_list)} conversations")
        
        # Process in smaller batches with memory cleanup
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        # Start with minimal batch size
        actual_batch_size = 1
        num_batches = (len(messages_list) + actual_batch_size - 1) // actual_batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(messages_list), actual_batch_size), total=num_batches, desc="Processing batches"):
                # Aggressive memory cleanup before each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                batch_messages = messages_list[i:i + actual_batch_size]
                
                # Apply chat template directly to message lists
                formatted_texts = []
                for idx, messages in enumerate(batch_messages):
                    print(f"DEBUG: Processing sample {idx}")
                    print(f"DEBUG: Sample {idx} has {len(messages)} messages")
                    
                    if not messages:  # Skip empty message lists
                        formatted_texts.append("")
                        continue
                    
                    # DEBUG: Show original messages before applying chat template
                    print(f"DEBUG: Original messages for sample {idx}:")
                    for msg_idx, msg in enumerate(messages):
                        print(f"  Message {msg_idx}: role={msg['role']}")
                        print(f"    Content length: {len(msg['content'])}")
                        print(f"    Content first 200 chars: {repr(msg['content'][:200])}")
                        
                        # Check for duplicate prefix in original content
                        if msg['role'] == 'system' and 'Cutting Knowledge Date' in msg['content']:
                            prefix_count = msg['content'].count('Cutting Knowledge Date')
                            print(f"    WARNING: Found {prefix_count} instances of 'Cutting Knowledge Date' in original system message!")
                            if prefix_count > 1:
                                print(f"    DUPLICATE DETECTED: Original dataset already has duplicate prefix!")
                    
                    # Apply chat template directly to the message structure
                    try:
                        # First try with add_generation_prompt=False to match original
                        formatted_text = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        
                        # Check if we need to add the double <|begin_of_text|> like the original
                        if not formatted_text.startswith("<|begin_of_text|><|begin_of_text|>"):
                            if formatted_text.startswith("<|begin_of_text|>"):
                                # Add the missing second <|begin_of_text|>
                                formatted_text = "<|begin_of_text|>" + formatted_text
                                print(f"DEBUG: Added missing second <|begin_of_text|> token")
                            else:
                                # Add both tokens
                                formatted_text = "<|begin_of_text|><|begin_of_text|>" + formatted_text
                                print(f"DEBUG: Added both <|begin_of_text|> tokens")
                        
                        print(f"DEBUG: Chat template applied successfully to sample {idx}")
                        print(f"DEBUG: Sample {idx} formatted length: {len(formatted_text)}")
                        print(f"DEBUG: Sample {idx} first 200 chars: {repr(formatted_text[:200])}")
                        
                        # Check for duplicates in final formatted text
                        if 'Cutting Knowledge Date' in formatted_text:
                            final_prefix_count = formatted_text.count('Cutting Knowledge Date')
                            print(f"DEBUG: Final formatted text has {final_prefix_count} instances of 'Cutting Knowledge Date'")
                            if final_prefix_count > 1:
                                print(f"DEBUG: DUPLICATION CONFIRMED in final output!")
                        
                        formatted_texts.append(formatted_text)
                        
                    except Exception as e:
                        print(f"ERROR: Chat template failed for sample {idx}: {e}")
                        print(f"DEBUG: Messages that failed: {messages}")
                        
                        # Fallback to simple concatenation
                        fallback_text = ""
                        for msg in messages:
                            fallback_text += f"{msg['role']}: {msg['content']}\n"
                        formatted_texts.append(fallback_text.strip())
                
                # Tokenize with chat template applied
                tokens = self.tokenizer(
                    formatted_texts, 
                    return_tensors="pt", 
                    padding=True,
                    truncation=True,
                    add_special_tokens=False  # Chat template already adds them
                ).to(self.device)
                
                try:
                    # Run model and extract activations from specified layers
                    outputs = self.model(
                        tokens["input_ids"],
                        attention_mask=tokens.get("attention_mask"),
                        output_hidden_states=True,
                        return_dict=True
                    )
                
                    # Extract hidden states for each requested layer
                    hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
                    
                    for layer in layers:
                        # Get activations for this layer - move to CPU immediately
                        if layer < len(hidden_states):
                            layer_activations = hidden_states[layer].cpu()
                            all_activations[layer].append(layer_activations)
                    
                    # Clear outputs reference and force cleanup
                    del outputs, hidden_states
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Store tokens
                    for j, messages in enumerate(batch_messages):
                        text_tokens = tokens["input_ids"][j]
                        if self.tokenizer.pad_token_id is not None:
                            pad_positions = torch.where(text_tokens == self.tokenizer.pad_token_id)[0]
                            if len(pad_positions) > 0:
                                first_pad_pos = pad_positions[0].item()
                                actual_tokens = text_tokens[:first_pad_pos]
                            else:
                                actual_tokens = text_tokens
                        else:
                            actual_tokens = text_tokens
                        
                        token_texts = self.tokenizer.convert_ids_to_tokens(actual_tokens)
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
        
        return result
    
    def evaluate_all_probes(self, messages_list=None, texts=None, labels: List[int] = None, 
                          probe_configs: Dict[Tuple[int, str], BaseProbe] = None) -> Dict[Tuple[int, str], Dict]:
        """Evaluate all probes efficiently using cached activations
        
        Args:
            messages_list: List of message lists (new streaming approach)
            texts: List of text strings (old batch approach)
            labels: List of binary labels
            probe_configs: Dictionary mapping (layer, probe_type) to probe instances
            
        Note: Provide either messages_list OR texts, not both
        """
        
        # Backward compatibility
        if texts is not None and messages_list is not None:
            raise ValueError("Provide either messages_list OR texts, not both")
        
        if texts is not None:
            print("DEBUG: evaluate_all_probes using backward compatibility mode")
            messages_list = []
            for text in texts:
                messages = self._parse_text_to_messages(text)
                messages_list.append(messages)
        
        if messages_list is None:
            raise ValueError("Must provide either messages_list or texts")
        
        # Extract unique layers from probe configs
        layers = list(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations for all required layers at once
        print("Getting activations for all layers...")
        activation_data = self.get_batch_activations(messages_list=messages_list, layers=layers)
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
                    messages_list, labels, tokens_by_text
                )
                
                results[(layer, probe_type)] = probe_results
        
        return results

    
    
    def _evaluate_single_probe_batch(self, probe: BaseProbe, 
                                   layer_activations: torch.Tensor,
                                   mean_activations: torch.Tensor,
                                   messages_list: List[List[Dict[str, str]]], labels: List[int],
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
            for i, (messages, true_label) in tqdm(enumerate(zip(messages_list, labels)), total=len(messages_list), desc="Processing texts"):
                # Clear memory before each sample
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    "messages": messages,
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
                "messages": sample["messages"],
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
        predictions = [1 if score > 0.5 else 0 for score in mean_scores]
        
        # Get precision, recall, f1 for binary classification with zero_division handling
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0.0
        )
        
        # Handle ROC AUC for single-class scenarios
        try:
            auroc = float(roc_auc_score(labels, mean_scores))
        except ValueError:
            # Single class case - ROC AUC is undefined
            auroc = 0.5  # Neutral value for single class
        
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auroc': auroc
        }
        
        # Prepare token_details for visualization
        token_details = []
        for sample in final_samples:
            # Clean tokens for visualization
            clean_tokens = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in sample["tokens"]]
            
            token_detail = {
                "messages": sample["messages"],
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
            'predictions': predictions
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
