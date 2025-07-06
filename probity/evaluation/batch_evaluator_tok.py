from typing import Dict, List, Optional, Tuple, Union
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from .batch_evaluator import OptimizedBatchProbeEvaluator

class TokenBasedBatchEvaluator(OptimizedBatchProbeEvaluator):
    """Evaluator that accepts pre-tokenized input to ensure exact token matching."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        super().__init__(model_name, device)
        
        # Set model dtype for memory efficiency
        if torch.cuda.is_available() and 'cuda' in device:
            self.model_dtype = torch.bfloat16
        else:
            self.model_dtype = torch.float32
        print(f"Using model dtype: {self.model_dtype}")
        
    def get_batch_activations_from_tokens(
        self,
        token_lists: List[List[int]],  # List of token ID lists
        layers: List[int],
        batch_size: int = 1
    ) -> Dict:
        """Get activations from pre-tokenized input to ensure exact token matching."""
        print(f"\nProcessing {len(token_lists)} pre-tokenized sequences")
        print(f"Using batch size: {batch_size}")
        
        # Initialize storage
        all_activations = {layer: [] for layer in layers}
        max_length = max(len(tokens) for tokens in token_lists)
        
        # Process in batches with aggressive memory cleanup
        for batch_start in tqdm(range(0, len(token_lists), batch_size)):
            # Clear memory before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            batch_end = min(batch_start + batch_size, len(token_lists))
            batch_tokens = token_lists[batch_start:batch_end]
            
            # Pad batch to max length in this batch
            batch_max_len = max(len(tokens) for tokens in batch_tokens)
            padded_batch = [
                tokens + [self.tokenizer.pad_token_id] * (batch_max_len - len(tokens))
                for tokens in batch_tokens
            ]
            
            # Convert to tensors with proper dtype
            input_ids = torch.tensor(padded_batch, device=self.device)
            attention_mask = torch.where(
                input_ids == self.tokenizer.pad_token_id,
                0, 1
            )
            
            # Get model outputs
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # Extract activations for each layer
            hidden_states = outputs.hidden_states
            
            for layer in layers:
                # Move to CPU immediately to free GPU memory
                layer_activations = hidden_states[layer + 1].cpu()  # +1 for embeddings
                all_activations[layer].append(layer_activations)
                
                # Clear layer data
                del layer_activations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Clear GPU memory
            del outputs, hidden_states, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Process final activations in chunks
        print("\nProcessing final activations...")
        combined_activations = {}
        for layer in layers:
            print(f"Combining layer {layer} activations...")
            
            # Get dimensions
            total_samples = sum(batch.shape[0] for batch in all_activations[layer])
            max_seq_len = max(batch.shape[1] for batch in all_activations[layer])
            hidden_size = all_activations[layer][0].shape[2]
            
            # Initialize empty tensor on CPU
            final_tensor = torch.zeros(
                (total_samples, max_seq_len, hidden_size),
                dtype=all_activations[layer][0].dtype,
                device='cpu'
            )
            
            # Fill tensor batch by batch
            current_idx = 0
            for batch in all_activations[layer]:
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
            
            combined_activations[layer] = final_tensor
            
            # Clear layer data
            del all_activations[layer]
            gc.collect()
        
        # Clear intermediate results
        del all_activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'activations': combined_activations,
            'tokens_by_text': token_lists
        }
        
    def evaluate_all_probes_from_tokens(
        self,
        token_lists: List[List[int]],  # List of token ID lists
        labels: List[int],
        probe_configs: Dict,
        batch_size: int = 1
    ) -> Dict:
        """Evaluate all probes using pre-tokenized input.
        
        Args:
            token_lists: List of lists of token IDs
            labels: List of labels (0/1)
            probe_configs: Dict mapping (layer, probe_type) to probe
            batch_size: Batch size for processing
            
        Returns:
            Dict mapping (layer, probe_type) to results
        """
        # Get unique layers
        layers = sorted(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations
        activation_data = self.get_batch_activations_from_tokens(
            token_lists=token_lists,
            layers=layers,
            batch_size=batch_size
        )
        
        # Evaluate each probe
        results = {}
        for (layer, probe_type), probe in probe_configs.items():
            print(f"\nEvaluating {probe_type} probe on layer {layer}")
            
            layer_results = {
                'metrics': {},
                'all_samples': [],
                'token_details': []
            }
            
            # Get layer activations
            layer_activations = activation_data['activations'][layer]
            
            # Process each sequence
            for i, (tokens, label) in enumerate(zip(token_lists, labels)):
                # Get activations for this sequence
                seq_activations = layer_activations[i, :len(tokens), :].to(self.device)
                
                # Apply probe
                with torch.no_grad():
                    token_scores = probe(seq_activations)
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)
                    
                token_scores = token_scores.cpu().squeeze().tolist()
                if isinstance(token_scores, float):
                    token_scores = [token_scores]
                
                # Store results
                sample_result = {
                    'tokens': tokens,
                    'token_scores': token_scores,
                    'label': label,
                    'mean_score': sum(token_scores) / len(token_scores)
                }
                layer_results['all_samples'].append(sample_result)
                
                # Store token details
                token_details = []
                for tok_idx, (token, score) in enumerate(zip(tokens, token_scores)):
                    token_text = self.tokenizer.decode([token])
                    token_details.append({
                        'token_id': token,
                        'token_text': token_text,
                        'position': tok_idx,
                        'score': score
                    })
                layer_results['token_details'].append(token_details)
            
            # Calculate metrics
            predictions = [
                1 if sample['mean_score'] > 0.5 else 0
                for sample in layer_results['all_samples']
            ]
            correct = sum(p == l for p, l in zip(predictions, labels))
            accuracy = correct / len(predictions)
            layer_results['metrics']['accuracy'] = accuracy
            
            results[(layer, probe_type)] = layer_results
            
            print(f"Accuracy: {accuracy:.4f}")
            
        return results 