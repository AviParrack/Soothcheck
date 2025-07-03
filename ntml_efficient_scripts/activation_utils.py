"""
Efficient activation caching and extraction for NTML binary token training.

Adapts the existing smart caching infrastructure for assistant token extraction.
"""

import torch
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm
import logging

from .config import NTMLBinaryTrainingConfig
from .data_loading import NTMLBinaryDataset, BinaryTokenExample

logger = logging.getLogger(__name__)


class NTMLActivationCache:
    """Manages activation caching and extraction for NTML binary training."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        self.model = None
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_dataset_hash(self, dataset: NTMLBinaryDataset) -> str:
        """Generate a hash for the dataset to enable smart caching."""
        # Create hash based on dataset characteristics
        hash_components = [
            self.config.dataset_path,
            self.config.model_name,
            self.config.hook_point,
            self.config.max_length,
            len(dataset.examples),
            str(dataset.label_distribution),
        ]
        
        # Sample first few examples for content-based hash
        if dataset.examples:
            for ex in dataset.examples[:3]:
                hash_components.extend([
                    ex.conversation_id,
                    str(ex.tokens[:10]),  # First 10 tokens
                    str(ex.labels[:10]),  # First 10 labels
                ])
        
        content_str = "|".join(str(comp) for comp in hash_components)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def load_model(self):
        """Load the transformer model for activation extraction."""
        if self.model is not None:
            return
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            self.model = HookedTransformer.from_pretrained_no_processing(
                self.config.model_name,
                device=self.config.device,
                dtype=self.config.torch_dtype,
            )
        except Exception as e:
            logger.warning(f"Error with from_pretrained_no_processing: {e}")
            logger.info("Attempting alternative loading method...")
            
            from transformers import AutoModelForCausalLM
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device
            )
            self.model = HookedTransformer.from_pretrained(
                self.config.model_name,
                hf_model=hf_model,
                device=self.config.device,
                dtype=self.config.torch_dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
        
        self.model.eval()
        logger.info(f"Model loaded with dtype: {self.config.torch_dtype}")
    
    def get_cache_paths(self, dataset_hash: str) -> Dict[str, Path]:
        """Get cache file paths for this dataset."""
        cache_base = self.cache_dir / f"ntml_{dataset_hash}"
        
        return {
            "activations": cache_base / "activations.pt",
            "metadata": cache_base / "metadata.json",
            "dataset_info": cache_base / "dataset_info.json",
        }
    
    def check_cache_validity(self, dataset: NTMLBinaryDataset, dataset_hash: str) -> bool:
        """Check if cached activations are valid for this dataset."""
        cache_paths = self.get_cache_paths(dataset_hash)
        
        # Check if all required files exist
        if not all(path.exists() for path in cache_paths.values()):
            return False
        
        try:
            # Load and verify metadata
            with open(cache_paths["metadata"], 'r') as f:
                metadata = json.load(f)
            
            # Verify compatibility
            required_matches = [
                metadata.get("model_name") == self.config.model_name,
                metadata.get("hook_point") == self.config.hook_point,
                metadata.get("dataset_size") == len(dataset.examples),
                metadata.get("dtype") == self.config.dtype,
                metadata.get("max_length") == self.config.max_length,
            ]
            
            if not all(required_matches):
                logger.info("Cache metadata mismatch, will recollect activations")
                return False
            
            # Check dataset info
            with open(cache_paths["dataset_info"], 'r') as f:
                dataset_info = json.load(f)
            
            if dataset_info.get("label_distribution") != dataset.label_distribution:
                logger.info("Dataset label distribution changed, will recollect activations")
                return False
            
            logger.info("Found valid activation cache")
            return True
            
        except Exception as e:
            logger.warning(f"Error validating cache: {e}")
            return False
    
    def collect_activations(self, dataset: NTMLBinaryDataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Collect activations for all assistant tokens in the dataset."""
        
        dataset_hash = self.get_dataset_hash(dataset)
        cache_paths = self.get_cache_paths(dataset_hash)
        
        # Check cache validity
        if not self.config.force_recache and self.check_cache_validity(dataset, dataset_hash):
            return self.load_cached_activations(cache_paths)
        
        # Load model if needed
        self.load_model()
        
        logger.info(f"Collecting activations from {self.config.hook_point}")
        logger.info(f"Processing {len(dataset.examples)} examples in batches of {self.config.activation_batch_size}")
        
        all_activations = []
        all_labels = []
        all_assistant_masks = []
        
        hook_point = self.config.hook_point
        
        # Process in batches
        with torch.no_grad():
            for batch_start in tqdm(
                range(0, len(dataset.examples), self.config.activation_batch_size),
                desc="Collecting activations"
            ):
                batch_end = min(batch_start + self.config.activation_batch_size, len(dataset.examples))
                batch_examples = dataset.examples[batch_start:batch_end]
                
                # Prepare batch tensors
                batch_input_ids = []
                batch_attention_masks = []
                batch_labels = []
                batch_assistant_masks = []
                
                for example in batch_examples:
                    batch_input_ids.append(example.tokens)
                    batch_attention_masks.append(example.attention_mask)
                    batch_labels.append(example.labels)
                    batch_assistant_masks.append(example.assistant_mask)
                
                # Convert to tensors
                input_ids = torch.tensor(batch_input_ids, dtype=torch.long).to(self.config.device)
                attention_mask = torch.tensor(batch_attention_masks, dtype=torch.long).to(self.config.device)
                
                # Run model with caching
                _, cache = self.model.run_with_cache(
                    input_ids,
                    names_filter=[hook_point],
                    return_cache_object=True,
                    stop_at_layer=self.config.hook_layer + 1
                )
                
                # Extract activations
                batch_activations = cache[hook_point]  # Shape: [batch_size, seq_len, hidden_size]
                
                # Store activations (move to CPU to save GPU memory)
                all_activations.append(batch_activations.cpu())
                all_labels.append(torch.tensor(batch_labels, dtype=torch.long))
                all_assistant_masks.append(torch.tensor(batch_assistant_masks, dtype=torch.long))
        
        # Concatenate all batches
        all_activations = torch.cat(all_activations, dim=0)  # [total_examples, seq_len, hidden_size]
        all_labels = torch.cat(all_labels, dim=0)  # [total_examples, seq_len]
        all_assistant_masks = torch.cat(all_assistant_masks, dim=0)  # [total_examples, seq_len]
        
        logger.info(f"Collected activations: {all_activations.shape}")
        logger.info(f"Labels shape: {all_labels.shape}")
        logger.info(f"Assistant masks shape: {all_assistant_masks.shape}")
        
        # Save to cache
        self.save_activations_to_cache(
            all_activations, all_labels, all_assistant_masks, 
            dataset, dataset_hash, cache_paths
        )
        
        return all_activations, all_labels, all_assistant_masks
    
    def save_activations_to_cache(self, activations: torch.Tensor, labels: torch.Tensor, 
                                assistant_masks: torch.Tensor, dataset: NTMLBinaryDataset,
                                dataset_hash: str, cache_paths: Dict[str, Path]):
        """Save activations and metadata to cache."""
        
        logger.info(f"Saving activations to cache: {cache_paths['activations']}")
        
        # Create cache directory
        cache_paths["activations"].parent.mkdir(parents=True, exist_ok=True)
        
        # Save activations, labels, and masks
        torch.save({
            "activations": activations,
            "labels": labels,
            "assistant_masks": assistant_masks,
            "hidden_size": activations.shape[-1],
        }, cache_paths["activations"])
        
        # Save metadata
        metadata = {
            "model_name": self.config.model_name,
            "hook_point": self.config.hook_point,
            "hook_layer": self.config.hook_layer,
            "dataset_size": len(dataset.examples),
            "dtype": self.config.dtype,
            "max_length": self.config.max_length,
            "dataset_hash": dataset_hash,
            "activation_shape": list(activations.shape),
            "cache_version": "1.0",
        }
        
        with open(cache_paths["metadata"], 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save dataset info
        dataset_info = {
            "label_distribution": dataset.label_distribution,
            "num_examples": len(dataset.examples),
            "dataset_path": self.config.dataset_path,
        }
        
        with open(cache_paths["dataset_info"], 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        logger.info("Activations saved to cache successfully")
    
    def load_cached_activations(self, cache_paths: Dict[str, Path]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load activations from cache."""
        
        logger.info(f"Loading cached activations from {cache_paths['activations']}")
        
        cached_data = torch.load(cache_paths["activations"], map_location="cpu")
        
        activations = cached_data["activations"]
        labels = cached_data["labels"]
        assistant_masks = cached_data["assistant_masks"]
        
        logger.info(f"Loaded cached activations: {activations.shape}")
        return activations, labels, assistant_masks
    
    def extract_training_data(self, activations: torch.Tensor, labels: torch.Tensor, 
                            assistant_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract only assistant tokens for training."""
        
        logger.info("Extracting assistant token data for training")
        
        # Flatten all dimensions except the last (hidden_size)
        batch_size, seq_len, hidden_size = activations.shape
        
        flat_activations = activations.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
        flat_labels = labels.view(-1)  # [batch_size * seq_len]
        flat_assistant_masks = assistant_masks.view(-1)  # [batch_size * seq_len]
        
        # Extract only assistant tokens
        assistant_indices = (flat_assistant_masks == 1).nonzero(as_tuple=True)[0]
        
        assistant_activations = flat_activations[assistant_indices]  # [num_assistant_tokens, hidden_size]
        assistant_labels = flat_labels[assistant_indices]  # [num_assistant_tokens]
        
        logger.info(f"Extracted {len(assistant_activations)} assistant tokens for training")
        logger.info(f"Label distribution: {torch.bincount(assistant_labels).tolist()}")
        
        return assistant_activations, assistant_labels
    
    def cleanup_model(self):
        """Free model memory."""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Model memory freed")


def prepare_ntml_training_data(config: NTMLBinaryTrainingConfig) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Main function to prepare training data from NTML dataset.
    
    Returns:
        assistant_activations: Activations for assistant tokens [num_tokens, hidden_size]
        assistant_labels: Binary labels for assistant tokens [num_tokens]
        metadata: Dictionary with dataset and processing information
    """
    
    # Load and process dataset
    logger.info("Loading NTML dataset...")
    dataset = NTMLBinaryDataset(config)
    dataset.load_and_process()
    
    # Save debug info if requested
    if config.verbose:
        debug_dir = Path(config.output_dir) / "debug"
        dataset.save_debug_info(str(debug_dir))
    
    # Collect activations
    logger.info("Collecting activations...")
    activation_cache = NTMLActivationCache(config)
    
    try:
        # Get full activations for all tokens
        all_activations, all_labels, all_assistant_masks = activation_cache.collect_activations(dataset)
        
        # Extract only assistant tokens for training
        assistant_activations, assistant_labels = activation_cache.extract_training_data(
            all_activations, all_labels, all_assistant_masks
        )
        
        # Prepare metadata
        metadata = {
            "dataset_info": dataset.label_distribution,
            "num_examples": len(dataset.examples),
            "num_assistant_tokens": len(assistant_activations),
            "hidden_size": assistant_activations.shape[-1],
            "config": config.to_dict(),
        }
        
        return assistant_activations, assistant_labels, metadata
        
    finally:
        # Always cleanup model memory
        activation_cache.cleanup_model()