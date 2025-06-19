"""
Custom data collator for packing multiple tokenized samples into efficient training batches.

This collator is designed to work with pre-tokenized datasets (those with input_ids, attention_mask, labels)
and packs multiple short sequences into single training examples to maximize GPU utilization.

Key features:
- Works with pre-tokenized data from custom chunking with overlap
- Packs multiple samples into sequences up to max_length
- Inserts EOS tokens between sequences for proper separation
- Maintains proper attention mask and label alignment
- Handles edge cases like single sequences and padding
- Supports blocking cross-document attention for document-level isolation
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class PackedSequenceCollator:
    """
    Custom data collator that packs multiple tokenized samples into batches up to max_length.
    
    This collator works with pre-tokenized datasets (those with input_ids, attention_mask, labels)
    and packs multiple short sequences into single training examples to maximize GPU utilization.
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        pad_token_id: Optional[int] = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        packing_efficiency_threshold: float = 0.8,
        pack_single_sequences: bool = False,
        block_cross_document_attention: bool = False,
    ):
        """
        Initialize the PackedSequenceCollator.
        
        Args:
            tokenizer: The tokenizer to use for special tokens
            max_length: Maximum sequence length for packed sequences
            pad_token_id: Token ID to use for padding (defaults to tokenizer.pad_token_id)
            label_pad_token_id: Token ID to use for padding labels (typically -100)
            return_tensors: Format for returned tensors ("pt" for PyTorch)
            packing_efficiency_threshold: Minimum ratio of used tokens to max_length to create a packed sequence
            pack_single_sequences: Whether to pack even when only one sequence fits
            block_cross_document_attention: Whether to block attention between different documents in packed sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id if pad_token_id is not None else tokenizer.pad_token_id
        self.label_pad_token_id = label_pad_token_id
        self.return_tensors = return_tensors
        self.packing_efficiency_threshold = packing_efficiency_threshold
        self.pack_single_sequences = pack_single_sequences
        self.block_cross_document_attention = block_cross_document_attention
        
        # Use EOS token for sequence separation
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
            logger.warning("No EOS token found in tokenizer. Using pad_token_id for sequence separation.")
            self.eos_token_id = self.pad_token_id
            
        # Ensure we have a pad token
        if self.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                self.pad_token_id = tokenizer.unk_token_id
                logger.warning("No pad token found. Using unk_token_id for padding.")
            else:
                raise ValueError("No pad token available in tokenizer. Please set pad_token_id manually.")
                
        logger.info(f"PackedSequenceCollator initialized with max_length={max_length}, "
                   f"eos_token_id={self.eos_token_id}, pad_token_id={self.pad_token_id}, "
                   f"block_cross_document_attention={block_cross_document_attention}")

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Pack multiple tokenized samples into efficient batches.
        
        Args:
            features: List of tokenized samples, each containing 'input_ids', 'attention_mask', 'labels'
            
        Returns:
            Dictionary with packed 'input_ids', 'attention_mask', 'labels' tensors
            If block_cross_document_attention is True, also includes 'attention_mask_2d'
        """
        if not features:
            raise ValueError("Cannot create batch from empty features list")
            
        # Validate that all features have required keys
        required_keys = {'input_ids', 'attention_mask', 'labels'}
        for i, feature in enumerate(features):
            missing_keys = required_keys - set(feature.keys())
            if missing_keys:
                raise ValueError(f"Feature {i} missing required keys: {missing_keys}")
        
        # Convert to tensors if needed and get lengths
        processed_features = []
        for feature in features:
            processed_feature = {}
            for key in required_keys:
                if isinstance(feature[key], torch.Tensor):
                    processed_feature[key] = feature[key]
                else:
                    processed_feature[key] = torch.tensor(feature[key])
            processed_features.append(processed_feature)
        
        # Pack sequences
        packed_sequences = self._pack_sequences(processed_features)
        
        # Convert to batch format
        batch = self._create_batch(packed_sequences)
        
        return batch
    
    def _pack_sequences(self, features: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """
        Pack multiple sequences into longer sequences up to max_length.
        
        Args:
            features: List of tokenized features
            
        Returns:
            List of packed sequences with EOS positions tracked for attention masking
        """
        packed_sequences = []
        current_pack = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
            'eos_positions': []  # Track EOS positions for cross-document attention blocking
        }
        current_length = 0
        
        for feature in features:
            seq_length = len(feature['input_ids'])
            
            # Check if this sequence would fit in current pack
            # +1 for EOS token if we're adding to an existing pack
            needed_length = seq_length + (1 if current_length > 0 else 0)
            
            if current_length + needed_length <= self.max_length:
                # Add EOS token between sequences if this isn't the first sequence
                if current_length > 0:
                    current_pack['input_ids'].append(self.eos_token_id)
                    current_pack['attention_mask'].append(1)
                    current_pack['labels'].append(self.label_pad_token_id)  # Don't train on EOS between sequences
                    current_pack['eos_positions'].append(current_length)  # Track EOS position
                    current_length += 1
                
                # Add the sequence
                current_pack['input_ids'].extend(feature['input_ids'].tolist())
                current_pack['attention_mask'].extend(feature['attention_mask'].tolist())
                current_pack['labels'].extend(feature['labels'].tolist())
                current_length += seq_length
                
            else:
                # Current sequence doesn't fit, finalize current pack if it meets threshold
                if self._should_finalize_pack(current_length):
                    packed_sequences.append(self._finalize_pack(current_pack, current_length))
                
                # Start new pack with current sequence
                current_pack = {
                    'input_ids': feature['input_ids'].tolist(),
                    'attention_mask': feature['attention_mask'].tolist(),
                    'labels': feature['labels'].tolist(),
                    'eos_positions': []  # Reset EOS positions for new pack
                }
                current_length = seq_length
        
        # Finalize the last pack if it exists
        if current_length > 0 and self._should_finalize_pack(current_length):
            packed_sequences.append(self._finalize_pack(current_pack, current_length))
        
        logger.debug(f"Packed {len(features)} sequences into {len(packed_sequences)} packed sequences")
        
        return packed_sequences
    
    def _should_finalize_pack(self, current_length: int) -> bool:
        """
        Determine if a pack should be finalized based on efficiency threshold.
        
        Args:
            current_length: Current length of the pack
            
        Returns:
            True if pack should be finalized
        """
        if current_length == 0:
            return False
            
        efficiency = current_length / self.max_length
        
        # Always finalize if we're at or near max_length
        if efficiency >= 0.95:
            return True
            
        # Check efficiency threshold, but allow single sequences if configured
        if efficiency >= self.packing_efficiency_threshold:
            return True
            
        # If pack_single_sequences is True, finalize even single sequences
        if self.pack_single_sequences and current_length > 0:
            return True
            
        return False
    
    def _finalize_pack(self, pack: Dict[str, List], current_length: int) -> Dict[str, Any]:
        """
        Finalize a pack by padding to max_length and converting to tensors.
        
        Args:
            pack: Dictionary containing lists of token IDs, attention mask, labels, and EOS positions
            current_length: Current length of the pack
            
        Returns:
            Dictionary with padded tensors and optional 2D attention mask
        """
        # Pad to max_length
        padding_length = self.max_length - current_length
        
        if padding_length > 0:
            pack['input_ids'].extend([self.pad_token_id] * padding_length)
            pack['attention_mask'].extend([0] * padding_length)  # Padding tokens get 0 attention
            pack['labels'].extend([self.label_pad_token_id] * padding_length)
        
        # Convert to tensors
        finalized_pack = {}
        for key, values in pack.items():
            if key == 'eos_positions':
                finalized_pack[key] = values  # Keep as list for now
            else:
                finalized_pack[key] = torch.tensor(values, dtype=torch.long)
        
        # Create 2D attention mask if cross-document attention blocking is enabled
        if self.block_cross_document_attention:
            finalized_pack['attention_mask_2d'] = self._create_document_attention_mask(
                finalized_pack['attention_mask'], 
                pack['eos_positions']
            )
        
        return finalized_pack
    
    def _create_document_attention_mask(self, attention_mask_1d: torch.Tensor, eos_positions: List[int]) -> torch.Tensor:
        """
        Create a 2D attention mask that blocks attention between different documents.
        
        Args:
            attention_mask_1d: 1D attention mask (1 for real tokens, 0 for padding)
            eos_positions: List of positions where EOS tokens are located
            
        Returns:
            2D attention mask of shape (seq_len, seq_len) where 0 blocks attention
        """
        seq_len = len(attention_mask_1d)
        
        # Start with standard causal mask (lower triangular)
        attention_mask_2d = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.long))
        
        # Apply padding mask (tokens can't attend to padding)
        padding_mask = attention_mask_1d.unsqueeze(0).expand(seq_len, -1)
        attention_mask_2d = attention_mask_2d * padding_mask
        
        # Block cross-document attention
        if eos_positions:
            # Define document boundaries
            doc_starts = [0] + [pos + 1 for pos in eos_positions]  # Start of each document
            doc_ends = eos_positions + [attention_mask_1d.sum().item()]  # End of each document (last non-padding token)
            
            # For each pair of documents, block attention between them
            for i, (start_i, end_i) in enumerate(zip(doc_starts, doc_ends)):
                for j, (start_j, end_j) in enumerate(zip(doc_starts, doc_ends)):
                    if i != j:  # Different documents
                        # Block attention from document i to document j
                        attention_mask_2d[start_i:end_i, start_j:end_j] = 0
        
        logger.debug(f"Created 2D attention mask with {len(eos_positions)} document boundaries")
        
        return attention_mask_2d
    
    def _create_batch(self, packed_sequences: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Create a batch from packed sequences.
        
        Args:
            packed_sequences: List of packed sequence dictionaries
            
        Returns:
            Batch dictionary with stacked tensors
        """
        if not packed_sequences:
            raise ValueError("Cannot create batch from empty packed sequences")
        
        # Stack all sequences into batch tensors
        batch = {}
        tensor_keys = ['input_ids', 'attention_mask', 'labels']
        
        for key in tensor_keys:
            batch[key] = torch.stack([seq[key] for seq in packed_sequences])
        
        # Handle 2D attention masks if present
        if 'attention_mask_2d' in packed_sequences[0]:
            batch['attention_mask_2d'] = torch.stack([seq['attention_mask_2d'] for seq in packed_sequences])
            logger.debug(f"Added 2D attention masks to batch with shape: {batch['attention_mask_2d'].shape}")
        
        # Log some stats
        batch_size = batch['input_ids'].shape[0]
        seq_length = batch['input_ids'].shape[1]
        
        # Calculate efficiency (non-padding tokens / total tokens)
        non_padding_tokens = (batch['attention_mask'] == 1).sum().item()
        total_tokens = batch_size * seq_length
        efficiency = non_padding_tokens / total_tokens if total_tokens > 0 else 0
        
        cross_doc_status = "blocked" if self.block_cross_document_attention else "allowed"
        logger.debug(f"Created batch: {batch_size} sequences, {seq_length} max length, "
                    f"{efficiency:.2%} efficiency ({non_padding_tokens}/{total_tokens} tokens), "
                    f"cross-document attention {cross_doc_status}")
        
        return batch

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the collator configuration."""
        return {
            "max_length": self.max_length,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "label_pad_token_id": self.label_pad_token_id,
            "packing_efficiency_threshold": self.packing_efficiency_threshold,
            "pack_single_sequences": self.pack_single_sequences,
            "block_cross_document_attention": self.block_cross_document_attention,
        } 