"""
Tests for cross-document attention blocking in PackedSequenceCollator.

Tests cover:
- Basic 2D attention mask creation
- Document boundary detection from EOS positions
- Proper masking between different documents
- Causal masking preservation within documents
- Padding mask application
- Integration with regular packing functionality
"""

import pytest
import torch
from unittest.mock import Mock

from src.training.packed_sequence_collator import PackedSequenceCollator


class TestCrossDocumentAttention:
    """Test suite for cross-document attention blocking functionality."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with standard special tokens."""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1
        tokenizer.unk_token_id = 2
        return tokenizer

    @pytest.fixture
    def blocking_collator(self, mock_tokenizer):
        """Create a PackedSequenceCollator with cross-document attention blocking enabled."""
        return PackedSequenceCollator(
            tokenizer=mock_tokenizer,
            max_length=30,
            packing_efficiency_threshold=0.6,
            pack_single_sequences=True,
            block_cross_document_attention=True,
        )

    @pytest.fixture
    def regular_collator(self, mock_tokenizer):
        """Create a PackedSequenceCollator with regular (allowing) cross-document attention."""
        return PackedSequenceCollator(
            tokenizer=mock_tokenizer,
            max_length=30,
            packing_efficiency_threshold=0.6,
            pack_single_sequences=True,
            block_cross_document_attention=False,
        )

    @pytest.fixture
    def multi_doc_features(self):
        """Create features representing multiple documents for testing."""
        return [
            {
                "input_ids": [10, 11, 12],  # Doc 1: 3 tokens
                "attention_mask": [1, 1, 1],
                "labels": [10, 11, 12],
            },
            {
                "input_ids": [20, 21, 22, 23],  # Doc 2: 4 tokens  
                "attention_mask": [1, 1, 1, 1],
                "labels": [20, 21, 22, 23],
            },
            {
                "input_ids": [30, 31],  # Doc 3: 2 tokens
                "attention_mask": [1, 1],
                "labels": [30, 31],
            },
        ]

    def test_2d_attention_mask_creation(self, blocking_collator, multi_doc_features):
        """Test that 2D attention masks are created when blocking is enabled."""
        result = blocking_collator(multi_doc_features)
        
        # Should have regular keys plus 2D attention mask
        assert "input_ids" in result
        assert "attention_mask" in result  # 1D mask
        assert "attention_mask_2d" in result  # 2D mask
        assert "labels" in result
        
        # Check shapes
        batch_size, seq_length = result["input_ids"].shape
        assert result["attention_mask_2d"].shape == (batch_size, seq_length, seq_length)

    def test_no_2d_mask_when_disabled(self, regular_collator, multi_doc_features):
        """Test that 2D attention masks are NOT created when blocking is disabled."""
        result = regular_collator(multi_doc_features)
        
        # Should only have regular keys
        assert "input_ids" in result
        assert "attention_mask" in result  # 1D mask only
        assert "attention_mask_2d" not in result
        assert "labels" in result

    def test_document_boundary_detection(self, blocking_collator, multi_doc_features):
        """Test that document boundaries are correctly identified from EOS positions."""
        result = blocking_collator(multi_doc_features)
        
        # Get the packed sequence
        input_ids = result["input_ids"][0]
        attention_mask_2d = result["attention_mask_2d"][0]
        
        # Find EOS positions
        eos_positions = (input_ids == 1).nonzero(as_tuple=True)[0]
        
        # Should have EOS tokens between documents
        assert len(eos_positions) >= 1  # At least one EOS between docs
        
        # Verify EOS positions make sense (not at start or very end)
        for pos in eos_positions:
            assert pos > 0  # Not at the very start
            assert pos < len(input_ids) - 5  # Not at the very end (account for padding)

    def test_cross_document_attention_blocked(self, blocking_collator, multi_doc_features):
        """Test that attention between different documents is blocked."""
        result = blocking_collator(multi_doc_features)
        
        input_ids = result["input_ids"][0]
        attention_mask_2d = result["attention_mask_2d"][0]
        
        # Find document boundaries (EOS positions)
        eos_positions = (input_ids == 1).nonzero(as_tuple=True)[0].tolist()
        
        if len(eos_positions) >= 1:
            # Define document ranges
            doc_starts = [0] + [pos + 1 for pos in eos_positions]
            # Find last non-padding token
            last_token = (result["attention_mask"][0] == 1).sum().item()
            doc_ends = eos_positions + [last_token]
            
            # Check that cross-document attention is blocked
            for i, (start_i, end_i) in enumerate(zip(doc_starts, doc_ends)):
                for j, (start_j, end_j) in enumerate(zip(doc_starts, doc_ends)):
                    if i != j and start_i < len(attention_mask_2d) and start_j < len(attention_mask_2d):
                        # Different documents - attention should be blocked
                        cross_attention = attention_mask_2d[start_i:end_i, start_j:end_j]
                        assert torch.all(cross_attention == 0), f"Cross-attention not blocked between doc {i} and doc {j}"

    def test_intra_document_attention_preserved(self, blocking_collator, multi_doc_features):
        """Test that attention within the same document is preserved (causal masking)."""
        result = blocking_collator(multi_doc_features)
        
        input_ids = result["input_ids"][0]
        attention_mask_2d = result["attention_mask_2d"][0]
        
        # Find document boundaries
        eos_positions = (input_ids == 1).nonzero(as_tuple=True)[0].tolist()
        
        if len(eos_positions) >= 1:
            # Check first document (before first EOS)
            first_doc_end = eos_positions[0]
            
            # Within first document, should have causal attention (lower triangular)
            first_doc_attention = attention_mask_2d[0:first_doc_end, 0:first_doc_end]
            
            # Should be lower triangular (causal masking)
            for i in range(first_doc_end):
                for j in range(first_doc_end):
                    if i >= j:  # Lower triangular
                        assert first_doc_attention[i, j] == 1, f"Missing causal attention at ({i}, {j})"
                    else:  # Upper triangular should be 0
                        assert first_doc_attention[i, j] == 0, f"Unexpected future attention at ({i}, {j})"

    def test_padding_mask_applied(self, blocking_collator, multi_doc_features):
        """Test that padding tokens are properly masked in 2D attention."""
        result = blocking_collator(multi_doc_features)
        
        attention_mask_1d = result["attention_mask"][0]
        attention_mask_2d = result["attention_mask_2d"][0]
        
        # Find padding positions
        padding_positions = (attention_mask_1d == 0).nonzero(as_tuple=True)[0]
        
        if len(padding_positions) > 0:
            # No token should attend to padding positions
            for pad_pos in padding_positions:
                assert torch.all(attention_mask_2d[:, pad_pos] == 0), f"Tokens attending to padding at position {pad_pos}"

    def test_single_document_behavior(self, blocking_collator):
        """Test behavior with a single document (no EOS tokens)."""
        single_doc_features = [
            {
                "input_ids": [10, 11, 12, 13, 14],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [10, 11, 12, 13, 14],
            }
        ]
        
        result = blocking_collator(single_doc_features)
        
        # Should still create 2D mask
        assert "attention_mask_2d" in result
        
        attention_mask_2d = result["attention_mask_2d"][0]
        
        # With single document, should just be causal attention
        seq_len = (result["attention_mask"][0] == 1).sum().item()
        
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:  # Lower triangular
                    assert attention_mask_2d[i, j] == 1
                else:  # Upper triangular
                    assert attention_mask_2d[i, j] == 0

    def test_stats_include_blocking_config(self, blocking_collator, regular_collator):
        """Test that collator stats include the blocking configuration."""
        blocking_stats = blocking_collator.get_stats()
        regular_stats = regular_collator.get_stats()
        
        assert "block_cross_document_attention" in blocking_stats
        assert "block_cross_document_attention" in regular_stats
        
        assert blocking_stats["block_cross_document_attention"] is True
        assert regular_stats["block_cross_document_attention"] is False

    def test_attention_mask_shapes_consistent(self, blocking_collator, multi_doc_features):
        """Test that 1D and 2D attention mask shapes are consistent."""
        result = blocking_collator(multi_doc_features)
        
        batch_size, seq_length = result["input_ids"].shape
        
        # 1D mask shape
        assert result["attention_mask"].shape == (batch_size, seq_length)
        
        # 2D mask shape
        assert result["attention_mask_2d"].shape == (batch_size, seq_length, seq_length)

    def test_blocking_with_different_efficiency_thresholds(self, mock_tokenizer):
        """Test that blocking works with different efficiency thresholds."""
        thresholds = [0.5, 0.8, 0.95]
        
        features = [
            {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [10, 11]},
            {"input_ids": [20, 21], "attention_mask": [1, 1], "labels": [20, 21]},
        ]
        
        for threshold in thresholds:
            collator = PackedSequenceCollator(
                tokenizer=mock_tokenizer,
                max_length=20,
                packing_efficiency_threshold=threshold,
                pack_single_sequences=True,
                block_cross_document_attention=True,
            )
            
            result = collator(features)
            
            # Should always create 2D masks when blocking is enabled
            assert "attention_mask_2d" in result
            
            # Basic shape checks
            batch_size, seq_length = result["input_ids"].shape
            assert result["attention_mask_2d"].shape == (batch_size, seq_length, seq_length)

    def test_logging_includes_cross_document_status(self, blocking_collator, regular_collator, caplog):
        """Test that logging includes cross-document attention status."""
        import logging
        
        # Create features to trigger logging
        features = [
            {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [10, 11]},
        ]
        
        with caplog.at_level(logging.DEBUG):
            blocking_collator(features)
            
        # Check that debug logs mention cross-document attention
        debug_messages = [record.message for record in caplog.records if record.levelno == logging.DEBUG]
        
        # Should have logging about cross-document attention status
        cross_doc_logs = [msg for msg in debug_messages if "cross-document attention" in msg]
        assert len(cross_doc_logs) > 0

    def test_eos_position_tracking_accuracy(self, blocking_collator):
        """Test that EOS positions are tracked accurately during packing."""
        # Create features that will definitely result in EOS tokens
        features = [
            {"input_ids": [10, 11], "attention_mask": [1, 1], "labels": [10, 11]},
            {"input_ids": [20, 21], "attention_mask": [1, 1], "labels": [20, 21]},
            {"input_ids": [30, 31], "attention_mask": [1, 1], "labels": [30, 31]},
        ]
        
        result = blocking_collator(features)
        
        input_ids = result["input_ids"][0]
        
        # Find actual EOS positions in final sequence
        actual_eos_positions = (input_ids == 1).nonzero(as_tuple=True)[0].tolist()
        
        # Should have EOS tokens between documents
        assert len(actual_eos_positions) >= 1
        
        # EOS positions should be where we expect them (between document content)
        # Structure should be: [doc1] [EOS] [doc2] [EOS] [doc3] [padding...]
        for pos in actual_eos_positions:
            # EOS should not be at the very start
            assert pos > 0
            # Token before EOS should not be padding (0) or another EOS (1)
            assert input_ids[pos - 1] not in [0, 1] 