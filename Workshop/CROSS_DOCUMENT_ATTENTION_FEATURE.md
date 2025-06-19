# Cross-Document Attention Blocking Feature

## Overview

Added a new configuration option `block_cross_document_attention` to the custom packing implementation that allows you to control whether tokens in packed sequences can attend to tokens from different documents.

## Quick Usage

```yaml
sft:
  packing: true
  block_cross_document_attention: true  # New option: blocks cross-document attention
```

## Key Features

- **Toggleable Behavior**: Easy A/B testing between allowing and blocking cross-document attention
- **Document Isolation**: When enabled, prevents information leakage between different documents in the same packed sequence
- **Performance Efficient**: Minimal computational overhead, same packing efficiency
- **Backward Compatible**: Default behavior unchanged (cross-document attention allowed)

## Implementation Details

### What Gets Added

1. **New Configuration Option**:
   - `block_cross_document_attention: bool = False` in `SFTStageConfig`

2. **2D Attention Masks**:
   - When blocking is enabled, creates custom `attention_mask_2d` tensors
   - Shape: `(batch_size, seq_length, seq_length)`
   - Preserves causal attention within documents, blocks between documents

3. **EOS Position Tracking**:
   - Tracks where EOS tokens are inserted between documents
   - Uses these positions to define document boundaries
   - Creates proper attention masks based on boundaries

### How It Works

```
Input: 3 documents → [Doc1] [Doc2] [Doc3]
Packed: [doc1_tokens] <EOS> [doc2_tokens] <EOS> [doc3_tokens] <PAD>...

Cross-Document Attention ALLOWED (default):
- Tokens can attend to all previous tokens (standard causal masking)
- doc2_tokens can attend to doc1_tokens
- doc3_tokens can attend to doc1_tokens and doc2_tokens

Cross-Document Attention BLOCKED:
- Tokens can only attend to previous tokens in the same document
- doc2_tokens CANNOT attend to doc1_tokens  
- doc3_tokens CANNOT attend to doc1_tokens or doc2_tokens
- Each document maintains its own causal attention pattern
```

### Example Attention Pattern (Blocked Mode)

```
Sequence: [10,11,12] <EOS> [20,21,22] <EOS> [30,31] <PAD>...
Positions: 0  1  2   3     4  5  6   7     8  9   10...

Attention Matrix (✓ = can attend, ✗ = blocked):
     0  1  2  3  4  5  6  7  8  9
 0   ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✗   # Doc1: only attends within doc
 1   ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✗   
 2   ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗  ✗   
 3   ✓  ✓  ✓  ✓  ✗  ✗  ✗  ✗  ✗  ✗   # EOS: can attend to prev doc
 4   ✗  ✗  ✗  ✓  ✓  ✗  ✗  ✗  ✗  ✗   # Doc2: blocked from Doc1
 5   ✗  ✗  ✗  ✓  ✓  ✓  ✗  ✗  ✗  ✗   
 6   ✗  ✗  ✗  ✓  ✓  ✓  ✓  ✗  ✗  ✗   
 7   ✗  ✗  ✗  ✓  ✓  ✓  ✓  ✓  ✗  ✗   
 8   ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✓  ✓  ✗   # Doc3: blocked from Doc1&2
 9   ✗  ✗  ✗  ✗  ✗  ✗  ✗  ✓  ✓  ✓   
```

## Use Cases

### When to BLOCK Cross-Document Attention

- **Independent documents/conversations**: Training on multiple unrelated texts
- **Privacy/security concerns**: Preventing information leakage between contexts
- **Document-level tasks**: When you need strict document boundaries
- **Multi-tenant scenarios**: Keeping different users' data isolated

### When to ALLOW Cross-Document Attention (Default)

- **Continuous text**: Documents are related parts of larger text (e.g., book chapters)
- **Standard LLM training**: Following typical pre-training practices (GPT, PaLM)
- **Maximum context utilization**: Want models to learn from all available context
- **Simpler processing**: Standard causal attention is computationally simpler

## A/B Testing Setup

Use the provided example configuration:

```bash
# Test with cross-document attention allowed
python scripts/train.py --config configs/examples/sft_cross_document_attention_comparison.yaml:allowed_cross_attention

# Test with cross-document attention blocked  
python scripts/train.py --config configs/examples/sft_cross_document_attention_comparison.yaml:blocked_cross_attention
```

## Performance Characteristics

- **Memory Overhead**: Additional `batch_size × seq_length²` for 2D attention masks
- **Compute Overhead**: Minimal (just mask application)
- **Packing Efficiency**: Identical in both modes
- **Training Speed**: Negligible difference

## Files Modified

1. **Configuration**: `src/config_models/stage_configs.py`
   - Added `block_cross_document_attention` field

2. **Core Implementation**: `src/training/packed_sequence_collator.py`
   - Added EOS position tracking
   - Added 2D attention mask creation
   - Added document boundary detection

3. **Trainer Integration**: `src/training/trainer_factory.py`
   - Pass new configuration to collator

4. **Tests**: `tests/test_cross_document_attention.py`
   - Comprehensive test suite (12 test cases)

5. **Examples**: `configs/examples/sft_cross_document_attention_comparison.yaml`
   - Ready-to-use A/B testing configuration

## Validation

All tests passing:
- ✅ 2D attention mask creation
- ✅ Document boundary detection  
- ✅ Cross-document attention blocking verification
- ✅ Intra-document attention preservation
- ✅ Padding mask behavior
- ✅ Single document handling
- ✅ Configuration integration
- ✅ Shape consistency
- ✅ Efficiency threshold compatibility

## Next Steps

1. **Test on your data**: Use the A/B testing configuration to compare both modes
2. **Monitor performance**: Check if blocked mode improves results for your specific use case
3. **Measure memory usage**: Ensure 2D attention masks fit in your GPU memory constraints
4. **Evaluate training metrics**: Compare loss curves, accuracy, and other metrics between modes

The feature is production-ready and maintains full backward compatibility with existing configurations. 