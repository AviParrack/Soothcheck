# This file provides validation utilities to ensure training is working correctly,
# particularly for assistant-only training where we want to verify that loss is only
# calculated on assistant tokens.
#
# Functions:
# - validate_assistant_only_training: Validates that labels are correctly set for assistant-only training
# - inspect_training_batch: Inspects a training batch to show what tokens are being trained on

import torch
import logging
from typing import Dict, List, Any
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def validate_assistant_only_training(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    sample_idx: int = 0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate that a training batch is correctly set up for assistant-only training.

    Args:
        batch: Training batch with 'input_ids', 'labels', and 'attention_mask'
        tokenizer: Tokenizer used for decoding
        sample_idx: Which sample in the batch to inspect (default: 0)
        verbose: Whether to print detailed information

    Returns:
        Dict with validation results and statistics
    """
    if verbose:
        print(f"\n🔍 Validating Assistant-Only Training (Sample {sample_idx})")
        print("=" * 60)

    # Extract tensors for the specified sample
    input_ids = batch["input_ids"][sample_idx].cpu()
    labels = batch["labels"][sample_idx].cpu()
    attention_mask = batch["attention_mask"][sample_idx].cpu()

    # Convert to lists for easier processing
    input_ids_list = input_ids.tolist()
    labels_list = labels.tolist()
    attention_mask_list = attention_mask.tolist()

    # Calculate statistics
    total_tokens = len(input_ids_list)
    training_tokens = sum(1 for l in labels_list if l != -100)
    ignored_tokens = sum(1 for l in labels_list if l == -100)
    attention_tokens = sum(attention_mask_list)

    stats = {
        "total_tokens": total_tokens,
        "training_tokens": training_tokens,
        "ignored_tokens": ignored_tokens,
        "attention_tokens": attention_tokens,
        "training_percentage": (
            (training_tokens / total_tokens * 100) if total_tokens > 0 else 0
        ),
        "is_assistant_only": ignored_tokens > 0 and training_tokens > 0,
        "all_tokens_attended": attention_tokens == total_tokens,
    }

    if verbose:
        print(f"Token Statistics:")
        print(f"  Total tokens: {total_tokens}")
        print(
            f"  Training tokens: {training_tokens} ({stats['training_percentage']:.1f}%)"
        )
        print(
            f"  Ignored tokens: {ignored_tokens} ({ignored_tokens/total_tokens*100:.1f}%)"
        )
        print(f"  Attention tokens: {attention_tokens}")

        # Validation checks
        print(f"\nValidation Checks:")
        if stats["is_assistant_only"]:
            print("  ✅ Assistant-only training detected")
        else:
            print("  ⚠️  All tokens are being trained on (not assistant-only)")

        if stats["all_tokens_attended"]:
            print("  ✅ All tokens have attention (correct)")
        else:
            print("  ⚠️  Some tokens are not attended to")

        # Show token breakdown
        num_tokens_to_show = 500
        print(f"\nToken Breakdown (first {num_tokens_to_show} tokens):")
        print(
            f"{'Idx':<3} {'Input':<8} {'Label':<8} {'Attn':<4} {'Status':<8} {'Token'}"
        )
        print("-" * 65)

        for i in range(min(num_tokens_to_show, total_tokens)):
            input_id = input_ids_list[i]
            label = labels_list[i]
            attn = attention_mask_list[i]

            try:
                token_text = tokenizer.decode([input_id], skip_special_tokens=False)
            except:
                token_text = f"<UNK:{input_id}>"

            status = "TRAIN" if label != -100 else "IGNORE"
            print(
                f"{i:<3} {input_id:<8} {label:<8} {attn:<4} {status:<8} '{token_text}'"
            )

    return stats


def inspect_training_batch(
    trainer,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 2,
) -> List[Dict[str, Any]]:
    """
    Inspect a training batch from the trainer's dataloader to validate assistant-only training.

    Args:
        trainer: The trainer instance (SFTTrainer, etc.)
        tokenizer: Tokenizer for decoding
        num_samples: Number of samples to inspect

    Returns:
        List of validation results for each sample
    """
    print(f"\n🔬 Inspecting Training Batch")
    print("=" * 40)

    # Get a batch from the trainer's dataloader
    train_dataloader = trainer.get_train_dataloader()
    batch = next(iter(train_dataloader))

    # Move to CPU for inspection
    batch = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    print(f"Batch shape: {batch['input_ids'].shape}")
    print(f"Inspecting {min(num_samples, batch['input_ids'].shape[0])} samples...")

    results = []
    for i in range(min(num_samples, batch["input_ids"].shape[0])):
        print(f"\n--- Sample {i+1} ---")
        stats = validate_assistant_only_training(
            batch, tokenizer, sample_idx=i, verbose=True
        )
        results.append(stats)

    # Summary across all samples
    print(f"\n📊 Summary Across {len(results)} Samples")
    print("-" * 30)

    avg_training_pct = sum(r["training_percentage"] for r in results) / len(results)
    assistant_only_count = sum(1 for r in results if r["is_assistant_only"])

    print(f"Average training percentage: {avg_training_pct:.1f}%")
    print(
        f"Samples with assistant-only training: {assistant_only_count}/{len(results)}"
    )

    if assistant_only_count == len(results):
        print("✅ All samples are using assistant-only training!")
    elif assistant_only_count > 0:
        print("⚠️  Mixed: Some samples use assistant-only, others don't")
    else:
        print("❌ No samples are using assistant-only training")

    return results


def validate_loss_calculation(
    model,
    batch: Dict[str, torch.Tensor],
    sample_idx: int = 0,
) -> Dict[str, float]:
    """
    Validate that loss is only calculated on non-ignored tokens.

    Args:
        model: The model to test
        batch: Training batch
        sample_idx: Which sample to test

    Returns:
        Dict with loss calculation details
    """
    print(f"\n🧮 Validating Loss Calculation")
    print("-" * 30)

    # Extract single sample
    input_ids = batch["input_ids"][sample_idx : sample_idx + 1]
    labels = batch["labels"][sample_idx : sample_idx + 1]
    attention_mask = batch["attention_mask"][sample_idx : sample_idx + 1]

    model.eval()
    with torch.no_grad():
        # Get model outputs
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        model_loss = outputs.loss.item()

        # Manual loss calculation
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)

        per_token_losses = loss_fct(flat_logits, flat_labels)

        # Count contributing vs ignored tokens
        contributing = (flat_labels != -100).sum().item()
        ignored = (flat_labels == -100).sum().item()

        # Calculate manual loss
        manual_loss = per_token_losses[flat_labels != -100].mean().item()

        # Check that ignored tokens have zero loss
        ignored_losses = per_token_losses[flat_labels == -100]
        max_ignored_loss = (
            ignored_losses.max().item() if len(ignored_losses) > 0 else 0.0
        )

        results = {
            "model_loss": model_loss,
            "manual_loss": manual_loss,
            "loss_difference": abs(model_loss - manual_loss),
            "contributing_tokens": contributing,
            "ignored_tokens": ignored,
            "max_ignored_loss": max_ignored_loss,
            "loss_calculation_correct": max_ignored_loss < 1e-6,
        }

        print(f"Model loss: {model_loss:.6f}")
        print(f"Manual loss: {manual_loss:.6f}")
        print(f"Difference: {results['loss_difference']:.6f}")
        print(f"Contributing tokens: {contributing}")
        print(f"Ignored tokens: {ignored}")
        print(f"Max ignored loss: {max_ignored_loss:.6f}")

        if results["loss_calculation_correct"]:
            print("✅ Loss calculation is correct (ignored tokens have zero loss)")
        else:
            print("❌ Loss calculation error: ignored tokens are contributing to loss!")

    return results
