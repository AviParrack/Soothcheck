#!/usr/bin/env python
# Test script to demonstrate the improved evaluation system
# This shows how the unified inference and clear model loading work

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.model_resolver import resolve_model_specification
from src.inference.inference import (
    preload_model_and_tokenizer,
    generate_batch_responses,
)


def test_model_resolution():
    """Test that model resolution works correctly for different specifications."""
    print("\n" + "=" * 80)
    print("TESTING MODEL RESOLUTION")
    print("=" * 80)

    test_specs = [
        "gemma-3-1b-it",  # Config name
        "base:google/gemma-2b",  # Explicit base model
        # Add actual experiment paths if available
    ]

    for spec in test_specs:
        print(f"\nTesting spec: {spec}")
        try:
            resolution = resolve_model_specification(spec)
            print(f"  ✓ Resolved successfully")
            print(f"    Base model: {resolution.base_model_hf_id}")
            print(f"    Model type: {resolution.model_type}")
            print(f"    Is experiment: {resolution.is_experiment}")
            print(f"    PEFT adapter: {resolution.peft_adapter_path or 'None'}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")


def test_unified_inference():
    """Test the unified inference pipeline with a small model."""
    print("\n" + "=" * 80)
    print("TESTING UNIFIED INFERENCE")
    print("=" * 80)

    # Use a small model for testing
    model_spec = "gemma-3-1b-it"

    print(f"\nLoading model: {model_spec}")
    print("(This demonstrates the clear logging)")

    try:
        # Load model
        model, tokenizer = preload_model_and_tokenizer(model_spec, device="auto")

        # Get model type from resolution
        resolution = resolve_model_specification(model_spec)
        model_type = resolution.model_type

        # Test prompts
        test_prompts = [
            "What is 2+2?",
            "Write a haiku about Python.",
            "Translate 'Hello' to French.",
        ]

        print(f"\nGenerating responses for {len(test_prompts)} prompts...")

        # Generate responses
        responses = generate_batch_responses(
            prompts=test_prompts,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            max_new_tokens=50,
            temperature=0.7,
            show_progress=True,
        )

        print("\nResults:")
        for prompt, response in zip(test_prompts, responses):
            print(f"\nPrompt: {prompt}")
            print(f"Response: {response[:100]}...")  # Truncate for display

    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback

        traceback.print_exc()


def test_eval_dataset_loading():
    """Test loading evaluation datasets."""
    print("\n" + "=" * 80)
    print("TESTING EVAL DATASET LOADING")
    print("=" * 80)

    from src.config_models.eval_configs import EvalDataset

    dataset_path = PROJECT_ROOT / "evals" / "eval_datasets" / "example_basic.json"

    if dataset_path.exists():
        print(f"\nLoading dataset: example_basic")
        with open(dataset_path, "r") as f:
            import json

            data = json.load(f)

        dataset = EvalDataset.from_simple_json(data, "example_basic")
        print(f"  ✓ Loaded {len(dataset.prompts)} prompts")
        print(
            f"  Categories: {set(p.metadata.get('category', 'none') for p in dataset.prompts)}"
        )
    else:
        print(f"\n  ✗ Example dataset not found at {dataset_path}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("EVALUATION SYSTEM TEST")
    print("=" * 80)
    print("\nThis script demonstrates the improvements to the evaluation system:")
    print("1. Unified inference logic (no duplication)")
    print("2. Clear model loading with extensive logging")
    print("3. Standardized file organization")

    test_model_resolution()
    test_eval_dataset_loading()

    # Only run inference test if explicitly requested (it loads a model)
    response = input("\nRun inference test? (requires loading a model) [y/N]: ")
    if response.lower() == "y":
        test_unified_inference()

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nTo run a full evaluation, use:")
    print("  python -m scripts.benchmark example_basic --model gemma-3-1b-it")


if __name__ == "__main__":
    main()
