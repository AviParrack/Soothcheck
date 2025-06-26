#!/usr/bin/env python3
"""
Universal CLI for testing NTML Probity Extensions Integration

This script can be run from anywhere in the repository and will automatically
detect paths and test the integration between NTML datasets and Probity.

Usage:
    python test_ntml_probity_integration.py
    python test_ntml_probity_integration.py --verbose
"""

import argparse
import sys
from pathlib import Path


def find_repo_root():
    """Find the repository root directory."""
    current = Path.cwd()
    
    # Look for key indicators that uniquely identify the working root
    # The actual content is in the probity/ subdirectory
    primary_indicators = [
        "Statement-level-probe-implementation-plan.md",
        "data/NTML-datasets",
        "probity_extensions"
    ]
    
    # Check current directory and parents
    for path in [current] + list(current.parents):
        if all((path / indicator).exists() for indicator in primary_indicators):
            return path
    
    # Fallback: look for the most distinctive indicator
    for path in [current] + list(current.parents):
        if (path / "Statement-level-probe-implementation-plan.md").exists():
            return path
    
    return current


def setup_paths():
    """Setup Python paths for imports."""
    repo_root = find_repo_root()
    probity_dir = repo_root / "probity"
    
    if not probity_dir.exists():
        raise FileNotFoundError(f"Probity directory not found at {probity_dir}")
    
    # Add repo root to Python path for probity_extensions
    sys.path.insert(0, str(repo_root))
    
    return repo_root, probity_dir


def find_test_datasets(repo_root):
    """Find available test datasets."""
    # Use the correct single path for NTML datasets
    data_dir = repo_root / "data" / "NTML-datasets"
    
    if not data_dir.exists():
        return []
    
    # Look for test datasets - updated patterns to match actual files
    test_patterns = ["test_*.jsonl", "*_2samples.jsonl", "*_3samples.jsonl", "*_5samples.jsonl"]
    test_files = []
    
    for pattern in test_patterns:
        test_files.extend(data_dir.glob(pattern))
    
    return sorted(test_files)


def test_ntml_loading(test_files, verbose=False):
    """Test loading NTML JSONL files."""
    print("🧪 Testing NTML Dataset Loading")
    
    if not test_files:
        print("❌ No test JSONL files found. Please generate test datasets first.")
        print("   Run: python generate_ntml_datasets.py --test")
        return False
    
    # Use the first available test file
    test_file = test_files[0]
    print(f"📁 Using test file: {test_file.name}")
    
    try:
        from probity_extensions import ConversationalProbingDataset
        
        # Load the dataset
        dataset = ConversationalProbingDataset.from_ntml_jsonl(str(test_file))
        print(f"✅ Successfully loaded {len(dataset.examples)} conversations")
        
        # Test dataset summary
        summary = dataset.get_conversation_summary()
        print(f"📊 Dataset summary:")
        print(f"   • Total conversations: {summary['total_conversations']}")
        print(f"   • Total statements: {summary['total_statements']}")
        print(f"   • Truth/lie split: {summary['total_truths']}/{summary['total_lies']}")
        
        if verbose:
            print(f"   • Ratio distribution: {summary['ratio_distribution']}")
            print(f"   • Statement count distribution: {summary['statement_count_distribution']}")
        
        # Test position validation
        validation = dataset.validate_positions()
        if validation['validation_success']:
            print("✅ Position validation: PASSED")
        else:
            print(f"❌ Position validation: FAILED")
            if verbose and validation['errors']:
                print(f"   • Errors: {validation['errors'][:3]}")
            return False
        
        # Test first example
        if dataset.examples and verbose:
            ex = dataset.examples[0]
            print(f"\n📝 First example:")
            print(f"   • ID: {ex.group_id}")
            print(f"   • System prompt: {ex.system_prompt[:50]}...")
            print(f"   • Assistant response: {ex.assistant_response[:50]}...")
            print(f"   • Statement count: {len(ex.statement_labels)}")
            print(f"   • Statement labels: {ex.statement_labels}")
            print(f"   • Lie positions: {ex.attributes.get('lie_positions', [])}")
            
            # Test position extraction
            print(f"\n🔍 Testing position extraction:")
            for i, (text, pos) in enumerate(zip(ex.statement_texts, ex.statement_positions)):
                extracted = ex.text[pos[0]:pos[1]]
                match = extracted.strip() == text.strip()
                status = "✅" if match else "❌"
                print(f"   • Statement {i}: {status} '{text[:30]}...'")
                if not match:
                    print(f"     Expected: '{text}'")
                    print(f"     Extracted: '{extracted}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_statement_dataset_creation(test_files, verbose=False):
    """Test creating statement-level datasets."""
    print("\n🧪 Testing Statement Dataset Creation")
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    test_file = test_files[0]
    
    try:
        from probity_extensions import ConversationalProbingDataset
        
        # Load conversational dataset
        conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(test_file))
        print(f"📁 Loaded {len(conv_dataset.examples)} conversations")
        
        # Test all-statements dataset
        all_stmt_dataset = conv_dataset.get_statement_dataset()
        print(f"✅ All-statements dataset: {len(all_stmt_dataset.examples)} examples")
        
        # Test single-statement dataset
        single_stmt_dataset = conv_dataset.get_statement_dataset(statement_idx=0)
        print(f"✅ Single-statement dataset (pos 0): {len(single_stmt_dataset.examples)} examples")
        
        # Verify the examples have correct structure
        if all_stmt_dataset.examples:
            ex = all_stmt_dataset.examples[0]
            print(f"📝 First statement example:")
            print(f"   • Label: {ex.label} ({ex.label_text})")
            print(f"   • Has target position: {'target' in ex.character_positions.keys()}")
            print(f"   • Statement text: {ex.attributes.get('statement_text', 'N/A')[:50]}...")
            print(f"   • Probing mode: {ex.attributes.get('probing_mode', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating statement datasets: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_tokenization_compatibility(test_files, verbose=False):
    """Test that our datasets work with Probity's tokenization."""
    print("\n🧪 Testing Tokenization Compatibility")
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    test_file = test_files[0]
    
    try:
        from transformers import AutoTokenizer
        from probity.datasets.tokenized import TokenizedProbingDataset
        from probity_extensions import ConversationalProbingDataset
        
        # Load and create statement dataset
        conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(test_file))
        stmt_dataset = conv_dataset.get_statement_dataset()
        
        # Test tokenization
        print("   • Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
            dataset=stmt_dataset,
            tokenizer=tokenizer,
            padding=True,
            max_length=256,
            add_special_tokens=True
        )
        
        print(f"✅ Tokenization successful: {len(tokenized_dataset.examples)} examples")
        
        # Test token position mapping
        if tokenized_dataset.examples:
            ex = tokenized_dataset.examples[0]
            print(f"📝 First tokenized example:")
            print(f"   • Token count: {len(ex.tokens)}")
            print(f"   • Has attention mask: {ex.attention_mask is not None}")
            print(f"   • Has token positions: {ex.token_positions is not None}")
            
            if ex.token_positions and "target" in ex.token_positions.keys():
                target_pos = ex.token_positions["target"]
                print(f"   • Target token position: {target_pos}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with tokenization: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_probity_pipeline_compatibility(test_files, verbose=False):
    """Test that our datasets work with Probity's full pipeline (without actual training)."""
    print("\n🧪 Testing Probity Pipeline Compatibility")
    
    if not test_files:
        print("❌ No test files found")
        return False
    
    test_file = test_files[0]
    
    try:
        from transformers import AutoTokenizer
        from probity.datasets.tokenized import TokenizedProbingDataset
        from probity.probes.logistic import LogisticProbe, LogisticProbeConfig
        from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
        from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
        from probity_extensions import ConversationalProbingDataset
        
        # Load and prepare dataset
        conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(test_file))
        stmt_dataset = conv_dataset.get_statement_dataset()
        
        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
            dataset=stmt_dataset,
            tokenizer=tokenizer,
            padding=True,
            max_length=256,
            add_special_tokens=True
        )
        
        # Configure probe and trainer (but don't train)
        probe_config = LogisticProbeConfig(
            input_size=768,
            normalize_weights=True,
            bias=False,
            model_name="gpt2",
            hook_point="blocks.7.hook_resid_pre",
            hook_layer=7,
            name="test_ntml_probe"
        )
        
        trainer_config = SupervisedTrainerConfig(
            batch_size=4,  # Small batch for testing
            learning_rate=1e-3,
            num_epochs=1,  # Just 1 epoch for testing
            weight_decay=0.01,
            train_ratio=0.8,
            handle_class_imbalance=True,
            show_progress=False  # Disable progress for testing
        )
        
        pipeline_config = ProbePipelineConfig(
            dataset=tokenized_dataset,
            probe_cls=LogisticProbe,
            probe_config=probe_config,
            trainer_cls=SupervisedProbeTrainer,
            trainer_config=trainer_config,
            position_key="target",
            model_name="gpt2",
            hook_points=["blocks.7.hook_resid_pre"],
            cache_dir=None  # No caching for test
        )
        
        # Create pipeline (but don't run training)
        pipeline = ProbePipeline(pipeline_config)
        print("✅ Pipeline configuration successful")
        print(f"   • Dataset: {len(tokenized_dataset.examples)} examples")
        print(f"   • Probe: {probe_config.name}")
        print(f"   • Position key: {pipeline_config.position_key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with pipeline setup: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test NTML Probity Extensions Integration"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip pipeline compatibility test (faster)"
    )
    
    args = parser.parse_args()
    
    print("🚀 NTML Probity Extensions Integration Test")
    print("=" * 60)
    
    try:
        # Setup paths
        repo_root, probity_dir = setup_paths()
        print(f"📁 Repository root: {repo_root}")
        print(f"🔧 Probity directory: {probity_dir}")
        
        # Find test datasets
        test_files = find_test_datasets(repo_root)
        if test_files:
            print(f"📋 Found {len(test_files)} test dataset(s)")
            if args.verbose:
                for f in test_files:
                    print(f"   • {f.name}")
        else:
            print("⚠️  No test datasets found. Run: python generate_ntml_datasets.py --test")
        
        # Define tests
        tests = [
            ("NTML Loading", test_ntml_loading),
            ("Statement Dataset Creation", test_statement_dataset_creation),
            ("Tokenization Compatibility", test_tokenization_compatibility),
        ]
        
        if not args.skip_pipeline:
            tests.append(("Pipeline Compatibility", test_probity_pipeline_compatibility))
        
        # Run tests
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func(test_files, args.verbose)
            results.append((test_name, success))
        
        # Summary
        print(f"\n{'='*60}")
        print("🏁 Test Results Summary:")
        all_passed = True
        for test_name, success in results:
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   • {test_name}: {status}")
            if not success:
                all_passed = False
        
        if all_passed:
            print("\n🎉 All tests passed! The integration is working correctly.")
            print("\n📋 Next Steps:")
            print("   • Generate full datasets: python generate_ntml_datasets.py")
            print("   • Train probes: python train_ntml_probes.py --jsonl_path <dataset.jsonl>")
            return 0
        else:
            print("\n💥 Some tests failed. Please check the errors above.")
            if not args.verbose:
                print("   💡 Try running with --verbose for more details")
            return 1
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 