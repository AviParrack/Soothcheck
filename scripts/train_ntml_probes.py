#!/usr/bin/env python3
"""
Universal CLI for Training NTML Statement-Level Probes

This script can be run from anywhere in the repository and will automatically
detect paths and train statement-level probes on NTML conversational datasets.

Usage:
    python train_ntml_probes.py --dataset 2T1L_500samples
    python train_ntml_probes.py --jsonl_path data/NTML-datasets/5T5L_500samples.jsonl --model_name gpt2
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


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


def find_dataset_file(repo_root, dataset_name):
    """Find a dataset file by name or pattern."""
    data_dir = repo_root / "data" / "NTML-datasets"
    
    if not data_dir.exists():
        return None
    
    # Try exact match first
    exact_path = data_dir / f"{dataset_name}.jsonl"
    if exact_path.exists():
        return exact_path
    
    # Try pattern matching
    patterns = [
        f"{dataset_name}*.jsonl",
        f"*{dataset_name}*.jsonl"
    ]
    
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            return matches[0]  # Return first match
    
    return None


def list_available_datasets(repo_root):
    """List all available NTML datasets."""
    data_dir = repo_root / "data" / "NTML-datasets"
    
    if not data_dir.exists():
        return []
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    return sorted([f.stem for f in jsonl_files])


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train statement-level probes on NTML conversational datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train_ntml_probes.py --dataset 2T1L_500samples
    python train_ntml_probes.py --dataset 5T5L --statement_idx 0
    python train_ntml_probes.py --jsonl_path data/NTML-datasets/custom.jsonl
    python train_ntml_probes.py --list-datasets
        """
    )
    
    # Dataset selection (mutually exclusive)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (e.g., '2T1L_500samples', '5T5L'). Will auto-find the .jsonl file."
    )
    dataset_group.add_argument(
        "--jsonl_path",
        type=str,
        help="Full path to NTML JSONL dataset file"
    )
    dataset_group.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name for activation collection (default: gpt2)"
    )
    parser.add_argument(
        "--hook_point",
        type=str,
        default="blocks.7.hook_resid_pre",
        help="Hook point for activation collection (default: blocks.7.hook_resid_pre)"
    )
    parser.add_argument(
        "--hook_layer",
        type=int,
        default=7,
        help="Hook layer number (default: 7)"
    )
    
    # Probing arguments
    parser.add_argument(
        "--statement_idx",
        type=int,
        default=None,
        help="Specific statement position to probe (default: all statements)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: 512)"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Training/validation split ratio (default: 0.8)"
    )
    
    # Output arguments
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for activation caching (default: ./cache/ntml_cache)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save trained probes (default: ./trained_probes)"
    )
    parser.add_argument(
        "--probe_name",
        type=str,
        default=None,
        help="Name for the trained probe (default: auto-generated)"
    )
    
    # Utility arguments
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only validate the dataset without training"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed pipeline tracing"
    )
    
    return parser.parse_args()


def validate_dataset(dataset, verbose=False):
    """Validate the loaded dataset."""
    print("🔍 Validating dataset...")
    
    # Get dataset summary
    summary = dataset.get_conversation_summary()
    print(f"📊 Dataset Summary:")
    print(f"   • Total conversations: {summary['total_conversations']}")
    print(f"   • Total statements: {summary['total_statements']}")
    print(f"   • Truth/lie ratio: {summary['total_truths']}/{summary['total_lies']} ({summary['truth_ratio']:.2%} truths)")
    print(f"   • Avg statements per conversation: {summary['avg_statements_per_conversation']:.1f}")
    
    if verbose:
        print(f"   • Ratio distribution: {summary['ratio_distribution']}")
        print(f"   • Statement count distribution: {summary['statement_count_distribution']}")
    
    # Validate positions
    validation = dataset.validate_positions()
    if validation['validation_success']:
        print("✅ Position validation: PASSED")
    else:
        print(f"❌ Position validation: FAILED ({validation['invalid_examples']} invalid examples)")
        if verbose and validation['errors']:
            print("   First few errors:")
            for error in validation['errors'][:3]:
                print(f"     • Example {error.get('example_idx', '?')}: {error}")
    
    return validation['validation_success']


def debug_print_first_conversation(dataset, stage_name):
    """Print detailed information about the first conversation for debugging."""
    print(f"\n🔍 DEBUG: {stage_name}")
    print("=" * 50)
    
    if not dataset.examples:
        print("❌ No examples in dataset")
        return
    
    ex = dataset.examples[0]
    print(f"📝 First Example Details:")
    print(f"   • Example type: {type(ex).__name__}")
    print(f"   • Group ID: {getattr(ex, 'group_id', 'N/A')}")
    
    if hasattr(ex, 'system_prompt'):
        # Conversational example
        print(f"   • System prompt: {ex.system_prompt[:100]}...")
        print(f"   • Assistant response: {ex.assistant_response[:100]}...")
        print(f"   • Statement count: {len(ex.statement_labels)}")
        print(f"   • Statement labels: {ex.statement_labels}")
        print(f"   • Statement texts:")
        for i, text in enumerate(ex.statement_texts):
            print(f"     [{i}] {text[:80]}...")
        print(f"   • Statement positions: {ex.statement_positions}")
    else:
        # Statement-level example
        print(f"   • Label: {ex.label} ({ex.label_text})")
        print(f"   • Text length: {len(ex.text)} chars")
        print(f"   • Full text: {ex.text[:200]}...")
        if hasattr(ex, 'character_positions') and ex.character_positions:
            positions = ex.character_positions.positions
            print(f"   • Character positions: {positions}")
            if 'target' in positions:
                target_pos = positions['target']
                extracted = ex.text[target_pos.start:target_pos.end]
                print(f"   • Target extraction: '{extracted}'")
        if hasattr(ex, 'attributes') and ex.attributes:
            print(f"   • Statement text: {ex.attributes.get('statement_text', 'N/A')[:80]}...")
            print(f"   • Statement idx: {ex.attributes.get('statement_idx', 'N/A')}")
            print(f"   • Probing mode: {ex.attributes.get('probing_mode', 'N/A')}")


def debug_print_tokenized_example(dataset, stage_name):
    """Print detailed information about the first tokenized example."""
    print(f"\n🔍 DEBUG: {stage_name}")
    print("=" * 50)
    
    if not dataset.examples:
        print("❌ No examples in dataset")
        return
    
    try:
        ex = dataset.examples[0]
        print(f"📝 First Tokenized Example:")
        print(f"   • Token count: {len(ex.tokens)}")
        print(f"   • Label: {ex.label}")
        print(f"   • Tokens (first 20): {ex.tokens[:20]}")
        
        # Check for different possible token ID attributes
        if hasattr(ex, 'token_ids'):
            print(f"   • Token IDs (first 20): {ex.token_ids[:20]}")
        elif hasattr(ex, 'input_ids'):
            print(f"   • Input IDs (first 20): {ex.input_ids[:20]}")
        else:
            print(f"   • Token ID attribute not found")
        
        if hasattr(ex, 'token_positions') and ex.token_positions:
            print(f"   • Token positions: {ex.token_positions}")
            try:
                # Try to access target position
                if hasattr(ex.token_positions, 'positions'):
                    positions = ex.token_positions.positions
                    if 'target' in positions:
                        target_token_pos = positions['target']
                        print(f"   • Target token position: {target_token_pos}")
                        if isinstance(target_token_pos, int) and target_token_pos < len(ex.tokens):
                            print(f"   • Target token: '{ex.tokens[target_token_pos]}'")
                elif hasattr(ex.token_positions, '__getitem__'):
                    # Try dictionary-like access
                    target_token_pos = ex.token_positions['target']
                    print(f"   • Target token position: {target_token_pos}")
                    if isinstance(target_token_pos, int) and target_token_pos < len(ex.tokens):
                        print(f"   • Target token: '{ex.tokens[target_token_pos]}'")
            except Exception as e:
                print(f"   • Error accessing token positions: {e}")
        
        if hasattr(ex, 'attention_mask') and ex.attention_mask is not None:
            active_tokens = sum(ex.attention_mask)
            print(f"   • Active tokens (attention): {active_tokens}/{len(ex.attention_mask)}")
        
        # Print the full text to understand the structure
        print(f"   • Full text: {getattr(ex, 'text', 'N/A')[:150]}...")
        
        # Show how tokens map to text
        if hasattr(ex, 'tokens') and len(ex.tokens) > 0:
            print(f"   • Token mapping example:")
            for i in range(min(10, len(ex.tokens))):
                print(f"     [{i}] '{ex.tokens[i]}'")
        
        # Show character to token mapping if available
        if hasattr(ex, 'character_positions') and ex.character_positions:
            try:
                char_positions = ex.character_positions.positions
                print(f"   • Character positions: {char_positions}")
                if 'target' in char_positions:
                    char_pos = char_positions['target']
                    print(f"   • Target char range: {char_pos.start}-{char_pos.end}")
                    if hasattr(ex, 'text'):
                        target_text = ex.text[char_pos.start:char_pos.end]
                        print(f"   • Target text: '{target_text}'")
            except Exception as e:
                print(f"   • Error accessing character positions: {e}")
        
        print(f"   • Available attributes: {[attr for attr in dir(ex) if not attr.startswith('_')]}")
        
    except Exception as e:
        print(f"❌ Error in debug function: {e}")
        import traceback
        traceback.print_exc()


def debug_print_pipeline_config(pipeline_config):
    """Print detailed pipeline configuration."""
    print(f"\n🔍 DEBUG: Pipeline Configuration")
    print("=" * 50)
    print(f"   • Model: {pipeline_config.model_name}")
    print(f"   • Hook points: {pipeline_config.hook_points}")
    print(f"   • Position key: {pipeline_config.position_key}")
    print(f"   • Cache dir: {pipeline_config.cache_dir}")
    print(f"   • Dataset size: {len(pipeline_config.dataset.examples)}")
    
    # Print probe config
    probe_config = pipeline_config.probe_config
    print(f"   • Probe type: {pipeline_config.probe_cls.__name__}")
    print(f"   • Probe name: {probe_config.name}")
    print(f"   • Input size: {probe_config.input_size}")
    print(f"   • Hook point: {probe_config.hook_point}")
    print(f"   • Hook layer: {probe_config.hook_layer}")
    
    # Print trainer config
    trainer_config = pipeline_config.trainer_config
    print(f"   • Batch size: {trainer_config.batch_size}")
    print(f"   • Learning rate: {trainer_config.learning_rate}")
    print(f"   • Epochs: {trainer_config.num_epochs}")
    print(f"   • Train ratio: {trainer_config.train_ratio}")


def debug_print_training_data_split(train_examples, val_examples):
    """Print information about the training/validation split."""
    print(f"\n🔍 DEBUG: Training Data Split")
    print("=" * 50)
    print(f"   • Training examples: {len(train_examples)}")
    print(f"   • Validation examples: {len(val_examples)}")
    
    if train_examples:
        train_labels = [ex.label for ex in train_examples]
        train_true_count = sum(train_labels)
        print(f"   • Training: {train_true_count} true, {len(train_labels) - train_true_count} false")
    
    if val_examples:
        val_labels = [ex.label for ex in val_examples]
        val_true_count = sum(val_labels)
        print(f"   • Validation: {val_true_count} true, {len(val_labels) - val_true_count} false")
    
    # Show first training example
    if train_examples:
        ex = train_examples[0]
        print(f"   • First training example:")
        print(f"     - Label: {ex.label}")
        print(f"     - Text: {ex.text[:100]}...")
        if hasattr(ex, 'token_positions') and ex.token_positions and 'target' in ex.token_positions:
            pos = ex.token_positions['target']
            print(f"     - Target token position: {pos}")
            if hasattr(ex, 'tokens') and isinstance(pos, int) and pos < len(ex.tokens):
                print(f"     - Target token: '{ex.tokens[pos]}'")


def debug_training_wrapper(pipeline, debug_mode=False):
    """Wrapper around pipeline training that adds debugging information."""
    if not debug_mode:
        return pipeline.run()
    
    print(f"\n🔍 DEBUG: Starting Training Process")
    print("=" * 50)
    
    # Get the dataset for debugging
    dataset = pipeline.config.dataset
    print(f"   • Total examples: {len(dataset.examples)}")
    
    # Show first few examples without calling train_test_split
    print(f"\n📊 Dataset Overview:")
    labels = [ex.label for ex in dataset.examples]
    true_count = sum(labels)
    print(f"   • Total examples: {len(labels)}")
    print(f"   • True labels: {true_count}, False labels: {len(labels) - true_count}")
    print(f"   • Train ratio: {pipeline.config.trainer_config.train_ratio}")
    
    # Show first example details
    if dataset.examples:
        ex = dataset.examples[0]
        print(f"   • First example:")
        print(f"     - Label: {ex.label}")
        print(f"     - Text: {ex.text[:100]}...")
        if hasattr(ex, 'token_positions') and ex.token_positions and 'target' in ex.token_positions.positions:
            pos = ex.token_positions.positions['target']
            print(f"     - Target token position: {pos}")
            if hasattr(ex, 'tokens') and isinstance(pos, int) and pos < len(ex.tokens):
                print(f"     - Target token: '{ex.tokens[pos]}'")
    
    print(f"\n🚀 Starting actual training...")
    print(f"   • This will collect activations from {pipeline.config.model_name}")
    print(f"   • Hook point: {pipeline.config.hook_points[0]}")
    print(f"   • Each forward pass processes the full conversation text")
    print(f"   • Activations are extracted at the target token positions")
    print(f"   • Probe learns to classify: activation -> true/false")
    
    # Run the actual training
    probe, history = pipeline.run()
    
    print(f"\n🔍 DEBUG: Training Complete")
    print("=" * 50)
    if history:
        print(f"   • Training history keys: {list(history.keys())}")
        if 'train_loss' in history:
            print(f"   • Final train loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"   • Final val loss: {history['val_loss'][-1]:.4f}")
        if 'train_accuracy' in history:
            print(f"   • Final train accuracy: {history['train_accuracy'][-1]:.4f}")
        if 'val_accuracy' in history:
            print(f"   • Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    
    return probe, history


def main():
    """Main training function."""
    args = parse_args()
    
    try:
        # Setup paths
        repo_root, probity_dir = setup_paths()
        
        # Handle --list-datasets
        if args.list_datasets:
            datasets = list_available_datasets(repo_root)
            if datasets:
                print("📋 Available NTML Datasets:")
                for dataset in datasets:
                    print(f"   • {dataset}")
                print(f"\nUsage: python train_ntml_probes.py --dataset <name>")
            else:
                print("❌ No NTML datasets found.")
                print("   Run: python generate_ntml_datasets.py")
            return 0
        
        # Determine dataset path
        if args.dataset:
            dataset_path = find_dataset_file(repo_root, args.dataset)
            if not dataset_path:
                print(f"❌ Dataset '{args.dataset}' not found.")
                available = list_available_datasets(repo_root)
                if available:
                    print("Available datasets:")
                    for dataset in available[:5]:  # Show first 5
                        print(f"   • {dataset}")
                    if len(available) > 5:
                        print(f"   ... and {len(available) - 5} more")
                return 1
        elif args.jsonl_path:
            dataset_path = Path(args.jsonl_path)
            if not dataset_path.is_absolute():
                dataset_path = repo_root / dataset_path
            if not dataset_path.exists():
                print(f"❌ JSONL file not found: {dataset_path}")
                return 1
        else:
            print("❌ Must specify either --dataset or --jsonl_path")
            print("   Use --list-datasets to see available options")
            return 1
        
        # Set default cache and output directories relative to repo root
        cache_dir = args.cache_dir or str(repo_root / "cache" / "ntml_cache")
        output_dir = args.output_dir or str(repo_root / "trained_probes")
        
        print("🚀 NTML Statement-Level Probe Training")
        print(f"📁 Repository root: {repo_root}")
        print(f"📄 Dataset: {dataset_path.name}")
        print(f"🤖 Model: {args.model_name}")
        print(f"🎯 Hook Point: {args.hook_point}")
        
        # Import here after path setup
        from transformers import AutoTokenizer
        from probity.datasets.tokenized import TokenizedProbingDataset
        from probity.probes.logistic import LogisticProbe, LogisticProbeConfig
        from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
        from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
        from probity_extensions import ConversationalProbingDataset
        
        # Load NTML dataset
        print("\n📥 Loading NTML dataset...")
        try:
            ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(dataset_path))
            print(f"✅ Loaded {len(ntml_dataset.examples)} conversations")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return 1
        
        # Validate dataset
        if not validate_dataset(ntml_dataset, args.verbose):
            print("❌ Dataset validation failed. Please check your JSONL file.")
            return 1
        
        # DEBUG: Print first conversation
        if args.debug:
            debug_print_first_conversation(ntml_dataset, "NTML Conversational Dataset")
        
        if args.validate_only:
            print("✅ Dataset validation complete. Exiting (--validate_only specified).")
            return 0
        
        # Create statement-level dataset
        print(f"\n🔄 Creating statement-level dataset...")
        if args.statement_idx is not None:
            print(f"   • Probing statement position: {args.statement_idx}")
            statement_dataset = ntml_dataset.get_statement_dataset(args.statement_idx)
        else:
            print("   • Probing all statements as separate examples")
            statement_dataset = ntml_dataset.get_statement_dataset()
        
        print(f"✅ Created dataset with {len(statement_dataset.examples)} statement examples")
        
        # DEBUG: Print first statement-level example
        if args.debug:
            debug_print_first_conversation(statement_dataset, "Statement-Level Dataset")
        
        # Tokenize dataset
        print(f"\n🔤 Tokenizing dataset...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
                dataset=statement_dataset,
                tokenizer=tokenizer,
                padding=True,
                max_length=args.max_length,
                add_special_tokens=True
            )
            print(f"✅ Tokenized dataset (max_length: {args.max_length})")
        except Exception as e:
            print(f"❌ Error tokenizing dataset: {e}")
            return 1
        
        # DEBUG: Print first tokenized example
        if args.debug:
            debug_print_tokenized_example(tokenized_dataset, "Tokenized Dataset")
        
        # Configure probe
        probe_name = args.probe_name or f"ntml_{dataset_path.stem}_{args.hook_point.replace('.', '_')}"
        if args.statement_idx is not None:
            probe_name += f"_stmt{args.statement_idx}"
        
        probe_config = LogisticProbeConfig(
            input_size=768 if "gpt2" in args.model_name else 768,  # TODO: Get actual model size
            normalize_weights=True,
            bias=False,
            model_name=args.model_name,
            hook_point=args.hook_point,
            hook_layer=args.hook_layer,
            name=probe_name
        )
        
        # Configure trainer
        trainer_config = SupervisedTrainerConfig(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            weight_decay=0.01,
            train_ratio=args.train_ratio,
            handle_class_imbalance=True,
            show_progress=True
        )
        
        # Configure pipeline
        pipeline_config = ProbePipelineConfig(
            dataset=tokenized_dataset,
            probe_cls=LogisticProbe,
            probe_config=probe_config,
            trainer_cls=SupervisedProbeTrainer,
            trainer_config=trainer_config,
            position_key="target",  # Probe at statement positions
            model_name=args.model_name,
            hook_points=[args.hook_point],
            cache_dir=cache_dir
        )
        
        # DEBUG: Print pipeline configuration
        if args.debug:
            debug_print_pipeline_config(pipeline_config)
        
        # Train probe
        print(f"\n🏋️ Training probe...")
        print(f"   • Probe name: {probe_name}")
        print(f"   • Batch size: {args.batch_size}")
        print(f"   • Learning rate: {args.learning_rate}")
        print(f"   • Epochs: {args.num_epochs}")
        print(f"   • Train/val split: {args.train_ratio:.1%}/{1-args.train_ratio:.1%}")
        print(f"   • Cache directory: {cache_dir}")
        print(f"   • Output directory: {output_dir}")
        
        try:
            pipeline = ProbePipeline(pipeline_config)
            probe, history = debug_training_wrapper(pipeline, args.debug)
            
            print("✅ Training completed!")
            
            # Save probe
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            probe_path = output_path / f"{probe_name}.pt"
            probe.save(str(probe_path))
            print(f"💾 Probe saved to: {probe_path}")
            
            # Print training summary
            if history and 'val_accuracy' in history:
                final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
                print(f"📈 Final validation accuracy: {final_val_acc:.3f}")
            
            print(f"\n🎉 Training complete!")
            print(f"📋 Next Steps:")
            print(f"   • Analyze probe: Load from {probe_path}")
            print(f"   • Train more ratios: python train_ntml_probes.py --dataset <other_ratio>")
            
            return 0
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        
    except Exception as e:
        print(f"❌ Setup error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main()) 