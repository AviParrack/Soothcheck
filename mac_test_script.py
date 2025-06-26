#!/usr/bin/env python3
"""
🧪 Simple Mac Test Script for NTML Probity System with GPT2
A lightweight test using GPT2-small (~500MB) that won't brick your laptop

This script will:
1. Check basic dependencies
2. Load GPT2-small model (lightweight)
3. Generate a 2T1L dataset with 20 samples using NTML module
4. Train a simple probe on the generated dataset
5. Test basic functionality

Usage: python mac_test_script.py
"""

import sys
import os
import time
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch

def find_repo_root():
    """Find the repository root directory."""
    current = Path.cwd()
    
    # Look for key indicators
    indicators = [
        "Statement-level-probe-implementation-plan.md",
        "data/NTML-datasets",
        "probity_extensions"
    ]
    
    # Check current directory first
    if all((current / indicator).exists() for indicator in indicators):
        return current
    
    # Check parent directories
    for path in list(current.parents):
        if all((path / indicator).exists() for indicator in indicators):
            return path
    
    raise RuntimeError("Could not find repository root")

# Setup paths
repo_root = find_repo_root()
sys.path.insert(0, str(repo_root))

def print_header(text: str):
    """Print a formatted header"""
    print(f"\n{'='*50}")
    print(f"🧪 {text}")
    print(f"{'='*50}")

def print_step(step: int, text: str):
    """Print a step header"""
    print(f"\n{'='*40}")
    print(f"Step {step}: {text}")
    print(f"{'='*40}")

def print_success(text: str):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text: str):
    """Print error message"""
    print(f"❌ {text}")

def print_warning(text: str):
    """Print warning message"""
    print(f"⚠️  {text}")

def check_basic_setup():
    """Check basic setup - much simpler than before"""
    print_step(1, "Basic Setup Check")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print_error(f"Python {python_version.major}.{python_version.minor} - Need Python 3.8+")
        return False
    
    # Check basic imports
    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
        
        import transformers
        print_success(f"Transformers {transformers.__version__}")
        
        # Don't require transformer_lens to be installed, we'll install if needed
        try:
            import transformer_lens
            print_success("TransformerLens available")
        except ImportError:
            print_warning("TransformerLens not found - will install if needed")
    
    except ImportError as e:
        print_error(f"Missing required package: {e}")
        return False
    
    return True

def install_minimal_dependencies():
    """Install only what we absolutely need"""
    print_step(2, "Installing Minimal Dependencies")
    
    required_packages = [
        "transformer-lens>=1.14.0",
        "datasets",
    ]
    
    try:
        import transformer_lens
        print_success("TransformerLens already installed")
        return True
    except ImportError:
        pass
    
    print("Installing TransformerLens (this may take a moment)...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "transformer-lens"
        ], check=True, capture_output=True)
        print_success("TransformerLens installed")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install TransformerLens: {e}")
        return False

def test_gpt2_access():
    """Test GPT2 access - much lighter than Llama"""
    print_step(3, "Testing GPT2 Access (~500MB download)")
    
    try:
        from transformer_lens import HookedTransformer
        
        print("Loading GPT2-small (this is lightweight)...")
        model = HookedTransformer.from_pretrained(
            "gpt2",
            device='cpu',  # Keep on CPU to be safe
            torch_dtype=torch.float32  # Use float32 for better compatibility
        )
        print_success(f"GPT2 loaded successfully ({model.cfg.n_layers} layers, {model.cfg.d_model} dimensions)")
        
        # Test a simple forward pass
        test_input = "Hello world"
        tokens = model.to_tokens(test_input)
        print(f"Test tokens shape: {tokens.shape}")
        
        with torch.no_grad():
            logits = model(tokens)
        print_success(f"Model forward pass successful (output shape: {logits.shape})")
        
        # Test getting activations from a middle layer
        hook_point = "blocks.6.hook_resid_pre"  # Middle layer for GPT2 (12 layers)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            activations = cache[hook_point]
        print_success(f"Hook point '{hook_point}' activations: {activations.shape}")
        
        # Clean up memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return True
        
    except Exception as e:
        print_error(f"GPT2 access failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_ntml_dataset():
    """Generate a 2T1L dataset with 20 samples using the NTML module"""
    print_step(4, "Generating 2T1L Dataset (20 samples)")
    
    print("   Using NTML module to generate dataset...")
    print("   Format: 2 Truths, 1 Lie per example")
    print("   Samples: 20")
    
    # Use the convenience script to generate the dataset
    generation_script = repo_root / "generate_ntml_datasets.py"
    
    generation_command = [
        sys.executable, str(generation_script),
        "--samples", "20",
        "--ratios", "2T1L",
        "--seed", "42"
    ]
    
    print(f"   Running: {' '.join(generation_command)}")
    
    try:
        result = subprocess.run(
            generation_command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout for generation
        )
        
        if result.returncode == 0:
            print_success("NTML dataset generation completed")
            
            # Check if the file was created
            expected_file = repo_root / "data" / "NTML-datasets" / "2T1L_20samples.jsonl"
            if expected_file.exists():
                print_success(f"Dataset file created: {expected_file.name}")
                
                # Show a sample of what was generated
                with open(expected_file, 'r') as f:
                    lines = f.readlines()
                    print(f"   Generated {len(lines)} examples")
                    
                    if lines:
                        # Show first example structure
                        first_example = json.loads(lines[0])
                        print(f"   Example structure: {len(first_example['labels']['statement_level'])} statements per example")
                
                return True
            else:
                print_error(f"Expected dataset file not found: {expected_file}")
                return False
                
        else:
            print_error(f"Dataset generation failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-500:])  # Last 500 chars
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Dataset generation timed out after 3 minutes")
        return False
    except Exception as e:
        print_error(f"Dataset generation failed: {e}")
        return False

def train_simple_probe():
    """Train a simple probe with minimal settings using the generated dataset"""
    print_step(5, "Training Simple Probe on Generated Dataset")
    
    model_name = "gpt2"
    hook_point = "blocks.6.hook_resid_pre"  # Middle layer for GPT2
    dataset_name = "2T1L_20samples"  # Use the generated dataset
    
    print(f"   Model: {model_name}")
    print(f"   Dataset: {dataset_name} (NTML generated)")
    print(f"   Hook point: {hook_point}")
    
    # Use the working training script but with minimal settings
    train_script_path = repo_root / "scripts" / "train_ntml_probes.py"
    
    train_command = [
        sys.executable, str(train_script_path),
        "--dataset", dataset_name,
        "--model_name", model_name,
        "--hook_point", hook_point,
        "--batch_size", "1",  # Very small batch
        "--num_epochs", "5",  # A few more epochs since we have real data
        "--verbose"
    ]
    
    print(f"   Running: {' '.join(train_command)}")
    
    try:
        result = subprocess.run(
            train_command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for training
        )
        
        if result.returncode == 0:
            print_success("Probe training on generated dataset completed")
            if result.stdout:
                # Show last few lines of output
                lines = result.stdout.strip().split('\n')
                print("Training output (last 3 lines):")
                for line in lines[-3:]:
                    print(f"   {line}")
            return True
        else:
            print_error(f"Training failed with return code {result.returncode}")
            if result.stderr:
                print("Error output:")
                print(result.stderr[-300:])  # Last 300 chars
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Training timed out after 10 minutes")
        return False
    except Exception as e:
        print_error(f"Training failed: {e}")
        return False

def test_probe_on_generated_data():
    """Test the trained probe with examples from the generated dataset"""
    print_step(6, "Testing Probe on Generated Data")
    
    # Find the trained probe file
    probe_dir = repo_root / "trained_probes"
    probe_files = list(probe_dir.glob("*2T1L_20samples*"))
    
    if not probe_files:
        print_error("No trained probe files found for 2T1L_20samples dataset")
        return False
    
    probe_path = probe_files[0]
    print_success(f"Found probe: {probe_path.name}")
    
    try:
        # Load and test the probe
        import torch
        from probity.probes import LogisticProbe
        from probity.probes.inference import ProbeInference
        
        # Load probe
        probe_data = torch.load(probe_path, map_location='cpu')
        config = probe_data['config']
        state_dict = probe_data['state_dict']
        
        probe = LogisticProbe(config)
        probe.load_state_dict(state_dict)
        probe.eval()
        
        print_success("Probe loaded successfully")
        
        # Create inference object
        inference = ProbeInference(
            model_name="gpt2",
            hook_point=config.hook_point,
            probe=probe,
            device='cpu'
        )
        
        # Load some examples from the generated dataset to test on
        dataset_path = repo_root / "data" / "NTML-datasets" / "2T1L_20samples.jsonl"
        test_statements = []
        expected_labels = []
        
        with open(dataset_path, 'r') as f:
            # Take first example to test on
            example = json.loads(f.readline())
            statements = example['labels']['statement_level']
            
            for stmt in statements[:3]:  # Test on first 3 statements
                test_statements.append(stmt['text'])
                expected_labels.append("TRUE" if stmt['is_true'] else "FALSE")
        
        print(f"\n🧪 Testing probe on {len(test_statements)} statements from generated dataset:")
        probabilities = inference.get_probabilities(test_statements)
        
        correct_predictions = 0
        for i, (statement, expected) in enumerate(zip(test_statements, expected_labels)):
            tokens = inference.model.to_str_tokens(statement, prepend_bos=False)
            token_probs = probabilities[i]
            max_tokens = min(len(tokens), len(token_probs) - 1)
            overall_prob = token_probs[1:max_tokens+1].mean().item()
            
            prediction = 'TRUE' if overall_prob > 0.5 else 'FALSE'
            confidence = abs(overall_prob - 0.5) * 2
            correct = "✓" if prediction == expected else "✗"
            
            if prediction == expected:
                correct_predictions += 1
            
            print(f"   {i+1}. \"{statement}\"")
            print(f"      → {prediction} (expected: {expected}) {correct}")
            print(f"        prob: {overall_prob:.3f}, confidence: {confidence:.3f}")
        
        accuracy = correct_predictions / len(test_statements)
        print(f"\n   Accuracy: {correct_predictions}/{len(test_statements)} ({accuracy:.1%})")
        
        print_success("Probe testing on generated data completed")
        return True
        
    except Exception as e:
        print_error(f"Probe testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the NTML-integrated GPT2 test"""
    print_header("NTML-Integrated Mac Test with GPT2")
    print("This test generates a real 2T1L dataset and trains a probe on it")
    print(f"Repository root: {repo_root}")
    print("Expected runtime: 10-15 minutes")
    
    # Change to repo root directory
    os.chdir(repo_root)
    
    # Track test results
    test_results = {}
    
    try:
        # Run test steps including NTML generation
        test_results["basic_setup"] = check_basic_setup()
        test_results["dependencies"] = install_minimal_dependencies()
        test_results["gpt2_access"] = test_gpt2_access()
        test_results["ntml_generation"] = generate_ntml_dataset()
        test_results["probe_training"] = train_simple_probe()
        test_results["probe_testing"] = test_probe_on_generated_data()
        
        # Print summary
        print_header("Test Results Summary")
        passed = sum(test_results.values())
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print_success("🎉 All tests passed! NTML-integrated system works with GPT2.")
            print("✅ Successfully generated dataset, trained probe, and tested on real data!")
        else:
            print_error(f"⚠️  {total - passed} tests failed. Check errors above.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 