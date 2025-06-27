#!/usr/bin/env python3
"""
🧪 Comprehensive Llama 70B Setup Test Script

This script tests the complete NTML-Probity pipeline with Llama 3.3 70B,
including quantization support, memory optimization, and error handling.
"""

import sys
import os
import torch
import time
from pathlib import Path
from typing import Optional

# Add probity to path
sys.path.insert(0, str(Path(__file__).parent))

def print_step(step: int, description: str):
    print(f"\n{'='*60}")
    print(f"🔍 Step {step}: {description}")
    print('='*60)

def print_success(message: str):
    print(f"✅ {message}")

def print_warning(message: str):
    print(f"⚠️  {message}")

def print_error(message: str):
    print(f"❌ {message}")

def check_system_requirements():
    """Check if system meets requirements for Llama 70B"""
    print_step(1, "System Requirements Check")
    
    # Check CUDA
    if not torch.cuda.is_available():
        print_error("CUDA not available - Llama 70B requires GPU")
        return False
    
    # Check GPU memory
    total_memory = 0
    for i in range(torch.cuda.device_count()):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
        total_memory += gpu_memory
        print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    print(f"Total GPU Memory: {total_memory:.1f}GB")
    
    # Memory recommendations
    if total_memory >= 160:
        print_success("Excellent! Can run Llama 70B with full precision")
        return "full_precision"
    elif total_memory >= 80:
        print_success("Good! Can run Llama 70B with 8-bit quantization")
        return "8bit"
    elif total_memory >= 40:
        print_warning("Marginal. Will try 4-bit quantization")
        return "4bit"
    else:
        print_error("Insufficient memory for Llama 70B")
        return False

def check_dependencies():
    """Check required dependencies"""
    print_step(2, "Dependencies Check")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('transformer_lens', 'transformer_lens'),
        ('accelerate', 'accelerate'),
        ('bitsandbytes', 'bitsandbytes'),
    ]
    
    missing = []
    for package, import_name in required_packages:
        try:
            __import__(import_name)
            print_success(f"{package} available")
        except ImportError:
            print_error(f"{package} missing")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages: pip install {' '.join(missing)}")
        return False
    
    return True

def check_huggingface_access():
    """Check HuggingFace access to Llama 3.3 70B"""
    print_step(3, "HuggingFace Access Check")
    
    try:
        from transformers import AutoTokenizer
        model_name = "meta-llama/Llama-3.3-70B-Instruct"
        
        print("Testing tokenizer access...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_success("Llama 3.3 70B access confirmed")
        return True
        
    except Exception as e:
        print_error(f"Access denied: {e}")
        print("Please ensure:")
        print("1. You have a HuggingFace token with Llama 3.3 access")
        print("2. Run: huggingface-cli login")
        print("3. Request access at: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct")
        return False

def test_quantized_model_loading(memory_mode: str):
    """Test loading model with appropriate quantization"""
    print_step(4, f"Model Loading Test ({memory_mode})")
    
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    
    try:
        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM
        
        print(f"Loading {model_name} with {memory_mode} configuration...")
        start_time = time.time()
        
        if memory_mode == "full_precision":
            print("Loading with full precision...")
            model = HookedTransformer.from_pretrained_no_processing(
                model_name,
                device="cuda",
                dtype=torch.bfloat16
            )
        elif memory_mode == "8bit":
            print("Loading with 8-bit quantization...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            model = HookedTransformer.from_pretrained(
                model_name,
                hf_model=hf_model,
                dtype=torch.bfloat16,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
        elif memory_mode == "4bit":
            print("Loading with 4-bit quantization...")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            model = HookedTransformer.from_pretrained(
                model_name,
                hf_model=hf_model,
                dtype=torch.bfloat16,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
        
        load_time = time.time() - start_time
        print_success(f"Model loaded in {load_time:.1f} seconds")
        print(f"Model layers: {model.cfg.n_layers}")
        print(f"Hidden size: {model.cfg.d_model}")
        print(f"Model dtype: {model.cfg.dtype}")
        
        # Test a simple forward pass
        test_text = "The capital of France is"
        tokens = model.to_tokens(test_text)
        print(f"Test tokens shape: {tokens.shape}")
        
        with torch.no_grad():
            logits = model(tokens)
        print_success(f"Forward pass successful (output shape: {logits.shape})")
        
        # Test activation extraction
        hook_point = "blocks.40.hook_resid_pre"  # Middle layer
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_point])
            activations = cache[hook_point]
        print_success(f"Activation extraction successful: {activations.shape}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print_error(f"Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ntml_integration():
    """Test NTML dataset integration"""
    print_step(5, "NTML Integration Test")
    
    try:
        from probity_extensions import ConversationalProbingDataset
        
        # Check for test dataset
        test_dataset_path = Path("data/NTML-datasets/2T1L_20samples.jsonl")
        if not test_dataset_path.exists():
            print_warning("No test dataset found - creating minimal test...")
            test_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create minimal test dataset
            minimal_dataset = '''{"conversation": [{"role": "system", "content": "Give 2 truths and 1 lie. You will lie on statement 3."}, {"role": "assistant", "content": "1. Paris is the capital of France. 2. Water boils at 100°C. 3. The Great Wall of China is visible from space."}], "statements": [{"text": "Paris is the capital of France.", "label": true, "position": [0, 31]}, {"text": "Water boils at 100°C.", "label": true, "position": [33, 54]}, {"text": "The Great Wall of China is visible from space.", "label": false, "position": [56, 103]}], "metadata": {"conversation_id": "test_001", "lie_positions": [3]}}'''
            
            with open(test_dataset_path, 'w') as f:
                f.write(minimal_dataset)
        
        # Load dataset
        conv_dataset = ConversationalProbingDataset.from_ntml_jsonl(str(test_dataset_path))
        stmt_dataset = conv_dataset.get_statement_dataset()
        
        print_success(f"Loaded {len(conv_dataset.examples)} conversations")
        print_success(f"Created {len(stmt_dataset.examples)} statement examples")
        
        return True
        
    except Exception as e:
        print_error(f"NTML integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_compatibility(memory_mode: str):
    """Test the training script with appropriate parameters"""
    print_step(6, "Training Script Compatibility Test")
    
    try:
        # Determine training parameters based on memory mode
        if memory_mode == "full_precision":
            cmd_args = [
                "--dataset", "2T1L_20samples",
                "--model_name", "meta-llama/Llama-3.3-70B-Instruct",
                "--hook_point", "blocks.40.hook_resid_pre",
                "--batch_size", "2",
                "--num_epochs", "1",  # Just 1 epoch for testing
                "--debug"
            ]
        elif memory_mode == "8bit":
            cmd_args = [
                "--dataset", "2T1L_20samples", 
                "--model_name", "meta-llama/Llama-3.3-70B-Instruct",
                "--hook_point", "blocks.40.hook_resid_pre",
                "--load_in_8bit",
                "--batch_size", "1",
                "--num_epochs", "1",
                "--debug"
            ]
        elif memory_mode == "4bit":
            cmd_args = [
                "--dataset", "2T1L_20samples",
                "--model_name", "meta-llama/Llama-3.3-70B-Instruct", 
                "--hook_point", "blocks.40.hook_resid_pre",
                "--load_in_4bit",
                "--batch_size", "1",
                "--num_epochs", "1",
                "--debug"
            ]
        
        print("Testing training script parameter parsing...")
        
        # Import the training script
        sys.path.insert(0, str(Path(__file__).parent / "scripts"))
        from train_ntml_probes import parse_args
        
        # Test argument parsing
        test_args = parse_args(cmd_args)
        print_success("Argument parsing successful")
        print(f"Model: {test_args.model_name}")
        print(f"Hook point: {test_args.hook_point}")
        print(f"8-bit: {test_args.load_in_8bit}")
        print(f"4-bit: {test_args.load_in_4bit}")
        print(f"Batch size: {test_args.batch_size}")
        
        return True
        
    except Exception as e:
        print_error(f"Training script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run comprehensive Llama 70B setup test"""
    print("🧪 Comprehensive Llama 70B Setup Test")
    print("This script will test the complete NTML-Probity pipeline with Llama 3.3 70B")
    
    # Step 1: System requirements
    memory_mode = check_system_requirements()
    if not memory_mode:
        print("\n❌ System requirements not met. Exiting.")
        return 1
    
    # Step 2: Dependencies
    if not check_dependencies():
        print("\n❌ Dependencies missing. Exiting.")
        return 1
    
    # Step 3: HuggingFace access
    if not check_huggingface_access():
        print("\n❌ HuggingFace access failed. Exiting.")
        return 1
    
    # Step 4: Model loading (this is the big test)
    print(f"\n⚠️  About to test model loading with {memory_mode} configuration.")
    print("This will download ~140GB if not cached. Continue? (y/n): ", end="")
    
    try:
        response = input().strip().lower()
        if response != 'y':
            print("Skipping model loading test.")
            memory_mode = None
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        return 1
    
    if memory_mode and not test_quantized_model_loading(memory_mode):
        print("\n❌ Model loading failed. Check GPU memory and quantization settings.")
        return 1
    
    # Step 5: NTML integration
    if not test_ntml_integration():
        print("\n❌ NTML integration failed. Check probity_extensions.")
        return 1
    
    # Step 6: Training script compatibility
    if memory_mode and not test_training_script_compatibility(memory_mode):
        print("\n❌ Training script compatibility failed.")
        return 1
    
    # Success!
    print("\n" + "="*60)
    print("🎉 All tests passed! Llama 70B setup is ready!")
    print("="*60)
    
    if memory_mode:
        print(f"\n📋 Recommended training command for your system ({memory_mode}):")
        
        if memory_mode == "full_precision":
            cmd = """python scripts/train_ntml_probes.py \\
    --dataset 2T1L_20samples \\
    --model_name meta-llama/Llama-3.3-70B-Instruct \\
    --hook_point blocks.40.hook_resid_pre \\
    --batch_size 4 \\
    --debug"""
        elif memory_mode == "8bit":
            cmd = """python scripts/train_ntml_probes.py \\
    --dataset 2T1L_20samples \\
    --model_name meta-llama/Llama-3.3-70B-Instruct \\
    --hook_point blocks.40.hook_resid_pre \\
    --load_in_8bit \\
    --batch_size 2 \\
    --debug"""
        elif memory_mode == "4bit":
            cmd = """python scripts/train_ntml_probes.py \\
    --dataset 2T1L_20samples \\
    --model_name meta-llama/Llama-3.3-70B-Instruct \\
    --hook_point blocks.40.hook_resid_pre \\
    --load_in_4bit \\
    --batch_size 1 \\
    --debug"""
        
        print(f"\n{cmd}")
        
        print(f"\n💡 Tips for Llama 70B:")
        print("• Try different layers: blocks.20, blocks.40, blocks.60.hook_resid_pre")
        print("• Llama 3.3 70B has 80 layers total (blocks.0 to blocks.79)")
        print("• Monitor GPU memory with: watch -n 1 nvidia-smi")
        print("• Use larger datasets for better probe performance")
    
    return 0

if __name__ == "__main__":
    exit(main()) 