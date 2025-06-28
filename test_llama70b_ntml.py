#!/usr/bin/env python3
"""
NTML probe training with Llama 3.3 70B and multi-GPU support.
Uses quantization for memory efficiency and tests on 2 samples.
"""

import torch
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probity_extensions.conversational import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from probity.utils.multigpu import MultiGPUConfig
from transformers import AutoTokenizer

def test_llama70b_ntml():
    """Test NTML probe training with Llama 3.3 70B"""
    print("=== NTML Probe Training with Llama 3.3 70B (Multi-GPU) ===\n")
    
    # 1. Load NTML dataset (3T3L 20 samples)
    print("1. Loading NTML dataset...")
    ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(
        "data/NTML-datasets/3T3L_20samples.jsonl"
    )
    print(f"   Loaded {len(ntml_dataset.examples)} conversations")
    
    # 2. Convert to statement-level dataset
    print("\n2. Converting to statement-level dataset...")
    statement_dataset = ntml_dataset.get_statement_dataset()  # None = all statements
    print(f"   Created {len(statement_dataset.examples)} statement examples")
    
    # 3. Tokenize with Llama tokenizer
    print("\n3. Tokenizing with Llama 3.3 70B tokenizer...")
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=statement_dataset,
        tokenizer=tokenizer,
        padding=True,
        max_length=4096,  # Full context length for Llama (supports up to 128K, but 4K is standard)
        add_special_tokens=True,
    )
    print(f"   Tokenized dataset size: {len(tokenized_dataset.examples)}")
    
    # 4. Configure multi-GPU
    print("\n4. Configuring multi-GPU...")
    multi_gpu_config = MultiGPUConfig(
        enabled=True,
        backend="DataParallel",
        device_ids=[0, 1]  # Use both H100s
    )
    print(f"   Multi-GPU enabled: {multi_gpu_config.enabled}")
    print(f"   Backend: {multi_gpu_config.backend}")
    print(f"   Device IDs: {multi_gpu_config.device_ids}")
    
    # 5. Configure probe and trainer for Llama 70B
    print("\n5. Configuring probe and trainer for Llama 70B...")
    hook_point = "blocks.22.hook_resid_pre"  # Layer 22 of 70B model
    
    probe_config = LogisticProbeConfig(
        input_size=8192,  # Llama 70B hidden size
        normalize_weights=True,
        bias=False,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=22,
        name="llama70b_3t3l_layer22_probe",
    )
    
    trainer_config = SupervisedTrainerConfig(
        batch_size=1,  # Small batch size for large Llama 70B model
        learning_rate=1e-3,
        num_epochs=3,  # Few epochs for quick test
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=True,
        device="cuda",
        multi_gpu=multi_gpu_config
    )
    
    # 6. Create pipeline (no quantization)
    print("\n6. Creating pipeline (no quantization)...")
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="target",  # Use the statement position key
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir="./test_llama70b_3t3l_layer22_cache",
        device="cuda",
        device_map="auto",  # Let the library handle device mapping
        low_cpu_mem_usage=True,  # Reduce memory usage during loading
        multi_gpu=multi_gpu_config
    )
    
    # 7. Train the probe
    print("\n7. Training probe...")
    print("   Note: This will download Llama 3.3 70B (~140GB) on first run")
    print("   Model will be cached for future use")
    print("   Using batch size 1 with memory optimizations (device_map='auto')")
    
    pipeline = ProbePipeline(pipeline_config)
    probe, training_history = pipeline.run()
    
    print(f"   Training completed!")
    print(f"   Final train loss: {training_history['train_loss'][-1]:.4f}")
    if 'val_loss' in training_history:
        print(f"   Final val loss: {training_history['val_loss'][-1]:.4f}")
    
    # 8. Test the probe
    print("\n8. Testing probe...")
    sentiment_direction = probe.get_direction()
    print(f"   Probe direction norm: {torch.norm(sentiment_direction):.4f}")
    print(f"   Probe direction shape: {sentiment_direction.shape}")

    # 9. Plot training history
    try:
        import matplotlib.pyplot as plt
        print("\n9. Plotting training history...")
        plt.figure(figsize=(8, 4))
        plt.plot(training_history["train_loss"], label="Train Loss")
        if "val_loss" in training_history:
            plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Llama 70B 3T3L Layer 22 Probe Training History (Multi-GPU)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("   Matplotlib not available, skipping plot")
    
    # 10. Save the probe
    print("\n10. Saving probe...")
    probe.save("test_llama70b_3t3l_layer22_probe.pt")
    print("   Probe saved to test_llama70b_3t3l_layer22_probe.pt")
    
    print("\n✅ Llama 70B 3T3L Layer 22 probe training completed successfully!")
    print("\nModel download info:")
    print("- Model will be cached in ~/.cache/huggingface/hub/")
    print("- Future runs will use the cached model")
    print("- Total model size: ~140GB (full precision, distributed across GPUs)")
    print("- Batch size: 1 (memory optimized for large model)")

if __name__ == "__main__":
    test_llama70b_ntml() 