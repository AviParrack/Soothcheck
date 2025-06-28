#!/usr/bin/env python3
"""
Simple NTML probe training with GPT-2 and multi-GPU support.
Follows the exact pattern from the tutorial but with NTML data.
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

def test_ntml_probe_training():
    """Test NTML probe training with GPT-2 and multi-GPU support"""
    print("=== NTML Probe Training with GPT-2 (Multi-GPU) ===\n")
    
    # 1. Load NTML dataset
    print("1. Loading NTML dataset...")
    ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(
        "data/NTML-datasets/2T1L_2samples.jsonl"
    )
    print(f"   Loaded {len(ntml_dataset.examples)} conversations")
    
    # 2. Convert to statement-level dataset
    print("\n2. Converting to statement-level dataset...")
    statement_dataset = ntml_dataset.get_statement_dataset()  # None = all statements
    print(f"   Created {len(statement_dataset.examples)} statement examples")
    
    # 3. Tokenize with GPT-2
    print("\n3. Tokenizing with GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=statement_dataset,
        tokenizer=tokenizer,
        padding=True,
        max_length=128,
        add_special_tokens=True,
    )
    print(f"   Tokenized dataset size: {len(tokenized_dataset.examples)}")
    
    # 4. Configure multi-GPU
    print("\n4. Configuring multi-GPU...")
    multi_gpu_config = MultiGPUConfig(
        enabled=True,
        backend="DataParallel",
        device_ids=[0, 1]  # Use both GPUs
    )
    print(f"   Multi-GPU enabled: {multi_gpu_config.enabled}")
    print(f"   Backend: {multi_gpu_config.backend}")
    print(f"   Device IDs: {multi_gpu_config.device_ids}")
    
    # 5. Configure probe and trainer (following tutorial pattern)
    print("\n5. Configuring probe and trainer...")
    model_name = "gpt2"
    hook_point = "blocks.7.hook_resid_pre"
    
    probe_config = LogisticProbeConfig(
        input_size=768,  # GPT-2 hidden size
        normalize_weights=True,
        bias=False,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=7,
        name="ntml_truth_probe",
    )
    
    trainer_config = SupervisedTrainerConfig(
        batch_size=16,  # Smaller batch size for multi-GPU
        learning_rate=1e-3,
        num_epochs=5,  # Few epochs for quick test
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=True,
        device="cuda",
        multi_gpu=multi_gpu_config
    )
    
    # 6. Create pipeline
    print("\n6. Creating pipeline...")
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="target",  # Use the statement position key
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir="./test_ntml_cache",
        multi_gpu=multi_gpu_config
    )
    
    # 7. Train the probe
    print("\n7. Training probe...")
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
        plt.title("NTML Probe Training History (Multi-GPU)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("   Matplotlib not available, skipping plot")
    
    # 10. Save the probe
    print("\n10. Saving probe...")
    probe.save("test_ntml_probe.pt")
    print("   Probe saved to test_ntml_probe.pt")
    
    print("\n✅ Multi-GPU NTML probe training completed successfully!")

if __name__ == "__main__":
    test_ntml_probe_training() 