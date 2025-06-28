#!/usr/bin/env python3
"""
Minimal test script for NTML probe training with GPT-2.
Uses the smallest NTML dataset (2T1L_2samples.jsonl) for quick iteration.
Now with multi-GPU support enabled!
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
    print("=== Testing NTML Probe Training with GPT-2 (Multi-GPU) ===\n")
    
    # 1. Load NTML dataset
    print("1. Loading NTML dataset...")
    ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(
        "data/NTML-datasets/2T1L_2samples.jsonl"
    )
    print(f"   Loaded {len(ntml_dataset)} examples")
    
    # 2. Convert to statement-level examples
    print("\n2. Converting to statement-level examples...")
    statement_dataset = ntml_dataset.get_statement_dataset()  # None = all statements
    print(f"   Created {len(statement_dataset.examples)} statement-level examples")
    
    # 3. Tokenize with GPT-2
    print("\n3. Tokenizing with GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        statement_dataset, 
        tokenizer, 
        max_length=128
    )
    print(f"   Tokenized dataset: {len(tokenized_dataset)} examples")
    
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
    
    # 5. Configure probe and trainer
    print("\n5. Configuring probe and trainer...")
    probe_config = LogisticProbeConfig(
        input_size=768,  # GPT-2 hidden size
        device="cuda"
    )
    
    trainer_config = SupervisedTrainerConfig(
        device="cuda",
        batch_size=16,  # Smaller batch size for multi-GPU
        num_epochs=5,
        learning_rate=1e-3,
        show_progress=True,
        multi_gpu=multi_gpu_config
    )
    
    # 6. Create and run pipeline
    print("\n6. Creating and running pipeline...")
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="statement_end",
        model_name="gpt2",
        hook_points=["blocks.7.hook_resid_post"],  # Middle layer
        activation_batch_size=16,
        device="cuda",
        multi_gpu=multi_gpu_config
    )
    
    pipeline = ProbePipeline(pipeline_config)
    probe, training_history = pipeline.run()
    
    # 7. Test the probe
    print("\n7. Testing probe...")
    sentiment_direction = probe.get_direction()
    print(f"   Probe direction norm: {torch.norm(sentiment_direction):.4f}")
    print(f"   Probe direction shape: {sentiment_direction.shape}")

    # 7b. Plot training history
    try:
        import matplotlib.pyplot as plt
        print("\n7b. Plotting training history...")
        plt.figure(figsize=(8, 4))
        plt.plot(training_history["train_loss"], label="Train Loss")
        if "val_loss" in training_history:
            plt.plot(training_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Probe Training History (Multi-GPU)")
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("   Matplotlib not available, skipping plot")
    
    # 8. Save the probe
    print("\n8. Saving probe...")
    # Save in probity's standard format for evaluation
    probe_dir = Path("trained_probes/logistic")
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as .pt file (our format)
    probe.save("test_ntml_probe.pt")
    print("   Probe saved to test_ntml_probe.pt")
    
    # Save as .json file (probity's evaluation format)
    probe.save_json(str(probe_dir / "layer_7_probe.json"))
    print("   Probe saved to trained_probes/logistic/layer_7_probe.json (for evaluation)")
    
    print("\n✅ Multi-GPU NTML probe training completed successfully!")
    print("\nTo evaluate using probity's built-in system:")
    print("python scripts/probe_eval.py --model_name gpt2 --eval_dataset_dir data/NTML-datasets --probe_dir trained_probes --results_save_dir evaluation_results")

if __name__ == "__main__":
    test_ntml_probe_training() 