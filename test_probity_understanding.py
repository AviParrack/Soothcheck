#!/usr/bin/env python3
"""
Simple test script to understand the probity library components
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from probity.datasets.templated import TemplatedDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.probes import LogisticProbe, LogisticProbeConfig, BaseProbe
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig
from transformers import AutoTokenizer

def test_basic_workflow():
    """Test the basic probity workflow"""
    print("=== Testing Basic Probity Workflow ===\n")
    
    # 1. Create a simple dataset
    print("1. Creating dataset...")
    adjectives = {
        "positive": ["good", "great", "excellent"],
        "negative": ["bad", "terrible", "awful"]
    }
    
    movie_dataset = TemplatedDataset.from_movie_sentiment_template(
        adjectives=adjectives, verbs={"positive": ["loved"], "negative": ["hated"]}
    )
    
    # Convert to probing dataset
    probing_dataset = movie_dataset.to_probing_dataset(
        label_from_attributes="sentiment",
        label_map={"positive": 1, "negative": 0},
        auto_add_positions=True,
    )
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding=True,
        max_length=64,
        add_special_tokens=True,
    )
    
    print(f"   Dataset size: {len(tokenized_dataset.examples)}")
    print(f"   Example: {tokenized_dataset.examples[0].text}")
    print(f"   Label: {tokenized_dataset.examples[0].label}")
    print(f"   Token positions: {tokenized_dataset.examples[0].token_positions}")
    
    # 2. Configure probe and trainer
    print("\n2. Configuring probe and trainer...")
    model_name = "gpt2"
    hook_point = "blocks.7.hook_resid_pre"
    
    probe_config = LogisticProbeConfig(
        input_size=768,  # GPT-2 hidden size
        normalize_weights=True,
        bias=False,
        model_name=model_name,
        hook_point=hook_point,
        hook_layer=7,
        name="test_sentiment_probe",
    )
    
    trainer_config = SupervisedTrainerConfig(
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=5,  # Short training for testing
        weight_decay=0.01,
        train_ratio=0.8,
        handle_class_imbalance=True,
        show_progress=True,
        device="cpu",  # Use CPU for testing
    )
    
    # 3. Create pipeline
    print("\n3. Creating pipeline...")
    pipeline_config = ProbePipelineConfig(
        dataset=tokenized_dataset,
        probe_cls=LogisticProbe,
        probe_config=probe_config,
        trainer_cls=SupervisedProbeTrainer,
        trainer_config=trainer_config,
        position_key="ADJ",
        model_name=model_name,
        hook_points=[hook_point],
        cache_dir="./test_cache",
    )
    
    # 4. Train the probe
    print("\n4. Training probe...")
    pipeline = ProbePipeline(pipeline_config)
    probe, training_history = pipeline.run()
    
    print(f"   Training completed!")
    print(f"   Final train loss: {training_history['train_loss'][-1]:.4f}")
    print(f"   Final val loss: {training_history['val_loss'][-1]:.4f}")
    
    # 5. Test the probe
    print("\n5. Testing probe...")
    sentiment_direction = probe.get_direction()
    print(f"   Probe direction norm: {torch.norm(sentiment_direction):.4f}")
    print(f"   Probe direction shape: {sentiment_direction.shape}")
    
    # 6. Save and load
    print("\n6. Testing save/load...")
    probe.save("./test_probe.pt")
    loaded_probe = BaseProbe.load("./test_probe.pt")
    
    original_direction = probe.get_direction()
    loaded_direction = loaded_probe.get_direction()
    
    cos_sim = torch.nn.functional.cosine_similarity(
        original_direction, loaded_direction, dim=0
    )
    print(f"   Cosine similarity between original and loaded: {cos_sim.item():.6f}")
    
    # Cleanup
    import shutil
    if os.path.exists("./test_cache"):
        shutil.rmtree("./test_cache")
    if os.path.exists("./test_probe.pt"):
        os.remove("./test_probe.pt")
    
    print("\n=== Test completed successfully! ===")

def test_device_handling():
    """Test how devices are handled"""
    print("\n=== Testing Device Handling ===\n")
    
    # Check available devices
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected device: {device}")
    
    # Test probe configuration with device
    probe_config = LogisticProbeConfig(
        input_size=768,
        device=device,
        model_name="gpt2",
        hook_point="blocks.7.hook_resid_pre",
        hook_layer=7,
        name="device_test_probe",
    )
    
    trainer_config = SupervisedTrainerConfig(
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=1,
        device=device,
        show_progress=False,
    )
    
    print(f"Probe config device: {probe_config.device}")
    print(f"Trainer config device: {trainer_config.device}")

if __name__ == "__main__":
    test_device_handling()
    test_basic_workflow() 