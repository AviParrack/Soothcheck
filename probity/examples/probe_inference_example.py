"""
Example demonstrating the standardized pattern for doing inference with probes.

This shows the recommended workflow for training, saving, loading, and
getting inference results from probes using the ProbeInference class.
"""

import torch
from probity.probes.linear_probe import LogisticProbe, LogisticProbeConfig
from probity.probes.inference import ProbeInference


def example_workflow():
    """Full example workflow for probe inference."""
    # This example assumes you have already collected activations and have a dataset
    
    # === Step 1: Train a probe ===
    # (This part would typically use a trainer and dataset)
    input_size = 768  # For a model with 768-dim embeddings
    config = LogisticProbeConfig(input_size=input_size)
    probe = LogisticProbe(config)
    
    # Simulate training with random weights
    with torch.no_grad():
        probe.linear.weight.data = torch.randn(1, input_size)
        probe.linear.bias.data = torch.zeros(1)
    
    # === Step 2: Save the probe ===
    # Option A: Save as regular probe
    # probe.save("path/to/probe.pt")
    
    # Option B (recommended): Save as ProbeVector with metadata
    probe_vector = probe.to_probe_vector(
        model_name="gpt2-small", 
        hook_point="blocks.8.hook_resid_pre", 
        hook_layer=8
    )
    # probe_vector.save("path/to/probe_vector.json")
    
    # === Step 3: Create ProbeInference instance ===
    # From a probe object:
    inference = ProbeInference(
        model_name="gpt2-small",
        hook_point="blocks.8.hook_resid_pre",
        probe=probe
    )
    
    # Or from a saved ProbeVector file:
    # inference = ProbeInference.from_saved_probe(
    #     model_name="gpt2-small",
    #     hook_point="blocks.8.hook_resid_pre",
    #     probe_path="path/to/probe_vector.json"
    # )
    
    # Or from a directory of ProbeVectors:
    # inference = ProbeInference.from_directory(
    #     model_name="gpt2-small",
    #     hook_point="blocks.8.hook_resid_pre",
    #     directory="path/to/probe_vectors/"
    # )
    
    # === Step 4: Do inference ===
    texts = ["This is a positive example.", "This is a negative example."]
    
    # Get raw activations along the probe direction:
    raw_scores = inference.get_direction_activations(texts)
    print(f"Raw activations shape: {raw_scores.shape}")
    
    # Get probabilities (applies sigmoid for logistic probes):
    probabilities = inference.get_probabilities(texts)
    print(f"Probabilities shape: {probabilities.shape}")
    
    # Note: Avoid using the deprecated direct calling method:
    # outputs = inference(texts)  # Deprecated


if __name__ == "__main__":
    # Skip actual running to avoid requiring dependencies
    print("This is an example file demonstrating the recommended pattern")
    print("for probe inference using the ProbeInference class.")
    
    # Uncomment to run the example (requires transformer-lens and other dependencies)
    # example_workflow() 