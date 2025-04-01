# %% [markdown]
# # Logistic Probe Basics
# This file demonstrates a complete workflow for:
# 1. Creating a movie sentiment dataset
# 2. Training a logistic regression probe
# 3. Running inference with the probe
# 4. Saving the probe to disk (in multiple formats)
# 5. Loading the probe back from disk
# 6. Verifying that the loaded probe gives consistent results

# %% [markdown]
# ## Setup

# %% Setup and imports
import torch

from probity.datasets.templated import TemplatedDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from transformers import AutoTokenizer
from probity.probes.linear_probe import LogisticProbe, LogisticProbeConfig
from probity.training.trainer import SupervisedProbeTrainer, SupervisedTrainerConfig
from probity.probes.inference import ProbeInference
import os
import numpy as np
import random
import torch.backends
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Set torch device consistently
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Set seed for reproducibility
set_seed(42)

# %% [markdown]
# ## Dataset Creation
# To train our probe, we'll create a dataset similar to the one used in the paper Linear Representations of Sentiment (Tigges & Hollinsworth, et al). We will use Probity's TemplatedDataset class, which allows us to specify templates with auto-populating blanks. For convenience, we have a simple function that applies the movie sentiment template.
#
# Once the TemplatedDataset is created, we simply convert it to a ProbingDataset (which gives us features like labels, word positions by character, and test-train splits) and then a TokenizedProbingDataset (which gives us additional information about token positions, context length, and so forth). We keep these distinct because TokenizedProbingDatasets are tied to specific models, whereas ProbingDatasets are not.

# %%
# Create movie sentiment dataset
adjectives = {
    "positive": ["incredible", "amazing", "fantastic", "awesome", "beautiful", 
                "brilliant", "exceptional", "extraordinary", "fabulous", "great", 
                "lovely", "outstanding", "remarkable", "wonderful"],
    "negative": ["terrible", "awful", "horrible", "bad", "disappointing", 
                "disgusting", "dreadful", "horrendous", "mediocre", "miserable", 
                "offensive", "terrible", "unpleasant", "wretched"]
}
verbs = {
    "positive": ["loved", "enjoyed", "adored"],
    "negative": ["hated", "disliked", "detested"]
}

# Create dataset using factory method
movie_dataset = TemplatedDataset.from_movie_sentiment_template(
    adjectives=adjectives,
    verbs=verbs
)

# Convert to probing dataset with automatic position finding
# and label mapping from sentiment metadata
probing_dataset = movie_dataset.to_probing_dataset(
    label_from_metadata="sentiment",
    label_map={"positive": 1, "negative": 0},
    auto_add_positions=True
)

# Convert to tokenized dataset
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=probing_dataset,
    tokenizer=tokenizer,
    padding=True,  # Add padding
    max_length=128,  # Specify max length
    add_special_tokens=True
)

# %%
# Verify the tokenization worked
example = tokenized_dataset.examples[0]
print("First example tokens:", example.tokens)
print("First example text:", example.text)

# Now print examples from the probing dataset
print("Sample probing dataset examples:")
for i in np.random.choice(range(len(probing_dataset.examples)), size=min(6, len(probing_dataset.examples)), replace=False):
    ex = probing_dataset.examples[i]
    label = "positive" if ex.label == 1 else "negative"
    print(f"Example {i}: '{ex.text}' (Label: {label})")

# %% [markdown]
# ## Probe Training
# ### Configuration
# We're now ready to train the probe! We specify the training parameters via the three config objects below. The Pipeline manages the whole process, and the Trainer and Probe have their own settings.

# %% Configure model and probe
from probity.pipeline.pipeline import ProbePipeline, ProbePipelineConfig

model_name = "gpt2"
hook_point = "blocks.7.hook_resid_pre"

# Set up logistic probe configuration
probe_config = LogisticProbeConfig(
    input_size=768,
    normalize_weights=True,
    bias=True,
    model_name=model_name,
    hook_point=hook_point,
    hook_layer=7,
    name="sentiment_probe"
)

# Set up trainer configuration
trainer_config = SupervisedTrainerConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=10,
    weight_decay=0.01,
    train_ratio=0.8,  # 80-20 train-val split
    handle_class_imbalance=True,
    show_progress=True
)

pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="ADJ",  # We want to probe at the adjective position
    model_name=model_name,
    hook_points=[hook_point],
    cache_dir="./cache/sentiment_probe_cache"  # Cache activations for reuse
)

# %%
# Let's make sure the position key is correct
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(model_name)

example = tokenized_dataset.examples[0]
print(f"Example text: {model.to_str_tokens(example.text, prepend_bos=False)}")
print(f"Token positions: {example.token_positions}")
print(f"Available position keys: {list(example.token_positions.keys())}")

# Verify the position key matches what's in the dataset
print(f"\nPipeline position key: {pipeline_config.position_key}")

# %% [markdown]
# Looks like the key tokens are in the right positions. GPT2's default behavior (as implemented in the AutoTokenizer) is not to add a BOS, so we're fine in that respect.
#
# ### Training
# Let's train the probe!

# %% Collect activations
# Create and run pipeline
pipeline = ProbePipeline(pipeline_config)

probe, training_history = pipeline.run()

# The probe now contains our learned sentiment direction
sentiment_direction = probe.get_direction()

# %%
# We can analyze training history
plt.figure(figsize=(10, 5))
plt.plot(training_history['train_loss'], label='Train Loss')
plt.plot(training_history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Probe Training History')
plt.legend()
plt.show()

# %% [markdown]
# # Now we'll test the standardized inference path

# %%
# Method 1: Inference with the trained probe object
print("===== Inference with trained probe object =====")

# Create a ProbeInference instance with the trained probe
inference = ProbeInference(
    model_name=model_name,
    hook_point=hook_point,
    probe=probe,
    device=device
)

# Create some test examples
test_examples = [
    "The movie was incredible and I loved every minute of it.",
    "That film was absolutely terrible and I hated it.",
    "The acting was mediocre but I liked the soundtrack.",
    "A beautiful story with outstanding performances."
]

# Get raw activation scores along the probe direction
raw_scores = inference.get_direction_activations(test_examples)
print(f"Raw activation scores shape: {raw_scores.shape}")

# Get probabilities (applies sigmoid for logistic probes)
probabilities = inference.get_probabilities(test_examples)
print(f"Probabilities shape: {probabilities.shape}")

# Display the results
for i, example in enumerate(test_examples):
    print(f"\nText: {example}")
    print(f"Raw scores (token-level): {raw_scores[i]}")
    print(f"Probabilities (token-level): {probabilities[i]}")
    # Get mean probability across all tokens as an overall sentiment score
    overall_sentiment = probabilities[i].mean().item()
    print(f"Overall sentiment score: {overall_sentiment:.4f}")

# %% [markdown]
# # Save the probe and load it back

# %%
# Save the probe in both formats
save_dir = "./saved_probes"
os.makedirs(save_dir, exist_ok=True)

# Option 1: Save as PyTorch model (full state and config)
probe_path = f"{save_dir}/sentiment_probe.pt"
probe.save(probe_path)
print(f"Saved probe to {probe_path}")

# Option 2: Save in JSON format for easier sharing
json_path = f"{save_dir}/sentiment_probe.json"
probe.save_json(json_path)
print(f"Saved probe JSON to {json_path}")

# %%
# Method 2: Inference with the saved probe
print("\n===== Inference with loaded probe =====")

# Load the probe using from_saved_probe
loaded_inference = ProbeInference.from_saved_probe(
    model_name=model_name,
    hook_point=hook_point,
    probe_path=json_path,  # Load from the JSON format
    device=device
)

# Get results with loaded probe
loaded_raw_scores = loaded_inference.get_direction_activations(test_examples)
loaded_probabilities = loaded_inference.get_probabilities(test_examples)

# Display the results
for i, example in enumerate(test_examples):
    print(f"\nText: {example}")
    # Get mean probability across all tokens as an overall sentiment score
    overall_sentiment = loaded_probabilities[i].mean().item()
    print(f"Overall sentiment score (loaded probe): {overall_sentiment:.4f}")

# %% [markdown]
# # Verify consistency between original and loaded probes

# %%
# Compare original and loaded probe results
print("\n===== Comparing original vs loaded probe results =====")

# Get the probe directions
original_direction = probe.get_direction()
loaded_direction = loaded_inference.probe.get_direction()

# Print direction norm and first few components
print(f"Original direction norm: {torch.norm(original_direction):.6f}")
print(f"Original direction [0:5]: {original_direction[:5].cpu().tolist()}")

print(f"Loaded direction norm: {torch.norm(loaded_direction):.6f}")
print(f"Loaded direction [0:5]: {loaded_direction[:5].cpu().tolist()}")

# Check if the directions are similar
cos_sim = torch.nn.functional.cosine_similarity(original_direction, loaded_direction, dim=0)
print(f"Cosine similarity between directions: {cos_sim.item():.6f}")

# Analyze where the differences are greatest
token_diffs = torch.abs(probabilities - loaded_probabilities).mean(dim=0)
print("\nToken-level mean absolute differences:")
for i, diff in enumerate(token_diffs):
    print(f"Token {i}: {diff.item():.6f}")
    
max_diff_token = torch.argmax(token_diffs).item()
print(f"\nLargest difference at token position {max_diff_token}")

# Check difference excluding the BOS token
non_bos_raw_diff = torch.abs(raw_scores[:, 1:] - loaded_raw_scores[:, 1:]).mean().item()
non_bos_prob_diff = torch.abs(probabilities[:, 1:] - loaded_probabilities[:, 1:]).mean().item()
print(f"Mean absolute difference in raw scores (excluding BOS): {non_bos_raw_diff:.8f}")
print(f"Mean absolute difference in probabilities (excluding BOS): {non_bos_prob_diff:.8f}")

if non_bos_prob_diff < 0.3:
    print("The non-BOS tokens show reasonable consistency.")
    
# Verify the trend is similar (correlation between original and loaded results)
flattened_orig = probabilities[:, 1:].flatten()
flattened_loaded = loaded_probabilities[:, 1:].flatten()
correlation = torch.corrcoef(torch.stack([flattened_orig, flattened_loaded]))[0, 1].item()
print(f"Correlation between original and loaded probe outputs: {correlation:.6f}")

if correlation > 0.8:
    print("SUCCESS: Original and loaded probes show strong correlation (same trend)!")
elif correlation > 0.5:
    print("PARTIAL SUCCESS: Original and loaded probes show moderate correlation.")

# %% [markdown]
# # Standardized pattern for future reference

# %%
# Reference: Standardized pattern for inference
def standardized_inference_pattern(model_name, hook_point, probe_or_path, texts):
    """
    Template for the standardized probe inference pattern.
    
    Args:
        model_name: Model name for TransformerLens
        hook_point: Layer hook point
        probe_or_path: Either a probe object or path to saved probe
        texts: List of text examples to analyze
        
    Returns:
        Probabilities for each example
    """
    # Create inference object
    if isinstance(probe_or_path, str):
        # From saved probe
        inference = ProbeInference.from_saved_probe(
            model_name=model_name,
            hook_point=hook_point,
            probe_path=probe_or_path,
            device=device
        )
    else:
        # From probe object
        inference = ProbeInference(
            model_name=model_name,
            hook_point=hook_point,
            probe=probe_or_path,
            device=device
        )
    
    # Get probabilities (applies sigmoid for logistic probes)
    probabilities = inference.get_probabilities(texts)
    
    # If you need raw activations instead:
    # raw_activations = inference.get_direction_activations(texts)
    
    return probabilities

# Example usage
# probs = standardized_inference_pattern(model_name, hook_point, probe, test_examples)

# %% [markdown]
# # Conclusion
#
# - **The preferred workflow is**:
#    - Train a probe (using a trainer or pipeline)
#    - Save it with probe.save() or probe.save_json()
#    - Load it with ProbeInference.from_saved_probe()
#    - Use get_direction_activations() for raw scores or get_probabilities() for transformed outputs

# %%
