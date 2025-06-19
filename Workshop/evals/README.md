# Model Evaluation System

This directory contains the evaluation infrastructure for benchmarking trained models on various datasets.

## Directory Structure

```
evals/
├── eval_datasets/      # Evaluation datasets (JSON files)
│   └── example_basic.json
└── model_responses/    # Model outputs organized by dataset and run
    └── {dataset_name}/
        └── {run_name}/
            ├── eval_config.json
            ├── evaluation_summary.json
            └── {model_id}_responses.json
```

## Usage

### Running Evaluations

Evaluate models using the benchmark script:

```bash
# Evaluate a single model on a dataset
python -m scripts.benchmark example_basic --model gemma-3-27b-it

# Evaluate all models from an experiment suite
python -m scripts.benchmark example_basic --suite my_experiment_suite

# Evaluate multiple models with custom settings
python -m scripts.benchmark example_basic \
    --model adam/my_run \
    --model base:google/gemma-2-9b-it \
    --model config:gemma-3-27b-it-4bit \
    --max-new-tokens 1024 \
    --temperature 0.8 \
    --run-name comparison_v1
```

### Model Specification Formats

Models can be specified in several ways:

1. **Experiment runs**: `dataset/run_name` or just `run_name`
   - Example: `adam/sft_run_001` or `sft_run_001`
   
2. **Model configs**: Config name from `configs/model/`
   - Example: `gemma-3-27b-it-4bit`
   
3. **Base models**: `base:hf_model_id`
   - Example: `base:google/gemma-2-9b-it`
   
4. **Explicit configs**: `config:config_name`
   - Example: `config:gemma-3-27b-it-8bit`

### Creating Evaluation Datasets

Evaluation datasets are JSON files in `eval_datasets/` with this structure:

```json
{
    "description": "Description of the dataset",
    "prompts": [
        {
            "prompt": "The actual prompt text",
            "category": "optional_category",
            "any_other": "metadata_fields"
        }
    ]
}
```

Or a simpler format with just prompt strings:

```json
{
    "prompts": [
        "First prompt",
        "Second prompt"
    ]
}
```

## Output Format

Results are saved in `model_responses/{dataset_name}/{run_name}/`:

- `eval_config.json`: Configuration used for the evaluation
- `evaluation_summary.json`: Summary of all models evaluated
- `{model_id}_responses.json`: Detailed responses for each model

### Response File Structure

Each model's response file contains:

```json
{
    "model_id": "unique_identifier",
    "model_path": "path_to_model",
    "model_type": "trained|base|config",
    "base_model": "base_model_name",
    "peft_applied": true/false,
    "responses": [
        {
            "prompt": "original_prompt",
            "response": "model_response",
            "metadata": {...}
        }
    ],
    "metrics": {
        "total_prompts": 10,
        "successful_responses": 10,
        "average_response_length": 45.2,
        "total_time_seconds": 120.5
    }
}
```

## Model Loading Clarity

The system provides extensive logging to ensure clarity about which model is being loaded:

```
================================================================================
UNIFIED MODEL LOADING PIPELINE
================================================================================
Model specification: adam/my_sft_run
Target device: auto

Model Resolution Details:
  ✓ Base model HF ID: google/gemma-2b-it
  ✓ Model type: chat
  ✓ Is experiment: True
  ✓ PEFT adapter path: experiments/adam/my_sft_run
    → This is a FINE-TUNED model with LoRA/PEFT adapters
  ✓ Experiment path: experiments/adam/my_sft_run

========================================
LOADING MODEL COMPONENTS
========================================
→ Loading fresh model instance from HuggingFace
  → Loading base model: google/gemma-2b-it
    Device map: auto
    Dtype: bfloat16
  ✓ Base model loaded successfully
  → Applying PEFT adapters...
  ✓ PEFT adapters applied successfully
  ✓ PEFT model ready: google/gemma-2b-it + experiments/adam/my_sft_run
```

## Best Practices

1. **Dataset Naming**: Use descriptive names for datasets (e.g., `coding_problems_v2.json`)

2. **Run Names**: When using `--run-name`, use descriptive names that help identify the evaluation:
   - `baseline_comparison`
   - `temperature_sweep_0.7`
   - `suite_eval_20240115`

3. **Model Selection**: Be explicit about model types when there's ambiguity:
   - Use `config:model_name` to explicitly load a config
   - Use `base:model_id` to explicitly load a base model

4. **Quantization**: Models with quantization configs (4-bit, 8-bit) will automatically apply the correct settings from their model configs. 