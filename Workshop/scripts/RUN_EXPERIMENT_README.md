# Experiment System Documentation

This document explains how the experiment system works, where outputs are stored, and how to work with the results.

## Overview

The experiment system supports three types of training runs:
- **Single Stage**: Individual training stages (SFT, DPO, etc.)
- **Pipeline**: Multi-stage training sequences (e.g., SFT → DPO)
- **Suite**: Systematic parameter exploration with multiple experiments

## Experiment Output Structure

### Base Directory Structure

All experiments are written to the `experiments/` directory with the following structure:

```
experiments/
├── {dataset_name}/
│   ├── {run_name}/                    # Individual experiment outputs
│   │   ├── run_metadata.json         # Complete training metadata
│   │   ├── README.md                 # Generated model card
│   │   ├── adapter_model.safetensors # LoRA adapter weights
│   │   ├── model_config.json         # Model configuration
│   │   ├── peft_config.json          # PEFT/LoRA configuration
│   │   ├── trainer_state.json        # Detailed training logs
│   │   └── logs/                     # Training logs directory
│   └── {another_run}/
└── suites/                           # Experiment suite tracking
    └── {suite_name}/
        ├── experiment_plan.json      # Complete experiment plan
        ├── experiments.json          # Experiment status tracking
        ├── suite_summary.json        # Final results summary
        └── logs/                     # Individual experiment logs (concurrent runs)
```

### Run Naming Convention

- **Manual runs**: User-specified or auto-generated based on model/stage/timestamp
- **Pipeline runs**: Auto-generated as `{model_short}_{stage1}-{stage2}_{timestamp}`
- **Suite experiments**: `{suite_name}_{experiment_id}`

Example paths:
- `experiments/adam/g3-27b_sft_1205-143022/`
- `experiments/rudolf/model_comparison_g3-27b_sft-dpo_adam/`

## Run Metadata System

### Complete Metadata (`run_metadata.json`)

Each experiment produces a comprehensive metadata file containing:

```json
{
  "run_name": "model_comparison_g3-27b_sft-dpo_adam",
  "dataset_name": "adam",
  "run_type": "pipeline",  // single_stage, pipeline, suite_experiment
  "created_at": "2024-12-05T14:30:22",
  "completed_at": "2024-12-05T16:45:33",
  "model": "gemma-3-27b-it",
  "peft_config_name": "lora_default",
  "stages": [
    {
      "stage_type": "sft",
      "stage_config_name": "sft_default",
      "dataset_name": "adam_sft",
      "starting_model": "gemma-3-27b-it",
      "final_loss": 0.1234,
      "eval_loss": 0.1456,
      "total_steps": 1000,
      "learning_rate": 2e-4,
      "num_epochs": 3.0,
      "batch_size": 8
    },
    {
      "stage_type": "dpo",
      "stage_config_name": "dpo_default", 
      "dataset_name": "adam_dpo",
      "starting_model": "experiments/adam/model_comparison_g3-27b_sft-dpo_adam/sft",
      "final_loss": 0.0876,
      "eval_loss": 0.0923,
      "total_steps": 500,
      "learning_rate": 5e-5,
      "num_epochs": 1.0,
      "batch_size": 4
    }
  ],
  "chain_info": {
    "parent_run": "adam/previous_sft_run",
    "parent_stage": "sft",
    "chain_type": "pipeline"
  },
  "pipeline_config_name": "sft_then_dpo",
  "suite_name": "model_comparison",
  "experiment_id": "g3-27b_sft-dpo_adam",
  "wandb_project": "pollux",
  "wandb_run_urls": [
    "https://wandb.ai/project/pollux/runs/abc123",
    "https://wandb.ai/project/pollux/runs/def456"
  ],
  "tags": ["suite:model_comparison", "model:gemma-3-27b-it", "dataset:adam"],
  "status": "completed"
}
```

### Generated Model Card (`README.md`)

Each run gets an auto-generated model card with:
- Model details and training configuration
- Stage-by-stage training information
- Usage instructions
- Links to W&B training logs
- File descriptions

## Finding and Managing Runs

### RunManager Interface

The `RunManager` class provides unified access to all runs:

```python
from src.training.run_manager import RunManager, find_run, list_runs

# Find a specific run
run_path, metadata = find_run("adam/my_run_name")
run_path, metadata = find_run("my_run_name")  # searches all datasets
run_path, metadata = find_run("/full/path/to/run")

# List runs with filtering
runs = list_runs(
    dataset_name="adam",           # optional filter
    run_type="pipeline",           # single_stage, pipeline, suite_experiment
    status="completed",            # running, completed, failed
    stage_type="dpo"              # find runs containing specific stages
)

# Find runs that can be chained to DPO
chainable_runs = manager.find_chainable_runs(
    target_stage_type="dpo",
    dataset_name="adam",
    model="gemma-3-27b-it"
)
```

### Run Discovery Patterns

The system supports flexible run identification:
- `dataset/run_name`: Exact specification
- `run_name`: Searches across all datasets
- Full path: Direct path specification

## Inference on Trained Models

### Basic Inference Command

```bash
python -m scripts.inference {dataset_name}/{run_name}
```

Examples:
```bash
# Single stage run
python -m scripts.inference adam/g3-27b_sft_1205-143022

# Pipeline run  
python -m scripts.inference rudolf/sft-dpo_model_1205-160000

# Suite experiment
python -m scripts.inference adam/model_comparison_g3-27b_sft-dpo_adam
```

### Model Loading

Each run directory contains everything needed for inference:
- `adapter_model.safetensors`: LoRA adapter weights
- `peft_config.json`: PEFT configuration for loading adapters
- `model_config.json`: Base model information

The inference script automatically:
1. Loads the base model specified in metadata
2. Applies PEFT adapters from the final training stage
3. Sets up the tokenizer with proper chat templates

### Multi-Stage Runs

For pipeline runs with multiple stages:
- Final model uses the last stage's adapters (typically DPO if SFT→DPO)
- Intermediate stage models are also available in subdirectories:
  - `{run_dir}/sft/` - SFT stage output
  - `{run_dir}/dpo/` - DPO stage output

## Experiment Suite Results

### Suite Organization

Experiment suites create additional tracking in `experiments/suites/{suite_name}/`:

- **`experiment_plan.json`**: Complete planned experiments
- **`experiments.json`**: Real-time status tracking
- **`suite_summary.json`**: Final results with status counts
- **`logs/{experiment_id}.log`**: Individual experiment logs (concurrent runs)

### Suite Analysis

The suite summary provides experiment-level results:

```json
{
  "suite_name": "model_comparison",
  "completed_at": "2024-12-05T18:30:00",
  "total_experiments": 12,
  "status_counts": {
    "completed": 10,
    "failed": 2
  },
  "experiments": {
    "experiment_id_1": {
      "status": "completed",
      "updated_at": "2024-12-05T16:45:00",
      "final_model_path": "experiments/adam/model_comparison_experiment_id_1"
    }
  }
}
```

## Integration Points for Evaluation

### For Building an Eval Harness

The system provides several integration points:

1. **Run Discovery**: Use `RunManager.list_runs()` to find completed experiments
2. **Metadata Access**: Each run's `run_metadata.json` contains complete configuration
3. **Model Loading**: Standardized inference interface for any completed run
4. **Result Organization**: Suite summaries provide experiment groupings and status

### Recommended Eval Harness Workflow

1. **Discovery Phase**:
   ```python
   # Find all completed runs for evaluation
   completed_runs = list_runs(status="completed")
   
   # Or focus on specific suites
   suite_runs = list_runs(status="completed", suite_name="model_comparison")
   ```

2. **Inference Phase**:
   ```python
   # For each run, load model and run standard prompts
   for run_path, metadata in completed_runs:
       run_id = f"{metadata.dataset_name}/{metadata.run_name}"
       # Call inference script or load model directly
   ```

3. **Analysis Phase**:
   - Compare responses across different models/configurations
   - Analyze by suite groupings for systematic comparisons
   - Use metadata for grouping (model type, dataset, hyperparameters)

### Available Metadata for Analysis

Each run provides rich metadata for evaluation analysis:
- **Training config**: Model, PEFT settings, hyperparameters
- **Training results**: Losses, metrics from each stage
- **Organizational info**: Suite membership, tags, experiment parameters
- **Reproducibility**: Seeds, exact configurations, W&B links
- **Chaining info**: Which runs built on previous results

This comprehensive metadata enables sophisticated evaluation analysis across parameter sweeps and systematic experiments.
