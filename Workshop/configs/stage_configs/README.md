# Stage Configurations

This directory contains reusable stage configuration files that can be referenced by training pipelines.

## Structure

Each stage config is a JSON file containing the hyperparameters and settings for a specific training stage. These configs are referenced by name in the main pipeline configuration files.

**Note**: Stage configs focus on training hyperparameters only. The `processed_dataset_name` and `starting_model` fields are automatically set at runtime based on command-line arguments and pipeline structure, so you don't need to specify them in your stage configs.

## Available Stage Configs

### SFT (Supervised Fine-Tuning) Configs

- `sft_default.json` - Standard SFT configuration with 2 epochs, moderate learning rate
- `sft_fast.json` - Fast SFT configuration with 1 epoch, higher learning rate, and packing enabled

### DPO (Direct Preference Optimization) Configs

- `dpo_default.json` - Standard DPO configuration with conservative beta (0.1)
- `dpo_aggressive.json` - Aggressive DPO configuration with higher beta (0.3) and more training

## Usage

To use these stage configs in a pipeline, reference them by name in your training configuration:

```json
{
    "pipeline_name": "my_pipeline",
    "model": "gemma-3-27b-it",
    "peft_config_name": "lora_default",
    "stages": [
        {
            "type": "sft",
            "config_name": "sft_fast"
        },
        {
            "type": "dpo", 
            "config_name": "dpo_aggressive"
        }
    ],
    "output_dir": "experiments",
    "report_to": ["wandb"],
    "wandb_project": "pollux",
    "seed": 42
}
```

## Creating New Stage Configs

To create a new stage config:

1. Create a new JSON file in this directory (e.g., `sft_my_config.json`)
2. Include only the training hyperparameters you want to customize
3. Follow the schema defined in `src/config_models/stage_configs.py`
4. Reference it in your pipeline configs using the filename (without .json extension)

**What to include**: Training hyperparameters like learning rates, batch sizes, epochs, etc.
**What to omit**: `processed_dataset_name` and `starting_model` (these are set automatically)

## Benefits

- **Reusability**: Stage configs can be shared across multiple pipelines
- **Modularity**: Easy to swap different configurations for experimentation
- **Maintainability**: Stage-specific settings are isolated and easier to manage
- **Flexibility**: Mix and match different stage configs to create custom pipelines
- **Clean configs**: No placeholder values cluttering the configuration files 