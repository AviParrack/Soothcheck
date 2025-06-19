# pollux

## Scripts Overview

This project provides scripts for dataset preparation, training, and inference:

- **`scripts/prepare_dataset.py`**: Converts raw data to Arrow format with tokenization
- **`scripts/train_stage.py`**: Runs individual training stages (SFT, DPO)
- **`scripts/train_pipeline.py`**: Runs multi-stage training pipelines
- **`scripts/run_experiment_suite.py`**: Runs systematic parameter exploration
- **`scripts/analyze_experiments.py`**: Analyzes experiment suite results
- **`scripts/inference.py`**: Command-line interface for model inference
- **`scripts/web_interface.py`**: Web-based interface for model comparison
- **`scripts/datalook.py`**: Web-based interface for exploring processed datasets.
- **`scripts/augment_dataset.py`**: Create more data for a dataset with synthetic data augmentations.

## Quick Start

### 1. Environment Setup
We use Conda for heavy ML dependencies, and pip for simple Python packages.
Run `envsetup.sh` to setup your environment quickly & easily.

### 2. Dataset Preparation
```bash
python -m scripts.prepare_dataset dataset_config_name dataset_base_name dataset_build
```

- `dataset_config_name` is the name of the config in configs/data.
- `dataset_base_name` is the name of the directory in datasets/raw that contains the raw data for the directory
- `dataset_build` (e.g. `sft` or `dpo`) is the name of the "build"; a single raw dataset can have multiple builds that extract different parts of it, in particular for different types of training (these end up in `datasets/processed/{dataset_base_name}_{build_name}`.)


### 3. Training
```bash
# Single stage
python -m scripts.train_stage sft_default --dataset_name adam_sft

# Multi-stage pipeline
python -m scripts.train_pipeline sft_then_dpo --dataset adam

# Experiment suite (parameter sweeps)
python -m scripts.run_experiment_suite model_comparison
```

### 4. Inference
```bash
python -m scripts.inference my_experiment
```

### 5. Web Interface
```bash
python -m scripts.web_interface my_experiment
```

---

## Detailed Script Documentation

### Dataset Preparation (`scripts/prepare_dataset.py`)

**Purpose**: Processes raw data files (text or JSONL) into Arrow format, applying chat templates and tokenization as defined by the dataset configuration.

**What it does**:
- Deletes any existing processed dataset with the same name
- Loads raw data from `datasets/raw/<dataset_name>/`
- Applies tokenization and chat templating using the tokenizer specified in the dataset config
- Saves processed data to `datasets/processed/<dataset_name>/`
- Supports document chunking for long texts

**Usage**:
```bash
python -m scripts.prepare_dataset <dataset_name>
```

**Example**:
```bash
python -m scripts.prepare_dataset sample_dataset
```

**Requirements**:
- Dataset configuration file must exist at `configs/data/<dataset_name>.json`
- Configuration must specify `tokenizer_name_or_path` field
- Raw data must be placed in `datasets/raw/<dataset_name>/`

---

### Training System

The training system supports three approaches: individual stages, multi-stage pipelines, and systematic experiment suites.

#### Individual Stage Training (`scripts/train_stage.py`)

**Purpose**: Run individual training stages (SFT, DPO) with full control over parameters.

**Usage**:
```bash
python -m scripts.train_stage <stage_config_name> --dataset_name <dataset> [options]
```

**Examples**:
```bash
# Basic SFT training
python -m scripts.train_stage sft_default --dataset_name adam_sft

# DPO training on SFT output
python -m scripts.train_stage dpo_default --dataset_name adam_dpo \
  --model adam/my_sft_run

# Custom run name and model config  
python -m scripts.train_stage sft_default --dataset_name adam_sft \
  --run_name baseline_sft --model gemma-3-1b-it
```

**Key Arguments**:
- `stage_config_name`: Stage configuration (e.g., `sft_default`, `dpo_aggressive`)
- `--dataset_name`: Processed dataset name (e.g., `adam_sft`, `adam_dpo`)
- `--model`: Model config name or experiment path (required)
- `--run_name`: Custom name for this run (auto-generated if not provided)
- `--peft_config_name`: PEFT configuration to use (default: `lora_default`)

#### Multi-Stage Pipeline Training (`scripts/train_pipeline.py`)

**Purpose**: Run complete training pipelines that automatically chain multiple stages.

**Usage**:
```bash
python -m scripts.train_pipeline <pipeline_config_name> --dataset <dataset> [options]
```

**Examples**:
```bash
# SFT followed by DPO
python -m scripts.train_pipeline sft_then_dpo --dataset adam

# Custom run name
python -m scripts.train_pipeline sft_then_dpo --dataset adam --run_name experiment_v2
```

**Benefits**:
- Automatic stage chaining (SFT output → DPO input)
- Consistent experiment organization
- Single command for complete workflows

#### Experiment Suite Training (`scripts/run_experiment_suite.py`)

**Purpose**: Run systematic parameter exploration with multiple experiments.

**Usage**:
```bash
python -m scripts.run_experiment_suite <suite_config_name> [options]
```

**Examples**:
```bash
# Compare different models
python -m scripts.run_experiment_suite model_comparison

# Hyperparameter sweep with concurrent runs
python -m scripts.run_experiment_suite hyperparameter_sweep --max_concurrent 2

# Preview experiment plan without running
python -m scripts.run_experiment_suite model_comparison --dry_run
```

**Configuration Flow**:
1. **Prepare datasets**: Create stage-specific datasets (e.g., `adam_sft`, `adam_dpo`)
2. **Define configs**: Create stage, model, and pipeline configurations
3. **Set up W&B**: Configure Weights & Biases for experiment tracking
4. **Run experiments**: Use appropriate training script for your needs

---

### Command-Line Inference (`scripts/inference.py`)

**Purpose**: Command-line interface for running inference with language models, supporting both base models and fine-tuned adapters.

**What it does**:
- Loads model configurations from `configs/model/` or experiment directories
- Supports both chat and text generation model types
- Provides interactive chat mode or single prompt mode
- Handles prompt templating automatically based on model type

**Usage**:
```bash
python -m scripts.inference <model_identifier> [options]
```

**Examples**:
```bash
# Interactive mode with a fine-tuned experiment
python -m scripts.inference my_experiment

# Interactive mode with a base model config
python -m scripts.inference gemma-3-27b-it

# Single prompt mode
python -m scripts.inference my_experiment --prompt "What are the key challenges in AI alignment?"

# Custom generation parameters
python -m scripts.inference gemma-3-27b-pt --temperature 0.8 --max_length 1024 --device cpu
```

**Key Arguments**:
- `model_identifier`: Experiment name (from `experiments/`) or config name (from `configs/model/`)
- `--model`: Optional explicit config name if different from identifier
- `--prompt_name`: System prompt template (for chat models)
- `--prompt`: Single prompt for non-interactive mode
- `--device`: Device selection (`auto`, `cpu`, `cuda`)
- `--max_length`, `--temperature`, `--top_p`: Generation parameters

**Model Types**:
- **Chat models** (suffix `-it`): Use conversation templates, support system prompts
- **Text generation models** (suffix `-pt`): Raw text completion

---

### Web Interface (`scripts/web_interface.py`)

**Purpose**: Provides a Gradio-based web interface for comparing three models: a base local model, a fine-tuned adapter, and a Gemini API model.

**What it does**:
- Preloads local models for fast inference
- Streams responses in real-time from all three models
- Provides interactive parameter tuning
- Supports password protection
- Can create publicly shareable links

**Usage**:
```bash
python -m scripts.web_interface <finetuned_experiment_name> [options]
```

**Examples**:
```bash
# Basic usage
python -m scripts.web_interface my_adam_clone_experiment

# With custom base model and sharing enabled
python -m scripts.web_interface my_experiment --base_local_model_id mistralai/Mistral-7B-Instruct-v0.2 --share_gradio

# Password protected with custom parameters
python -m scripts.web_interface my_experiment --ui_password mysecret --temperature 0.8 --max_new_tokens 1024

# Using system prompts
python -m scripts.web_interface my_experiment --prompt_name analytical_assistant
```

**Key Arguments**:
- `finetuned_experiment_name`: Required. Name of experiment in `experiments/`
- `--base_local_model_id`: Base model to compare against (default: `google/gemma-3-27b-it`)
- `--gemini_model_id`: Gemini API model (default: `gemini-2.0-flash`)
- `--ui_password`: Password protection for the interface
- `--prompt_name`: System prompt template to use
- `--share_gradio`: Create publicly shareable link
- `--device`, `--max_new_tokens`, `--temperature`, `--top_p`: Model parameters

**Requirements**:
- Fine-tuned experiment must exist in `experiments/<name>/`
- For Gemini API: `GEMINI_API_KEY` in `.env` file
- Gradio and required dependencies installed

---

## Experiment Tracking & Model Chaining

### Unified, Simple System

The new system uses a **flat, consistent directory structure** with comprehensive metadata for robust experiment tracking:

#### Directory Structure
```
experiments/
├── {dataset}/                    # Dataset name (e.g., 'adam', 'rudolf')  
│   └── {run_name}/              # All runs in flat structure
│       ├── adapter_model.safetensors
│       ├── model_config.json
│       ├── peft_config.json  
│       ├── run_metadata.json   # 🆕 Complete run information
│       └── README.md           # 🆕 Auto-generated model card
└── suites/                       # Experiment suite results
    └── {suite_name}/             # Suite directory
        └── {experiment_id}/      # Individual experiment
```

#### Key Improvements
- **Flat structure**: No more nested `single_stage_sft/` directories
- **Unified metadata**: Single `run_metadata.json` contains all information
- **Smart run names**: Descriptive names encode stage/pipeline information
- **Auto-generated model cards**: Each run gets a proper README
- **Simple inference**: `python -m scripts.inference {dataset}/{run_name}`

#### Comprehensive Run Metadata
Every run saves complete information in `run_metadata.json`:

- **Identity**: Run name, dataset, type (single_stage/pipeline/suite)
- **Configuration**: Model config, PEFT config, stage configs used
- **Training details**: All stages, hyperparameters, results
- **Chaining info**: Parent runs, dependencies, compatibility
- **W&B tracking**: Links to training logs and metrics
- **Status**: Completion status, timing, error handling

### Discovering and Managing Runs

The new system includes powerful tools for managing experiments:

#### List All Runs
```bash
# List all runs
python -m scripts.manage_runs list

# Filter by dataset
python -m scripts.manage_runs list --dataset adam

# Filter by stage type
python -m scripts.manage_runs list --stage_type sft --status completed
```

#### Find Specific Runs
```bash
# Find a run by name or path
python -m scripts.manage_runs find adam/baseline_sft
python -m scripts.manage_runs find baseline_sft  # searches all datasets

# Get detailed information
python -m scripts.manage_runs inspect adam/baseline_sft
```

#### Find Chainable Runs
```bash
# Find SFT runs that can be chained to DPO
python -m scripts.manage_runs chainable --stage dpo --dataset adam
```

### Chaining Training Stages

The new system makes chaining much simpler:

#### Method 1: Smart Discovery and Chaining
```bash
# Find runs that can be chained to DPO
python -m scripts.manage_runs chainable --stage dpo --dataset adam

# Chain to DPO using the discovered path
python -m scripts.train_stage dpo_default --dataset_name adam_dpo \
    --starting_model adam/g3-27b_sft_1201-1200 \
    --model gemma-3-27b-it
```

#### Method 2: Manual Chaining with Simple Paths
```bash
# Step 1: Train SFT  
python -m scripts.train_stage sft_default --dataset_name adam --model gemma-3-27b-it
# Creates: experiments/adam/g3-27b_sft_1201-1200/

# Step 2: Chain to DPO
python -m scripts.train_stage dpo_default --dataset_name adam_dpo \
    --starting_model adam/g3-27b_sft_1201-1200 \
    --model gemma-3-27b-it
```

#### Method 3: Pipeline Training (Automatic)
```bash
# Multi-stage training with automatic chaining
python -m scripts.train_pipeline sft_then_dpo --dataset_name adam --model gemma-3-27b-it
# Creates: experiments/adam/g3-27b_sft-dpo_1201-1200/
```

### Running Inference

With the new unified system, inference is much simpler:

```bash
# Standard format: dataset/run_name
python -m scripts.inference adam/g3-27b_sft_1201-1200

# Short format: just run_name (searches all datasets)  
python -m scripts.inference g3-27b_sft_1201-1200

# Interactive chat mode
python -m scripts.inference adam/baseline_sft

# Single prompt mode
python -m scripts.inference adam/baseline_sft --prompt "Explain quantum computing"
```

#### Method 3: Starting from Previous Pipeline Stages
```bash
# Continue from a pipeline stage
python -m scripts.train_stage dpo_aggressive --dataset_name adam_dpo \
  --starting_model experiments/adam/sft_then_dpo/stage_sft
```

### Running Inference on Trained Models

The inference system automatically resolves model paths and loads the correct configurations:

#### For Single Stage Runs
```bash
# The full path works directly
python -m scripts.inference experiments/adam/single_stage_sft/my_sft_run

# You can also create a symlink for easier access
ln -s experiments/adam/single_stage_sft/my_sft_run experiments/my_sft_model
python -m scripts.inference my_sft_model
```

#### For Pipeline Runs  
```bash
# Use the final stage of a pipeline
python -m scripts.inference experiments/adam/sft_then_dpo/stage_dpo

# Or intermediate stages
python -m scripts.inference experiments/adam/sft_then_dpo/stage_sft
```

#### For Legacy Experiments
```bash
# Old format experiments work directly
python -m scripts.inference adam-clone
```

### Finding and Managing Experiments

#### List Available Experiments
```bash
# See all experiments for a dataset
find experiments/adam -name "*.json" -path "*/adapter_model.safetensors" -exec dirname {} \;

# See all completed experiments
find experiments -name "all_results.json" -exec dirname {} \;
```

#### Query Experiment Metadata
```bash
# Check what config was used for a run
cat experiments/adam/single_stage_sft/my_run/stage_run_config.json

# See training results
cat experiments/adam/single_stage_sft/my_run/all_results.json

# Check model configuration  
cat experiments/adam/single_stage_sft/my_run/model_config.json
```

#### Experiment Suite Analysis
```bash
# Analyze suite results
python -m scripts.analyze_experiments model_comparison --metric eval_loss

# Export results to CSV for further analysis
python -m scripts.analyze_experiments hyperparameter_sweep --export_csv results.csv

# Generate plots
python -m scripts.analyze_experiments model_comparison --plot
```

### Running Experiment Suites

For systematic parameter exploration, use experiment suites:

#### Define an Experiment Suite
Create a configuration in `configs/experiment_suites/` that specifies parameter variations:
```json
{
    "suite_name": "model_comparison",
    "parameter_sweep": {
        "model_configs": ["gemma-3-1b-it", "gemma-3-27b-it"],
        "datasets": ["adam", "rudolf"],
        "seeds": [42, 123]
    }
}
```

#### Run the Suite
```bash
# Preview what will run (dry run)
python -m scripts.run_experiment_suite model_comparison --dry_run

# Run all experiments
python -m scripts.run_experiment_suite model_comparison

# Run with parallel execution
python -m scripts.run_experiment_suite model_comparison --max_concurrent 4
```

#### Analyze Results
```bash
# Generate analysis report
python -m scripts.analyze_experiments model_comparison

# Export to CSV
python -m scripts.analyze_experiments model_comparison --export_csv results.csv
```

### Best Practices for Experiment Organization

1. **Use Descriptive Run Names**: Instead of timestamps, use meaningful names
   ```bash
   python -m scripts.train_stage sft_default --dataset_name adam_sft --run_name baseline_sft
   ```

2. **Consistent Dataset Naming**: Use suffixes to indicate stage-specific datasets
   - `adam_sft` for SFT data
   - `adam_dpo` for DPO preference pairs

3. **Document Your Experiments**: The `README.md` file in each experiment directory captures the training configuration and results

4. **Use Experiment Suites for Systematic Exploration**: Rather than manual runs, use parameter sweeps for systematic comparison

5. **Chain Experiments Logically**: Build up from base model → SFT → DPO → specialized fine-tuning

### Troubleshooting Common Issues

#### Inference Path Resolution
If inference can't find your model:
```bash
# Check the exact path exists
ls -la experiments/adam/single_stage_sft/my_run/

# Verify adapter files are present  
ls -la experiments/adam/single_stage_sft/my_run/adapter_model.safetensors

# Use full path if needed
python -m scripts.inference experiments/adam/single_stage_sft/my_run
```

#### Missing Dependencies for Chaining
When chaining stages, ensure:
- Previous stage completed successfully (check for `adapter_model.safetensors`)
- Dataset for next stage exists (`datasets/processed/{dataset_name}`)
- Configurations are compatible (same base model)

---

## Configuration System

### File Structure
```
configs/
├── train/              # Pipeline configurations (sft_then_dpo.json)
├── stage_configs/      # Stage configurations (sft_default.json, dpo_aggressive.json)
├── model/              # Model configurations (gemma-3-27b-it.json)
├── data/               # Dataset configurations (general.json)
├── peft/               # PEFT configurations (lora_default.json)
└── experiment_suites/  # Experiment suite configurations (model_comparison.json)

datasets/
├── raw/                # Raw input data
└── processed/          # Processed Arrow datasets

experiments/
├── {dataset}/          # Per-dataset experiments
│   ├── single_stage_sft/
│   ├── single_stage_dpo/
│   └── {pipeline_name}/
└── suites/             # Experiment suite results
    └── {suite_name}/
```

### Configuration Inheritance
Configurations support inheritance and composition. Training configs reference model and dataset configs by name.

### Prompt Templates
System prompts are stored in `prompts/` directory and can be referenced by name in model configurations or command-line arguments.

---

## Generation Parameters

All inference interfaces support these parameters:

- **Max Length/Tokens**: Maximum number of new tokens to generate
- **Temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **Top-p**: Nucleus sampling threshold (controls diversity)

**Recommended settings**:
- **Creative writing**: `temperature=0.8, top_p=0.9`
- **Analytical tasks**: `temperature=0.3, top_p=0.7`
- **Code generation**: `temperature=0.2, top_p=0.95`

---

## Tips for Effective Usage

### For Dataset Preparation
1. Ensure raw data files are properly formatted (text or JSONL)
2. Test with a small subset first
3. Check tokenizer compatibility with your model

### For Training
1. Start with existing configuration examples
2. Monitor training in W&B dashboard
3. Use appropriate learning rates for your model size

### For Inference
1. Use chat models (`-it`) for conversational tasks
2. Use text models (`-pt`) for completion tasks
3. Experiment with generation parameters for your use case
4. Use system prompts to guide model behavior

### Example Prompts for Fine-tuned Models
- "What are the most important considerations for designing an AI alignment research agenda?"
- "Analyze the trade-offs between different approaches to AI governance."
- "What are the practical steps that AI labs should take to ensure responsible development?"
- "Critique the current discourse around AI safety and suggest improvements."

---

## Quick Reference

### Common Commands

```bash
# 1. Prepare datasets
python -m scripts.prepare_dataset adam_sft
python -m scripts.prepare_dataset adam_dpo

# 2. Train single stage
python -m scripts.train_stage sft_default --dataset_name adam_sft --run_name baseline

# 3. Chain stages manually  
python -m scripts.train_stage dpo_default --dataset_name adam_dpo \
  --starting_model experiments/adam/single_stage_sft/baseline --run_name baseline_dpo

# 4. Run complete pipeline
python -m scripts.train_pipeline sft_then_dpo --dataset adam --run_name pipeline_v1

# 5. Run inference
python -m scripts.inference experiments/adam/single_stage_sft/baseline
python -m scripts.inference experiments/adam/sft_then_dpo/stage_dpo

# 6. Systematic experiments
python -m scripts.run_experiment_suite model_comparison --dry_run
python -m scripts.run_experiment_suite model_comparison --max_concurrent 2

# 7. Analyze results
python -m scripts.analyze_experiments model_comparison --metric eval_loss --plot
```

### Directory Quick Lookup

| Experiment Type | Location | Example |
|----------------|----------|---------|
| Single SFT | `experiments/{dataset}/single_stage_sft/{run_name}/` | `experiments/adam/single_stage_sft/baseline/` |
| Single DPO | `experiments/{dataset}/single_stage_dpo/{run_name}/` | `experiments/adam/single_stage_dpo/baseline_dpo/` |
| Pipeline | `experiments/{dataset}/{pipeline_name}/stage_{stage}/` | `experiments/adam/sft_then_dpo/stage_dpo/` |
| Suite | `experiments/suites/{suite_name}/{experiment_id}/` | `experiments/suites/model_comparison/g3-27b_sft-dpo_adam_s42/` |

### Configuration Quick Lookup

| Config Type | Location | Purpose |
|-------------|----------|---------|
| Stage | `configs/stage_configs/` | Training hyperparameters (learning rate, epochs, etc.) |
| Model | `configs/model/` | Model loading settings (dtype, quantization, etc.) |
| PEFT | `configs/peft/` | LoRA/adapter parameters (rank, alpha, target modules) |
| Pipeline | `configs/train/` | Multi-stage workflow definitions |
| Suite | `configs/experiment_suites/` | Parameter sweep definitions |
