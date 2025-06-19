# Model Configuration System

This directory contains model configuration files that define how models are loaded and configured for training and inference. The system separates **model identity** (what model to load) from **model settings** (how to load it).

Core principle: we load models either from the name of a config in configs/model (this folder), that specifies a base model (via HF string, e.g. "google/gemma-3-27b-it"), OR by specifying the directory location of an existing run (which implicitly specifies the base model and the loading config).

Model resolution logic ground truth is in src.utils.model_resolver

## Architecture Overview

### Separation of Concerns

1. **Model Config Files** (`*.json`): Define loading settings (dtype, quantization, tokenizer, etc.)
2. **Model Parameter**: Specifies which actual model to load (HuggingFace ID, config name, or experiment path)
3. **PEFT Configs**: Define LoRA/adapter settings (separate from model configs)

### Model Resolution Logic

When you specify a `--model` parameter, the system resolves it using this priority:

1. **Local Path** (e.g., `experiments/rudolf/single_stage_sft/run_name` or `rudolf/single_stage_sft/run_name`) → Load fine-tuned model directly
2. **Config Name Match** (e.g., `gemma-3-27b-it`) → Use `model_name_or_path` from corresponding config

**Note**: If you specify just a run name without a full path (e.g., `my_run_name`), the system will search across all dataset subdirectories in the experiments folder to find a matching run.

## File Structure

```
configs/model/
├── README.md                    # This file
├── gemma-3-27b-it.json         # Large model config with quantization
├── gemma-3-1b-it.json          # Small model config for testing
└── other-model-configs.json    # Additional model configs
```

## Configuration Fields

### Required Fields

- **`model_name_or_path`**: HuggingFace model identifier (e.g., `"google/gemma-3-27b-it"`)
- **`model_type`**: Either `"chat"` or `"text_generation"` (affects prompt formatting)

### Loading Settings

- **`torch_dtype`**: Precision for model weights (`"bfloat16"`, `"float16"`, `"auto"`)
- **`trust_remote_code`**: Whether to trust remote code in model repos
- **`tokenizer_name_or_path`**: Tokenizer to use (usually same as model)
- **`use_fast_tokenizer`**: Enable fast tokenizer implementation
- **`max_position_embeddings`**: Maximum sequence length
- **`attn_implementation`**: Attention mechanism (`"eager"`, `"sdpa"`)

### Quantization (Optional)

For example:

```json
"quantization_config": {
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": true
}
```

## Usage Examples

### 1. Simple Training with Default Model Config

```bash
# Uses model_name_or_path from gemma-3-27b-it.json automatically
python -m scripts.train_stage sft_default --dataset_name my_data --model gemma-3-27b-it
```

### 2. Continue from Previous Fine-tune

```bash
# First stage: SFT
python -m scripts.train_stage sft_default --dataset_name my_sft \
    --model gemma-3-27b-it

# Second stage: DPO using SFT output
python -m scripts.train_stage dpo_default --dataset_name my_dpo \
    --model experiments/my/single_stage_sft/run_20241201_120000

# Alternative: Use just the run name (system will search for it)
python -m scripts.train_stage dpo_default --dataset_name my_dpo \
    --model run_20241201_120000
```

### 3. Pipeline Training

```bash
# Full pipeline using pipeline config
python -m scripts.train_pipeline sft_then_dpo --dataset adam

# Override model in pipeline
python -m scripts.train_pipeline sft_then_dpo --dataset adam --model gemma-3-1b-it
```

# Integration with Pipelines

Pipeline configs reference model configs by name:
```json
{
    "pipeline_name": "sft_then_dpo",
    "model": "gemma-3-27b-it",  // Points to this directory
    "peft_config_name": "lora_default",
    "stages": [...] 
}
```