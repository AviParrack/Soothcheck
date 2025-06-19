# Template Management System

The template management system handles chat template formatting for both training and inference in a unified way.

## Core Interface

There is now **one function** that handles all cases:

```python
from src.utils.template_manager import TemplateManager

# The universal function
result = TemplateManager.format_for_model(
    data=your_data,           # text string OR messages list
    tokenizer=tokenizer,
    model_type="chat",        # or "text_generation"
    purpose="training"        # or "inference"
)
```

## Input/Model Validation

The system validates that your data matches your model type:

- **Chat models** (`model_type="chat"`) require **messages format**: `[{"role": "user", "content": "..."}]`
- **Text generation models** (`model_type="text_generation"`) require **text format**: `"your text"`

Mismatched combinations will raise clear errors immediately.

## Examples

### Chat Model + Messages (Training)
```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
]

result = TemplateManager.format_for_model(
    data=messages,
    tokenizer=tokenizer,
    model_type="chat",
    purpose="training"
)
# Returns: {"input_ids": [...], "attention_mask": [...]}
```

### Chat Model + Messages (Inference)
```python
messages = [{"role": "user", "content": "What is 2+2?"}]

result = TemplateManager.format_for_model(
    data=messages,
    tokenizer=tokenizer,
    model_type="chat",
    purpose="inference"
)
# Returns: [2, 105, 1987, 106, ...]  # Token list
```

### Text Generation Model
```python
text = "Complete this sentence: The weather today is"

result = TemplateManager.format_for_model(
    data=text,
    tokenizer=tokenizer,
    model_type="text_generation",
    purpose="inference"
)
# Returns: [2, 25034, 1234, ...]  # Token list with BOS
```

## Error Cases
```python
# ❌ This will error - chat model needs messages
TemplateManager.format_for_model(
    data="some text",
    model_type="chat",
    ...
)

# ❌ This will error - text model needs text
TemplateManager.format_for_model(
    data=[{"role": "user", "content": "hi"}],
    model_type="text_generation",
    ...
)
```

## Key Features

- **Always returns tokens** - no confusion about text vs tokens
- **Validates input/model compatibility** - fails fast with helpful errors
- **One function handles all cases** - no need to remember multiple functions
- **Proper tokenizer integration** - uses tokenizer's built-in chat templates
- **Automatic special token handling** - BOS tokens, generation prompts, etc.

## Behind the Scenes

The system relies on transformers tokenizers having correct `chat_template` attributes. For example, Gemma's tokenizer properly handles:
- BOS token (ID 2)
- Special tokens like `<start_of_turn>` (ID 105) and `<end_of_turn>` (ID 106)
- Proper message formatting: `<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n`

This ensures your models see the exact format they expect during both training and inference.

## Integration Points

1. **Training Pipeline**: Uses `format_for_model()` with `purpose="training"`
2. **Inference Pipeline**: Uses `format_for_model()` with `purpose="inference"`  
3. **Dataset Processing**: Automatically detects data format and applies appropriate handling

## Model Support

The system automatically detects model types and applies correct templates:
- **Gemma**: `<start_of_turn>user` / `<start_of_turn>model` format
- **Llama**: `[INST]` / `[/INST]` format  
- **Mistral**: Native Mistral format
- **Others**: Uses tokenizer's built-in chat template

If a tokenizer lacks a chat template, the system raises a clear error instead of guessing.

## Dataset Formats

### SFT Datasets

Chat format datasets should have a `messages` field:
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

Text format datasets should have a `text` field:
```json
{
  "text": "This is a plain text document for training..."
}
```

### DPO Datasets

For preference learning with chat models:
```json
{
  "prompt": [{"role": "user", "content": "Explain quantum physics"}],
  "response_accepted": "Quantum physics is the study of matter and energy...",
  "response_rejected": "Quantum physics is too complicated to explain."
}
```

## Debugging

To inspect how templates are applied during training:

```bash
python -m scripts.train_stage sft_default --dataset_name my_dataset --model gemma-3-1b-it --inspect_batches 2
```

This will show the tokenized output with proper formatting applied.

## Summary

The new template system is **much simpler**:

✅ **One function** handles all cases: `format_for_model()`  
✅ **Clear validation** with helpful error messages  
✅ **Always tokens** - no text/token confusion  
✅ **Explicit control** - no hidden automatic behavior  
✅ **Fail fast** - errors instead of silent failures  

```python
# Everything you need:
TemplateManager.format_for_model(
    data=your_data,           # text or messages
    tokenizer=tokenizer,
    model_type="chat",        # or "text_generation"  
    purpose="training"        # or "inference"
)
``` 