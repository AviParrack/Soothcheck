# %%

import os
from pathlib import Path
from transformers import AutoTokenizer

if not "notebooks" in os.listdir():
    # Get the current directory of the notebook
    notebook_dir = Path(os.getcwd())
    # Assume the project root is one level up from the notebook directory
    project_root = notebook_dir.parent
    # Change the current working directory to the project root
    if Path(os.getcwd()).resolve() != project_root.resolve():
        os.chdir(project_root)
        print(f"Changed CWD to project root: {project_root}")

from src.utils.template_manager import TemplateManager

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# %%
print("=== TESTING SPECIAL TOKENS ===")
print()

# Check what's in the tokenizer
print("Tokenizer special tokens:")
print(f"  bos_token: {repr(tokenizer.bos_token)} -> ID: {tokenizer.bos_token_id}")
print(f"  eos_token: {repr(tokenizer.eos_token)} -> ID: {tokenizer.eos_token_id}")
print(f"  pad_token: {repr(tokenizer.pad_token)} -> ID: {tokenizer.pad_token_id}")
print()

# Check special tokens map
print("Special tokens map:")
for token_str, token_id in tokenizer.special_tokens_map_extended.items():
    print(f"  {token_str}: {repr(token_id)}")
print()

# Test the problematic approach (current code)
print("=== TESTING THE BUG ===")
test_tokens = ["<|eot_id|>", "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]

for token in test_tokens:
    # This is what the current code does (WRONG)
    token_id_wrong = tokenizer.convert_tokens_to_ids(token)
    
    # Show what actually happens
    tokenized = tokenizer.tokenize(token)
    token_ids_actual = tokenizer.convert_tokens_to_ids(tokenized)
    
    print(f"Token: {token}")
    print(f"  convert_tokens_to_ids('{token}'): {token_id_wrong}")
    print(f"  tokenize('{token}'): {tokenized}")
    print(f"  token IDs: {token_ids_actual}")
    print(f"  Length: {len(token_ids_actual) if isinstance(token_ids_actual, list) else 1}")
    
    # Check if it's in special tokens
    if token in tokenizer.special_tokens_map_extended:
        special_id = tokenizer.special_tokens_map_extended[token]
        print(f"  Special token ID: {special_id}")
    
    print()

# %%
print("=== TESTING CORRECT APPROACH ===")
print()

# Check for actual special token attributes
special_attrs = [attr for attr in dir(tokenizer) if 'token' in attr.lower() and not attr.startswith('_')]
print(f"Tokenizer attributes with 'token': {special_attrs}")
print()

# Test direct special token access
print("Direct special token access:")
special_tokens_to_check = {
    "bos_token_id": getattr(tokenizer, 'bos_token_id', None),
    "eos_token_id": getattr(tokenizer, 'eos_token_id', None),
    "pad_token_id": getattr(tokenizer, 'pad_token_id', None),
}

# Check for Llama-specific tokens
llama_specific = ["eot_id", "start_header_id", "end_header_id"]
for attr in llama_specific:
    if hasattr(tokenizer, attr):
        special_tokens_to_check[attr] = getattr(tokenizer, attr)

for name, token_id in special_tokens_to_check.items():
    if token_id is not None:
        try:
            decoded = tokenizer.decode([token_id])
            print(f"  {name}: {token_id} -> {repr(decoded)}")
        except:
            print(f"  {name}: {token_id} -> (decode failed)")

print()

# Test what our current TemplateManager.get_termination_tokens returns
print("=== CURRENT TEMPLATE MANAGER TERMINATION TOKENS ===")
current_termination = TemplateManager.get_termination_tokens(tokenizer)
print(f"Current termination tokens: {current_termination}")
for token_id in current_termination:
    try:
        decoded = tokenizer.decode([token_id])
        print(f"  {token_id} -> {repr(decoded)}")
    except:
        print(f"  {token_id} -> (decode failed)")

print()

# %%
print("=== TESTING TEMPLATE APPLICATION ===")
print()

# Test a simple conversation
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]

# Use our template manager
result = TemplateManager.format_for_model(
    data=messages, tokenizer=tokenizer, model_type="chat", purpose="training"
)

print("Template manager result:")
print(f"Input IDs: {result['input_ids']}")
print(f"First 10 tokens: {result['input_ids'][:10]}")
print(f"Last 10 tokens: {result['input_ids'][-10:]}")
print()

# Decode and check for special tokens
decoded = tokenizer.decode(result['input_ids'])
print(f"Decoded: {repr(decoded)}")
print()

# Check specific token IDs
token_meanings = {}
for i, token_id in enumerate(result['input_ids'][:10]):
    try:
        decoded_token = tokenizer.decode([token_id])
        token_meanings[i] = f"{token_id} -> {repr(decoded_token)}"
    except:
        token_meanings[i] = f"{token_id} -> (decode failed)"

print("First 10 token breakdown:")
for i, meaning in token_meanings.items():
    print(f"  Position {i}: {meaning}")

# %%
print("=== TESTING DIFFERENT APPROACHES ===")
print()

# Check what parameters apply_chat_template accepts
import inspect
sig = inspect.signature(tokenizer.apply_chat_template)
print(f"apply_chat_template parameters: {list(sig.parameters.keys())}")
print()

# Try to see if the tokenizer has a default_system_message attribute
print(f"Tokenizer attributes with 'system' or 'default': {[attr for attr in dir(tokenizer) if 'system' in attr.lower() or 'default' in attr.lower()]}")
print()

# Check the chat template
print(f"Chat template type: {type(getattr(tokenizer, 'chat_template', None))}")
if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
    print(f"Chat template (first 200 chars): {tokenizer.chat_template[:200]}...")
print()

# Test with explicit system message set to empty
print("Testing with explicit empty system message:")
messages_with_empty_system = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]

try:
    result = tokenizer.apply_chat_template(
        messages_with_empty_system,
        tokenize=True,
        add_generation_prompt=False,
    )
    decoded = tokenizer.decode(result)
    print(f"With empty system message: {repr(decoded)}")
except Exception as e:
    print(f"Error with empty system: {e}")
print()

# Test with no system message at all  
print("Testing with no system message:")
messages_no_system = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]

try:
    result = tokenizer.apply_chat_template(
        messages_no_system,
        tokenize=True,
        add_generation_prompt=False,
    )
    decoded = tokenizer.decode(result)
    print(f"No system message: {repr(decoded)}")
except Exception as e:
    print(f"Error with no system: {e}")
print()

# %%
print("=== NEW UNIFIED INTERFACE ===")
print("One function handles all cases: format_for_model()")
print()

# Case 1: Chat model + Messages + Training
print("1. Chat model + Messages + Training:")
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
]
result = TemplateManager.format_for_model(
    data=messages, tokenizer=tokenizer, model_type="chat", purpose="training"
)
print(f"Result type: {type(result)}")
print(f"Input IDs: {result['input_ids'][:10]}...")
print(f"Decoded: {repr(tokenizer.decode(result['input_ids']))}")
print()

# Case 2: Chat model + Messages + Inference
print("2. Chat model + Messages + Inference:")
user_only = [{"role": "user", "content": "What is 2+2?"}]
result = TemplateManager.format_for_model(
    data=user_only, tokenizer=tokenizer, model_type="chat", purpose="inference"
)
print(f"Result type: {type(result)}")
print(f"Tokens: {result[:10]}...")
print(f"Decoded: {repr(tokenizer.decode(result))}")
print()

# Case 3: Text model + Text + Training
print("3. Text model + Text + Training:")
text_data = "This is some training text."
try:
    result = TemplateManager.format_for_model(
        data=text_data,
        tokenizer=tokenizer,
        model_type="text_generation",
        purpose="training",
    )
    print(f"Result type: {type(result)}")
    print(f"Input IDs: {result['input_ids'][:10]}...")
except Exception as e:
    print(f"Error (expected): {e}")
print()

# Case 4: ERROR - Chat model + Text (should fail)
print("4. ERROR Case - Chat model + Text:")
try:
    result = TemplateManager.format_for_model(
        data="Some text", tokenizer=tokenizer, model_type="chat", purpose="training"
    )
except ValueError as e:
    print(f"✓ Correctly failed: {e}")
print()

# Case 5: ERROR - Text model + Messages (should fail)
print("5. ERROR Case - Text model + Messages:")
try:
    result = TemplateManager.format_for_model(
        data=messages,
        tokenizer=tokenizer,
        model_type="text_generation",
        purpose="training",
    )
except ValueError as e:
    print(f"✓ Correctly failed: {e}")
print()

# %%
