import torch
import json
import os
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions, Position
from probity.datasets.tokenized import TokenizedProbingDataset, TokenizedProbingExample, TokenPositions
from transformers import AutoTokenizer

def load_lie_truth_dataset(json_path: str, tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> TokenizedProbingDataset:
    """Load the lie/truth dataset from JSON format"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]
    
    def find_lie_statement_character_positions(formatted_text, lie_statement):
        start_pos = formatted_text.find(lie_statement)
        if start_pos == -1:
            return None
        end_pos = start_pos + len(lie_statement)
        return Position(start=start_pos, end=end_pos)
    
    probing_examples = []
    
    for item in data:
        lie_statement = item["lie_statement"]
        
        # Truth version
        truth_messages = create_chat_messages(
            item["truth_version"]["system"],
            USER_PROMPT,
            item["truth_version"]["model"]
        )
        
        # Lie version
        lie_messages = create_chat_messages(
            item["lie_version"]["system"],
            USER_PROMPT,
            item["lie_version"]["model"]
        )
        
        # Get formatted text
        truth_conv = tokenizer.apply_chat_template(
            truth_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        lie_conv = tokenizer.apply_chat_template(
            lie_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Find positions
        truth_lie_pos = find_lie_statement_character_positions(truth_conv, lie_statement)
        lie_lie_pos = find_lie_statement_character_positions(lie_conv, lie_statement)
        
        if truth_lie_pos is None or lie_lie_pos is None:
            continue
        
        # Create examples
        truth_example = ProbingExample(
            text=truth_conv,
            label=0,  # Truth
            label_text="truth",
            character_positions=CharacterPositions({"LIE_SPAN": truth_lie_pos})
        )
        
        lie_example = ProbingExample(
            text=lie_conv,
            label=1,  # Lie
            label_text="lie",
            character_positions=CharacterPositions({"LIE_SPAN": lie_lie_pos})
        )
        
        probing_examples.extend([truth_example, lie_example])
    
    # Create probing dataset
    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={"description": "Truth vs Lie classification in Llama chat format"}
    )
    
    # Convert to tokenized dataset
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=False  # Chat template already adds special tokens
    )
    
    return tokenized_dataset

def get_model_dtype(model_name: str) -> torch.dtype:
    """Determine the appropriate dtype for the model"""
    # Models that typically use bfloat16
    bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
    
    if any(m in model_name.lower() for m in bfloat16_models):
        return torch.bfloat16
    return torch.float32