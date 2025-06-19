# This file provides a simple chat session manager for maintaining conversation history.
# It handles:
# - Efficient KV cache management using HuggingFace iterative generation pattern
# - Adding user and assistant messages to conversation history
# - Formatting conversations using proper chat templates
# - Reusing persistent cache across multiple turns
#
# Key functions:
# - ChatManager.add_message: Add messages to history
# - ChatManager.generate_response: Generate response using the persistent KV cache
# - ChatManager.clear_history: Reset the conversation and cache

from typing import List, Dict, Optional
from transformers import PreTrainedTokenizer, PreTrainedModel, TextStreamer
from transformers.cache_utils import DynamicCache
import torch


class ChatManager:
    """Efficient conversation history manager using persistent KV cache."""
    
    def __init__(self, max_history_turns: Optional[int] = None, max_sequence_length: int = 4096):
        """
        Initialize chat manager.
        
        Args:
            max_history_turns: Maximum number of conversation turns to keep.
                              If None, keeps all history.
            max_sequence_length: Maximum sequence length for KV cache (default 4096 tokens).
        """
        self.messages: List[Dict[str, str]] = []
        self.max_history_turns = max_history_turns
        self.max_sequence_length = max_sequence_length
        self.kv_cache = DynamicCache()  # Persistent cache across turns
        
    def generate_response(
        self,
        user_input: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        prompt_name: Optional[str] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate response efficiently using persistent KV cache.
        Follows HuggingFace iterative generation pattern CORRECTLY.
        """
        from ..utils.template_manager import TemplateManager
        
        # Add the user message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        
        # Add system prompt if this is the first message and prompt_name is provided
        if len(self.messages) == 1 and prompt_name:
            from .inference_utils import _load_system_prompt_content
            system_prompt_content = _load_system_prompt_content(prompt_name)
            if system_prompt_content:
                # Insert system message at the beginning
                self.messages.insert(0, {"role": "system", "content": system_prompt_content})
        
        # CRITICAL: Format the ENTIRE conversation using chat template
        # This is the CORRECT way according to HF docs for iterative generation
        inputs = tokenizer.apply_chat_template(
            self.messages, 
            add_generation_prompt=True, 
            return_tensors="pt", 
            return_dict=True
        ).to(model.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Check if sequence is getting too long and trim if needed
        if input_length > self.max_sequence_length:
            print(f"Warning: Sequence length ({input_length}) exceeds max ({self.max_sequence_length}). Trimming history.")
            self._trim_history()
            self.kv_cache = DynamicCache()  # Reset cache after trimming
            # Re-format after trimming
            inputs = tokenizer.apply_chat_template(
                self.messages, 
                add_generation_prompt=True, 
                return_tensors="pt", 
                return_dict=True
            ).to(model.device)
            input_length = inputs["input_ids"].shape[1]
        
        # Set up generation parameters
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        
        termination_tokens = TemplateManager.get_termination_tokens(tokenizer)
        
        # Create streamer for real-time display
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate with KV cache reuse - CORRECT HF pattern
        with torch.no_grad():
            try:
                # Need return_dict_in_generate=True to get past_key_values back
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=termination_tokens,
                    past_key_values=self.kv_cache,  # Reuse persistent cache
                    use_cache=True,
                    cache_implementation=None,  # Explicitly disable default cache
                    streamer=streamer,
                    return_dict_in_generate=True,  # CRITICAL: Need this to get cache back
                )
                
                # The cache object is updated in-place, so we don't need to reassign it.
                # The 'past_key_values' output is the same object we passed in.
                
                # Get the generated sequence
                generated_tokens = outputs.sequences
                
            except Exception as e:
                print(f"Error during generation with cache: {e}")
                print("Falling back to generation without cache...")
                # Reset cache and try without it
                self.kv_cache = DynamicCache()
                
                generated_tokens = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=pad_token_id,
                    eos_token_id=termination_tokens,
                    use_cache=True,
                    streamer=streamer,
                )
        
        # Extract just the new response (everything after the input)
        response_token_ids = generated_tokens[0][input_length:]

        # Decode response WITH special tokens for history to maintain cache consistency.
        response_for_history = tokenizer.decode(
            response_token_ids,
            skip_special_tokens=False
        ).strip()
        
        # Add response to conversation history
        self.messages.append({"role": "assistant", "content": response_for_history})
        self._trim_history()
        
        # Decode response WITHOUT special tokens for a clean return value.
        # The live streaming is already clean thanks to TextStreamer settings.
        cleaned_response = tokenizer.decode(
            response_token_ids,
            skip_special_tokens=True
        ).strip()
        
        return cleaned_response
    
    def clear_history(self) -> None:
        """Clear all conversation history and reset KV cache."""
        self.messages.clear()
        self.kv_cache = DynamicCache()  # Reset the cache
    
    def get_history_summary(self) -> str:
        """Get a simple summary of conversation state."""
        return f"Conversation history: {len(self.messages)} messages"
    
    def _trim_history(self) -> None:
        """Trim history to max_history_turns if specified."""
        if self.max_history_turns is not None and len(self.messages) > self.max_history_turns:
            # Keep system message if it exists
            keep_messages = []
            if self.messages and self.messages[0]["role"] == "system":
                keep_messages.append(self.messages[0])
            
            # Keep recent messages
            recent_start = len(self.messages) - self.max_history_turns
            if recent_start > 0:
                keep_messages.extend(self.messages[recent_start:])
                self.messages = keep_messages
                # Reset cache when trimming for simplicity
                self.kv_cache = DynamicCache() 