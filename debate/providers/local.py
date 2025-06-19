import torch
import time
from typing import List, Dict, Any, Optional
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import gc

from .base import BaseModelProvider, GenerationResult, estimate_tokens
from ..types import ModelConfig


class LocalModelProvider(BaseModelProvider):
    """Provider for local HuggingFace/TransformerLens models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = self._get_model_dtype()
        
        # Load model and tokenizer
        self._load_model()
        
        # Generation defaults
        self.default_gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": None  # Will be set after tokenizer load
        }
        self.default_gen_kwargs.update(config.generation_kwargs or {})
    
    def _get_model_dtype(self) -> torch.dtype:
        """Determine appropriate dtype for model"""
        bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
        if any(m in self.config.model_name.lower() for m in bfloat16_models):
            return torch.bfloat16
        return torch.float32
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading local model {self.config.model_name}...")
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = HookedTransformer.from_pretrained_no_processing(
                self.config.model_name,
                device=self.device,
                dtype=self.model_dtype
            )
            
            # Update default generation kwargs
            self.default_gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
            print(f"Successfully loaded {self.config.model_name}")
            print(f"Model dtype: {self.model_dtype}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> GenerationResult:
        """Generate response from messages"""
        
        if not self.is_available():
            return GenerationResult(
                content="",
                tokens_used=0,
                latency=0.0,
                metadata={},
                success=False,
                error="Model not available"
            )
        
        start_time = time.time()
        
        try:
            # Format messages using chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False  # Chat template already adds
            ).to(self.device)
            
            input_length = inputs["input_ids"].shape[1]
            
            # Prepare generation kwargs
            gen_kwargs = self.default_gen_kwargs.copy()
            gen_kwargs.update(kwargs)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Calculate usage
            output_length = len(new_tokens)
            total_tokens = input_length + output_length
            latency = time.time() - start_time
            
            self._update_usage(total_tokens)
            
            # Clean up memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return GenerationResult(
                content=generated_text.strip(),
                tokens_used=total_tokens,
                latency=latency,
                metadata={
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "model_name": self.config.model_name,
                    "device": str(self.device),
                    "dtype": str(self.model_dtype),
                    "generation_kwargs": gen_kwargs
                },
                success=True
            )
            
        except Exception as e:
            return GenerationResult(
                content="",
                tokens_used=0,
                latency=time.time() - start_time,
                metadata={},
                success=False,
                error=f"Generation failed: {str(e)}"
            )
    
    def generate_with_activations(
        self,
        messages: List[Dict[str, str]],
        hook_points: List[str],
        **kwargs
    ) -> tuple[GenerationResult, Optional[Dict[str, torch.Tensor]]]:
        """
        Generate response and return activations for specified hook points.
        Useful for probe scoring during generation.
        """
        
        if not self.is_available():
            return GenerationResult(
                content="",
                tokens_used=0,
                latency=0.0,
                metadata={},
                success=False,
                error="Model not available"
            ), None
        
        start_time = time.time()
        
        try:
            # Format messages
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # We want to run through the model
            )
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False
            ).to(self.device)
            
            # Run with cache to get activations
            self.model.eval()
            with torch.no_grad():
                logits, cache = self.model.run_with_cache(
                    inputs["input_ids"],
                    names_filter=hook_points,
                    return_cache_object=True
                )
            
            # For generation, we need to extract just the last response
            # This is a simplified version - you might need more sophisticated parsing
            input_text = formatted_text
            
            # Find where assistant response starts
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
            if assistant_marker in input_text:
                # Find the last occurrence (most recent assistant response)
                last_assistant_pos = input_text.rfind(assistant_marker)
                if last_assistant_pos != -1:
                    response_start = last_assistant_pos + len(assistant_marker)
                    generated_text = input_text[response_start:].strip()
                else:
                    generated_text = input_text
            else:
                # Fallback to last portion
                generated_text = input_text.split("assistant")[-1].strip()
            
            # Clean up generated text
            if "<|eot_id|>" in generated_text:
                generated_text = generated_text.split("<|eot_id|>")[0].strip()
            
            # Calculate tokens
            total_tokens = inputs["input_ids"].shape[1]
            output_tokens = estimate_tokens(generated_text)
            
            latency = time.time() - start_time
            self._update_usage(total_tokens)
            
            # Convert cache to CPU to save memory
            cpu_cache = {}
            for hook_point in hook_points:
                if hook_point in cache:
                    cpu_cache[hook_point] = cache[hook_point].cpu()
            
            # Clean up
            del logits, cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return GenerationResult(
                content=generated_text,
                tokens_used=total_tokens,
                latency=latency,
                metadata={
                    "input_tokens": total_tokens - output_tokens,
                    "output_tokens": output_tokens,
                    "model_name": self.config.model_name,
                    "hook_points": hook_points,
                    "has_activations": True
                },
                success=True
            ), cpu_cache
            
        except Exception as e:
            return GenerationResult(
                content="",
                tokens_used=0,
                latency=time.time() - start_time,
                metadata={},
                success=False,
                error=f"Generation with activations failed: {str(e)}"
            ), None
    
    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return (
            self.model is not None and 
            self.tokenizer is not None and
            torch.cuda.is_available() if "cuda" in str(self.device) else True
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_available():
            return {}
        
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(
                self.model.cfg, "n_ctx", "unknown"
            ),
            "hidden_size": getattr(self.model.cfg, "d_model", "unknown"),
            "num_layers": getattr(self.model.cfg, "n_layers", "unknown"),
            "num_attention_heads": getattr(
                self.model.cfg, "n_heads", "unknown"
            )
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        self.clear_cache()
        print(f"Unloaded model {self.config.model_name}")


# Utility functions for local models

def get_available_local_models() -> List[str]:
    """Get list of available local models (placeholder)"""
    # This could scan HuggingFace cache or model directories
    return [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct", 
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-128k-instruct"
    ]


def estimate_memory_usage(model_name: str) -> Dict[str, float]:
    """Estimate memory usage for a model"""
    
    # Rough estimates in GB
    model_sizes = {
        "7b": 14,    # ~14GB for 7B model in bfloat16
        "8b": 16,    # ~16GB for 8B model
        "13b": 26,   # ~26GB for 13B model
        "70b": 140,  # ~140GB for 70B model
    }
    
    model_lower = model_name.lower()
    
    for size_key, memory_gb in model_sizes.items():
        if size_key in model_lower:
            return {
                "estimated_memory_gb": memory_gb,
                "recommended_gpu_memory_gb": memory_gb * 1.2,  # Add overhead
                "can_run_on_single_gpu": memory_gb <= 80
            }
    
    return {
        "estimated_memory_gb": "unknown",
        "recommended_gpu_memory_gb": "unknown", 
        "can_run_on_single_gpu": False
    }