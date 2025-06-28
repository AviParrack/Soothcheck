import torch
from dataclasses import dataclass
from typing import List, Dict, Optional
from transformer_lens import HookedTransformer
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.collection.activation_store import ActivationStore
from probity.utils.multigpu import MultiGPUConfig, wrap_model_for_multigpu


@dataclass
class TransformerLensConfig:
    """Configuration for TransformerLensCollector."""

    model_name: str
    hook_points: List[str]  # e.g. ["blocks.12.hook_resid_post"]
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Memory optimization parameters
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    low_cpu_mem_usage: bool = True
    device_map: str = "auto"
    multi_gpu: Optional[MultiGPUConfig] = None


class TransformerLensCollector:
    """Collects activations using TransformerLens."""

    def __init__(self, config: TransformerLensConfig):
        self.config = config
        print(f"Initializing collector with device: {config.device}")
        
        # Load model with quantization support if needed
        if config.load_in_8bit or config.load_in_4bit:
            print(f"⚠️  Quantization requested: 8bit={config.load_in_8bit}, 4bit={config.load_in_4bit}")
            print("   Loading via HuggingFace with proper quantization config...")
            
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
            import torch
            
            # Determine dtype
            bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
            dtype = torch.bfloat16 if any(m in config.model_name.lower() for m in bfloat16_models) else torch.float32
            
            # Create quantization config
            if config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",  # Use NF4 (normalized float 4) for better performance
                    bnb_4bit_use_double_quant=True,  # Double quantization saves additional memory
                    bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computation
                )
                print("   Using 4-bit NF4 quantization with double quantization")
            elif config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                print("   Using 8-bit quantization")
            else:
                quantization_config = None
            
            # Load with quantization
            hf_model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                quantization_config=quantization_config,
                device_map=config.device_map,
                torch_dtype=dtype,
                low_cpu_mem_usage=config.low_cpu_mem_usage,
                trust_remote_code=True
            )
            
            # Convert to HookedTransformer
            self.model = HookedTransformer.from_pretrained(
                config.model_name,
                hf_model=hf_model,
                device=None,  # Device already set by quantization
                dtype=dtype,
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            print(f"✅ Model loaded with quantization: 8bit={config.load_in_8bit}, 4bit={config.load_in_4bit}")
        else:
            # Standard loading
            self.model = HookedTransformer.from_pretrained_no_processing(config.model_name)
            print(f"Moving model to device: {config.device}")
            self.model.to(config.device)
        
        # --- Multi-GPU support ---
        if config.multi_gpu and config.multi_gpu.enabled:
            self.model = wrap_model_for_multigpu(self.model, config.multi_gpu)
            print(f"✅ Model wrapped for multi-GPU: {config.multi_gpu.backend}")

    @staticmethod
    def get_layer_from_hook_point(hook_point: str) -> int:
        """Extract layer number from hook point string.
        
        Args:
            hook_point: Hook point string (e.g. "blocks.12.hook_resid_post")
            
        Returns:
            Layer number
        """
        try:
            # Extract number after "blocks."
            layer = int(hook_point.split(".")[1])
            return layer
        except (IndexError, ValueError):
            raise ValueError(f"Could not extract layer from hook point: {hook_point}")

    def collect(
        self,
        dataset: TokenizedProbingDataset,
    ) -> Dict[str, ActivationStore]:
        """Collect activations for each hook point.

        Returns:
            Dictionary mapping hook points to ActivationCache objects
        """
        all_activations = {}

        # Set model to evaluation mode
        self.model.eval()

        # Get maximum layer needed
        max_layer = max(
            self.get_layer_from_hook_point(hook)
            for hook in self.config.hook_points
        )

        # Process in batches
        with torch.no_grad():  # Disable gradient computation for determinism
            for batch_start in range(0, len(dataset.examples), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(dataset.examples))
                batch_indices = list(range(batch_start, batch_end))

                # Get batch tensors
                batch = dataset.get_batch_tensors(batch_indices)

                # Run model with caching
                _, cache = self.model.run_with_cache(
                    batch["input_ids"].to(self.config.device),
                    names_filter=self.config.hook_points,
                    return_cache_object=True,
                    stop_at_layer=max_layer + 1
                )

                # Store activations for each hook point
                for hook in self.config.hook_points:
                    if hook not in all_activations:
                        all_activations[hook] = []
                    all_activations[hook].append(cache[hook].cpu())

        # Create ActivationCache objects
        return {
            hook: ActivationStore(
                raw_activations=torch.cat(activations, dim=0),
                hook_point=hook,
                example_indices=torch.arange(len(dataset.examples)),
                sequence_lengths=torch.tensor(dataset.get_token_lengths()),
                hidden_size=activations[0].shape[-1],
                dataset=dataset,
                labels=torch.tensor([ex.label for ex in dataset.examples]),
                label_texts=[ex.label_text for ex in dataset.examples],
            )
            for hook, activations in all_activations.items()
        }
