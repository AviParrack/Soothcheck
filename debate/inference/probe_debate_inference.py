import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import time

from probity.probes import BaseProbe
from ..types import ProbeScore, ConversationTurn
from ..config import ProbeInferenceConfig


class ProbeDebateInference:
    """Real-time probe inference for debate conversations"""
    
    def __init__(self, config: ProbeInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model once for reuse
        print(f"Loading model {config.model_name} for probe inference...")
        self.model_dtype = self._get_model_dtype(config.model_name)
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            config.model_name,
            device=config.device,
            dtype=self.model_dtype
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load probes
        self.probes = self._load_probes()
        
        # Cache for conversation activations
        self._conversation_cache = {}
        self._last_conversation_length = 0
        
        print(f"Loaded {len(self.probes)} probes for layer {config.layer}")
    
    def _get_model_dtype(self, model_name: str) -> torch.dtype:
        """Determine model dtype"""
        bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
        if any(m in model_name.lower() for m in bfloat16_models):
            return torch.bfloat16
        return torch.float32
    
    def _load_probes(self) -> Dict[str, BaseProbe]:
        """Load all specified probes for the layer"""
        probes = {}
        probe_dir = Path(self.config.probe_dir)
        
        for probe_type in self.config.probe_types:
            probe_path = probe_dir / probe_type / f"layer_{self.config.layer}_probe.json"
            
            if probe_path.exists():
                try:
                    probe = BaseProbe.load_json(str(probe_path), device=self.config.device)
                    probes[probe_type] = probe
                    print(f"Loaded {probe_type} probe from {probe_path}")
                except Exception as e:
                    print(f"Failed to load {probe_type} probe: {e}")
            else:
                print(f"Probe not found: {probe_path}")
        
        return probes
    
    def score_new_response(
        self, 
        conversation_history: List[ConversationTurn],
        new_response: str,
        speaker: str
    ) -> List[ProbeScore]:
        """
        Score a new response in the context of conversation history.
        Returns probe scores for just the new response tokens.
        """
        # Convert conversation to text format
        full_conversation = self._format_conversation(conversation_history, new_response)
        
        # Get activations for the full conversation
        activations, tokens = self._get_conversation_activations(full_conversation)
        
        # Find where the new response starts in the token sequence
        new_response_start_idx = self._find_new_response_start(
            conversation_history, tokens, new_response
        )
        
        if new_response_start_idx is None:
            print("Warning: Could not locate new response in token sequence")
            return []
        
        # Extract tokens and activations for new response only
        new_response_tokens = tokens[new_response_start_idx:]
        new_response_activations = activations[new_response_start_idx:]
        
        # Score with all probes
        probe_scores = []
        
        with torch.no_grad():
            for probe_type, probe in self.probes.items():
                try:
                    # Ensure dtype compatibility
                    if new_response_activations.dtype != probe.dtype:
                        new_response_activations = new_response_activations.to(dtype=probe.dtype)
                    
                    # Apply probe
                    token_scores = probe(new_response_activations)
                    
                    # Apply sigmoid for LogisticProbe
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)
                    
                    # Convert to list
                    scores_list = token_scores.cpu().squeeze().tolist()
                    if isinstance(scores_list, float):
                        scores_list = [scores_list]
                    
                    # Normalize non-logistic probes to [0,1]
                    if probe.__class__.__name__ != 'LogisticProbe':
                        scores_list = self._normalize_scores(scores_list)
                    
                    probe_score = ProbeScore(
                        probe_type=probe_type,
                        layer=self.config.layer,
                        tokens=new_response_tokens,
                        token_scores=scores_list,
                        mean_score=float(np.mean(scores_list)),
                        metadata={
                            "timestamp": time.time(),
                            "speaker": speaker,
                            "probe_model": self.config.model_name,
                            "num_tokens": len(new_response_tokens)
                        }
                    )
                    
                    probe_scores.append(probe_score)
                    
                except Exception as e:
                    print(f"Error scoring with {probe_type} probe: {e}")
        
        return probe_scores
    
    def _format_conversation(
        self, 
        history: List[ConversationTurn], 
        new_response: str
    ) -> str:
        """Format conversation into chat template format"""
        messages = []
        
        # Add conversation history
        for turn in history:
            if turn.speaker == "system":
                messages.append({"role": "system", "content": turn.content})
            elif turn.speaker in ["debater1", "debater2"]:
                messages.append({"role": "user", "content": turn.content})
            elif turn.speaker == "judge":
                messages.append({"role": "assistant", "content": turn.content})
        
        # Add new response
        messages.append({"role": "assistant", "content": new_response})
        
        # Apply chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return formatted
    
    def _get_conversation_activations(self, conversation_text: str) -> Tuple[torch.Tensor, List[str]]:
        """Get activations for the conversation, using cache when possible"""
        
        # Check if we can reuse cached activations
        if (conversation_text.startswith(self._conversation_cache.get("text", "")) and
            len(conversation_text) > len(self._conversation_cache.get("text", ""))):
            
            # We can extend the existing cache
            cached_length = len(self._conversation_cache.get("tokens", []))
            return self._extend_conversation_cache(conversation_text, cached_length)
        
        else:
            # Need to compute from scratch
            return self._compute_full_conversation_activations(conversation_text)
    
    def _compute_full_conversation_activations(self, text: str) -> Tuple[torch.Tensor, List[str]]:
        """Compute activations for full conversation"""
        
        # Tokenize
        tokens_dict = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,  # Chat template already adds
            padding=False,
            truncation=True,
            max_length=self.config.max_sequence_length
        ).to(self.device)
        
        # Get hook point
        hook_point = f"blocks.{self.config.layer}.hook_resid_pre"
        
        # Run model with caching
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens_dict["input_ids"],
                names_filter=[hook_point],
                return_cache_object=True,
                stop_at_layer=self.config.layer + 1
            )
        
        # Extract activations and tokens
        activations = cache[hook_point].squeeze(0)  # Remove batch dimension
        token_ids = tokens_dict["input_ids"].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        # Update cache
        self._conversation_cache = {
            "text": text,
            "activations": activations.cpu(),
            "tokens": tokens,
            "token_ids": token_ids.cpu()
        }
        
        return activations, tokens
    
    def _extend_conversation_cache(
        self, 
        new_text: str, 
        cached_length: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extend existing conversation cache with new tokens"""
        
        # For now, recompute everything (can optimize later with incremental caching)
        return self._compute_full_conversation_activations(new_text)
    
    def _find_new_response_start(
        self,
        history: List[ConversationTurn],
        all_tokens: List[str],
        new_response: str
    ) -> Optional[int]:
        """Find where the new response starts in the token sequence"""
        
        # Tokenize just the new response to get target tokens
        new_response_tokens = self.tokenizer.tokenize(new_response)
        
        if not new_response_tokens:
            return None
        
        # Look for the start of new response tokens in the full sequence
        # Start from the end and work backwards
        for i in range(len(all_tokens) - len(new_response_tokens), -1, -1):
            if all_tokens[i:i+len(new_response_tokens)] == new_response_tokens:
                return i
        
        # Fallback: try to find assistant response markers
        assistant_markers = ["<|start_header_id|>assistant<|end_header_id|>", "assistant<|end_header_id|>"]
        
        for marker in assistant_markers:
            marker_tokens = self.tokenizer.tokenize(marker)
            for i in range(len(all_tokens) - len(marker_tokens) + 1):
                if all_tokens[i:i+len(marker_tokens)] == marker_tokens:
                    # Return position after the marker
                    return i + len(marker_tokens)
        
        # Final fallback: return last portion of tokens
        if len(new_response_tokens) > 0:
            return max(0, len(all_tokens) - len(new_response_tokens))
        
        return None
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        if not scores:
            return scores
        
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def clear_cache(self):
        """Clear conversation cache"""
        self._conversation_cache = {}
        self._last_conversation_length = 0
    
    def save_scores(self, probe_scores: List[ProbeScore], save_path: str):
        """Save probe scores to JSON"""
        scores_data = []
        
        for score in probe_scores:
            scores_data.append({
                "probe_type": score.probe_type,
                "layer": score.layer,
                "tokens": score.tokens,
                "token_scores": score.token_scores,
                "mean_score": score.mean_score,
                "metadata": score.metadata
            })
        
        with open(save_path, 'w') as f:
            json.dump(scores_data, f, indent=2)


def main():
    """CLI interface for probe debate inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time probe inference for debates')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--probe_types', nargs='+', default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--conversation_json', type=str, required=True,
                       help='JSON file with conversation history and new response')
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_all_probes_in_dir', action='store_true',
                       help='Test all available probes in directory')
    
    args = parser.parse_args()
    
    # Load conversation data
    with open(args.conversation_json, 'r') as f:
        conv_data = json.load(f)
    
    history = [ConversationTurn(**turn) for turn in conv_data.get('history', [])]
    new_response = conv_data['new_response']
    speaker = conv_data['speaker']
    
    # Create config
    if args.test_all_probes_in_dir:
        # Find all available probes
        probe_dir = Path(args.probe_dir)
        available_types = []
        for probe_type_dir in probe_dir.iterdir():
            if probe_type_dir.is_dir():
                probe_file = probe_type_dir / f"layer_{args.layer}_probe.json"
                if probe_file.exists():
                    available_types.append(probe_type_dir.name)
        probe_types = available_types
    else:
        probe_types = args.probe_types
    
    config = ProbeInferenceConfig(
        model_name=args.model_name,
        probe_dir=args.probe_dir,
        probe_types=probe_types,
        layer=args.layer,
        device=args.device
    )
    
    # Initialize inference
    inferencer = ProbeDebateInference(config)
    
    # Score the new response
    probe_scores = inferencer.score_new_response(history, new_response, speaker)
    
    # Save results
    save_dir = Path(args.results_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = save_dir / f"probe_scores_{timestamp}.json"
    
    inferencer.save_scores(probe_scores, str(save_path))
    
    print(f"Scored {len(probe_scores)} probe types")
    print(f"Results saved to {save_path}")
    
    # Print summary
    for score in probe_scores:
        print(f"{score.probe_type}: mean={score.mean_score:.3f}, tokens={len(score.tokens)}")


if __name__ == "__main__":
    main()