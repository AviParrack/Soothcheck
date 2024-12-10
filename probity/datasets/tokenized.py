from typing import Optional, List, Dict, Set, Union, cast, Any
import dataclasses
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .position_finder import Position
from dataclasses import dataclass, field
import json
import torch
from datasets import Dataset
from .base import ProbingDataset, ProbingExample
from .position_finder import PositionFinder


@dataclass
class TokenizationConfig:
    """Stores information about how the dataset was tokenized."""

    tokenizer_name: str
    tokenizer_kwargs: Dict
    vocab_size: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]


@dataclass
class TokenPositions:
    """Container for token positions of interest."""

    positions: Dict[str, Union[int, List[int]]]

    def __getitem__(self, key: str) -> Union[int, List[int]]:
        return self.positions[key]

    def keys(self) -> Set[str]:
        return set(self.positions.keys())


@dataclass
class TokenizedProbingExample(ProbingExample):
    """Extends ProbingExample with tokenization information."""

    tokens: List[int] = field(default_factory=list)
    attention_mask: Optional[List[int]] = None
    token_positions: Optional[TokenPositions] = None


class TokenizedProbingDataset(ProbingDataset):
    """ProbingDataset with tokenization information and guarantees."""

    def __init__(
        self,
        examples: List[TokenizedProbingExample],
        tokenization_config: TokenizationConfig,
        position_types: Optional[Set[str]] = None,
        **kwargs,
    ):
        # Call parent constructor with the examples
        super().__init__(examples=examples, **kwargs)  # type: ignore
        self.tokenization_config = tokenization_config
        self.position_types = position_types or set()

        # Validate that all examples are properly tokenized
        self._validate_tokenization()

    def _validate_tokenization(self):
        """Ensure all examples have valid tokens."""
        for example in self.examples:
            if not isinstance(example, TokenizedProbingExample):
                raise TypeError(
                    "All examples must be TokenizedProbingExample instances"
                )
            if any(t >= self.tokenization_config.vocab_size for t in example.tokens):
                raise ValueError(f"Invalid token ID found in example: {example.text}")

    def _to_hf_dataset(self) -> Dataset:
        """Override parent method to include tokenization data."""
        # Get base dictionary from parent
        data_dict = super()._to_hf_dataset().to_dict()

        # Add tokenization data
        examples = [cast(TokenizedProbingExample, ex) for ex in self.examples]
        data_dict["tokens"] = [ex.tokens for ex in examples]
        if all(ex.attention_mask is not None for ex in examples):
            data_dict["attention_mask"] = [ex.attention_mask for ex in examples]

        return Dataset.from_dict(data_dict)

    @classmethod
    def from_probing_dataset(
        cls,
        dataset: ProbingDataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        **tokenizer_kwargs,
    ) -> "TokenizedProbingDataset":
        """Create TokenizedProbingDataset from ProbingDataset."""
        tokenized_examples = []

        for example in dataset.examples:
            # Tokenize the text
            tokens = tokenizer(example.text, return_tensors="pt", **tokenizer_kwargs)

            # Convert character positions to token positions if they exist
            token_positions = None
            if example.character_positions:
                positions = {}
                for key, pos in example.character_positions.positions.items():
                    if isinstance(pos, Position):
                        positions[key] = PositionFinder.convert_to_token_position(
                            pos, example.text, tokenizer
                        )
                    else:  # List[Position]
                        positions[key] = [
                            PositionFinder.convert_to_token_position(
                                p, example.text, tokenizer
                            )
                            for p in pos
                        ]
                token_positions = TokenPositions(positions)

            # Create tokenized example
            tokenized_example = TokenizedProbingExample(
                text=example.text,
                label=example.label,
                label_text=example.label_text,
                character_positions=example.character_positions,
                token_positions=token_positions,
                group_id=example.group_id,
                metadata=example.metadata,
                tokens=tokens["input_ids"][0].tolist(),
                attention_mask=(
                    tokens["attention_mask"][0].tolist()
                    if "attention_mask" in tokens
                    else None
                ),
            )
            tokenized_examples.append(tokenized_example)

        # Create tokenization config
        tokenization_config = TokenizationConfig(
            tokenizer_name=tokenizer.name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        return cls(
            examples=tokenized_examples,
            tokenization_config=tokenization_config,
            task_type=dataset.task_type,
            valid_layers=dataset.valid_layers,
            label_mapping=dataset.label_mapping,
            metadata=dataset.metadata,
        )

    def get_token_lengths(self) -> List[int]:
        """Get length of each sequence in tokens."""
        return [len(cast(TokenizedProbingExample, ex).tokens) for ex in self.examples]

    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length in dataset."""
        return max(self.get_token_lengths())

    def validate_positions(self) -> bool:
        """Check if all token positions are valid given the sequence lengths."""
        for example in self.examples:
            tokenized_ex = cast(TokenizedProbingExample, example)
            if tokenized_ex.token_positions is None:
                continue
            seq_len = len(tokenized_ex.tokens)
            for pos in tokenized_ex.token_positions.positions.values():
                if isinstance(pos, int) and pos >= seq_len:
                    return False
                elif isinstance(pos, list) and any(p >= seq_len for p in pos):
                    return False
        return True

    def get_batch_tensors(
        self, indices: List[int], pad: bool = True
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get a batch of examples as tensors.

        Args:
            indices: Which examples to include in batch
            pad: Whether to pad sequences to max length in batch

        Returns:
            Dictionary with keys 'input_ids', 'attention_mask', and 'positions'
        """
        # Get examples and cast them to TokenizedProbingExample
        examples = [cast(TokenizedProbingExample, self.examples[i]) for i in indices]

        # Get max length for this batch
        max_len = max(len(ex.tokens) for ex in examples)

        # Prepare tensors
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        positions: Dict[str, List[Union[int, List[int]]]] = {pt: [] for pt in self.position_types}

        for ex in examples:
            if pad:
                # Pad tokens
                padded_tokens = ex.tokens + [self.tokenization_config.pad_token_id] * (
                    max_len - len(ex.tokens)
                )
                input_ids.append(padded_tokens)

                # Pad attention mask
                if ex.attention_mask is not None:
                    padded_mask = ex.attention_mask + [0] * (
                        max_len - len(ex.attention_mask)
                    )
                    attention_mask.append(padded_mask)
            else:
                input_ids.append(ex.tokens)
                if ex.attention_mask is not None:
                    attention_mask.append(ex.attention_mask)

            # Add positions
            for pt in self.position_types:
                if ex.token_positions is not None:
                    positions[pt].append(ex.token_positions[pt])
                else:
                    positions[pt].append(0)  # or whatever default value makes sense

        # Convert to tensors
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {
            "input_ids": torch.tensor(input_ids),
            "positions": {pt: torch.tensor(positions[pt]) for pt in self.position_types},
        }

        if attention_mask:
            batch["attention_mask"] = torch.tensor(attention_mask)

        return batch

    def save(self, path: str) -> None:
        """Override save to include tokenization info."""
        super().save(path)

        # Save tokenization info
        with open(f"{path}/tokenization_config.json", "w") as f:
            json.dump(dataclasses.asdict(self.tokenization_config), f)

    @classmethod
    def load(cls, path: str) -> "TokenizedProbingDataset":
        """Override load to include tokenization info."""
        # Load base dataset
        base_dataset = ProbingDataset.load(path)

        # Load tokenization info
        with open(f"{path}/tokenization_config.json", "r") as f:
            tokenization_config = TokenizationConfig(**json.load(f))

        # Convert examples to TokenizedProbingExamples
        tokenized_examples = []
        hf_dataset = Dataset.load_from_disk(f"{path}/hf_dataset")

        for base_ex, hf_item in zip(base_dataset.examples, hf_dataset):
            tokenized_examples.append(
                TokenizedProbingExample(
                    text=base_ex.text,
                    label=base_ex.label,
                    label_text=base_ex.label_text,
                    character_positions=base_ex.character_positions,
                    group_id=base_ex.group_id,
                    metadata=base_ex.metadata,
                    tokens=cast(Dict[str, Any], hf_item)["tokens"],
                    attention_mask=cast(Dict[str, Any], hf_item).get("attention_mask"),
                )
            )

        return cls(
            examples=tokenized_examples,
            tokenization_config=tokenization_config,
            task_type=base_dataset.task_type,
            valid_layers=base_dataset.valid_layers,
            label_mapping=base_dataset.label_mapping,
            position_types=base_dataset.position_types,
            metadata=base_dataset.metadata,
        )
