from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Set, Callable, Any
from datasets import Dataset
import json
from .position_finder import Position


@dataclass
class CharacterPositions:
    """Container for character positions of interest."""

    positions: Dict[str, Union[Position, List[Position]]]

    def __getitem__(self, key: str) -> Union[Position, List[Position]]:
        return self.positions[key]

    def keys(self) -> Set[str]:
        return set(self.positions.keys())


@dataclass
class ProbingExample:
    """Single example for probing."""

    text: str
    label: Union[int, float]  # Numeric label
    label_text: str  # Original text label
    character_positions: Optional[CharacterPositions] = None
    group_id: Optional[str] = None  # For cross-validation/grouping
    metadata: Optional[Dict] = None  # Any additional metadata


class ProbingDataset:
    """Base dataset for probing experiments."""

    def __init__(
        self,
        examples: List[ProbingExample],
        task_type: str = "classification",  # or "regression"
        valid_layers: Optional[List[str]] = None,
        label_mapping: Optional[Dict[str, int]] = None,
        metadata: Optional[Dict] = None,
    ):
        self.examples = examples
        self.task_type = task_type
        self.valid_layers = valid_layers or []
        self.label_mapping = label_mapping or {}
        self.metadata = metadata or {}
        self.dataset = self._to_hf_dataset()

    def add_target_positions(
        self, key: str, finder: Callable[[str], Union[Position, List[Position]]]
    ) -> None:
        """Add target positions using a finding strategy.

        Args:
            key: Name for this set of positions
            finder: Function that finds positions in text
        """
        for example in self.examples:
            positions = finder(example.text)

            # Initialize character_positions if needed
            if example.character_positions is None:
                example.character_positions = CharacterPositions({})

            # Add new positions
            example.character_positions.positions[key] = positions

    def _to_hf_dataset(self) -> Dataset:
        """Convert to HuggingFace dataset format."""
        data_dict: Dict[str, List[Optional[Union[str, int, float, List]]]] = {
            "text": [],
            "label": [],
            "label_text": [],
            "group_id": [],
        }

        # Add position data if available
        position_keys = set()
        for ex in self.examples:
            if ex.character_positions:
                position_keys.update(ex.character_positions.keys())

        for key in position_keys:
            data_dict[f"char_pos_{key}_start"] = []
            data_dict[f"char_pos_{key}_end"] = []
            data_dict[f"char_pos_{key}_multi"] = []  # For multiple matches

        # Populate data
        for ex in self.examples:
            data_dict["text"].append(ex.text)
            data_dict["label"].append(ex.label)
            data_dict["label_text"].append(ex.label_text)
            data_dict["group_id"].append(ex.group_id)

            # Add position data
            if ex.character_positions:
                for key in position_keys:
                    pos = ex.character_positions.positions.get(key)
                    if pos is None:
                        data_dict[f"char_pos_{key}_start"].append(None)
                        data_dict[f"char_pos_{key}_end"].append(None)
                        data_dict[f"char_pos_{key}_multi"].append([])
                    elif isinstance(pos, Position):
                        data_dict[f"char_pos_{key}_start"].append(pos.start)
                        data_dict[f"char_pos_{key}_end"].append(pos.end)
                        data_dict[f"char_pos_{key}_multi"].append([])
                    else:  # List[Position]
                        # Store first match in main columns
                        data_dict[f"char_pos_{key}_start"].append(
                            pos[0].start if pos else None
                        )
                        data_dict[f"char_pos_{key}_end"].append(
                            pos[0].end if pos else None
                        )
                        # Store all matches in multi column
                        data_dict[f"char_pos_{key}_multi"].append(
                            [(p.start, p.end) for p in pos]
                        )
            else:
                for key in position_keys:
                    data_dict[f"char_pos_{key}_start"].append(None)
                    data_dict[f"char_pos_{key}_end"].append(None)
                    data_dict[f"char_pos_{key}_multi"].append([])

        return Dataset.from_dict(data_dict)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset: Dataset,
        position_types: List[str],
        label_mapping: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> "ProbingDataset":
        """Create ProbingDataset from a HuggingFace dataset."""
        examples = []

        for item in dataset:  # type: ignore
            # Update position handling to use character positions
            positions = {}
            for pt in position_types:
                # Check for single position
                start = item.get(f"char_pos_{pt}_start")
                end = item.get(f"char_pos_{pt}_end")
                multi = item.get(f"char_pos_{pt}_multi", [])

                if multi:
                    # Handle multiple positions
                    positions[pt] = [Position(start=s, end=e) for s, e in multi]
                elif start is not None and end is not None:
                    # Handle single position
                    positions[pt] = Position(start=start, end=end)

            examples.append(
                ProbingExample(
                    text=str(item["text"]),
                    label=float(item["label"]),
                    label_text=str(item["label_text"]),
                    character_positions=CharacterPositions(positions),
                    group_id=item.get("group_id"),
                )
            )

        return cls(
            examples=examples,
            label_mapping=label_mapping,  # Remove position_types from kwargs
            **kwargs,
        )

    def save(self, path: str) -> None:
        """Save dataset to disk."""
        # Save HF dataset
        self.dataset.save_to_disk(f"{path}/hf_dataset")

        # Get position types from the examples
        position_types = set()
        for example in self.examples:
            if example.character_positions:
                position_types.update(example.character_positions.keys())

        # Save metadata
        metadata = {
            "task_type": self.task_type,
            "valid_layers": self.valid_layers,
            "label_mapping": self.label_mapping,
            "metadata": self.metadata,
            "position_types": list(position_types)  # Add position_types to metadata
        }

        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: str) -> "ProbingDataset":
        """Load dataset from disk."""
        # Load HF dataset
        dataset = Dataset.load_from_disk(f"{path}/hf_dataset")

        # Load metadata
        with open(f"{path}/metadata.json", "r") as f:
            metadata = json.load(f)

        return cls.from_hf_dataset(
            dataset=dataset,
            position_types=metadata["position_types"],
            label_mapping=metadata["label_mapping"],
            task_type=metadata["task_type"],
            valid_layers=metadata["valid_layers"],
            metadata=metadata["metadata"],
        )

    def train_test_split(
        self, test_size: float = 0.2, shuffle: bool = True, seed: Optional[int] = None
    ) -> tuple["ProbingDataset", "ProbingDataset"]:
        """Split into train and test datasets."""
        split = self.dataset.train_test_split(
            test_size=test_size, shuffle=shuffle, seed=seed
        )

        # Remove position_types from both train and test creation
        train = self.from_hf_dataset(
            dataset=split["train"],
            position_types=[
                k.replace("char_pos_", "")
                for k in split["train"].features.keys()
                if k.endswith("_start")
            ],  # Derive position types from columns
            label_mapping=self.label_mapping,
            task_type=self.task_type,
            valid_layers=self.valid_layers,
            metadata=self.metadata,
        )

        test = self.from_hf_dataset(
            dataset=split["test"],
            position_types=[
                k.replace("char_pos_", "")
                for k in split["test"].features.keys()
                if k.endswith("_start")
            ],  # Derive position types from columns
            label_mapping=self.label_mapping,
            task_type=self.task_type,
            valid_layers=self.valid_layers,
            metadata=self.metadata,
        )

        return train, test
