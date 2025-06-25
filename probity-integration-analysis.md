# Probity Integration Analysis for NTML Datasets

## Executive Summary

After analyzing the Probity library structure, I've identified the most efficient path to implement statement-level probing for NTML (N Truths, M Lies) conversational datasets. The library's architecture is well-designed for extension, and we can achieve our goals with **minimal modifications** to existing code.

## Key Probity Architecture Insights

### 1. Dataset Layer (`probity/datasets/`)

**Core Classes:**
- `ProbingExample`: Single example with text, label, character positions, and attributes
- `ProbingDataset`: Collection of examples with HuggingFace dataset conversion
- `TokenizedProbingExample`: Extends with tokenization info (tokens, attention_mask, token_positions)
- `TokenizedProbingDataset`: Tokenized version with character→token position mapping

**Key Features:**
- **Position Tracking**: Character positions → Token positions via `PositionFinder`
- **Flexible Labels**: Supports single labels (int/float) and label text
- **Attributes System**: Extensible metadata via `attributes` field
- **HF Integration**: Seamless conversion to HuggingFace Dataset format

### 2. Probe Layer (`probity/probes/`)

**Core Classes:**
- `BaseProbe`: Abstract base with direction extraction and encoding
- `LogisticProbe`: Binary classification (perfect for true/false statements)
- `MultiClassLogisticProbe`: Multi-class classification
- `LinearProbe`: Regression-style probing

**Key Features:**
- **Direction Extraction**: `get_direction()` for interpretability
- **Encoding**: `encode(acts)` computes dot products with probe direction
- **Device Management**: Automatic GPU/CPU handling
- **Serialization**: Save/load with JSON export for analysis

### 3. Training Layer (`probity/training/`)

**Core Classes:**
- `SupervisedProbeTrainer`: Handles train/val splits, early stopping, class imbalance
- `BaseTrainerConfig`: Learning rate, epochs, batch size, device management

**Key Features:**
- **Activation Standardization**: Optional feature normalization during training
- **Class Imbalance**: Automatic positive weight calculation for imbalanced datasets
- **LR Scheduling**: Exponential decay from start_lr to end_lr
- **Progress Tracking**: Built-in tqdm progress bars

### 4. Pipeline Layer (`probity/pipeline/`)

**Core Classes:**
- `ProbePipeline`: End-to-end orchestration (activation collection → training)
- `ProbePipelineConfig`: Unified configuration for entire pipeline

**Key Features:**
- **Activation Caching**: Smart caching with compatibility validation
- **TransformerLens Integration**: Built-in activation collection
- **Device Synchronization**: Ensures consistent device usage across components

## Optimal Integration Strategy

### Option 1: Extend Existing Classes (RECOMMENDED)

**Why This Approach:**
- **Minimal Code Duplication**: Reuse 90% of existing functionality
- **Full Compatibility**: Works with existing pipeline and training infrastructure
- **Easy Maintenance**: Changes are isolated and well-defined

**Implementation Plan:**

#### 1. Create `ConversationalProbingExample` (extends `ProbingExample`)
```python
@dataclass
class ConversationalProbingExample(ProbingExample):
    """NTML conversational example with statement-level labels."""
    
    # Core conversation structure
    system_prompt: str
    assistant_response: str
    
    # Statement-level information
    statement_labels: List[bool]  # [True, False, True, ...]
    statement_texts: List[str]    # Individual statement texts
    statement_positions: List[Tuple[int, int]]  # (char_start, char_end)
    
    # Enhanced metadata
    conversation_metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Set the main text as the full conversation
        self.text = f"System: {self.system_prompt}\nAssistant: {self.assistant_response}"
        
        # Set overall label (could be ratio-based or majority vote)
        self.label = sum(self.statement_labels) / len(self.statement_labels)
        self.label_text = f"{sum(self.statement_labels)}T{len(self.statement_labels)-sum(self.statement_labels)}L"
        
        # Add statement positions to character_positions
        positions = {}
        for i, (start, end) in enumerate(self.statement_positions):
            positions[f"statement_{i}"] = Position(start, end)
        self.character_positions = CharacterPositions(positions)
        
        # Store metadata
        self.attributes = {
            "ratio": self.conversation_metadata.get("ratio"),
            "truth_count": sum(self.statement_labels),
            "lie_count": len(self.statement_labels) - sum(self.statement_labels),
            "lie_positions": self.conversation_metadata.get("lie_positions", []),
            "statement_count": len(self.statement_labels),
            **self.conversation_metadata
        }
```

#### 2. Create `ConversationalProbingDataset` (extends `ProbingDataset`)
```python
class ConversationalProbingDataset(ProbingDataset):
    """NTML dataset with conversational structure."""
    
    @classmethod
    def from_ntml_jsonl(cls, jsonl_path: str) -> "ConversationalProbingDataset":
        """Load NTML dataset from JSONL file."""
        examples = []
        
        with open(jsonl_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                
                # Extract conversation components
                system_msg = next(m for m in data["messages"] if m["role"] == "system")
                assistant_msg = next(m for m in data["messages"] if m["role"] == "assistant")
                
                # Extract statement information
                statements = data["labels"]["statement_level"]
                statement_texts = [s["text"] for s in statements]
                statement_labels = [s["is_true"] for s in statements]
                
                # Calculate positions in assistant response
                statement_positions = []
                for stmt in statements:
                    # Add offset for "System: ...\nAssistant: " prefix
                    prefix_len = len(f"System: {system_msg['content']}\nAssistant: ")
                    start = prefix_len + stmt["char_start"]
                    end = prefix_len + stmt["char_end"]
                    statement_positions.append((start, end))
                
                example = ConversationalProbingExample(
                    system_prompt=system_msg["content"],
                    assistant_response=assistant_msg["content"],
                    statement_labels=statement_labels,
                    statement_texts=statement_texts,
                    statement_positions=statement_positions,
                    conversation_metadata=data["labels"],
                    group_id=data["id"]
                )
                examples.append(example)
        
        return cls(
            examples=examples,
            task_type="classification",
            label_mapping={"false": 0, "true": 1},
            dataset_attributes={"source": "ntml_jsonl", "path": jsonl_path}
        )
    
    def get_statement_dataset(self, statement_idx: Optional[int] = None) -> "ProbingDataset":
        """Create a dataset for probing specific statements."""
        if statement_idx is not None:
            # Single statement dataset
            statement_examples = []
            for ex in self.examples:
                if isinstance(ex, ConversationalProbingExample):
                    if statement_idx < len(ex.statement_labels):
                        # Create new example for this statement
                        stmt_example = ProbingExample(
                            text=ex.text,  # Full conversation text
                            label=int(ex.statement_labels[statement_idx]),
                            label_text=str(ex.statement_labels[statement_idx]),
                            character_positions=CharacterPositions({
                                "target": Position(*ex.statement_positions[statement_idx])
                            }),
                            group_id=ex.group_id,
                            attributes={
                                **ex.attributes,
                                "statement_idx": statement_idx,
                                "statement_text": ex.statement_texts[statement_idx]
                            }
                        )
                        statement_examples.append(stmt_example)
            
            return ProbingDataset(
                examples=statement_examples,
                task_type="classification",
                label_mapping={"false": 0, "true": 1},
                dataset_attributes={
                    **self.dataset_attributes,
                    "statement_idx": statement_idx
                }
            )
        else:
            # Multi-statement dataset (all statements as separate examples)
            all_statement_examples = []
            for ex in self.examples:
                if isinstance(ex, ConversationalProbingExample):
                    for i, (label, text, pos) in enumerate(zip(
                        ex.statement_labels, ex.statement_texts, ex.statement_positions
                    )):
                        stmt_example = ProbingExample(
                            text=ex.text,
                            label=int(label),
                            label_text=str(label),
                            character_positions=CharacterPositions({
                                "target": Position(*pos)
                            }),
                            group_id=f"{ex.group_id}_stmt_{i}",
                            attributes={
                                **ex.attributes,
                                "statement_idx": i,
                                "statement_text": text
                            }
                        )
                        all_statement_examples.append(stmt_example)
            
            return ProbingDataset(
                examples=all_statement_examples,
                task_type="classification",
                label_mapping={"false": 0, "true": 1},
                dataset_attributes={
                    **self.dataset_attributes,
                    "all_statements": True
                }
            )
```

#### 3. Training Integration (NO CHANGES NEEDED!)
The existing `SupervisedProbeTrainer` and `ProbePipeline` will work seamlessly:

```python
# Load NTML dataset
ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl("data/NTML-datasets/2T1L_500samples.jsonl")

# Get statement-level dataset (all statements as separate examples)
statement_dataset = ntml_dataset.get_statement_dataset()

# Tokenize for training
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
    dataset=statement_dataset,
    tokenizer=tokenizer,
    padding=True,
    max_length=512,
    add_special_tokens=True
)

# Configure probe and training (EXISTING PROBITY CODE!)
probe_config = LogisticProbeConfig(
    input_size=768,
    normalize_weights=True,
    bias=False,
    model_name="gpt2",
    hook_point="blocks.7.hook_resid_pre",
    hook_layer=7,
    name="ntml_statement_probe"
)

trainer_config = SupervisedTrainerConfig(
    batch_size=32,
    learning_rate=1e-3,
    num_epochs=20,
    weight_decay=0.01,
    train_ratio=0.8,
    handle_class_imbalance=True,
    show_progress=True
)

pipeline_config = ProbePipelineConfig(
    dataset=tokenized_dataset,
    probe_cls=LogisticProbe,
    probe_config=probe_config,
    trainer_cls=SupervisedProbeTrainer,
    trainer_config=trainer_config,
    position_key="target",  # Probe at statement positions
    model_name="gpt2",
    hook_points=["blocks.7.hook_resid_pre"],
    cache_dir="./cache/ntml_cache"
)

# Train (EXISTING PROBITY PIPELINE!)
pipeline = ProbePipeline(pipeline_config)
probe, history = pipeline.run()
```

## Implementation Benefits

### 1. **Minimal Code Changes**
- Only need to create 2 new classes that extend existing ones
- Zero changes to existing Probity code
- Full backward compatibility

### 2. **Leverages All Existing Features**
- ✅ Automatic activation collection and caching
- ✅ Character → token position mapping
- ✅ Train/validation splits with early stopping
- ✅ Class imbalance handling
- ✅ Device management (GPU/CPU)
- ✅ Progress tracking and logging
- ✅ Probe serialization and analysis

### 3. **Enhanced NTML Features**
- ✅ Statement-level position tracking
- ✅ Conversation metadata preservation
- ✅ Flexible probing (single statement vs all statements)
- ✅ Enhanced system prompts with lie positions
- ✅ Rich attribution tracking

### 4. **Research Flexibility**
- Can probe individual statement positions
- Can probe all statements as separate examples
- Can analyze position bias across conversations
- Can compare performance across different N:M ratios
- Can leverage existing probe analysis tools

## File Structure

```
probity/
├── probity-README.md                    ✅ CREATED
├── probity-integration-analysis.md     ✅ CREATED  
├── probity_extensions/                  🎯 TO CREATE
│   ├── __init__.py
│   ├── conversational.py               # ConversationalProbingExample/Dataset
│   └── analysis.py                     # NTML-specific analysis tools
└── scripts/
    └── train_ntml_probes.py            # Training script using extensions
```

## Next Steps

1. **Create `probity_extensions/conversational.py`** with the classes outlined above
2. **Test JSONL loading** with our generated NTML datasets  
3. **Validate position mapping** from character to token positions
4. **Run end-to-end training** using existing Probity pipeline
5. **Create analysis tools** for NTML-specific metrics

This approach gives us a **production-ready statement-level probing system** that fully leverages Probity's mature infrastructure while adding minimal complexity. 