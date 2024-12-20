Here's a structured overview of the repository:

# Probity Repository Structure

## 📁 probity/
### 📁 collection/
- **collectors.py**
  - `@dataclass TransformerLensConfig`
  - `class TransformerLensCollector`
    - Handles collection of activations using TransformerLens
- **activation_store.py**
  - `@dataclass ActivationStore`
    - Stores and manages model activations

### 📁 datasets/
- **base.py**
  - `@dataclass CharacterPositions`
  - `@dataclass ProbingExample`
  - `class ProbingDataset`
    - Base dataset class for probing experiments
- **tokenized.py**
  - `@dataclass TokenizationConfig`
  - `@dataclass TokenPositions`
  - `@dataclass TokenizedProbingExample`
  - `class TokenizedProbingDataset`
    - Extends ProbingDataset with tokenization capabilities
- **templated.py**
  - `@dataclass TemplateVariable`
  - `@dataclass Template`
  - `class TemplatedDataset`
    - Handles template-based dataset generation
- **position_finder.py**
  - `@dataclass Position`
  - `class PositionFinder`
    - Strategies for finding target positions in text

### 📁 utils/
- **dataset_loading.py** (empty)

## 📁 Root Files
- **pyproject.toml** - Project configuration
- **LICENSE** - MIT License
- **.gitignore** - Git ignore rules
- **README.md** - Project documentation
- **testing.ipynb** - Jupyter notebook for testing

The codebase appears to be focused on probing neural networks, with a particular emphasis on:
1. Collecting and storing activations from transformer models
2. Managing datasets with various levels of abstraction (base, tokenized, templated)
3. Handling position tracking in text for analysis purposes

The structure follows a clean hierarchical organization with clear separation of concerns between data management, activation collection, and position tracking functionality.
