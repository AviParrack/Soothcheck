# This file provides unified model resolution for all training and inference components.
# It implements the two core ways of specifying models:
# 1. Via config name (from configs/model/*.json)
# 2. Via experiment directory path (from experiments/*)
#
# Functions:
# - resolve_model_specification: Main function that resolves model strings to paths and configs

from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from ..config_models import ModelConfig


@dataclass
class ModelResolution:
    """Result of model resolution process."""

    model_config: ModelConfig
    is_experiment: bool  # True if resolved from experiment directory
    experiment_path: Optional[Path] = None  # Path to experiment if applicable

    @property
    def base_model_hf_id(self) -> str:
        """HuggingFace model ID for the base model."""
        return self.model_config.model_name_or_path

    @property
    def peft_adapter_path(self) -> Optional[str]:
        """Path to PEFT adapter weights, if this is a PEFT experiment."""
        if self.is_experiment and self.experiment_path:
            if (self.experiment_path / "adapter_config.json").exists():
                return str(self.experiment_path)
        return None

    @property
    def model_type(self) -> str:
        """Get model type from config."""
        if not self.model_config.model_type:
            raise ValueError(
                f"Model config is missing required 'model_type' field. "
                f"Please add 'model_type': 'chat' or 'text_generation' to the config."
            )
        return self.model_config.model_type


def resolve_model_specification(
    model_spec: str, project_root: Optional[Path] = None
) -> ModelResolution:
    """
    Resolve a model specification string using the two core methods:
    1. Experiment path → loads from experiments/{path}/model_config.json
    2. Config name → loads from configs/model/{name}.json

    Args:
        model_spec: Model specification string. Can be:
            - Experiment path (e.g., 'adam/my_run' or 'experiments/adam/my_run')
            - Config name (e.g., 'gemma-3-27b-it')
        project_root: Project root directory. Auto-detected if None.

    Returns:
        ModelResolution object with resolved config and paths

    Raises:
        ValueError: If model specification cannot be resolved
    """
    if project_root is None:
        project_root = _detect_project_root()

    # Remove leading 'experiments/' if present
    clean_spec = model_spec
    if model_spec.startswith("experiments/"):
        clean_spec = model_spec[12:]  # Remove 'experiments/' prefix

    # Step 1: Try as experiment path
    experiment_resolution = _try_resolve_as_experiment(clean_spec, project_root)
    if experiment_resolution:
        return experiment_resolution

    # Step 2: Try as config name
    config_resolution = _try_resolve_as_config(clean_spec, project_root)
    if config_resolution:
        return config_resolution

    raise ValueError(
        f"Could not resolve model specification '{model_spec}'. Tried:\n"
        f"  - Experiment path: experiments/{clean_spec}\n"
        f"  - Config name: configs/model/{clean_spec}.json\n"
        f"Ensure the model specification exists in one of these forms."
    )


def _try_resolve_as_experiment(
    spec: str, project_root: Path
) -> Optional[ModelResolution]:
    """Try to resolve as experiment directory path."""

    # Try as dataset/run_name format first
    if "/" in spec:
        experiment_path = project_root / "experiments" / spec
    else:
        # Search for run_name in all dataset subdirectories
        experiments_dir = project_root / "experiments"
        experiment_path = None

        if experiments_dir.exists():
            for dataset_dir in experiments_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name != "suites":
                    potential_path = dataset_dir / spec
                    if (
                        potential_path.exists()
                        and (potential_path / "model_config.json").exists()
                    ):
                        experiment_path = potential_path
                        break

        if not experiment_path:
            return None

    if not experiment_path.exists():
        return None

    model_config_path = experiment_path / "model_config.json"
    if not model_config_path.exists():
        return None

    try:
        # Load model config from experiment
        model_config = ModelConfig.from_json(str(model_config_path))

        return ModelResolution(
            model_config=model_config,
            is_experiment=True,
            experiment_path=experiment_path,
        )

    except Exception:
        return None


def _try_resolve_as_config(spec: str, project_root: Path) -> Optional[ModelResolution]:
    """Try to resolve as model config name."""

    config_name = spec if spec.endswith(".json") else f"{spec}.json"
    config_path = project_root / "configs" / "model" / config_name

    if not config_path.exists():
        return None

    try:
        model_config = ModelConfig.from_json(str(config_path))

        return ModelResolution(
            model_config=model_config,
            is_experiment=False,
        )

    except Exception:
        return None


def _detect_project_root() -> Path:
    """Auto-detect project root directory."""

    # Try from current working directory
    cwd = Path.cwd()
    if (cwd / "configs").exists() and (cwd / "src").exists():
        return cwd

    # Try from this file's location (assuming src/utils/model_resolver.py)
    file_path = Path(__file__).resolve()
    potential_root = file_path.parent.parent.parent
    if (potential_root / "configs").exists() and (potential_root / "src").exists():
        return potential_root

    # Fallback to current working directory with warning
    print(f"Warning: Could not auto-detect project root. Using: {cwd}")
    return cwd


# Legacy compatibility functions
def load_model_config_from_spec(model_spec: str) -> Tuple[ModelConfig, str]:
    """
    Legacy compatibility function.
    Returns (model_config, actual_model_path) tuple.
    """
    resolution = resolve_model_specification(model_spec)
    # For legacy compatibility, return the path that should be loaded
    actual_path = resolution.peft_adapter_path or resolution.base_model_hf_id
    return resolution.model_config, actual_path
