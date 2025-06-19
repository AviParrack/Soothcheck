# This file defines experiment suite configurations for systematic parameter exploration.
# It allows users to define parameter grids, launch multiple runs, and track experiments
# with consistent metadata for easy comparison and evaluation.
#
# Classes:
# - ParameterSweep: Define parameter variations for systematic exploration
# - ExperimentSuite: Configuration for running multiple related experiments
# - ExperimentRun: Individual experiment instance with metadata
#
# Key features:
# - Parameter grid expansion for systematic sweeps
# - Automatic experiment naming and organization
# - Consistent W&B tagging and grouping
# - Metadata tracking for evaluation and analysis

from pydantic import Field, validator
from typing import Optional, List, Dict, Any, Union, Literal
import os
import itertools
from datetime import datetime
from pathlib import Path

from .base_config import BaseConfig


class ParameterSweep(BaseConfig):
    """Define parameter variations for systematic exploration."""

    # Core configuration variations
    model_configs: Optional[List[str]] = Field(
        default=None,
        description="List of model config names to test (e.g., ['gemma-3-27b-it', 'gemma-3-1b-it'])",
    )

    pipeline_configs: Optional[List[str]] = Field(
        default=None,
        description="List of pipeline config names to test (e.g., ['sft_then_dpo', 'sft_only'])",
    )

    datasets: Optional[List[str]] = Field(
        default=None,
        description="List of dataset names to test (e.g., ['adam', 'rudolf'])",
    )

    peft_configs: Optional[List[str]] = Field(
        default=None,
        description="List of PEFT config names to test (e.g., ['lora_default', 'lora_aggressive'])",
    )

    # Stage-specific parameter overrides
    stage_overrides: Optional[Dict[str, Dict[str, List[Any]]]] = Field(
        default=None,
        description="Stage-specific parameter variations. Format: {stage_name: {param_name: [values]}}",
    )

    # Stage config file sweeps
    stage_config_sweeps: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Stage config file sweeps. Format: {stage_name: [config_file_names]}. Mutually exclusive with stage_overrides for the same stage.",
    )

    # Global parameter overrides
    seeds: Optional[List[int]] = Field(
        default=None,
        description="List of random seeds for multiple runs (e.g., [42, 123, 456])",
    )

    @validator("stage_config_sweeps")
    def validate_stage_config_conflicts(cls, v, values):
        """Ensure stage_config_sweeps and stage_overrides don't conflict."""
        if v is not None and values.get("stage_overrides") is not None:
            # Check for overlapping stage names
            sweep_stages = set(v.keys())
            override_stages = set(values["stage_overrides"].keys())
            conflicts = sweep_stages & override_stages
            if conflicts:
                raise ValueError(
                    f"stage_config_sweeps and stage_overrides cannot specify the same stages. "
                    f"Conflicting stages: {sorted(conflicts)}"
                )
        return v


class ExperimentMetadata(BaseConfig):
    """Metadata for experiment tracking and organization."""

    experiment_id: str = Field(..., description="Unique experiment identifier")
    suite_name: str = Field(..., description="Name of the experiment suite")
    created_at: str = Field(
        ..., description="ISO timestamp when experiment was created"
    )

    # Configuration parameters
    model: str = Field(..., description="Model configuration used")
    pipeline_config_name: str = Field(..., description="Pipeline configuration used")
    dataset_name: str = Field(..., description="Dataset used")
    peft_config_name: Optional[str] = Field(
        default=None, description="PEFT configuration used"
    )
    seed: int = Field(..., description="Random seed used")

    # Stage overrides applied
    stage_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None, description="Stage-specific parameter overrides applied"
    )

    # Stage config files selected
    stage_config_selections: Optional[Dict[str, str]] = Field(
        default=None, description="Stage config file names selected for each stage"
    )

    # Tracking information
    wandb_run_url: Optional[str] = Field(default=None, description="W&B run URL")
    status: Literal["planned", "running", "completed", "failed"] = Field(
        default="planned", description="Current status of the experiment"
    )

    # Results (populated after completion)
    final_model_path: Optional[str] = Field(
        default=None, description="Path to final trained model"
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None, description="Final metrics"
    )

    # Tags for organization and filtering
    tags: List[str] = Field(
        default_factory=list, description="Tags for experiment organization"
    )


class ExperimentSuite(BaseConfig):
    """Configuration for running multiple related experiments."""

    suite_name: str = Field(..., description="Name for this experiment suite")

    description: Optional[str] = Field(
        default=None, description="Description of the experiment suite's purpose"
    )

    # Parameter sweep definition
    parameter_sweep: ParameterSweep = Field(
        ..., description="Parameter variations to explore"
    )

    # Base configurations (can be overridden by sweep)
    base_model_config: Optional[str] = Field(
        default="gemma-3-27b-it", description="Default model config if not swept"
    )

    base_pipeline_config: Optional[str] = Field(
        default="sft_then_dpo", description="Default pipeline config if not swept"
    )

    base_dataset: Optional[str] = Field(
        default=None, description="Default dataset if not swept"
    )

    base_peft_config: Optional[str] = Field(
        default="lora_default", description="Default PEFT config if not swept"
    )

    # Organization settings
    output_base_dir: str = Field(
        default="experiments/suites", description="Base directory for suite outputs"
    )

    # W&B settings
    wandb_project: str = Field(default="pollux", description="W&B project name")
    wandb_tags: List[str] = Field(
        default_factory=list,
        description="Additional W&B tags for all runs in this suite",
    )

    # Execution settings
    max_concurrent_runs: int = Field(
        default=1, description="Maximum number of concurrent experiments"
    )

    dry_run: bool = Field(
        default=False,
        description="If True, only generate experiment plan without running",
    )

    def generate_experiment_plan(self) -> List[ExperimentMetadata]:
        """Generate the full list of experiments to run based on parameter sweep."""

        # Collect all parameter combinations
        param_combinations = []

        # Core configurations
        model_configs = self.parameter_sweep.model_configs or [self.base_model_config]
        pipeline_configs = self.parameter_sweep.pipeline_configs or [
            self.base_pipeline_config
        ]
        datasets = self.parameter_sweep.datasets or [self.base_dataset]
        peft_configs = self.parameter_sweep.peft_configs or [self.base_peft_config]
        seeds = self.parameter_sweep.seeds or [42]

        # Generate all combinations
        for (
            model_config,
            pipeline_config,
            dataset,
            peft_config,
            seed,
        ) in itertools.product(
            model_configs, pipeline_configs, datasets, peft_configs, seeds
        ):

            # Generate stage override combinations if specified
            stage_override_combinations = [{}]  # Default: no overrides
            stage_config_combinations = [{}]  # Default: no config sweeps

            # Handle parameter-level overrides
            if self.parameter_sweep.stage_overrides:
                stage_combos = []
                for (
                    stage_name,
                    param_dict,
                ) in self.parameter_sweep.stage_overrides.items():
                    param_combos = []
                    for param_name, values in param_dict.items():
                        param_combos.append([(param_name, value) for value in values])

                    # Generate all combinations for this stage
                    if param_combos:
                        stage_param_combinations = itertools.product(*param_combos)
                        stage_combos.append(
                            [
                                {stage_name: dict(combo)}
                                for combo in stage_param_combinations
                            ]
                        )

                if stage_combos:
                    stage_override_combinations = []
                    for combo in itertools.product(*stage_combos):
                        merged = {}
                        for stage_dict in combo:
                            merged.update(stage_dict)
                        stage_override_combinations.append(merged)

            # Handle stage config file sweeps
            if self.parameter_sweep.stage_config_sweeps:
                config_combos = []
                for stage_name, config_names in self.parameter_sweep.stage_config_sweeps.items():
                    config_combos.append([(stage_name, config_name) for config_name in config_names])

                if config_combos:
                    stage_config_combinations = []
                    for combo in itertools.product(*config_combos):
                        stage_config_combinations.append(dict(combo))

            # Create experiment for each combination of overrides and config selections
            for stage_overrides in stage_override_combinations:
                for stage_config_selections in stage_config_combinations:
                    experiment_id = self._generate_experiment_id(
                        model_config,
                        pipeline_config,
                        dataset,
                        peft_config,
                        seed,
                        stage_overrides,
                        stage_config_selections,
                    )

                    tags = self.wandb_tags.copy()
                    tags.extend(
                        [
                            f"suite:{self.suite_name}",
                            f"model:{model_config}",
                            f"pipeline:{pipeline_config}",
                            f"dataset:{dataset}",
                            f"seed:{seed}",
                        ]
                    )
                    if peft_config:
                        tags.append(f"peft:{peft_config}")

                    experiment = ExperimentMetadata(
                        experiment_id=experiment_id,
                        suite_name=self.suite_name,
                        created_at=datetime.now().isoformat(),
                        model=model_config,
                        pipeline_config_name=pipeline_config,
                        dataset_name=dataset,
                        peft_config_name=peft_config,
                        seed=seed,
                        stage_overrides=stage_overrides if stage_overrides else None,
                        stage_config_selections=stage_config_selections if stage_config_selections else None,
                        tags=tags,
                    )

                    param_combinations.append(experiment)

        return param_combinations

    def _generate_experiment_id(
        self,
        model_config: str,
        pipeline_config: str,
        dataset: str,
        peft_config: str,
        seed: int,
        stage_overrides: Dict[str, Dict[str, Any]],
        stage_config_selections: Dict[str, str] = None,
    ) -> str:
        """Generate a unique, readable experiment ID."""

        # Shorten common names for readability
        model_short = model_config.replace("gemma-3-", "g3-").replace("-it", "").replace("llama-3.", "l3-").replace("-70B-Instruct", "-70bi")
        pipeline_short = pipeline_config.replace("_then_", "-").replace("_", "-")

        base_id = f"{model_short}_{pipeline_short}_{dataset}"

        # Add PEFT config to ensure uniqueness
        if peft_config:
            # Shorten PEFT config names for readability
            peft_short = peft_config.replace("lora_", "").replace("_default", "")
            # Convert patterns like "r8_a16" to "r8a16", "r32_a64" to "r32a64"
            peft_short = peft_short.replace("_", "")
            base_id += f"_{peft_short}"

        # Add stage config selections suffix if present
        if stage_config_selections:
            config_parts = []
            for stage, config_name in stage_config_selections.items():
                # Extract meaningful parts from config name (e.g., "sft_l33_max_bf16" -> "max-bf16")
                config_short = config_name.replace(f"{stage}_", "").replace("l33_", "")
                config_parts.append(f"{stage[0]}{config_short}")
            if config_parts:
                base_id += "_" + "-".join(config_parts)

        # Add stage override suffix if present
        if stage_overrides:
            override_parts = []
            for stage, params in stage_overrides.items():
                for param, value in params.items():
                    override_parts.append(f"{stage[0]}{param[:2]}{value}")
            if override_parts:
                base_id += "_" + "-".join(override_parts)

        return base_id

    def get_suite_output_dir(self) -> Path:
        """Get the output directory for this experiment suite."""
        return Path(self.output_base_dir) / self.suite_name

    @classmethod
    def load(cls, suite_name: str) -> "ExperimentSuite":
        """Load an experiment suite configuration by name."""
        return super().load_from_name(suite_name, "experiment_suites")
