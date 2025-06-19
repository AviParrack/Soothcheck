# This file provides unified run management for all training types.
# It handles directory creation, metadata management, and run discovery
# with a consistent interface regardless of training type (single stage, pipeline, suite).
#
# Classes:
# - RunManager: Main interface for managing training runs
#
# Functions:
# - find_run: Find a run by various identifiers
# - list_runs: List available runs with filtering
# - create_run_name: Generate appropriate run names

import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

from ..config_models.run_metadata import RunMetadata, StageInfo, ChainInfo


class RunManager:
    """Unified run management for all training types."""

    def __init__(self, base_experiments_dir: str = "experiments"):
        self.base_dir = Path(base_experiments_dir)
        self.base_dir.mkdir(exist_ok=True)

    def create_run_directory(
        self, dataset_name: str, run_name: str, run_type: str, overwrite: bool = False
    ) -> Path:
        """Create a run directory with consistent structure."""

        run_dir = self.base_dir / dataset_name / run_name

        if run_dir.exists() and not overwrite:
            raise ValueError(f"Run directory already exists: {run_dir}")

        run_dir.mkdir(parents=True, exist_ok=overwrite)
        return run_dir

    def save_run_metadata(self, metadata: RunMetadata, run_dir: Path) -> None:
        """Save run metadata and generate model card."""

        # Save metadata
        metadata.save_to_directory(run_dir)

        # Generate and save model card
        model_card = metadata.generate_model_card()
        with open(run_dir / "README.md", "w") as f:
            f.write(model_card)

    def find_run(self, identifier: str) -> Optional[Tuple[Path, RunMetadata]]:
        """Find a run by various identifier formats.

        Supports:
        - dataset/run_name
        - run_name (searches all datasets)
        - full path
        """

        # Try as dataset/run_name
        if "/" in identifier:
            parts = identifier.split("/", 1)
            if len(parts) == 2:
                dataset, run_name = parts
                run_dir = self.base_dir / dataset / run_name
                if run_dir.exists():
                    try:
                        metadata = RunMetadata.from_run_directory(str(run_dir))
                        return run_dir, metadata
                    except:
                        pass

        # Try as just run_name (search all datasets)
        else:
            for dataset_dir in self.base_dir.iterdir():
                if dataset_dir.is_dir() and dataset_dir.name != "suites":
                    run_dir = dataset_dir / identifier
                    if run_dir.exists():
                        try:
                            metadata = RunMetadata.from_run_directory(str(run_dir))
                            return run_dir, metadata
                        except:
                            pass

        # Try as full path
        if Path(identifier).exists():
            try:
                metadata = RunMetadata.from_run_directory(identifier)
                return Path(identifier), metadata
            except:
                pass

        return None

    def list_runs(
        self,
        dataset_name: Optional[str] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
        stage_type: Optional[str] = None,
    ) -> List[Tuple[Path, RunMetadata]]:
        """List runs with optional filtering."""

        runs = []

        search_dirs = []
        if dataset_name:
            dataset_dir = self.base_dir / dataset_name
            if dataset_dir.exists():
                search_dirs = [dataset_dir]
        else:
            search_dirs = [
                d for d in self.base_dir.iterdir() if d.is_dir() and d.name != "suites"
            ]

        for dataset_dir in search_dirs:
            for run_dir in dataset_dir.iterdir():
                if run_dir.is_dir():
                    try:
                        metadata = RunMetadata.from_run_directory(str(run_dir))

                        # Apply filters
                        if run_type and metadata.run_type != run_type:
                            continue
                        if status and metadata.status != status:
                            continue
                        if stage_type and not metadata.get_stage_by_type(stage_type):
                            continue

                        runs.append((run_dir, metadata))
                    except:
                        # Skip runs without valid metadata
                        continue

        return runs

    def find_chainable_runs(
        self,
        target_stage_type: str,
        dataset_name: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Tuple[Path, RunMetadata]]:
        """Find runs that can be chained to a specific stage type."""

        all_runs = self.list_runs(dataset_name=dataset_name, status="completed")
        chainable_runs = []

        for run_dir, metadata in all_runs:
            if metadata.can_be_chained_to(target_stage_type):
                # Additional model compatibility check
                if model and metadata.model != model:
                    continue
                chainable_runs.append((run_dir, metadata))

        return chainable_runs

    def create_run_name(
        self,
        base_name: Optional[str],
        run_type: str,
        stage_types: List[str],
        model: str,
        dataset_name: str,
        include_timestamp: bool = True,
    ) -> str:
        """Generate an appropriate run name."""

        if base_name:
            return base_name

        # Auto-generate based on type
        parts = []

        # Add model indicator
        model_short = (
            model.lower()
            .replace("gemma-3-", "g3-")
            .replace("deepseek-", "ds-")
            .replace("llama-4-", "l4-")
            .replace("qwen3-", "q3-")
        )
        parts.append(model_short)

        # Add stage information
        if run_type == "single_stage":
            parts.append(stage_types[0])
        elif run_type == "pipeline":
            parts.append("-".join(stage_types))
        else:  # suite_experiment
            parts.append("suite")

        # Add timestamp if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%m%d-%H%M%S")
            parts.append(timestamp)

        return "_".join(parts)

    def get_run_summary(self, metadata: RunMetadata) -> Dict[str, Any]:
        """Get a summary dict for a run (useful for listings)."""

        final_stage = metadata.get_final_stage()

        return {
            "run_name": metadata.run_name,
            "dataset": metadata.dataset_name,
            "path": metadata.get_run_path(),
            "display_name": metadata.get_display_name(),
            "run_type": metadata.run_type,
            "stages": [s.stage_type for s in metadata.stages],
            "model_config": metadata.model,
            "status": metadata.status,
            "created_at": metadata.created_at,
            "final_loss": final_stage.final_loss,
            "eval_loss": final_stage.eval_loss,
            "can_chain_to_dpo": metadata.can_be_chained_to("dpo"),
        }


def find_run(identifier: str) -> Optional[Tuple[Path, RunMetadata]]:
    """Convenience function to find a run."""
    manager = RunManager()
    return manager.find_run(identifier)


def list_runs(**kwargs) -> List[Tuple[Path, RunMetadata]]:
    """Convenience function to list runs."""
    manager = RunManager()
    return manager.list_runs(**kwargs)


def create_run_name(**kwargs) -> str:
    """Convenience function to create run names."""
    manager = RunManager()
    return manager.create_run_name(**kwargs)
