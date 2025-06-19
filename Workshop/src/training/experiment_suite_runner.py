# This file contains the experiment suite runner for systematic parameter exploration.
# It manages multiple training runs with different parameter combinations,
# handles concurrent execution, and provides comprehensive experiment tracking.
#
# Classes:
# - ExperimentSuiteRunner: Main orchestrator for experiment suites
#
# Functions:
# - run_experiment_suite: Main entry point for running experiment suites

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import contextlib
import io

from ..config_models import TrainConfig, ModelConfig, PeftConfig as PydanticPeftConfig
from ..config_models.experiment_config import ExperimentSuite, ExperimentMetadata

# Import the programmatic pipeline runner
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from scripts.train_pipeline import run_pipeline_programmatic


class ExperimentSuiteRunner:
    """Main orchestrator for experiment suites with parameter sweeps."""

    def __init__(self, suite_config: ExperimentSuite):
        self.suite_config = suite_config
        self.suite_output_dir = suite_config.get_suite_output_dir()
        self.suite_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize experiment tracking
        self.experiments_db_path = self.suite_output_dir / "experiments.json"
        self.experiments_db = self._load_experiments_db()

    def run(self) -> None:
        """Run the complete experiment suite."""

        print(f"🚀 Starting experiment suite: {self.suite_config.suite_name}")
        if self.suite_config.description:
            print(f"Description: {self.suite_config.description}")

        # Generate experiment plan
        experiment_plan = self.suite_config.generate_experiment_plan()
        print(f"📋 Generated {len(experiment_plan)} experiments")

        if self.suite_config.dry_run:
            print("🔍 Dry run mode - showing experiment plan:")
            self._show_experiment_plan(experiment_plan)
            return

        # Save experiment plan
        self._save_experiment_plan(experiment_plan)

        # Filter to only run experiments that haven't completed
        pending_experiments = [
            exp
            for exp in experiment_plan
            if exp.experiment_id not in self.experiments_db
            or self.experiments_db[exp.experiment_id].get("status") != "completed"
        ]

        if not pending_experiments:
            print("✅ All experiments already completed!")
            return

        print(f"▶️ Running {len(pending_experiments)} pending experiments")
        print(f"⚙️ Max concurrent runs: {self.suite_config.max_concurrent_runs}")

        # Run experiments
        if self.suite_config.max_concurrent_runs == 1:
            self._run_experiments_sequential(pending_experiments)
        else:
            self._run_experiments_concurrent(pending_experiments)

        print(f"🎉 Experiment suite '{self.suite_config.suite_name}' completed!")
        self._generate_suite_summary()

    def _run_experiments_sequential(
        self, experiments: List[ExperimentMetadata]
    ) -> None:
        """Run experiments one at a time."""

        total_experiments = len(experiments)
        for i, experiment in enumerate(experiments, 1):
            print(f"\n{'='*60}")
            print(
                f"🔄 Running experiment {i}/{total_experiments}: {experiment.experiment_id}"
            )
            print(f"📊 Model: {experiment.model} | Dataset: {experiment.dataset_name}")
            print(
                f"🎯 Pipeline: {experiment.pipeline_config_name} | Seed: {experiment.seed}"
            )
            print(f"{'='*60}\n")

            try:
                self._run_single_experiment(experiment)
                print(
                    f"\n✅ Experiment {experiment.experiment_id} completed successfully"
                )

                # Show remaining experiments
                remaining = total_experiments - i
                if remaining > 0:
                    print(f"📋 {remaining} experiment(s) remaining")

            except Exception as e:
                print(f"\n❌ Experiment {experiment.experiment_id} failed: {e}")
                self._update_experiment_status(
                    experiment.experiment_id, "failed", str(e)
                )

                # Show remaining experiments even on failure
                remaining = total_experiments - i
                if remaining > 0:
                    print(f"📋 {remaining} experiment(s) remaining")

    def _run_experiments_concurrent(
        self, experiments: List[ExperimentMetadata]
    ) -> None:
        """Run experiments concurrently using ThreadPoolExecutor."""

        total_experiments = len(experiments)

        print(
            f"\n🚀 Starting {total_experiments} experiments with up to {self.suite_config.max_concurrent_runs} running concurrently"
        )
        print(f"{'='*60}\n")

        with ThreadPoolExecutor(
            max_workers=self.suite_config.max_concurrent_runs
        ) as executor:

            # Submit all experiments
            future_to_experiment = {
                executor.submit(
                    self._run_single_experiment_concurrent, exp, i, total_experiments
                ): exp
                for i, exp in enumerate(experiments, 1)
            }

            # Process completions
            completed = 0
            for future in as_completed(future_to_experiment):
                experiment = future_to_experiment[future]
                completed += 1

                try:
                    future.result()
                    print(
                        f"\n✅ [{completed}/{total_experiments}] Experiment {experiment.experiment_id} completed"
                    )

                except Exception as e:
                    print(
                        f"\n❌ [{completed}/{total_experiments}] Experiment {experiment.experiment_id} failed: {e}"
                    )
                    self._update_experiment_status(
                        experiment.experiment_id, "failed", str(e)
                    )

                # Show remaining
                remaining = total_experiments - completed
                if remaining > 0:
                    print(f"📋 {remaining} experiment(s) remaining")

    def _run_single_experiment_concurrent(
        self, experiment: ExperimentMetadata, index: int, total: int
    ) -> None:
        """Run a single experiment with concurrent-friendly output."""

        print(f"\n🔄 Starting experiment {index}/{total}: {experiment.experiment_id}")
        print(f"   Model: {experiment.model} | Dataset: {experiment.dataset_name}")

        # Run the experiment in quiet mode for concurrent execution
        self._run_single_experiment(experiment, show_output=False)

    def _run_single_experiment(
        self, experiment: ExperimentMetadata, show_output: bool = True
    ) -> None:
        """Run a single experiment directly in Python."""

        # Update status to running
        self._update_experiment_status(experiment.experiment_id, "running")

        # Use standard directory structure: experiments/dataset_name/run_name
        # The run_name will be the experiment_id prefixed with suite name
        run_name = f"{self.suite_config.suite_name}_{experiment.experiment_id}"

        try:
            if show_output:
                print(f"🏃 Starting training...")
                print(
                    f"💾 Run output will be saved to: experiments/{experiment.dataset_name}/{run_name}"
                )
                print("-" * 60 + "\n")

            # Set up experiment logging if needed
            if not show_output:
                # For concurrent runs, create a log directory in the suite output
                log_dir = self.suite_output_dir / "logs"
                log_dir.mkdir(exist_ok=True)

                # Redirect stdout/stderr to log file for this experiment
                log_file_path = log_dir / f"{experiment.experiment_id}.log"

                # Create a context manager to capture output
                log_buffer = io.StringIO()

                with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(
                    log_buffer
                ):
                    self._run_experiment_directly(experiment, run_name)

                # Write captured output to log file
                with open(log_file_path, "w") as log_file:
                    log_file.write(log_buffer.getvalue())
            else:
                # For sequential runs, let output go to console normally
                self._run_experiment_directly(experiment, run_name)

            # The actual experiment output will be in experiments/dataset_name/run_name
            actual_output_dir = Path("experiments") / experiment.dataset_name / run_name

            # Save experiment metadata to suite directory for tracking
            self._save_experiment_metadata(
                experiment, self.suite_output_dir / experiment.experiment_id
            )

            # Update status to completed with actual output path
            self._update_experiment_status(
                experiment.experiment_id,
                "completed",
                final_model_path=str(actual_output_dir),
            )

        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    def _run_experiment_directly(
        self, experiment: ExperimentMetadata, run_name: str
    ) -> None:
        """Run a single experiment directly using the programmatic interface."""

        # Call the pipeline directly
        run_pipeline_programmatic(
            config_name=experiment.pipeline_config_name,
            dataset_name=experiment.dataset_name,
            run_name=run_name,
            model_override=experiment.model,
            peft_config_override=experiment.peft_config_name,
            stage_overrides=experiment.stage_overrides,
            stage_config_selections=experiment.stage_config_selections,
        )

    def _build_experiment_command(
        self, experiment: ExperimentMetadata, run_name: str
    ) -> List[str]:
        """Build the command to run a single experiment (legacy - kept for reference)."""

        # Use train_pipeline.py for multi-stage experiments
        cmd = [
            sys.executable,
            "-m",
            "scripts.train_pipeline",
            experiment.pipeline_config_name,
            "--dataset",
            experiment.dataset_name,
            "--run_name",
            run_name,
        ]

        # Add model override if different from pipeline default
        if experiment.model:
            cmd.extend(["--model", experiment.model])

        # Add PEFT config override if specified
        if experiment.peft_config_name:
            cmd.extend(["--peft_config", experiment.peft_config_name])

        # Add stage overrides if specified
        if experiment.stage_overrides:
            # Convert to JSON string for command line
            import json

            overrides_json = json.dumps(experiment.stage_overrides)
            cmd.extend(["--stage_overrides", overrides_json])

        return cmd

    def _save_experiment_metadata(
        self, experiment: ExperimentMetadata, output_dir: Path
    ) -> None:
        """Save experiment metadata to the output directory."""

        # Ensure the output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata_dict = experiment.dict()

        with open(output_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2)

    def _load_experiments_db(self) -> Dict[str, Dict[str, Any]]:
        """Load the experiments database."""

        if self.experiments_db_path.exists():
            with open(self.experiments_db_path, "r") as f:
                return json.load(f)
        return {}

    def _save_experiments_db(self) -> None:
        """Save the experiments database."""

        with open(self.experiments_db_path, "w") as f:
            json.dump(self.experiments_db, f, indent=2)

    def _update_experiment_status(
        self,
        experiment_id: str,
        status: str,
        error_message: Optional[str] = None,
        final_model_path: Optional[str] = None,
    ) -> None:
        """Update experiment status in the database."""

        if experiment_id not in self.experiments_db:
            self.experiments_db[experiment_id] = {}

        self.experiments_db[experiment_id]["status"] = status
        self.experiments_db[experiment_id]["updated_at"] = datetime.now().isoformat()

        if error_message:
            self.experiments_db[experiment_id]["error_message"] = error_message

        if final_model_path:
            self.experiments_db[experiment_id]["final_model_path"] = final_model_path

        self._save_experiments_db()

    def _save_experiment_plan(self, experiments: List[ExperimentMetadata]) -> None:
        """Save the complete experiment plan."""

        plan_dict = {
            "suite_name": self.suite_config.suite_name,
            "description": self.suite_config.description,
            "created_at": datetime.now().isoformat(),
            "total_experiments": len(experiments),
            "experiments": [exp.dict() for exp in experiments],
        }

        with open(self.suite_output_dir / "experiment_plan.json", "w") as f:
            json.dump(plan_dict, f, indent=2)

    def _show_experiment_plan(self, experiments: List[ExperimentMetadata]) -> None:
        """Display the experiment plan in a readable format."""

        print(f"\n📊 Experiment Plan for '{self.suite_config.suite_name}':")
        print(f"Total experiments: {len(experiments)}")
        print()

        # Group by configuration combinations
        config_groups = {}
        for exp in experiments:
            key = (
                exp.model,
                exp.pipeline_config_name,
                exp.peft_config_name,
            )
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(exp)

        for (model, pipeline, peft), group_experiments in config_groups.items():
            print(f"📦 Config: {model} + {pipeline} + {peft or 'no-peft'}")

            # Group by dataset
            dataset_groups = {}
            for exp in group_experiments:
                if exp.dataset_name not in dataset_groups:
                    dataset_groups[exp.dataset_name] = []
                dataset_groups[exp.dataset_name].append(exp)

            for dataset, dataset_experiments in dataset_groups.items():
                seeds = [exp.seed for exp in dataset_experiments]
                print(f"   📁 Dataset: {dataset} (seeds: {seeds})")

                # Show any stage overrides or config selections
                for exp in dataset_experiments:
                    if exp.stage_overrides:
                        print(f"      🔧 {exp.experiment_id}: overrides={exp.stage_overrides}")
                    if exp.stage_config_selections:
                        print(f"      📋 {exp.experiment_id}: configs={exp.stage_config_selections}")
            print()

    def _generate_suite_summary(self) -> None:
        """Generate a summary of the experiment suite results."""

        summary = {
            "suite_name": self.suite_config.suite_name,
            "completed_at": datetime.now().isoformat(),
            "total_experiments": len(self.experiments_db),
            "status_counts": {},
            "experiments": self.experiments_db,
        }

        # Count statuses
        for exp_data in self.experiments_db.values():
            status = exp_data.get("status", "unknown")
            summary["status_counts"][status] = (
                summary["status_counts"].get(status, 0) + 1
            )

        with open(self.suite_output_dir / "suite_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📈 Suite Summary:")
        for status, count in summary["status_counts"].items():
            print(f"   {status}: {count}")
        print(f"\n📂 Results saved to: {self.suite_output_dir}")


def run_experiment_suite(suite_config: ExperimentSuite) -> None:
    """
    Main entry point for running experiment suites.

    Args:
        suite_config: Configuration for the experiment suite
    """

    runner = ExperimentSuiteRunner(suite_config)
    runner.run()
