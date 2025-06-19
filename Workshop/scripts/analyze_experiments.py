#!/usr/bin/env python
# This script analyzes and compares results from experiment suites.
# It can generate reports, compare metrics across different configurations,
# and export results for further analysis.
#
# Example usage:
# python -m scripts.analyze_experiments model_comparison --metric eval_loss
# python -m scripts.analyze_experiments hyperparameter_sweep --export_csv results.csv
# python -m scripts.analyze_experiments model_comparison --compare_models --plot
#
# Arguments:
#   suite_name (str): Name of the experiment suite to analyze
#   --metric (str): Primary metric to focus on (default: eval_loss)
#   --export_csv (str): Export results to CSV file
#   --compare_models: Compare results across different models
#   --compare_datasets: Compare results across different datasets
#   --plot: Generate plots (requires matplotlib)

import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Ensure src is in path for imports if running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))


class ExperimentAnalyzer:
    """Analyzer for experiment suite results."""

    def __init__(self, suite_name: str):
        self.suite_name = suite_name
        self.suite_dir = Path("experiments/suites") / suite_name

        if not self.suite_dir.exists():
            raise FileNotFoundError(
                f"Experiment suite directory not found: {self.suite_dir}"
            )

        # Load experiment metadata
        self.experiments_db = self._load_experiments_db()
        self.experiment_plan = self._load_experiment_plan()

    def _load_experiments_db(self) -> Dict[str, Any]:
        """Load the experiments database."""
        db_path = self.suite_dir / "experiments.json"
        if db_path.exists():
            with open(db_path, "r") as f:
                return json.load(f)
        return {}

    def _load_experiment_plan(self) -> Dict[str, Any]:
        """Load the experiment plan."""
        plan_path = self.suite_dir / "experiment_plan.json"
        if plan_path.exists():
            with open(plan_path, "r") as f:
                return json.load(f)
        return {}

    def get_experiment_results(self) -> pd.DataFrame:
        """Extract results from all completed experiments into a DataFrame."""

        results = []

        for experiment_id, exp_status in self.experiments_db.items():
            if exp_status.get("status") != "completed":
                continue

            # Find experiment metadata from plan
            exp_metadata = None
            for exp in self.experiment_plan.get("experiments", []):
                if exp["experiment_id"] == experiment_id:
                    exp_metadata = exp
                    break

            if not exp_metadata:
                continue

            # Load experiment results if available
            exp_dir = self.suite_dir / experiment_id
            metrics = self._extract_experiment_metrics(exp_dir)

            # Combine metadata and metrics
            result = {
                "experiment_id": experiment_id,
                "model_config": exp_metadata["model"],
                "pipeline_config": exp_metadata["pipeline_config_name"],
                "dataset": exp_metadata["dataset_name"],
                "peft_config": exp_metadata.get("peft_config_name"),
                "seed": exp_metadata["seed"],
                "status": exp_status["status"],
                "completed_at": exp_status.get("updated_at"),
                **metrics,
            }

            # Add stage overrides as separate columns
            if exp_metadata.get("stage_overrides"):
                for stage, params in exp_metadata["stage_overrides"].items():
                    for param, value in params.items():
                        result[f"{stage}_{param}"] = value

            results.append(result)

        return pd.DataFrame(results)

    def _extract_experiment_metrics(self, exp_dir: Path) -> Dict[str, Any]:
        """Extract metrics from an experiment directory."""
        metrics = {}

        # Look for final metrics in various files
        # 1. Check for trainer state (training metrics)
        trainer_state_path = exp_dir / "trainer_state.json"
        if trainer_state_path.exists():
            try:
                with open(trainer_state_path, "r") as f:
                    trainer_state = json.load(f)

                # Extract final metrics from log history
                if "log_history" in trainer_state and trainer_state["log_history"]:
                    final_logs = trainer_state["log_history"][-1]
                    for key, value in final_logs.items():
                        if isinstance(value, (int, float)):
                            metrics[key] = value

            except Exception as e:
                print(
                    f"Warning: Could not load trainer_state.json for {exp_dir.name}: {e}"
                )

        # 2. Check for all_results.json (final evaluation results)
        results_path = exp_dir / "all_results.json"
        if results_path.exists():
            try:
                with open(results_path, "r") as f:
                    all_results = json.load(f)
                    metrics.update(all_results)
            except Exception as e:
                print(
                    f"Warning: Could not load all_results.json for {exp_dir.name}: {e}"
                )

        # 3. Check for experiment metadata
        metadata_path = exp_dir / "experiment_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    if "metrics" in metadata:
                        metrics.update(metadata["metrics"])
            except Exception as e:
                print(
                    f"Warning: Could not load experiment_metadata.json for {exp_dir.name}: {e}"
                )

        return metrics

    def generate_summary_report(self, primary_metric: str = "eval_loss") -> str:
        """Generate a comprehensive summary report."""

        df = self.get_experiment_results()

        if df.empty:
            return "No completed experiments found."

        report = []
        report.append(f"# Experiment Suite Analysis: {self.suite_name}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Basic statistics
        total_experiments = len(self.experiment_plan.get("experiments", []))
        completed_experiments = len(df)

        report.append(f"## Overview")
        report.append(f"- Total planned experiments: {total_experiments}")
        report.append(f"- Completed experiments: {completed_experiments}")
        report.append(
            f"- Success rate: {completed_experiments/total_experiments*100:.1f}%"
        )
        report.append("")

        # Metrics summary
        if primary_metric in df.columns:
            report.append(f"## {primary_metric} Summary")
            metric_stats = df[primary_metric].describe()
            report.append(f"- Mean: {metric_stats['mean']:.4f}")
            report.append(f"- Std: {metric_stats['std']:.4f}")
            report.append(f"- Min: {metric_stats['min']:.4f}")
            report.append(f"- Max: {metric_stats['max']:.4f}")
            report.append("")

            # Best performing experiments
            if primary_metric.endswith("_loss") or "error" in primary_metric:
                best_df = df.nsmallest(3, primary_metric)
            else:
                best_df = df.nlargest(3, primary_metric)

            report.append(f"## Top 3 Experiments by {primary_metric}")
            for i, (_, row) in enumerate(best_df.iterrows(), 1):
                report.append(f"{i}. {row['experiment_id']}: {row[primary_metric]:.4f}")
                report.append(
                    f"   Model: {row['model_config']}, Dataset: {row['dataset']}, Seed: {row['seed']}"
                )
            report.append("")

        # Configuration analysis
        if len(df["model_config"].unique()) > 1:
            report.append("## Model Comparison")
            model_comparison = df.groupby("model_config")[primary_metric].agg(
                ["mean", "std", "count"]
            )
            for model, stats in model_comparison.iterrows():
                report.append(
                    f"- {model}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                )
            report.append("")

        if len(df["dataset"].unique()) > 1:
            report.append("## Dataset Comparison")
            dataset_comparison = df.groupby("dataset")[primary_metric].agg(
                ["mean", "std", "count"]
            )
            for dataset, stats in dataset_comparison.iterrows():
                report.append(
                    f"- {dataset}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                )
            report.append("")

        # Hyperparameter analysis
        hyperparam_cols = [
            col for col in df.columns if col.startswith(("sft_", "dpo_"))
        ]
        if hyperparam_cols and primary_metric in df.columns:
            report.append("## Hyperparameter Analysis")
            for param_col in hyperparam_cols:
                if len(df[param_col].unique()) > 1:
                    param_analysis = df.groupby(param_col)[primary_metric].agg(
                        ["mean", "std", "count"]
                    )
                    report.append(f"### {param_col}")
                    for param_val, stats in param_analysis.iterrows():
                        report.append(
                            f"- {param_val}: {stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})"
                        )
                    report.append("")

        return "\n".join(report)

    def export_to_csv(self, output_path: str) -> None:
        """Export results to CSV file."""
        df = self.get_experiment_results()
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

    def plot_results(self, primary_metric: str = "eval_loss") -> None:
        """Generate plots for experiment results."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Error: matplotlib and seaborn are required for plotting.")
            print("Install with: pip install matplotlib seaborn")
            return

        df = self.get_experiment_results()

        if df.empty:
            print("No completed experiments to plot.")
            return

        if primary_metric not in df.columns:
            print(f"Metric '{primary_metric}' not found in results.")
            return

        # Set up the plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Experiment Suite Results: {self.suite_name}", fontsize=16)

        # 1. Distribution of primary metric
        axes[0, 0].hist(df[primary_metric], bins=20, alpha=0.7, edgecolor="black")
        axes[0, 0].set_title(f"Distribution of {primary_metric}")
        axes[0, 0].set_xlabel(primary_metric)
        axes[0, 0].set_ylabel("Frequency")

        # 2. Model comparison (if multiple models)
        if len(df["model_config"].unique()) > 1:
            sns.boxplot(data=df, x="model_config", y=primary_metric, ax=axes[0, 1])
            axes[0, 1].set_title(f"{primary_metric} by Model")
            axes[0, 1].tick_params(axis="x", rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, "Single Model Used", ha="center", va="center")
            axes[0, 1].set_title("Model Comparison (N/A)")

        # 3. Dataset comparison (if multiple datasets)
        if len(df["dataset"].unique()) > 1:
            sns.boxplot(data=df, x="dataset", y=primary_metric, ax=axes[1, 0])
            axes[1, 0].set_title(f"{primary_metric} by Dataset")
            axes[1, 0].tick_params(axis="x", rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, "Single Dataset Used", ha="center", va="center")
            axes[1, 0].set_title("Dataset Comparison (N/A)")

        # 4. Seed stability (if multiple seeds)
        if len(df["seed"].unique()) > 1:
            sns.boxplot(data=df, x="seed", y=primary_metric, ax=axes[1, 1])
            axes[1, 1].set_title(f"{primary_metric} by Random Seed")
        else:
            axes[1, 1].text(0.5, 0.5, "Single Seed Used", ha="center", va="center")
            axes[1, 1].set_title("Seed Stability (N/A)")

        plt.tight_layout()

        # Save plot
        plot_path = self.suite_dir / f"{self.suite_name}_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment suite results.",
        epilog="Example usage:\n"
        "  python -m scripts.analyze_experiments model_comparison --metric eval_loss\n"
        "  python -m scripts.analyze_experiments hyperparameter_sweep --export_csv results.csv\n"
        "  python -m scripts.analyze_experiments model_comparison --plot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "suite_name",
        type=str,
        help="Name of the experiment suite to analyze",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="eval_loss",
        help="Primary metric to focus on for analysis (default: eval_loss)",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        help="Export results to CSV file with this name",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires matplotlib and seaborn)",
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = ExperimentAnalyzer(args.suite_name)

        # Generate and print summary report
        report = analyzer.generate_summary_report(args.metric)
        print(report)

        # Export to CSV if requested
        if args.export_csv:
            analyzer.export_to_csv(args.export_csv)

        # Generate plots if requested
        if args.plot:
            analyzer.plot_results(args.metric)

    except Exception as e:
        print(f"Analysis failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
