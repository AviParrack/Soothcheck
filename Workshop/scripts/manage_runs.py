#!/usr/bin/env python
# This script helps manage and discover training runs with the unified system.
# It provides commands to list, inspect, and find runs across all training types.
#
# Example usage:
# python -m scripts.manage_runs list --dataset adam
# python -m scripts.manage_runs find adam/baseline_sft
# python -m scripts.manage_runs chainable --stage dpo --dataset adam
# python -m scripts.manage_runs inspect adam/baseline_sft
#
# Arguments:
#   command: list, find, chainable, inspect
#   Various options depending on command

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import json

# Ensure src is in path for imports if running as a script
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.training.run_manager import RunManager, find_run, list_runs
from src.config_models.run_metadata import RunMetadata


def cmd_list(args):
    """List runs with optional filtering."""

    manager = RunManager()
    runs = manager.list_runs(
        dataset_name=args.dataset,
        run_type=args.run_type,
        status=args.status,
        stage_type=args.stage_type,
    )

    if not runs:
        print("No runs found matching criteria.")
        return

    print(f"Found {len(runs)} runs:\n")

    # Group by dataset for better organization
    by_dataset = {}
    for run_dir, metadata in runs:
        dataset = metadata.dataset_name
        if dataset not in by_dataset:
            by_dataset[dataset] = []
        by_dataset[dataset].append((run_dir, metadata))

    for dataset, dataset_runs in by_dataset.items():
        print(f"📁 Dataset: {dataset}")

        for run_dir, metadata in dataset_runs:
            summary = manager.get_run_summary(metadata)

            # Status indicator
            status_icon = {"completed": "✅", "running": "🔄", "failed": "❌"}.get(
                metadata.status, "❓"
            )

            # Stage info
            if metadata.run_type == "single_stage":
                stage_info = metadata.stages[0].stage_type
            else:
                stage_info = " → ".join(s.stage_type for s in metadata.stages)

            print(f"  {status_icon} {metadata.run_name} ({stage_info})")
            print(f"     Model: {metadata.model}")

            if summary["final_loss"]:
                print(f"     Final Loss: {summary['final_loss']:.4f}")
            if summary["eval_loss"]:
                print(f"     Eval Loss: {summary['eval_loss']:.4f}")

            # Chaining info
            if metadata.chain_info:
                print(f"     🔗 Chained from: {metadata.chain_info.parent_run}")

            if summary["can_chain_to_dpo"]:
                print(f"     🔄 Can chain to DPO")

            print(f"     📂 Path: {summary['path']}")
            print()


def cmd_find(args):
    """Find a specific run."""

    result = find_run(args.identifier)

    if not result:
        print(f"❌ Run not found: {args.identifier}")

        # Suggest similar runs
        manager = RunManager()
        all_runs = manager.list_runs()

        print("\nAvailable runs:")
        for run_dir, metadata in all_runs[:5]:  # Show first 5
            print(f"  {metadata.dataset_name}/{metadata.run_name}")

        if len(all_runs) > 5:
            print(f"  ... and {len(all_runs) - 5} more")

        return

    run_dir, metadata = result

    print(f"✅ Found run: {metadata.get_display_name()}")
    print(f"📂 Path: {run_dir}")
    print(f"🎯 Status: {metadata.status}")

    if args.verbose:
        cmd_inspect_detailed(metadata, run_dir)


def cmd_chainable(args):
    """Find runs that can be chained to a specific stage."""

    manager = RunManager()
    chainable_runs = manager.find_chainable_runs(
        target_stage_type=args.stage,
        dataset_name=args.dataset,
        model=args.model_config,
    )

    if not chainable_runs:
        print(f"❌ No runs found that can be chained to {args.stage}")
        return

    print(f"✅ Found {len(chainable_runs)} runs that can be chained to {args.stage}:\n")

    for run_dir, metadata in chainable_runs:
        final_stage = metadata.get_final_stage()

        print(f"🔗 {metadata.dataset_name}/{metadata.run_name}")
        print(f"   Final stage: {final_stage.stage_type}")
        print(f"   Model: {metadata.model}")
        if final_stage.eval_loss:
            print(f"   Eval loss: {final_stage.eval_loss:.4f}")
        print(f"   Usage: --starting_model {metadata.get_run_path()}")
        print()


def cmd_inspect(args):
    """Inspect a run in detail."""

    result = find_run(args.identifier)

    if not result:
        print(f"❌ Run not found: {args.identifier}")
        return

    run_dir, metadata = result
    cmd_inspect_detailed(metadata, run_dir)


def cmd_inspect_detailed(metadata: RunMetadata, run_dir: Path):
    """Show detailed information about a run."""

    print(f"\n{'='*60}")
    print(f"🔍 Run Details: {metadata.get_display_name()}")
    print(f"{'='*60}")

    print(f"📂 Directory: {run_dir}")
    print(f"🎯 Status: {metadata.status}")
    print(f"📅 Created: {metadata.created_at}")
    if metadata.completed_at:
        print(f"✅ Completed: {metadata.completed_at}")

    print(f"\n📋 Configuration:")
    print(f"  Model Config: {metadata.model}")
    if metadata.peft_config_name:
        print(f"  PEFT Config: {metadata.peft_config_name}")
    if metadata.pipeline_config_name:
        print(f"  Pipeline Config: {metadata.pipeline_config_name}")
    if metadata.suite_name:
        print(f"  Experiment Suite: {metadata.suite_name}")

    print(f"\n🎭 Training Stages:")
    for i, stage in enumerate(metadata.stages, 1):
        print(f"  Stage {i}: {stage.stage_type.upper()}")
        print(f"    Dataset: {stage.dataset_name}")
        print(f"    Starting Model: {stage.starting_model}")
        if stage.learning_rate:
            print(f"    Learning Rate: {stage.learning_rate}")
        if stage.num_epochs:
            print(f"    Epochs: {stage.num_epochs}")
        if stage.final_loss:
            print(f"    Final Loss: {stage.final_loss:.4f}")
        if stage.eval_loss:
            print(f"    Eval Loss: {stage.eval_loss:.4f}")
        print()

    if metadata.chain_info:
        print(f"🔗 Chaining:")
        print(f"  Parent: {metadata.chain_info.parent_run}")
        print(f"  Type: {metadata.chain_info.chain_type}")

    if metadata.tags:
        print(f"🏷️  Tags: {', '.join(metadata.tags)}")

    if metadata.wandb_run_urls:
        print(f"\n📊 W&B Runs:")
        for i, url in enumerate(metadata.wandb_run_urls):
            stage_name = (
                metadata.stages[i].stage_type
                if i < len(metadata.stages)
                else f"Stage {i+1}"
            )
            print(f"  {stage_name.upper()}: {url}")

    # Show files in directory
    files = list(run_dir.iterdir())
    if files:
        print(f"\n📁 Files ({len(files)}):")
        for file in sorted(files):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                if size_mb > 1:
                    print(f"  {file.name} ({size_mb:.1f} MB)")
                else:
                    print(f"  {file.name}")

    # Show usage examples
    print(f"\n💡 Usage Examples:")
    print(f"  # Run inference")
    print(f"  python -m scripts.inference {metadata.dataset_name}/{metadata.run_name}")

    final_stage = metadata.get_final_stage()
    if metadata.can_be_chained_to("dpo"):
        print(f"\n  # Chain to DPO")
        print(
            f"  python -m scripts.train_stage dpo_default --dataset_name {metadata.dataset_name}_dpo \\"
        )
        print(f"    --starting_model {metadata.get_run_path()}")


def main():
    parser = argparse.ArgumentParser(
        description="Manage and discover training runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser(
        "list", help="List runs with optional filtering"
    )
    list_parser.add_argument("--dataset", help="Filter by dataset name")
    list_parser.add_argument(
        "--run_type",
        choices=["single_stage", "pipeline", "suite_experiment"],
        help="Filter by run type",
    )
    list_parser.add_argument(
        "--status", choices=["running", "completed", "failed"], help="Filter by status"
    )
    list_parser.add_argument("--stage_type", help="Filter by stage type (sft, dpo)")

    # Find command
    find_parser = subparsers.add_parser("find", help="Find a specific run")
    find_parser.add_argument(
        "identifier", help="Run identifier (dataset/run_name, run_name, or path)"
    )
    find_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed information"
    )

    # Chainable command
    chainable_parser = subparsers.add_parser(
        "chainable", help="Find runs that can be chained"
    )
    chainable_parser.add_argument(
        "--stage", required=True, help="Target stage type (dpo)"
    )
    chainable_parser.add_argument("--dataset", help="Filter by dataset name")
    chainable_parser.add_argument("--model_config", help="Filter by model config")

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a run in detail")
    inspect_parser.add_argument(
        "identifier", help="Run identifier (dataset/run_name, run_name, or path)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "list":
            cmd_list(args)
        elif args.command == "find":
            cmd_find(args)
        elif args.command == "chainable":
            cmd_chainable(args)
        elif args.command == "inspect":
            cmd_inspect(args)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
