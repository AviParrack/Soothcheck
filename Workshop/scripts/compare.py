#!/usr/bin/env python
# This script runs pairwise comparisons between model responses on an evaluation dataset
# It supports both LLM-based judges (Gemini) and human judges, with interruptible sessions
# Results are saved with ELO ratings and win/loss statistics
#
# Example usage:
# python -m scripts.compare example_basic                    # Uses Gemini judge by default
# python -m scripts.compare example_basic --judge human      # Uses human judge via CLI
# python -m scripts.compare example_basic --judge gemini     # Explicitly use Gemini
# python -m scripts.compare example_basic --max-comparisons 10  # Limit comparisons (for testing)

import argparse
import sys
from pathlib import Path

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.compare_models import run_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare model responses on an evaluation dataset using pairwise judgments.",
        epilog="""Examples:
  python -m scripts.compare example_basic                    # Uses Gemini judge
  python -m scripts.compare example_basic --judge human      # Uses human judge
  python -m scripts.compare example_basic --max-comparisons 10  # Limit comparisons for testing
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "dataset_name",
        type=str,
        help="Name of the evaluation dataset to compare responses for",
    )

    # Optional arguments
    parser.add_argument(
        "--judge",
        type=str,
        choices=["human", "gemini"],
        default="gemini",
        help="Type of judge to use (default: gemini)",
    )

    parser.add_argument(
        "--max-comparisons",
        type=int,
        help="Maximum number of comparisons to run (useful for testing)",
    )

    args = parser.parse_args()

    # Map judge argument to internal judge type
    judge_type_map = {
        "gemini": "gemini-2.0-pro",
        "human": "human",
    }
    judge_type = judge_type_map[args.judge]

    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON: {args.dataset_name}")
    print(f"{'='*80}")
    print(f"Judge type: {judge_type}")

    if args.max_comparisons:
        print(f"Max comparisons: {args.max_comparisons}")

    # Check if responses exist for this dataset
    responses_dir = PROJECT_ROOT / "evals" / "model_responses" / args.dataset_name
    if not responses_dir.exists():
        print(f"\nError: No model responses found for dataset '{args.dataset_name}'")
        print(f"Expected directory: {responses_dir}")
        print("\nAvailable datasets with responses:")

        model_responses_dir = PROJECT_ROOT / "evals" / "model_responses"
        if model_responses_dir.exists():
            available = [d.name for d in model_responses_dir.iterdir() if d.is_dir()]
            if available:
                for dataset in sorted(available):
                    print(f"  - {dataset}")
            else:
                print("  (none found)")

        return 1

    # Run the comparison
    try:
        comparison_results = run_comparison(
            dataset_name=args.dataset_name,
            judge_type=judge_type,
            max_comparisons=args.max_comparisons,
        )

        if comparison_results:
            print("\n" + "=" * 80)
            print("COMPARISON COMPLETE!")
            print("=" * 80)

            grading_dir = PROJECT_ROOT / "evals" / "grading" / args.dataset_name
            print(f"\nResults saved to: {grading_dir}")
            print(f"  - Session file: session_{judge_type}.json")
            print(f"  - Results file: comparison_results_{judge_type}.json")
            print(f"  - ELO chart: elo_chart_{judge_type}.png")

            return 0
        else:
            # Comparison was interrupted or aborted
            return 1

    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user.")
        print("Progress has been saved. Run the same command to resume.")
        return 1
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
