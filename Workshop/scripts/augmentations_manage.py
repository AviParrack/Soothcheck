#!/usr/bin/env python
# This script manages augmentation files by merging them from multiple folders
# or randomly deleting lines from existing augmentation files.
#
# Example usage:
# python -m scripts.augmentations_manage brad merge augmentations_blogs augmentations_transcripts
# (merges augmentation files from augmentations_blogs and augmentations_transcripts into augmentations)
#
# python -m scripts.augmentations_manage brad delete_random 50
# (randomly deletes 50% of lines from all augmentation .jsonl files)
#
# Arguments for merge:
#   raw_dataset_name (str): Name of the raw dataset directory under datasets/raw/
#   merge (str): Command to merge augmentation files
#   source_folders (str...): Names of source augmentation folders to merge from
#
# Arguments for delete_random:
#   raw_dataset_name (str): Name of the raw dataset directory under datasets/raw/
#   delete_random (str): Command to randomly delete lines
#   percentage (int): Percentage of lines to delete (0-99)

import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def find_augmentation_files(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all .jsonl files in augmentation subfolders and group them by relative path.

    Args:
        base_dir: Path to the base directory containing augmentations folder

    Returns:
        Dictionary mapping relative paths to lists of absolute file paths
    """
    files_by_path = defaultdict(list)
    augmentations_dir = os.path.join(base_dir, "augmentations")

    if not os.path.exists(augmentations_dir):
        print(f"Warning: No augmentations directory found at {augmentations_dir}")
        return files_by_path

    # Walk through all subdirectories in augmentations
    for root, dirs, files in os.walk(augmentations_dir):
        for file in files:
            if file.endswith(".jsonl"):
                full_path = os.path.join(root, file)
                # Get relative path from augmentations directory
                rel_path = os.path.relpath(full_path, augmentations_dir)
                files_by_path[rel_path].append(full_path)

    return files_by_path


def merge_jsonl_files(input_files: List[str], output_file: str) -> int:
    """
    Merge multiple JSONL files into one output file.

    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path

    Returns:
        Number of lines written
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_lines = 0
    with open(output_file, "w", encoding="utf-8") as outf:
        for input_file in input_files:
            if os.path.exists(input_file):
                print(f"  Reading from {input_file}")
                with open(input_file, "r", encoding="utf-8") as inf:
                    for line in inf:
                        line = line.strip()
                        if line:  # Skip empty lines
                            outf.write(line + "\n")
                            total_lines += 1

    return total_lines


def merge_augmentations(raw_dataset_name: str, source_folders: List[str]) -> None:
    """
    Merge augmentation files from multiple source folders into a single augmentations folder.

    Args:
        raw_dataset_name: Name of the raw dataset
        source_folders: List of source folder names to merge from
    """
    base_raw_dir = os.path.join(os.getcwd(), "datasets", "raw", raw_dataset_name)
    target_augmentations_dir = os.path.join(base_raw_dir, "augmentations")

    if not os.path.exists(base_raw_dir):
        print(f"Error: Raw dataset directory not found: {base_raw_dir}")
        return

    print(f"Merging augmentations for dataset '{raw_dataset_name}'")
    print(f"Source folders: {', '.join(source_folders)}")
    print(f"Target directory: {target_augmentations_dir}")

    # Collect all files from all source folders
    all_files_by_path = defaultdict(list)

    for source_folder in source_folders:
        source_dir = os.path.join(base_raw_dir, source_folder)
        if not os.path.exists(source_dir):
            print(f"Warning: Source folder not found: {source_dir}")
            continue

        print(f"\nScanning source folder: {source_folder}")
        source_files = find_augmentation_files(source_dir)

        for rel_path, file_list in source_files.items():
            all_files_by_path[rel_path].extend(file_list)
            print(f"  Found {len(file_list)} files for {rel_path}")

    if not all_files_by_path:
        print("No augmentation files found to merge.")
        return

    # Merge files for each relative path
    print(f"\nMerging files:")
    total_files_created = 0
    total_lines_written = 0

    for rel_path, input_files in all_files_by_path.items():
        output_file = os.path.join(target_augmentations_dir, rel_path)
        print(f"\nMerging {len(input_files)} files into {rel_path}:")

        lines_written = merge_jsonl_files(input_files, output_file)
        print(f"  → Created {output_file} with {lines_written} lines")

        total_files_created += 1
        total_lines_written += lines_written

    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}")
    print(f"Created {total_files_created} merged files")
    print(f"Total lines written: {total_lines_written}")


def delete_random_lines(raw_dataset_name: str, percentage: int) -> None:
    """
    Randomly delete a percentage of lines from all augmentation JSONL files.

    Args:
        raw_dataset_name: Name of the raw dataset
        percentage: Percentage of lines to delete (0-99)
    """
    if percentage < 0 or percentage >= 100:
        print(f"Error: Percentage must be between 0 and 99, got {percentage}")
        return

    base_raw_dir = os.path.join(os.getcwd(), "datasets", "raw", raw_dataset_name)
    augmentations_dir = os.path.join(base_raw_dir, "augmentations")

    if not os.path.exists(augmentations_dir):
        print(f"Error: Augmentations directory not found: {augmentations_dir}")
        return

    print(
        f"Randomly deleting {percentage}% of lines from augmentation files in dataset '{raw_dataset_name}'"
    )
    print(f"Target directory: {augmentations_dir}")

    # Find all JSONL files
    jsonl_files = []
    for root, dirs, files in os.walk(augmentations_dir):
        for file in files:
            if file.endswith(".jsonl"):
                jsonl_files.append(os.path.join(root, file))

    if not jsonl_files:
        print("No JSONL files found in augmentations directory.")
        return

    print(f"Found {len(jsonl_files)} JSONL files to process")

    total_lines_before = 0
    total_lines_after = 0

    for file_path in jsonl_files:
        rel_path = os.path.relpath(file_path, augmentations_dir)
        print(f"\nProcessing {rel_path}:")

        # Read all lines
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        lines_before = len(lines)
        total_lines_before += lines_before

        if lines_before == 0:
            print(f"  Skipping empty file")
            continue

        # Calculate how many lines to keep
        lines_to_keep = int(lines_before * (100 - percentage) / 100)

        if lines_to_keep == 0:
            print(
                f"  Warning: Would delete all {lines_before} lines, keeping 1 line minimum"
            )
            lines_to_keep = 1

        # Randomly sample lines to keep
        random.shuffle(lines)
        lines_to_write = lines[:lines_to_keep]

        # Write back to file
        with open(file_path, "w", encoding="utf-8") as f:
            for line in lines_to_write:
                f.write(line + "\n")

        lines_after = len(lines_to_write)
        total_lines_after += lines_after
        deleted = lines_before - lines_after

        print(f"  Before: {lines_before} lines")
        print(f"  After: {lines_after} lines")
        print(f"  Deleted: {deleted} lines ({deleted/lines_before*100:.1f}%)")

    print(f"\n{'='*60}")
    print("RANDOM DELETION COMPLETE")
    print(f"{'='*60}")
    print(f"Processed {len(jsonl_files)} files")
    print(f"Total lines before: {total_lines_before}")
    print(f"Total lines after: {total_lines_after}")
    print(f"Total lines deleted: {total_lines_before - total_lines_after}")
    print(
        f"Overall deletion rate: {(total_lines_before - total_lines_after)/total_lines_before*100:.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Manage augmentation files by merging from multiple folders or randomly deleting lines.",
        epilog="Examples:\n"
        "  python -m scripts.augmentations_manage brad merge augmentations_blogs augmentations_transcripts\n"
        "  python -m scripts.augmentations_manage brad delete_random 50",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "raw_dataset_name",
        type=str,
        help="Name of the raw dataset directory under 'datasets/raw/' (e.g., 'brad').",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Merge command
    merge_parser = subparsers.add_parser(
        "merge", help="Merge augmentation files from multiple source folders"
    )
    merge_parser.add_argument(
        "source_folders",
        nargs="+",
        help="Names of source augmentation folders to merge from (e.g., augmentations_blogs augmentations_transcripts)",
    )

    # Delete random command
    delete_parser = subparsers.add_parser(
        "delete_random",
        help="Randomly delete a percentage of lines from augmentation files",
    )
    delete_parser.add_argument(
        "percentage", type=int, help="Percentage of lines to delete (0-99)"
    )

    args = parser.parse_args()

    if args.command == "merge":
        merge_augmentations(args.raw_dataset_name, args.source_folders)
    elif args.command == "delete_random":
        delete_random_lines(args.raw_dataset_name, args.percentage)
    else:
        parser.print_help()


if __name__ == "__main__":
    if "src" not in sys.path[0] and "src" not in "".join(sys.path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            print(f"Adjusted sys.path to include project root: {project_root}")
    main()
