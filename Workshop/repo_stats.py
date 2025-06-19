#!/usr/bin/env python3
"""Repository analysis script - run from repo root."""

import os
from pathlib import Path
from collections import defaultdict


def count_lines(file_path):
    """Count non-empty lines in a file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return len([line for line in f if line.strip()])
    except:
        return 0


def get_dir_size(path):
    """Get total size of directory in bytes."""
    total = 0
    try:
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    total += os.path.getsize(os.path.join(root, file))
                except:
                    pass
    except:
        pass
    return total


def format_bytes(bytes_size):
    """Format bytes into human readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f}TB"


def build_tree_structure(files_with_lines):
    """Build nested tree structure from file paths and line counts."""
    tree = defaultdict(
        lambda: {
            "lines": 0,
            "children": defaultdict(lambda: {"lines": 0, "children": {}}),
        }
    )

    for file_path, lines in files_with_lines:
        parts = file_path.parts
        if len(parts) == 1:  # Root level file
            tree["_root"]["lines"] += lines
        else:
            current = tree
            for i, part in enumerate(parts[:-1]):  # Exclude filename
                current[part]["lines"] += lines
                if i < len(parts) - 2:  # Not the last directory
                    current = current[part]["children"]

    return tree


def print_tree(tree, prefix="", is_source=False):
    """Print tree structure with proper formatting."""
    items = sorted(tree.items(), key=lambda x: x[1]["lines"], reverse=True)

    for i, (name, data) in enumerate(items):
        is_last = i == len(items) - 1

        # Format the name
        display_name = "[root files]" if name == "_root" else name

        # Choose the right connector
        if is_source:
            connector = "  │  └─" if is_last else "  │  ├─"
        else:
            connector = "└─" if is_last else "├─"

        print(f"{prefix}{connector} {display_name}: {data['lines']:,}")

        # Print children if they exist
        if data["children"]:
            child_prefix = prefix + ("  │  " if not is_last else "     ")
            if is_source:
                child_prefix = prefix + ("  │     " if not is_last else "       ")
            print_children(data["children"], child_prefix)


def print_children(children, prefix):
    """Print children with proper tree formatting."""
    items = sorted(children.items(), key=lambda x: x[1]["lines"], reverse=True)

    for i, (name, data) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└─" if is_last else "├─"

        print(f"{prefix}{connector} {name}: {data['lines']:,}")

        if data["children"]:
            child_prefix = prefix + ("   " if is_last else "│  ")
            print_children(data["children"], child_prefix)


def analyze_repo():
    root = Path(".")

    # Python code analysis
    py_lines = defaultdict(int)
    src_files = []
    config_lines = 0

    # Define patterns
    config_exts = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}

    # Walk through relevant directories
    code_dirs = ["src", "scripts", "notebooks"]
    config_dirs = ["configs"]

    for dir_name in code_dirs:
        if (root / dir_name).exists():
            for file_path in (root / dir_name).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() == ".py":
                    lines = count_lines(file_path)

                    if dir_name == "notebooks":
                        py_lines["notebooks"] += lines
                    elif dir_name == "src":
                        relative_path = file_path.relative_to(root / "src")
                        src_files.append((relative_path, lines))
                        py_lines["source"] += lines
                    elif dir_name == "scripts":
                        py_lines["scripts"] += lines

    # Handle notebooks (.ipynb files)
    if (root / "notebooks").exists():
        for file_path in (root / "notebooks").rglob("*.ipynb"):
            if file_path.is_file():
                py_lines["notebooks"] += count_lines(file_path)

    # Config analysis
    for dir_name in config_dirs:
        if (root / dir_name).exists():
            for file_path in (root / dir_name).rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in config_exts:
                    config_lines += count_lines(file_path)

    # Dataset analysis
    datasets_dir = root / "datasets"
    raw_count = processed_count = 0
    raw_size = processed_size = 0

    if datasets_dir.exists():
        raw_dir = datasets_dir / "raw"
        processed_dir = datasets_dir / "processed"

        if raw_dir.exists():
            raw_count = len([d for d in raw_dir.iterdir() if d.is_dir()])
            raw_size = get_dir_size(raw_dir)

        if processed_dir.exists():
            processed_count = len([d for d in processed_dir.iterdir() if d.is_dir()])
            processed_size = get_dir_size(processed_dir)

    # Build source tree
    src_tree = build_tree_structure(src_files)

    # Report
    total_py = sum(py_lines.values())

    print("📊 REPO ANALYSIS")
    print("=" * 40)
    print(f"Python Code Lines: {total_py:,}")

    # Source breakdown with full nested structure
    print(f"  ├─ Source: {py_lines['source']:,}")
    if src_tree:
        print_tree(src_tree, is_source=True)

    print(f"  ├─ Scripts: {py_lines['scripts']:,}")
    print(f"  └─ Notebooks: {py_lines['notebooks']:,}")
    print()
    print(f"Config Lines: {config_lines:,}")
    print()
    print(f"Datasets: {raw_count + processed_count} total")
    print(f"  ├─ Raw: {raw_count} ({format_bytes(raw_size)})")
    print(f"  └─ Processed: {processed_count} ({format_bytes(processed_size)})")


if __name__ == "__main__":
    analyze_repo()
