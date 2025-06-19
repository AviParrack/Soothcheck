#!/usr/bin/env python
"""
Test script to demonstrate the markdown chunking functionality.
This script loads a markdown file, chunks it at natural boundaries,
and displays the resulting chunks.
"""

import argparse
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer

# Ensure src is in path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.markdown_chunker import chunk_markdown_at_headers
from src.config_models.data_config import DatasetConfig


def main():
    parser = argparse.ArgumentParser(
        description="Test markdown chunking functionality.",
        epilog="Example usage: python scripts/test_chunking.py datasets/raw/adam_dataset/7-blogs-7-days.mdx",
    )
    parser.add_argument(
        "file_path", type=str, help="Path to the markdown file to chunk"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum sequence length for chunking (default: 2048)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2b",
        help="Model name or path for tokenizer (default: google/gemma-2b)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found: {args.file_path}")
        return

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Read the file
    with open(args.file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Get token count of the original file
    token_count = len(tokenizer.encode(content))
    print(f"Original file: {args.file_path}")
    print(f"Token count: {token_count}")
    print(f"Max length: {args.max_length}")

    if token_count <= args.max_length:
        print("File is within the maximum length. No chunking needed.")
        return

    # Chunk the file
    print(f"Chunking file at markdown headers...")
    chunks = chunk_markdown_at_headers(content, tokenizer, max_length=args.max_length)

    # Display the chunks
    print(f"File was split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        chunk_tokens = len(tokenizer.encode(chunk))
        print(f"  Chunk {i+1}: {chunk_tokens} tokens")

        # Print the first few lines of each chunk
        preview_lines = chunk.split("\n")[:5]
        preview = "\n".join(preview_lines)
        if len(preview_lines) < len(chunk.split("\n")):
            preview += "\n..."
        print(f"  Preview:\n{preview}\n")


if __name__ == "__main__":
    main()
