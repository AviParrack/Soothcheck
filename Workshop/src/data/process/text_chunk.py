# Text chunking utilities
#
# This module provides functions for chunking text based on regex patterns.
# Main function: chunk_text_by_pattern - splits text into numbered chunks based on regex

import re
from typing import List, Tuple


def chunk_text_by_pattern(text: str, pattern: str) -> List[Tuple[int, str]]:
    """
    Chunks text based on a regex pattern, with each chunk labeled with increasing numbers.

    The pattern matches serve as delimiters that become the start of the next chunk.
    Everything before the first match (if not empty) becomes chunk 1.

    Args:
        text: The input text to chunk
        pattern: Regex pattern to split on (e.g., r'\n\d+\s+' for newline-number-space)

    Returns:
        List of tuples (chunk_number, chunk_content) starting from 1

    Example:
        >>> text = "Introduction\\n1. First Chapter\\nContent here\\n2. Second Chapter\\nMore content"
        >>> chunks = chunk_text_by_pattern(text, r'\\n\\d+\\.')
        >>> # Returns: [(1, "Introduction"), (2, "1. First Chapter\\nContent here"), (3, "2. Second Chapter\\nMore content")]
    """
    if not text.strip():
        return []

    # Find all matches with their positions
    matches = list(re.finditer(pattern, text))

    if not matches:
        # No pattern found, return entire text as single chunk
        return [(1, text)]

    chunks = []
    chunk_number = 1

    # Handle content before first match
    first_match_start = matches[0].start()
    if first_match_start > 0:
        pre_content = text[:first_match_start].strip()
        if pre_content:
            chunks.append((chunk_number, pre_content))
            chunk_number += 1

    # Process each chunk from pattern match to next pattern match (or end)
    for i, match in enumerate(matches):
        start_pos = match.start()

        # Determine end position
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)

        chunk_content = text[start_pos:end_pos].strip()
        if chunk_content:
            chunks.append((chunk_number, chunk_content))
            chunk_number += 1

    return chunks
