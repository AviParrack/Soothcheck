"""
Module for chunking markdown documents at natural boundaries for fine-tuning.
This helps ensure that all content in long documents contributes to training,
while maintaining semantic coherence by splitting at headers.
"""

import re
from typing import List, Optional, Tuple
from transformers import PreTrainedTokenizer


def estimate_token_length(text: str, tokenizer: PreTrainedTokenizer) -> int:
    """
    Estimate the number of tokens in a text string using the provided tokenizer.

    Args:
        text: The text to tokenize
        tokenizer: The tokenizer to use

    Returns:
        The number of tokens in the text
    """
    return len(tokenizer.encode(text))


def get_header_priority(line: str) -> int:
    """
    Get priority for header line (1=highest priority for H1, 6=lowest for H6).
    
    Args:
        line: The line to analyze
        
    Returns:
        Priority number (1-6 for headers, 99 for non-headers)
    """
    match = re.match(r"^(#{1,6})\s", line.strip())
    return len(match.group(1)) if match else 99


def find_overlap_start_position(lines: List[str], overlap_tokens: int, tokenizer: PreTrainedTokenizer) -> int:
    """
    Find where overlap should start in the previous chunk by counting back from the end.
    
    Args:
        lines: List of lines in the chunk
        overlap_tokens: Target number of tokens for overlap
        tokenizer: Tokenizer for counting tokens
        
    Returns:
        Number of lines from the end to start overlap
    """
    if not lines or overlap_tokens <= 0:
        return 0
        
    current_tokens = 0
    lines_from_end = 0
    
    # Count backward from end until we reach target overlap tokens
    for i in range(len(lines) - 1, -1, -1):
        line_tokens = len(tokenizer.encode(lines[i] + "\n"))
        current_tokens += line_tokens
        lines_from_end += 1
        
        if current_tokens >= overlap_tokens:
            break
            
    # Ensure we don't overlap more than half the chunk
    max_overlap_lines = len(lines) // 2
    return min(lines_from_end, max_overlap_lines, 10)  # Cap at 10 lines


def find_best_split_point(lines: List[str], target_line_idx: int, current_start_line: int = 0, window_size: int = 20) -> int:
    """
    Find the best natural boundary near the target split position.
    
    Args:
        lines: All lines in the document
        target_line_idx: Preferred split position
        current_start_line: Starting line of the current chunk (to avoid empty chunks)
        window_size: How many lines around target to search
        
    Returns:
        Best line index to split at
    """
    # Ensure we don't go below current_start_line + 1 (to avoid empty chunks)
    # and focus search around the target position
    start_search = max(current_start_line + 1, target_line_idx - window_size)
    end_search = min(len(lines), target_line_idx + window_size)
    
    best_idx = target_line_idx
    best_priority = 99
    
    for i in range(start_search, end_search):
        if i >= len(lines):
            break
            
        line = lines[i]
        
        # Check for headers (highest priority)
        # But don't split at the current_start_line as that would create empty chunks
        if is_header_line(line) and i > current_start_line:
            priority = get_header_priority(line)
            # Prefer headers closer to the target position
            distance_penalty = abs(i - target_line_idx) * 0.1
            adjusted_priority = priority + distance_penalty
            if adjusted_priority < best_priority:
                best_priority = adjusted_priority
                best_idx = i
        
        # Check for paragraph breaks (medium priority)
        elif (line.strip() == "" and i > current_start_line and i < len(lines) - 1 and 
              lines[i-1].strip() != "" and lines[i+1].strip() != ""):
            if best_priority > 7:  # Only if no headers found
                best_priority = 7
                best_idx = i
    
    # Ensure we never return an index that would create an empty chunk
    if best_idx <= current_start_line:
        best_idx = min(target_line_idx, len(lines))
    
    return best_idx


def chunk_markdown_with_overlap(
    markdown_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    overlap_ratio: float = 0.15,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Split markdown into chunks with intelligent overlap for better context preservation.
    
    Args:
        markdown_text: The markdown text to chunk
        tokenizer: The tokenizer to use for token counting  
        max_length: Maximum tokens per chunk
        overlap_ratio: Fraction of chunk to overlap (0.0-0.5)
        min_chunk_size: Minimum tokens for a valid chunk
        
    Returns:
        List of overlapping chunks
    """
    # If no overlap requested, use original chunker
    if overlap_ratio <= 0:
        return chunk_markdown_at_headers(markdown_text, tokenizer, max_length)
    
    lines = markdown_text.split("\n")
    if not lines:
        return []
        
    chunks = []
    overlap_tokens = int(max_length * overlap_ratio)
    current_start_line = 0
    
    while current_start_line < len(lines):
        # Determine target end position based on token count
        current_tokens = 0
        target_end_line = current_start_line
        
        for i in range(current_start_line, len(lines)):
            line_tokens = len(tokenizer.encode(lines[i] + "\n"))
            if current_tokens + line_tokens > max_length and current_tokens > 0:
                target_end_line = i
                break
            current_tokens += line_tokens
            target_end_line = i + 1
            
        # If we've reached the end of the document
        if target_end_line >= len(lines):
            target_end_line = len(lines)
            
        # Find the best natural split point near our target
        actual_end_line = find_best_split_point(lines, target_end_line, current_start_line)
        
        # Extract chunk
        chunk_lines = lines[current_start_line:actual_end_line]
        if not chunk_lines:
            break
            
        chunk_text = "\n".join(chunk_lines)
        chunk_tokens = len(tokenizer.encode(chunk_text))
        
        # Only add chunk if it meets minimum size
        if chunk_tokens >= min_chunk_size:
            chunks.append(chunk_text)
            
            # Calculate overlap for next chunk (but not for the last chunk)
            if actual_end_line < len(lines):
                overlap_lines = find_overlap_start_position(chunk_lines, overlap_tokens, tokenizer)
                next_start = actual_end_line - overlap_lines
                
                # Ensure we make progress - don't get stuck
                if next_start <= current_start_line:
                    next_start = current_start_line + max(1, len(chunk_lines) // 2)
                    
                current_start_line = next_start
            else:
                break
        else:
            # Chunk too small, try to make progress
            current_start_line = min(current_start_line + max(1, len(chunk_lines) // 2), 
                                   actual_end_line)
            
        # Safety check to prevent infinite loops
        if current_start_line >= len(lines):
            break
            
    return chunks


def is_header_line(line: str) -> bool:
    """
    Check if a line is a markdown header.

    Args:
        line: The line to check

    Returns:
        True if the line is a markdown header, False otherwise
    """
    # Match lines that start with one or more # characters followed by a space
    return bool(re.match(r"^#{1,6}\s", line.strip()))


def chunk_markdown_at_headers(
    markdown_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> List[str]:
    """
    Split a markdown document into chunks at header boundaries, ensuring each chunk
    stays under the maximum token length.

    Args:
        markdown_text: The markdown text to chunk
        tokenizer: The tokenizer to use for estimating token lengths
        max_length: The maximum token length for each chunk

    Returns:
        A list of markdown chunks
    """
    # Split the text into lines
    lines = markdown_text.split("\n")

    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        # Check if this is a header line
        is_header = is_header_line(line)

        # Tokenize the line to get its token length (include newline)
        line_tokens = tokenizer.encode(line + "\n")
        line_length = len(line_tokens)

        # If adding this line would exceed max_length AND it's a header (natural break point)
        # OR if we're already at max length (e.g. a very long line of code without headers)
        # The check is simplified as we no longer need to account for marker_length.
        if (
            (current_length + line_length > max_length and is_header)
            or (
                current_length + line_length > max_length
                and not current_chunk  # Avoid tiny first chunk if first line is too long
            )
            or (current_length > 0 and current_length + line_length > max_length)
        ):  # General case: current chunk + new line > max
            # Finalize current chunk
            if (
                current_chunk
            ):  # Ensure we don't add an empty chunk if the first line itself is a new chunk point
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)

            # Start a new chunk
            current_chunk = [line]
            current_length = line_length
        else:
            # Add line to current chunk
            current_chunk.append(line)
            current_length += line_length

    # Add the final chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def chunk_file_content(
    file_content: str,
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
) -> List[Tuple[str, str]]:
    """
    Process a file's content, chunking it if necessary based on token length.

    Args:
        file_content: The content of the file
        file_path: The path to the file (for reference in the returned tuples)
        tokenizer: The tokenizer to use for estimating token lengths
        max_length: The maximum token length for each chunk

    Returns:
        A list of tuples (chunk_content, source_info) where source_info indicates
        the original file and chunk number
    """
    # Estimate token length
    tokens = tokenizer.encode(file_content)

    if len(tokens) <= max_length:
        # No chunking needed
        return [(file_content, file_path)]

    # Apply chunking for markdown files
    if file_path.endswith((".md", ".mdx", ".markdown")):
        chunks = chunk_markdown_at_headers(file_content, tokenizer, max_length)
    else:
        # For non-markdown files, use a simpler chunking approach:
        # Split by lines and then group lines into chunks.
        # This is a basic strategy and might need refinement for specific non-markdown formats.
        lines = file_content.split("\n")
        chunks = []
        current_chunk_lines = []
        current_token_count = 0
        for line in lines:
            line_token_count = estimate_token_length(line + "\n", tokenizer)
            if (
                current_token_count + line_token_count > max_length
                and current_chunk_lines
            ):
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [line]
                current_token_count = line_token_count
            else:
                current_chunk_lines.append(line)
                current_token_count += line_token_count
        if current_chunk_lines:  # Add any remaining lines
            chunks.append("\n".join(current_chunk_lines))

        if (
            not chunks and file_content
        ):  # If file_content is not empty but no chunks were made (e.g. single very long line)
            # Fallback: take the first max_length tokens if line-based chunking fails for non-markdown
            print(
                f"Warning: Non-markdown file {file_path} could not be effectively line-chunked. Truncating to max_length tokens."
            )
            encoded_content = tokenizer.encode(
                file_content, max_length=max_length, truncation=True
            )
            chunks = [tokenizer.decode(encoded_content, skip_special_tokens=True)]

        if len(chunks) > 1:
            print(
                f"Note: Non-markdown file {file_path} was split into {len(chunks)} generic chunks."
            )
        elif not chunks and file_content:
            print(
                f"Warning: Non-markdown file {file_path} is very long but was not chunked. Check its content."
            )
            # If still no chunks, and there was content, we take the original content up to max_length as a single chunk.
            # This might happen if the file is one extremely long line with no newlines.
            text_to_encode = file_content
            encoded_tokens = tokenizer.encode(
                text_to_encode, truncation=True, max_length=max_length
            )
            decoded_chunk = tokenizer.decode(encoded_tokens, skip_special_tokens=True)
            chunks = [decoded_chunk]

    # Create tuples with source information
    result = []
    if (
        not chunks and file_content
    ):  # If, after all attempts, chunks is empty but there was content
        print(
            f"Warning: File {file_path} resulted in no chunks despite having content. Using original content (potentially truncated)."
        )
        # This is a safeguard. Ideally, previous logic should handle all cases.
        # We'll take the initial part of the file up to max_length.
        encoded_tokens = tokenizer.encode(
            file_content, truncation=True, max_length=max_length
        )
        decoded_chunk = tokenizer.decode(encoded_tokens, skip_special_tokens=True)
        result.append((decoded_chunk, f"{file_path} [chunk 1/1 - fallback truncation]"))
    else:
        for i, chunk in enumerate(chunks):
            source_info = f"{file_path} [chunk {i+1}/{len(chunks)}]"
            result.append((chunk, source_info))

    if not result and file_content:  # Final fallback if result is still empty
        print(
            f"Critical Warning: File {file_path} had content but produced no processable chunks. This should not happen."
        )
        # Add the original content as a single item to avoid downstream errors, though it might be too long.
        result.append((file_content, file_path + " [problematic processing]"))

    return result


def chunk_file_content_with_overlap(
    file_content: str,
    file_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    overlap_ratio: float = 0.15,
    min_chunk_size: int = 50,
) -> List[Tuple[str, str]]:
    """
    Process a file's content with overlapping chunks for better context preservation.

    Args:
        file_content: The content of the file
        file_path: The path to the file (for reference in the returned tuples)
        tokenizer: The tokenizer to use for estimating token lengths
        max_length: The maximum token length for each chunk
        overlap_ratio: The ratio of overlap between chunks (0.0-0.5)
        min_chunk_size: Minimum tokens for a valid chunk

    Returns:
        A list of tuples (chunk_content, source_info) where source_info indicates
        the original file and chunk number with overlap info
    """
    # Estimate token length
    tokens = tokenizer.encode(file_content)

    if len(tokens) <= max_length:
        # No chunking needed
        return [(file_content, file_path)]

    # Apply chunking for markdown files with overlap
    if file_path.endswith((".md", ".mdx", ".markdown")):
        chunks = chunk_markdown_with_overlap(
            file_content, tokenizer, max_length, overlap_ratio, min_chunk_size
        )
    else:
        # For non-markdown files, use simpler line-based overlap
        chunks = chunk_text_lines_with_overlap(
            file_content, tokenizer, max_length, overlap_ratio, min_chunk_size
        )

    # Create tuples with source information including overlap info
    result = []
    overlap_suffix = f" [overlap:{overlap_ratio:.0%}]" if overlap_ratio > 0 else ""
    
    for i, chunk in enumerate(chunks):
        source_info = f"{file_path} [chunk {i+1}/{len(chunks)}{overlap_suffix}]"
        result.append((chunk, source_info))

    # Fallback if no chunks produced
    if not result and file_content:
        print(f"Warning: File {file_path} with overlap chunking produced no chunks. Using fallback.")
        encoded_tokens = tokenizer.encode(
            file_content, truncation=True, max_length=max_length
        )
        decoded_chunk = tokenizer.decode(encoded_tokens, skip_special_tokens=True)
        result.append((decoded_chunk, f"{file_path} [fallback chunk]"))

    return result


def chunk_text_lines_with_overlap(
    text_content: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    overlap_ratio: float = 0.15,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Simple line-based chunking with overlap for non-markdown files.

    Args:
        text_content: The text to chunk
        tokenizer: Tokenizer for token counting
        max_length: Maximum tokens per chunk
        overlap_ratio: Fraction of chunk to overlap
        min_chunk_size: Minimum tokens for a valid chunk

    Returns:
        List of overlapping text chunks
    """
    if overlap_ratio <= 0:
        # Fall back to original logic for non-markdown
        lines = text_content.split("\n")
        chunks = []
        current_chunk_lines = []
        current_token_count = 0
        
        for line in lines:
            line_token_count = estimate_token_length(line + "\n", tokenizer)
            if current_token_count + line_token_count > max_length and current_chunk_lines:
                chunks.append("\n".join(current_chunk_lines))
                current_chunk_lines = [line]
                current_token_count = line_token_count
            else:
                current_chunk_lines.append(line)
                current_token_count += line_token_count
                
        if current_chunk_lines:
            chunks.append("\n".join(current_chunk_lines))
        return chunks

    lines = text_content.split("\n")
    if not lines:
        return []

    chunks = []
    overlap_tokens = int(max_length * overlap_ratio)
    current_start_line = 0

    while current_start_line < len(lines):
        # Build chunk up to max_length tokens
        current_tokens = 0
        end_line = current_start_line

        for i in range(current_start_line, len(lines)):
            line_tokens = len(tokenizer.encode(lines[i] + "\n"))
            if current_tokens + line_tokens > max_length and current_tokens > 0:
                end_line = i
                break
            current_tokens += line_tokens
            end_line = i + 1

        # Extract chunk
        chunk_lines = lines[current_start_line:end_line]
        if not chunk_lines:
            break

        chunk_text = "\n".join(chunk_lines)
        chunk_tokens = len(tokenizer.encode(chunk_text))

        # Only add if meets minimum size
        if chunk_tokens >= min_chunk_size:
            chunks.append(chunk_text)

            # Calculate overlap for next chunk
            if end_line < len(lines):
                overlap_lines = find_overlap_start_position(chunk_lines, overlap_tokens, tokenizer)
                next_start = end_line - overlap_lines

                # Ensure progress
                if next_start <= current_start_line:
                    next_start = current_start_line + max(1, len(chunk_lines) // 2)

                current_start_line = next_start
            else:
                break
        else:
            # Skip small chunk and make progress
            current_start_line = min(current_start_line + max(1, len(chunk_lines) // 2), end_line)

        # Safety check
        if current_start_line >= len(lines):
            break

    return chunks
