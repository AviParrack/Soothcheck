# This file defines utility data augmentations that process text files using
# regex patterns or simple text processing (no LLM required).
#
# Unlike LLM-based augmentations, these perform deterministic transformations:
# - TextCleanAugmentation: Uses Gemini to clean PDF-to-text conversions
# - TextChunkAugmentation: Splits text into numbered chunks using regex patterns
# - PunctuationSpaceAugmentation: Fixes missing spaces after punctuation marks
# - MarkdownHeaderChunkAugmentation: Splits text at top-level markdown headers

from typing import List, Dict, Any
import os
import re

from .augmentations import DataAugmentation
from ..process.text_clean import clean_full_text
from ..process.text_chunk import chunk_text_by_pattern


class TextCleanAugmentation(DataAugmentation):
    """
    Text cleaning augmentation that processes PDF-to-text conversions to remove
    spurious line breaks, page headers, page numbers, etc.

    Uses Gemini-2.5-flash-preview-05-20 with temperature=0 to process text in ~5k word chunks.
    Saves cleaned files to the "cleaned" folder.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_chunk_size = self.params.get("target_chunk_size", 5000)

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Cleans the text content of a file by processing it in chunks with Gemini.

        Args:
            file_path: The path to the original file being cleaned.
            file_content: The raw string content of the file.
            raw_data_dir: The absolute path to the raw data directory.
            **kwargs: Additional arguments (unused for this augmentation).

        Returns:
            A list containing the path to the newly created cleaned file.
        """
        if not file_content.strip():
            return []

        # Create cleaned directory if it doesn't exist
        target_folder = "augmentations/cleaned"
        cleaned_dir = os.path.join(raw_data_dir, target_folder)
        os.makedirs(cleaned_dir, exist_ok=True)

        # Generate output filename
        original_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        cleaned_filename = f"{name_without_ext}_cleaned.txt"
        output_path = os.path.join(cleaned_dir, cleaned_filename)

        try:
            # Clean the full text
            print(f"Starting text cleaning for: {original_filename}")
            cleaned_text = clean_full_text(file_content, self.target_chunk_size)

            # Write cleaned text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"Cleaned text saved to: {output_path}")
            return [output_path]

        except Exception as e:
            print(f"Error cleaning text file {file_path}: {e}")
            return []


class TextChunkAugmentation(DataAugmentation):
    """
    Pattern-based text chunking augmentation that splits text files into numbered chunks
    based on a regex pattern.

    The pattern serves as delimiters, with each match becoming the start of the next chunk.
    Useful for splitting documents by chapter headers, sections, etc.
    Saves chunked files to the "chunked" folder with numbered suffixes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pattern = self.params.get("pattern", r"\n\d+\s+")

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Chunks the text content of a file based on the configured pattern.

        Args:
            file_path: The path to the original file being chunked.
            file_content: The raw string content of the file.
            raw_data_dir: The absolute path to the raw data directory.
            **kwargs: Additional arguments (unused for this augmentation).

        Returns:
            A list of paths to the newly created chunk files.
        """
        if not file_content.strip():
            return []

        # Create chunked directory if it doesn't exist
        target_folder = "augmentations/chunked"
        chunked_dir = os.path.join(raw_data_dir, target_folder)
        os.makedirs(chunked_dir, exist_ok=True)

        # Generate base filename for chunks
        original_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_filename)[0]

        try:
            # Chunk the text based on pattern
            print(f"Starting pattern-based chunking for: {original_filename}")
            print(f"Using pattern: {self.pattern}")

            chunks = chunk_text_by_pattern(file_content, self.pattern)

            if not chunks:
                print(f"No chunks created for {original_filename}")
                return []

            created_files = []

            # Write each chunk to a separate file
            for chunk_number, chunk_content in chunks:
                chunk_filename = f"{name_without_ext}_chunk_{chunk_number:03d}.txt"
                chunk_path = os.path.join(chunked_dir, chunk_filename)

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk_content)

                created_files.append(chunk_path)
                print(f"Created chunk {chunk_number}: {chunk_filename}")

            print(
                f"Successfully created {len(created_files)} chunks from: {original_filename}"
            )
            return created_files

        except Exception as e:
            print(f"Error chunking text file {file_path}: {e}")
            return []


class PunctuationSpaceAugmentation(DataAugmentation):
    """
    Regex-based punctuation spacing augmentation that fixes missing spaces after punctuation marks.

    Identifies patterns like "word.Word" or "word?Word" and inserts the missing space to become
    "word. Word" or "word? Word". Preserves legitimate cases like decimal numbers, abbreviations,
    and URLs.

    Saves fixed files to the "cleaned" folder with "_spaced" suffix.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Pattern to match: letter + punctuation + capital letter (missing space case)
        # This pattern looks for:
        # - A word character (letter/digit)
        # - Followed by punctuation (.!?:;)
        # - Followed immediately by a capital letter (no space)
        self.punctuation_pattern = re.compile(r"(\w)([.!?:;])([A-Z])")

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Fixes missing spaces after punctuation marks in the text content.

        Args:
            file_path: The path to the original file being processed.
            file_content: The raw string content of the file.
            raw_data_dir: The absolute path to the raw data directory.
            **kwargs: Additional arguments (unused for this augmentation).

        Returns:
            A list containing the path to the newly created spaced file.
        """
        if not file_content.strip():
            return []

        # Create cleaned directory if it doesn't exist (same as TextCleanAugmentation)
        target_folder = "augmentations/cleaned"
        cleaned_dir = os.path.join(raw_data_dir, target_folder)
        os.makedirs(cleaned_dir, exist_ok=True)

        # Generate output filename
        original_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        spaced_filename = f"{name_without_ext}_spaced.txt"
        output_path = os.path.join(cleaned_dir, spaced_filename)

        try:
            # Fix punctuation spacing
            print(f"Starting punctuation spacing fix for: {original_filename}")

            # Apply the regex replacement: insert space between punctuation and capital letter
            # \1 = first group (word character)
            # \2 = second group (punctuation)
            # \3 = third group (capital letter)
            # Result: word character + punctuation + space + capital letter
            spaced_text = self.punctuation_pattern.sub(r"\1\2 \3", file_content)

            # Count the number of fixes made
            original_matches = len(self.punctuation_pattern.findall(file_content))
            if original_matches > 0:
                print(f"Fixed {original_matches} missing spaces after punctuation")
            else:
                print("No missing punctuation spaces found")

            # Write spaced text to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(spaced_text)

            print(f"Punctuation-spaced text saved to: {output_path}")
            return [output_path]

        except Exception as e:
            print(f"Error fixing punctuation spacing in file {file_path}: {e}")
            return []


class MarkdownHeaderChunkAugmentation(DataAugmentation):
    """
    Markdown header-based chunking augmentation that splits text files at top-level
    markdown headers (# Header).

    Each top-level header (starting with a single #) becomes the start of a new chunk.
    Content above the first header is included in the first chunk.
    Saves chunked files to the "chunked" folder with numbered suffixes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Chunks the text content of a file at top-level markdown headers.

        Args:
            file_path: The path to the original file being chunked.
            file_content: The raw string content of the file.
            raw_data_dir: The absolute path to the raw data directory.
            **kwargs: Additional arguments (unused for this augmentation).

        Returns:
            A list of paths to the newly created chunk files.
        """
        if not file_content.strip():
            return []

        # Create chunked directory if it doesn't exist
        target_folder = "augmentations/chunked"
        chunked_dir = os.path.join(raw_data_dir, target_folder)
        os.makedirs(chunked_dir, exist_ok=True)

        # Generate base filename for chunks
        original_filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(original_filename)[0]

        try:
            # Split content at top-level markdown headers
            print(f"Starting markdown header chunking for: {original_filename}")

            # Find all top-level headers (lines starting with exactly one #)
            lines = file_content.split("\n")
            header_indices = []

            for i, line in enumerate(lines):
                # Match lines that start with exactly one # followed by space
                if re.match(r"^# .+", line):
                    header_indices.append(i)

            if not header_indices:
                # No headers found, treat entire content as one chunk
                print(
                    f"No top-level headers found in {original_filename}, creating single chunk"
                )
                chunk_filename = f"{name_without_ext}_chunk_001.md"
                chunk_path = os.path.join(chunked_dir, chunk_filename)

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(file_content)

                print(f"Created single chunk: {chunk_filename}")
                return [chunk_path]

            created_files = []

            # Create chunks based on header positions
            for chunk_idx, start_idx in enumerate(header_indices):
                # Determine end of current chunk
                if chunk_idx + 1 < len(header_indices):
                    end_idx = header_indices[chunk_idx + 1]
                else:
                    end_idx = len(lines)

                # For the first chunk, include any content before the first header
                if chunk_idx == 0 and start_idx > 0:
                    # Include content from beginning to first header
                    chunk_lines = lines[:end_idx]
                else:
                    # Include content from current header to next header (or end)
                    chunk_lines = lines[start_idx:end_idx]

                chunk_content = "\n".join(chunk_lines).strip()

                if chunk_content:  # Only create file if there's content
                    chunk_filename = f"{name_without_ext}_chunk_{chunk_idx + 1:03d}.md"
                    chunk_path = os.path.join(chunked_dir, chunk_filename)

                    with open(chunk_path, "w", encoding="utf-8") as f:
                        f.write(chunk_content)

                    created_files.append(chunk_path)
                    print(f"Created chunk {chunk_idx + 1}: {chunk_filename}")

            print(
                f"Successfully created {len(created_files)} chunks from: {original_filename}"
            )
            return created_files

        except Exception as e:
            print(f"Error chunking markdown file {file_path}: {e}")
            return []
