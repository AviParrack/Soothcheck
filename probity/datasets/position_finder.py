from typing import Callable, List, Union, Optional
import re
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class Position:
    """Represents a position or span in text."""

    start: int  # Character start position
    end: Optional[int] = None  # Character end position (exclusive)

    def __post_init__(self):
        if self.start < 0:
            raise ValueError("Start position must be non-negative")
        if self.end is not None and self.end < self.start:
            raise ValueError("End position must be greater than start position")


class PositionFinder:
    """Strategies for finding target positions in text."""

    @staticmethod
    def from_template(template: str, marker: str) -> Callable[[str], Position]:
        """Create finder for template-based positions.

        Args:
            template: Template string with marker (e.g. "The movie was {ADJ}")
            marker: The marker to find (e.g. "{ADJ}")

        Returns:
            Function that finds the position in a string matching the template
        """
        def finder(text: str) -> Position:
            # First escape the entire template
            regex = re.escape(template)
            
            # Then replace the escaped marker with a capture group
            escaped_marker = re.escape(marker)
            regex = regex.replace(escaped_marker, "(.*?)")
            
            # Replace other template variables with non-capturing wildcards
            for var in re.findall(r'\\\{([^}]+)\\\}', regex):
                var_marker = f"\\{{{var}\\}}"
                regex = regex.replace(var_marker, ".*?")
            # print(f"Template: {template}")
            # print(f"Marker: {marker}")
            # print(f"Regex: {regex}")
            # print(f"Text: {text}")
            match = re.match(regex, text)
            if not match:
                raise ValueError(f"Text does not match template: {text}")

            # Get the position of the matching group
            start = match.start(1)
            end = match.end(1)
            return Position(start, end)

        return finder

    @staticmethod
    def from_regex(pattern: str, group: int = 0) -> Callable[[str], List[Position]]:
        """Create finder for regex-based positions.

        Args:
            pattern: Regex pattern to match
            group: Which capture group to use for position (0 = full match)

        Returns:
            Function that finds all matching positions in a string
        """
        compiled = re.compile(pattern)

        def finder(text: str) -> List[Position]:
            positions = []
            for match in compiled.finditer(text):
                start = match.start(group)
                end = match.end(group)
                positions.append(Position(start, end))
            return positions

        return finder

    @staticmethod
    def from_char_position(pos: int) -> Callable[[str], Position]:
        """Create finder for fixed character positions.

        Args:
            pos: Character position to find

        Returns:
            Function that returns the fixed position
        """

        def finder(text: str) -> Position:
            if pos >= len(text):
                raise ValueError(f"Position {pos} is beyond text length {len(text)}")
            return Position(pos)

        return finder

    @staticmethod
    def convert_to_token_position(
        position: Position,
        text: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        space_precedes_token: bool = True,
        add_special_tokens: bool = True,
    ) -> int:
        """
        Convert character position to token position, handling special tokens properly.
        
        Args:
            position: Character position to convert
            text: Input text
            tokenizer: Tokenizer to use
            space_precedes_token: Whether space precedes token (default: True)
            add_special_tokens: Whether to add special tokens (default: True)
            
        Returns:
            Token index corresponding to the character position
        """
        # 1) Tokenize with offset mapping
        encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
        offsets = encoding.offset_mapping  # List of (start_char, end_char) pairs
        
        # 2) Find the token whose span covers position.start
        for token_idx, (start_char, end_char) in enumerate(offsets):
            if start_char <= position.start < end_char:
                return token_idx
        
        # If not found, either raise an error or return -1 to indicate position not found:
        raise ValueError(f"Character position {position.start} not aligned with any token offset.")

    @staticmethod
    def validate_token_position(token_position: int, tokens: List[int]) -> bool:
        """Validate that a token position is valid for a sequence.

        Args:
            token_position: Position to validate
            tokens: Token sequence

        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= token_position < len(tokens)
