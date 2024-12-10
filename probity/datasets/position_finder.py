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
    ) -> int:
        """Convert a character position to a token position.

        Args:
            position: Character position to convert
            text: The full text string
            tokenizer: Tokenizer to use for conversion
            space_precedes_token: Whether spaces are included with the following token (True)
                                or the preceding token (False). Most tokenizers use True.

        Returns:
            Token position corresponding to the character position
        """
        # If position is at start, no need for space adjustment
        if position.start == 0:
            prefix = ""
        else:
            # Find the last space before the position
            prefix_end = position.start
            if space_precedes_token:
                # Include the previous space if it exists
                space_pos = text.rfind(" ", 0, position.start)
                if space_pos != -1:
                    prefix_end = space_pos
            
            prefix = text[:prefix_end]
            #print(f"Extracted prefix: '{prefix}'")

        # Check if tokenizer adds special tokens at start
        special_tokens_count = len(tokenizer("", add_special_tokens=True).input_ids) - len(tokenizer("", add_special_tokens=False).input_ids)
        #print(f"Number of special tokens added by tokenizer: {special_tokens_count}")

        # Access input_ids directly as a list
        prefix_tokens = tokenizer(prefix, add_special_tokens=False).input_ids
        #print(f"Prefix tokens: {prefix_tokens}")
        #print(f"Number of prefix tokens: {len(prefix_tokens)}")

        final_position = len(prefix_tokens) + special_tokens_count
        #print(f"Final token position: {final_position}")
        return final_position

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
