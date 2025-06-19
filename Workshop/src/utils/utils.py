# This file contains utility functions for the project.
# Major functions:
# - get_torch_dtype: Converts a string representation of a torch dtype to the actual torch.dtype object.
# - long_string_display: Displays a long string, formatted for readability in a console or notebook.
# - full_string_display: Displays a long string in its entirety, with line wrapping and optional indentation for the content.
# - resolve_model_specification: Unified model resolution for configs and experiments.
# - load_model_config_from_spec: Legacy compatibility function for model config loading.
import torch


def get_torch_dtype(dtype_str: str | None) -> torch.dtype | str | None:
    """
    Converts a string representation of a torch dtype to the actual torch.dtype object.
    Allows 'auto' for automatic selection.
    """
    if dtype_str == "auto":
        return "auto"  # Should be handled by a model's from_pretrained or similar
    if (
        dtype_str == "bfloat16"
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    if dtype_str == "float16" and torch.cuda.is_available():
        return torch.float16
    if dtype_str:
        try:
            return getattr(torch, dtype_str)
        except AttributeError:
            print(
                f"Warning: Unrecognized torch_dtype string '{dtype_str}'. Returning None."
            )
            return None
    return None


def long_string_display(
    s: str, col_width: int = 80, max_lines_preview: int = 5, indent_level: int = 0
) -> str:
    """
    Displays a long string:
    - Splits the original string by newlines.
    - If the number of lines exceeds 2 * max_lines_preview, it shows the first
      `max_lines_preview` lines, an omission message, and the last `max_lines_preview` lines.
    - For each of these selected lines, if a line's length exceeds `col_width`,
      it's wrapped into multiple sub-lines of at most `col_width`.
    - Prepends `indent_level` spaces to each displayed line.
    - Appends a character count at the end.
    """
    if not s:
        return "====BEGIN====\n\n====END====[0 chars]"

    indent_prefix = " " * indent_level
    original_lines = s.split("\n")
    num_original_lines = len(original_lines)

    lines_to_process = []
    if num_original_lines > 2 * max_lines_preview and max_lines_preview > 0:
        lines_to_process.extend(original_lines[:max_lines_preview])
        lines_to_process.append(
            f"... ({num_original_lines - 2 * max_lines_preview} lines omitted between start and end) ..."
        )
        lines_to_process.extend(original_lines[-max_lines_preview:])
    else:
        lines_to_process.extend(original_lines)

    processed_lines = []
    for line in lines_to_process:
        if len(line) <= col_width:
            processed_lines.append(indent_prefix + line)
        else:
            # Wrap long lines
            chunks = [line[i : i + col_width] for i in range(0, len(line), col_width)]
            for chunk in chunks:
                processed_lines.append(indent_prefix + chunk)

    # Join all lines and add the character count
    result = "\n".join(processed_lines)
    # The BEGIN/END markers and char count should not be indented as part of the content block.
    return f"====BEGIN====\n{result}\n====END====\n[{len(s)} chars]"


def full_string_display(
    s: str, col_width: int = 80, content_indent_str: str = ""
) -> str:
    """
    Displays a long string in its entirety, with line wrapping and optional indentation for the content.
    - Splits the original string by newlines.
    - For each line, if its length exceeds `col_width`, it's wrapped into multiple sub-lines.
    - Prepends `content_indent_str` to each displayed line of the string content.
    - Encloses the content with BEGIN/END markers and appends a character count.
    """
    if not s:
        # Consistent with long_string_display for empty strings, but no indent needed for empty content area.
        return "====BEGIN====\n\n====END====[0 chars]"

    original_lines = s.split("\n")
    processed_content_lines = []

    for line in original_lines:
        if not line:  # Preserve empty lines from original content, indented
            processed_content_lines.append(content_indent_str)
            continue

        if len(line) <= col_width:
            processed_content_lines.append(content_indent_str + line)
        else:
            # Wrap long lines
            current_pos = 0
            while current_pos < len(line):
                # For the first chunk of a wrapped line, use the provided indent.
                # For subsequent chunks of the same original line, also use the same indent.
                chunk = line[current_pos : current_pos + col_width]
                processed_content_lines.append(content_indent_str + chunk)
                current_pos += col_width

    result_content = "\n".join(processed_content_lines)
    return f"====BEGIN====\n{result_content}\n====END====\n[{len(s)} chars]"
