# This file defines the data processing pipeline interface and basic implementations.
#
# Data pipelines are responsible for taking raw file content (as strings) and
# transforming it into a list of structured samples.
# For "chat" format, this means a list of role/content dictionaries.
# For "text" format, this means a string.
# The actual application of chat templates and tokenization to IDs typically happens
# later in the training process (e.g., by SFTTrainer).
#
# Key Classes:
# - DataPipeline (ABC): Defines the interface for all data pipelines.
#   Requires a `process` method.
# - BasicPipeline: The default pipeline that handles chunking of text files
#   (like .txt, .md) and structuring them for either text or chat format.
# - ContinuesPipeline: A pipeline that creates "continue the text" style prompts,
#   where a segment of text is provided as a prompt, and the subsequent
#   text from the file forms the completion.
# - QADataPipeline: A pipeline that processes .jsonl files containing QA pairs
#   and converts them to chat or text format.
# - ContrastPairsPipeline: A pipeline that processes .jsonl files containing
#   prompt/response_accepted/response_rejected triplets for preference learning.
# - ConversationPipeline: A pipeline that processes .jsonl files containing
#   multi-turn conversations with "messages" field containing role/content pairs.
#   Always outputs in chat format.
# - AVAILABLE_PIPELINES (dict): Maps pipeline names to their classes.
# - get_pipeline_by_name (function): Instantiates a pipeline by its name.

from abc import ABC, abstractmethod

import json
import os
from typing import List, Dict, Any, Union, Optional
import random  # Added for ContinuesPipeline

from transformers import PreTrainedTokenizer

from .markdown_chunker import chunk_file_content, chunk_file_content_with_overlap

# Assuming DatasetConfig might be needed for pipeline-specific configs in the future,
# but for now, BasicPipeline takes parameters directly.
# from ..config_models.data_config import DatasetConfig


class DataPipeline(ABC):
    """
    Abstract base class for data processing pipelines.
    A pipeline takes a list of file contents and transforms them into processed samples.
    """

    @abstractmethod
    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int] = None,
        enable_chunking: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Processes the content of a single file.

        Args:
            file_path: The path to the file being processed (for source tracking).
            file_content: The raw string content of the file.
            tokenizer: The Hugging Face tokenizer (used for chunking, not for templating here).
            prompt_format: The desired output format ("text" or "chat").
            max_length: Maximum token length for chunking (if applicable).
            enable_chunking: Whether to enable chunking for this file.
            **kwargs: Additional arguments for specific pipeline implementations.

        Returns:
            A list of dictionaries. Each dictionary contains a "source" key.
            If prompt_format is "text", it also contains a "text" key with the string content.
            If prompt_format is "chat", it also contains a "messages" key with a list of
            role/content dictionaries.
        """
        pass


class BasicPipeline(DataPipeline):
    """
    A basic data pipeline that handles text-based files (e.g., .txt, .md):
        - If chunking is enabled, chunks the content.
        - If chunking is disabled, uses the entire file content.
    The content is then structured for either "text" or "chat" format.
    For "chat" format, it produces `[{"role": "user", ...}, {"role": "assistant", ...}]`.
    For "text" format, it produces the raw string.
    """

    def _format_item(
        self,
        text_content: str,
        source_info: str,
        prompt_format: str,
        filename_for_logging: str,
        item_description: str,
    ) -> Optional[Dict[str, Any]]:
        """Helper to format a single piece of text content based on prompt_format.

        Output Structure:
        - For "chat" format: Returns `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}], "source": ...}`.
          The tokenizer.apply_chat_template() is *not* called here; that is deferred to the training stage.
        - For "text" format: Returns `{"text": raw_text_content, "source": ...}`.
          If chunking was enabled, the tokenizer was already used by `chunk_file_content`
          to determine appropriate chunk boundaries based on token lengths.
        """
        if prompt_format == "text":
            return {"text": text_content, "source": source_info}
        elif prompt_format == "chat":
            messages = [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": text_content},
            ]
            return {"messages": messages, "source": source_info}
        else:
            print(
                f"      BasicPipeline Warning: Unknown prompt_format '{prompt_format}' for {item_description} from {filename_for_logging}. Skipping."
            )
            return None

    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int],
        enable_chunking: bool,
        chunk_overlap_ratio: float = 0.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Processes content from a text-based file (e.g., .txt, .md)."""
        processed_items: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        if not file_content.strip():
            print(f"      BasicPipeline: Skipping empty text file: {filename}")
            return []

        if enable_chunking and max_length:
            # Use overlapping chunker if overlap_ratio > 0
            if chunk_overlap_ratio > 0:
                chunks = chunk_file_content_with_overlap(
                    file_content=file_content,
                    file_path=file_path,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    overlap_ratio=chunk_overlap_ratio,
                )
                print(f"      BasicPipeline: Using overlapping chunker ({chunk_overlap_ratio:.0%} overlap) for {filename}")
            else:
            chunks = chunk_file_content(
                file_content=file_content,
                file_path=file_path,
                tokenizer=tokenizer,
                max_length=max_length,
            )
            
            if not chunks:
                print(
                    f"      BasicPipeline: No chunks produced for {filename}. Original content might be too short."
                )

            for chunk_idx, (chunk_text, source_info) in enumerate(chunks):
                formatted_item = self._format_item(
                    text_content=chunk_text,
                    source_info=source_info,
                    prompt_format=prompt_format,
                    filename_for_logging=filename,
                    item_description=f"chunk {chunk_idx}",
                )
                if formatted_item:
                    processed_items.append(formatted_item)
        else:  # Not chunking text-based files
            formatted_item = self._format_item(
                text_content=file_content,
                source_info=file_path,
                prompt_format=prompt_format,
                filename_for_logging=filename,
                item_description="full file",
            )
            if formatted_item:
                processed_items.append(formatted_item)

        return processed_items


class ContinuesPipeline(DataPipeline):
    """
    A data pipeline that creates "continue the text" style prompts.
    It iteratively grabs an initial segment of text from the input file content
    to form a prompt (prefixed with "Continue the text:\\n\\n"), and the subsequent
    text from the file forms the completion. The total length of prompt + completion
    is managed to fit within max_length.
    """

    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int],
        enable_chunking: bool,  # Unused by this pipeline's core logic but part of interface
        min_grab_length: int,
        max_grab_length: int,
        dataset_split_seed: int,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        processed_items: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        if not file_content.strip():
            # print(f"      ContinuesPipeline: Skipping empty text file: {filename}")
            return []

        if max_length is None:
            print(
                f"      ContinuesPipeline: max_length must be provided for {filename}. Skipping."
            )
            return []

        content_token_limit = (
            max_length - 10
        )  # Buffer for global special tokens/formatting

        all_file_tokens = tokenizer.encode(file_content, add_special_tokens=False)
        if not all_file_tokens:
            # print(f"      ContinuesPipeline: File {filename} resulted in no tokens. Skipping.")
            return []

        # Initialize RNG for grab_length, deterministic per file
        file_specific_seed = dataset_split_seed + hash(file_path)
        rng = random.Random(file_specific_seed)

        current_token_idx = 0
        sample_counter = 0

        while current_token_idx < len(all_file_tokens):
            if (
                len(all_file_tokens) - current_token_idx < min_grab_length
            ):  # Not enough tokens left for even a minimal prompt
                # print(f"      ContinuesPipeline: Not enough tokens remaining in {filename} for a min_grab_length prompt. Tokens left: {len(all_file_tokens) - current_token_idx}")
                break

            grab_length = rng.randint(min_grab_length, max_grab_length)

            # Ensure grab_length doesn't exceed remaining tokens
            actual_grab_length = min(
                grab_length, len(all_file_tokens) - current_token_idx
            )

            if (
                actual_grab_length == 0
            ):  # Should be caught by the check above, but as a safeguard
                break

            prompt_content_tokens = all_file_tokens[
                current_token_idx : current_token_idx + actual_grab_length
            ]

            if not prompt_content_tokens:  # Should not happen if actual_grab_length > 0
                # print(f"      ContinuesPipeline: Grabbed empty prompt_content_tokens for {filename}. This shouldn't happen. Breaking.")
                break

            prompt_text_decoded = tokenizer.decode(
                prompt_content_tokens, skip_special_tokens=True
            )

            if not prompt_text_decoded.strip():
                # print(f"      ContinuesPipeline: Decoded prompt is empty/whitespace for {filename} at token {current_token_idx}. Advancing past grab.")
                current_token_idx += actual_grab_length
                continue

            user_facing_prompt_str = f"Continue the text:\n\n{prompt_text_decoded}"
            # Encode to get token count, excluding special tokens as they are handled later
            user_facing_prompt_token_ids = tokenizer.encode(
                user_facing_prompt_str, add_special_tokens=False
            )
            tokens_for_user_prompt = len(user_facing_prompt_token_ids)

            start_idx_for_completion_content = current_token_idx + actual_grab_length
            max_tokens_for_completion_part = (
                content_token_limit - tokens_for_user_prompt
            )

            completion_text_decoded = ""
            completion_content_tokens_grabbed = []

            if prompt_format == "text":
                completion_interstitial_str = "\\n\\nCompletion:\\n"
                completion_interstitial_token_ids = tokenizer.encode(
                    completion_interstitial_str, add_special_tokens=False
                )
                max_tokens_for_completion_part -= len(completion_interstitial_token_ids)

            if max_tokens_for_completion_part <= 0:
                # print(f"      ContinuesPipeline: Prompt from {filename} (token {current_token_idx}) is too long ({tokens_for_user_prompt} tokens), no space for completion. Advancing past grab.")
                current_token_idx += actual_grab_length
                continue  # Try next segment of the file

            if start_idx_for_completion_content < len(all_file_tokens):
                completion_content_tokens_grabbed = all_file_tokens[
                    start_idx_for_completion_content : min(
                        len(all_file_tokens),
                        start_idx_for_completion_content
                        + max_tokens_for_completion_part,
                    )
                ]
                if completion_content_tokens_grabbed:
                    completion_text_decoded = tokenizer.decode(
                        completion_content_tokens_grabbed, skip_special_tokens=True
                    )

            # We need some completion text for a valid sample
            if not completion_text_decoded.strip():
                # print(f"      ContinuesPipeline: No valid completion text could be formed for prompt from {filename} (token {current_token_idx}). Advancing past grab.")
                current_token_idx += actual_grab_length  # Consume the prompt part only
                continue

            sample_idx_str = f"sample_{sample_counter}"
            prompt_slice_str = (
                f"{current_token_idx}-{current_token_idx + actual_grab_length -1}"
            )
            completion_slice_start = start_idx_for_completion_content
            completion_slice_end = (
                start_idx_for_completion_content
                + len(completion_content_tokens_grabbed)
                - 1
            )
            completion_slice_str = f"{completion_slice_start}-{completion_slice_end}"
            source_info = f"{file_path} (ContinuesPipeline {sample_idx_str}, prompt_tokens {prompt_slice_str}, completion_tokens {completion_slice_str})"

            item_created = False
            if prompt_format == "chat":
                messages = [
                    {"role": "user", "content": user_facing_prompt_str},
                    {"role": "assistant", "content": completion_text_decoded},
                ]
                processed_items.append({"messages": messages, "source": source_info})
                item_created = True
            elif prompt_format == "text":
                final_text_str = f"{user_facing_prompt_str}{completion_interstitial_str}{completion_text_decoded}"
                # Final check on length for text (though theoretically covered by max_tokens_for_completion_part)
                # final_text_token_ids = tokenizer.encode(final_text_str, add_special_tokens=False)
                # if len(final_text_token_ids) > content_token_limit:
                #     print(f"      ContinuesPipeline WARNING: Text mode generated content > content_token_limit for {filename}. This should not happen. Truncating (not ideal).")
                #     # This would require re-calculating and truncating completion_text_decoded.
                #     # For now, assume the budget calculation was correct.
                processed_items.append({"text": final_text_str, "source": source_info})
                item_created = True
            else:
                print(
                    f"      ContinuesPipeline Warning: Unknown prompt_format '{prompt_format}' for sample from {filename}. Skipping."
                )

            if item_created:
                sample_counter += 1

            # Advance by exactly the grab length to start the next prompt
            # where the previous prompt ended
            current_token_idx += actual_grab_length

        if sample_counter > 0:
            print(
                f"      ContinuesPipeline: Produced {sample_counter} items from {filename}."
            )
        elif file_content.strip():
            print(
                f"      ContinuesPipeline: No items produced for {filename} despite having content and tokens."
            )

        return processed_items


class QADataPipeline(DataPipeline):
    """
    A data pipeline that processes .jsonl files containing QA pairs.
    Each line should contain a JSON object with "prompt" and "response" keys.
    For "chat" format, creates user/assistant message pairs.
    For "text" format, separates prompt and response with two newlines.
    """

    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int] = None,
        enable_chunking: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Processes content from a .jsonl file containing QA pairs."""
        processed_items: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        if not file_content.strip():
            print(f"      QADataPipeline: Skipping empty file: {filename}")
            return []

        # Process each line as a JSON object
        for line_num, line in enumerate(file_content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                qa_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"      QADataPipeline: Error parsing JSON on line {line_num} in {filename}: {e}"
                )
                continue

            prompt = qa_data.get("prompt", None)
            response = qa_data.get("response", None)

            if prompt is None or response is None:
                print(
                    f"      QADataPipeline: Missing prompt or response on line {line_num} in {filename}"
                )
                continue

            # Create source info including line number
            source_info = f"{file_path} (line {line_num})"

            if prompt_format == "chat":
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ]
                item = {"messages": messages, "source": source_info}
            elif prompt_format == "text":
                text_content = f"{prompt}\n\n{response}"
                item = {"text": text_content, "source": source_info}
            else:
                print(
                    f"      QADataPipeline Warning: Unknown prompt_format '{prompt_format}' for line {line_num} from {filename}. Skipping."
                )
                continue

            # Preserve augmentation_name field if it exists in the input
            if "augmentation_name" in qa_data:
                item["augmentation_name"] = qa_data["augmentation_name"]

            processed_items.append(item)

        if processed_items:
            print(
                f"      QADataPipeline: Processed {len(processed_items)} QA pairs from {filename}"
            )

        return processed_items


class ContrastPairsPipeline(DataPipeline):
    """
    A data pipeline that processes .jsonl files containing
    prompt/response_accepted/response_rejected triplets for preference learning.
    """

    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int] = None,
        enable_chunking: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Processes content from a .jsonl file containing prompt/response_accepted/response_rejected triplets."""
        processed_items: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        if not file_content.strip():
            print(f"      ContrastPairsPipeline: Skipping empty file: {filename}")
            return []

        # Process each line as a JSON object
        for line_num, line in enumerate(file_content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                triplet_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"      ContrastPairsPipeline: Error parsing JSON on line {line_num} in {filename}: {e}"
                )
                continue

            prompt = triplet_data.get("prompt", "").strip()
            response_accepted = triplet_data.get("response_accepted", "").strip()
            response_rejected = triplet_data.get("response_rejected", "").strip()

            # NB: we don't always have a prompt here
            if not response_accepted or not response_rejected:
                print(
                    f"      ContrastPairsPipeline: Missing response_accepted or response_rejected on line {line_num} in {filename}"
                )
                continue

            # Create source info including line number
            source_info = f"{file_path} (line {line_num})"

            # For DPO, we use the standard format regardless of prompt_format
            item = {
                "prompt": prompt,
                "response_accepted": response_accepted,
                "response_rejected": response_rejected,
                "source": source_info,
            }

            # Preserve augmentation_name field if it exists in the input
            if "augmentation_name" in triplet_data:
                item["augmentation_name"] = triplet_data["augmentation_name"]

            processed_items.append(item)

        if processed_items:
            print(
                f"      ContrastPairsPipeline: Processed {len(processed_items)} prompt/response_accepted/response_rejected triplets from {filename}"
            )

        return processed_items


class ConversationPipeline(DataPipeline):
    """
    A data pipeline that processes .jsonl files containing conversations.
    Each line should contain a JSON object with a "messages" key containing
    a list of {"role": "user"|"assistant", "content": "..."} objects.
    This pipeline always outputs in "chat" format regardless of prompt_format.
    If enable_chunking is True and max_length is provided, long conversations
    will be split into smaller conversation chunks that fit within the token limit.
    """

    def _chunk_conversation(
        self,
        messages: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        source_info: str,
    ) -> List[List[Dict[str, str]]]:
        """
        Splits a long conversation into smaller chunks that fit within max_length.

        Args:
            messages: List of message dictionaries with role/content
            tokenizer: Tokenizer for measuring token length
            max_length: Maximum tokens per chunk
            source_info: Source information for logging

        Returns:
            List of message chunks, each fitting within max_length
        """
        # Calculate total conversation length
        full_conversation_text = ""
        for msg in messages:
            full_conversation_text += f"{msg['role']}: {msg['content']}\n\n"

        total_tokens = len(
            tokenizer.encode(full_conversation_text, add_special_tokens=False)
        )

        # If conversation fits within limit, return as single chunk
        if total_tokens <= max_length:
            return [messages]

        # Need to chunk the conversation
        chunks = []
        current_chunk = []
        current_chunk_tokens = 0

        for msg in messages:
            # Calculate tokens for this message (including role prefix)
            msg_text = f"{msg['role']}: {msg['content']}\n\n"
            msg_tokens = len(tokenizer.encode(msg_text, add_special_tokens=False))

            # If adding this message would exceed limit, start new chunk
            if current_chunk and current_chunk_tokens + msg_tokens > max_length:
                if current_chunk:  # Only add non-empty chunks
                    chunks.append(current_chunk)
                current_chunk = [msg]
                current_chunk_tokens = msg_tokens
            else:
                current_chunk.append(msg)
                current_chunk_tokens += msg_tokens

        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        # Log chunking info
        print(
            f"      ConversationPipeline: Split conversation from {os.path.basename(source_info)} into {len(chunks)} chunks ({total_tokens} total tokens, max {max_length})"
        )

        return chunks

    def process(
        self,
        file_path: str,
        file_content: str,
        tokenizer: PreTrainedTokenizer,
        prompt_format: str,
        max_length: Optional[int] = None,
        enable_chunking: bool = False,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Processes content from a .jsonl file containing conversations."""
        processed_items: List[Dict[str, Any]] = []
        filename = os.path.basename(file_path)

        if not file_content.strip():
            print(f"      ConversationPipeline: Skipping empty file: {filename}")
            return []

        # Process each line as a JSON object
        for line_num, line in enumerate(file_content.strip().split("\n"), 1):
            line = line.strip()
            if not line:
                continue

            try:
                conversation_data = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"      ConversationPipeline: Error parsing JSON on line {line_num} in {filename}: {e}"
                )
                continue

            messages = conversation_data.get("messages", [])

            if not messages or not isinstance(messages, list):
                print(
                    f"      ConversationPipeline: Missing or invalid messages on line {line_num} in {filename}"
                )
                continue

            # Validate message structure
            valid_messages = []
            for msg_idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    print(
                        f"      ConversationPipeline: Invalid message format at index {msg_idx} on line {line_num} in {filename}"
                    )
                    continue

                role = msg.get("role", "").strip()
                content = msg.get("content", "").strip()

                if role not in ["user", "assistant", "system"]:
                    print(
                        f"      ConversationPipeline: Invalid role '{role}' at message {msg_idx} on line {line_num} in {filename}"
                    )
                    continue

                if not content:
                    print(
                        f"      ConversationPipeline: Empty content at message {msg_idx} on line {line_num} in {filename}"
                    )
                    continue

                valid_messages.append({"role": role, "content": content})

            if not valid_messages:
                print(
                    f"      ConversationPipeline: No valid messages found on line {line_num} in {filename}"
                )
                continue

            # Create base source info
            base_source_info = f"{file_path} (line {line_num})"

            # Check if we need to chunk this conversation
            if enable_chunking and max_length:
                message_chunks = self._chunk_conversation(
                    valid_messages, tokenizer, max_length, base_source_info
                )
            else:
                message_chunks = [valid_messages]

            # Process each chunk as a separate conversation
            for chunk_idx, chunk_messages in enumerate(message_chunks):
                # Create source info with chunk information if chunked
                if len(message_chunks) > 1:
                    source_info = f"{base_source_info} (chunk {chunk_idx + 1}/{len(message_chunks)})"
                else:
                    source_info = base_source_info

                # Always output in chat format for conversations
                item = {"messages": chunk_messages, "source": source_info}

                # Preserve augmentation_name field if it exists in the input
                if "augmentation_name" in conversation_data:
                    item["augmentation_name"] = conversation_data["augmentation_name"]

                # Preserve run_index field if it exists in the input
                if "run_index" in conversation_data:
                    item["run_index"] = conversation_data["run_index"]

                # Preserve author_name field if it exists in the input
                if "author_name" in conversation_data:
                    item["author_name"] = conversation_data["author_name"]

                processed_items.append(item)

        if processed_items:
            print(
                f"      ConversationPipeline: Processed {len(processed_items)} conversation items from {filename}"
            )

        return processed_items


# Example of how one might select a pipeline (this logic would live elsewhere, e.g., in data.py)
# def get_pipeline(config: DatasetConfig) -> DataPipeline:
# For now, always return BasicPipeline.
# Later, this could inspect config to choose different pipelines.
# if config.pipeline_name == "basic":
#     return BasicPipeline()
# elif config.pipeline_name == "another_pipeline":
#     return AnotherPipeline() # Assuming AnotherPipeline is defined
# else:
#     raise ValueError(f"Unknown pipeline: {config.pipeline_name}")
# return BasicPipeline()

# Updated dictionary to map pipeline names to classes
AVAILABLE_PIPELINES = {
    "basic": BasicPipeline,
    "continues": ContinuesPipeline,
    "qa": QADataPipeline,
    "contrast_pairs": ContrastPairsPipeline,
    "conversation": ConversationPipeline,
    # Add other pipelines here as they are defined
}


def get_pipeline_by_name(pipeline_name: str, **kwargs) -> DataPipeline:
    """
    Instantiates and returns a data pipeline based on its name.
    Specific pipeline configuration (like min_grab_length for ContinuesPipeline)
    should be passed via **kwargs if they are not part of the standard 'process' signature
    or derivable from DatasetConfig more broadly.

    Args:
        pipeline_name: The registered name of the pipeline.
        **kwargs: Additional arguments to pass to the pipeline's constructor (if any).
                  Currently, pipelines are instantiated without constructor args specific to this function.

    Returns:
        An instance of the requested DataPipeline.

    Raises:
        ValueError: If the pipeline_name is unknown.
    """
    pipeline_class = AVAILABLE_PIPELINES.get(pipeline_name.lower())
    if pipeline_class:
        return (
            pipeline_class()
        )  # Assuming constructors don't take specific args from here yet
    else:
        raise ValueError(
            f"Unknown pipeline: {pipeline_name}. Available: {list(AVAILABLE_PIPELINES.keys())}"
        )
