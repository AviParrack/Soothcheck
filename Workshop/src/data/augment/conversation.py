# This file defines conversation augmentations that generate multi-turn conversations
# from text content using LLM-based augmentation, as well as transcript conversion.
#
# Key Classes:
# - BaseConversationAugmentation: Base class for conversation augmentations that
#   generates conversations with "messages" field containing role/content pairs
# - TopicChatAugmentation: Generates conversations between user and assistant
#   about the topic of the input text
# - TranscriptConversationAugmentation: Converts transcript files to chat format
#   by parsing speaker boundaries and roles

from typing import List, Dict, Any
import json

from .augmentations import BaseLLMAugmentation, DataAugmentation


class BaseConversationAugmentation(BaseLLMAugmentation):
    """
    Base class for conversation augmentations that generate multi-turn conversations.
    Unlike QA augmentations that generate prompt/response pairs, this generates
    full conversations with multiple messages.
    """

    def __init__(
        self,
        prompt_template: str,
        tool_name: str,
        tool_description: str,
        output_filename: str,
        num_runs: int = 1,
        augmentation_name: str = "BaseConversationAugmentation",
    ):
        super().__init__(augmentation_name=augmentation_name)
        self.prompt_template = prompt_template
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.output_filename = output_filename
        self.num_runs = num_runs

    def _get_tool_definition(self) -> Dict[str, Any]:
        """Returns the tool definition for Gemini to use for conversation generation."""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "conversations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "messages": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "role": {
                                                    "type": "string",
                                                    "enum": ["user", "assistant"],
                                                    "description": "The role of the speaker",
                                                },
                                                "content": {
                                                    "type": "string",
                                                    "description": "The message content",
                                                },
                                            },
                                            "required": ["role", "content"],
                                        },
                                    }
                                },
                                "required": ["messages"],
                            },
                        }
                    },
                    "required": ["conversations"],
                },
            },
        }

    def _generate_prompt(self, text_content: str) -> str:
        """Generates the prompt for the given text content."""
        return self.prompt_template.format(text_content=text_content)

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response to extract conversations."""
        if tool_call.function.name == self.tool_name:
            try:
                function_args = json.loads(tool_call.function.arguments)
                if "conversations" in function_args:
                    results = []
                    for conversation in function_args["conversations"]:
                        if "messages" in conversation and conversation["messages"]:
                            # Validate that messages have proper structure
                            valid_messages = []
                            for msg in conversation["messages"]:
                                if (
                                    isinstance(msg, dict)
                                    and "role" in msg
                                    and "content" in msg
                                    and msg["role"] in ["user", "assistant"]
                                    and msg["content"].strip()
                                ):
                                    valid_messages.append(
                                        {
                                            "role": msg["role"],
                                            "content": msg["content"].strip(),
                                        }
                                    )

                            if valid_messages:
                                results.append({"messages": valid_messages})
                    return results
            except json.JSONDecodeError:
                pass
        return []

    def _get_output_filename(self) -> str:
        """Returns the output filename."""
        return self.output_filename

    def _get_target_folder(self) -> str:
        """Returns the target folder for conversation augmentations."""
        return "augmentations/chats"

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that a conversation has proper messages structure."""
        if "messages" not in result:
            return False

        messages = result["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            return False

        # Check that all messages have role and content
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["user", "assistant"]:
                return False
            if not msg["content"].strip():
                return False

        return True

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Augments the content by running the conversation generation multiple times.
        """
        if not file_content.strip():
            return []

        # Create target directory if it doesn't exist
        import os

        target_dir = os.path.join(raw_data_dir, self._get_target_folder())
        os.makedirs(target_dir, exist_ok=True)

        # Set output file path if not already set (thread-safe)
        with self._lock:
            if self.output_file_path is None:
                self.output_file_path = os.path.join(
                    target_dir, self._get_output_filename()
                )

        # Run the augmentation multiple times
        all_results = []
        for run_idx in range(self.num_runs):
            results = self._call_gemini(file_content)

            # Add metadata to each result
            for result in results:
                if self._is_valid_result(result):
                    result_with_metadata = {
                        **result,
                        "source": file_path,
                        "augmentation_name": self.augmentation_name,
                        "run_index": run_idx,
                    }
                    all_results.append(result_with_metadata)

        # Add all valid results to accumulator in one atomic operation
        if all_results:
            with self._lock:
                self.results_accumulator.extend(all_results)

        return []  # We don't return files until finalize is called

    @staticmethod
    def extract_text_strings_from_file(file_path: str) -> List[str]:
        """
        Extracts all text strings from a conversation augmentation JSONL file.
        Returns a list of strings containing all message content.
        """
        import json

        text_strings = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Extract message content from conversations
                        messages = data.get("messages", [])
                        if isinstance(messages, list):
                            for message in messages:
                                if isinstance(message, dict):
                                    content = message.get("content", "").strip()
                                    if content:
                                        text_strings.append(content)

                    except json.JSONDecodeError:
                        continue
        except (IOError, UnicodeDecodeError):
            pass

        return text_strings


class TopicChatAugmentation(BaseConversationAugmentation):
    """
    Generates conversations between a user and assistant about the topic of the input text.
    The assistant responds with the author's takes and arguments from the text.
    """

    def __init__(self, **kwargs):
        # First get the author name and validate it
        self.author_name = kwargs.get("author_name", None)
        if not self.author_name:
            raise ValueError("Author name is required for TopicChatAugmentation")

        # Create the prompt template with the author name
        prompt_template = f"""Read this text:
{{text_content}}

Based on this text, write a plausible chat conversation about this topic that a smart interlocutor might've had with {self.author_name}. They should have a conversation of about 5-15 messages, which traces a thread through the content/arguments of the piece. The interlocutor should question and push back and ask for more, and the author role should respond with the takes and arguments that the author presents in the text. Try to copy the author's writing style, but do not use verbatim quotes. The user should not be too sycophantic towards the author, they should generally be neutral, questioning, and efficient in their words (imagine what someone might type into a chat with a friend or an LLM)

Please use the tool calls available to you to log this conversation."""

        # Initialize the parent class with required parameters
        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_topic_chat",
            tool_description="Log a conversation about the topic of the text between a user and the author",
            output_filename="topic_chat.jsonl",
            num_runs=kwargs.get("num_runs", 1),
            augmentation_name="topic_chat",
        )

        # Store all original kwargs including author_name in params for validation
        self.params.update(kwargs)

    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["author_name"]


class InterviewAugmentation(BaseConversationAugmentation):
    """
    Generates conversations between a user and assistant about the topic of the input text.
    The assistant responds with the author's takes and arguments from the text.
    """

    def __init__(self, **kwargs):
        # First get the author name and validate it
        self.author_name = kwargs.get("author_name", None)


class TranscriptConversationAugmentation(DataAugmentation):
    """
    Converts transcript files to chat format by parsing speaker boundaries.
    Does not use LLM - just parses the transcript structure.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.speaker_regex = kwargs.get("speaker_regex")
        self.assistant_speaker = kwargs.get("assistant_speaker")
        self.output_filename = kwargs.get("output_filename", "transcript_chat.jsonl")

        self.results = []

        # Store all parameters for validation
        self.params = kwargs

        # Validate required parameters immediately
        self.validate_params()

    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["speaker_regex", "assistant_speaker"]

    def _parse_transcript(self, content: str) -> List[Dict[str, Any]]:
        """
        Parses transcript content into messages using speaker regex.

        Args:
            content: Raw transcript content

        Returns:
            List of message dictionaries with role and content
        """
        import re

        # Find all speaker boundaries and their positions
        speaker_matches = list(
            re.finditer(self.speaker_regex, content, flags=re.MULTILINE)
        )

        if not speaker_matches:
            return []

        messages = []

        for i, match in enumerate(speaker_matches):
            speaker_name = match.group(1)  # First capture group

            # Find content start (after the speaker line)
            content_start = match.end()

            # Find content end (before next speaker or end of text)
            if i + 1 < len(speaker_matches):
                content_end = speaker_matches[i + 1].start()
            else:
                content_end = len(content)

            # Extract message content
            message_content = content[content_start:content_end].strip()

            if message_content:
                # Determine role
                role = "assistant" if speaker_name == self.assistant_speaker else "user"

                # For user messages, include speaker name as prefix
                if role == "user":
                    content_text = f"{speaker_name}: {message_content}"
                else:
                    content_text = message_content

                messages.append(
                    {
                        "role": role,
                        "content": content_text.strip(),
                        "speaker": speaker_name,
                    }
                )

        return messages

    def _post_process_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Post-processes messages to join consecutive user messages and ensure proper format.

        Args:
            messages: List of message dictionaries

        Returns:
            Post-processed list of messages
        """
        if not messages:
            return []

        # Join consecutive user messages
        processed = []
        current_user_content = []

        for msg in messages:
            if msg["role"] == "user":
                current_user_content.append(msg["content"])
            else:  # assistant message
                # If we have accumulated user content, add it as one message
                if current_user_content:
                    processed.append(
                        {"role": "user", "content": "\n\n".join(current_user_content)}
                    )
                    current_user_content = []

                # Add the assistant message
                processed.append({"role": "assistant", "content": msg["content"]})

        # Handle any remaining user content
        if current_user_content:
            processed.append(
                {"role": "user", "content": "\n\n".join(current_user_content)}
            )

        # If first message is assistant, add empty user message
        if processed and processed[0]["role"] == "assistant":
            processed.insert(0, {"role": "user", "content": ""})

        return processed

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Converts transcript file to chat format.
        """
        if not file_content.strip():
            return []

        # Parse transcript into messages
        raw_messages = self._parse_transcript(file_content)
        if not raw_messages:
            return []

        # Post-process messages
        processed_messages = self._post_process_messages(raw_messages)
        if not processed_messages:
            return []

        # Create the conversation record
        conversation = {
            "messages": processed_messages,
            "source": file_path,
            "augmentation_name": "transcript_conversation",
        }

        # Store result for later writing
        self.results.append(conversation)

        return []  # Files written in finalize()

    def finalize(self, raw_data_dir: str = None) -> List[str]:
        """
        Writes all accumulated conversations to JSONL file.
        """
        if not self.results:
            return []

        import os
        import json

        # Create target directory
        if raw_data_dir:
            target_dir = os.path.join(raw_data_dir, "augmentations", "chats")
        else:
            # Fallback: try to determine from source paths
            source_path = self.results[0]["source"]
            # Assume source is in raw/dataset/corpus/file.txt, go up to raw/dataset/
            dataset_dir = os.path.dirname(os.path.dirname(source_path))
            target_dir = os.path.join(dataset_dir, "augmentations", "chats")

        os.makedirs(target_dir, exist_ok=True)

        # Write results
        output_path = os.path.join(target_dir, self.output_filename)
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for result in self.results:
                    f.write(json.dumps(result) + "\n")

            print(
                f"Saved {len(self.results)} transcript conversations to {output_path}"
            )
            return [output_path]
        except Exception as e:
            print(f"Error writing transcript conversations to {output_path}: {e}")
            return []
