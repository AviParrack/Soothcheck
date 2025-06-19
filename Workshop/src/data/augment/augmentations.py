# This file defines data augmentations that act on raw data to create more raw data.
# Unlike data pipelines which process raw data during conversion to processed data,
# augmentations create additional raw files that can then be processed by pipelines.
#
# Augmented files are saved in the "augmented" or "contrast_pairs" folders
# within the raw dataset directory to differentiate them from original raw files.
#
# Key Classes:
# - DataAugmentation (ABC): Defines the interface for all data augmentations.
#   Requires an `augment` method.
# - BaseLLMAugmentation (ABC): Base class for LLM-based augmentations using Gemini.
#   Handles common logic while allowing customization of prompts and tool definitions.
# - SimpleQAAugmentation: Generates prompt/response pairs from text content using Gemini
#   via LiteLLM with function calling. Takes prompt template and tool definition as parameters.
# - ContrastPairAugmentation: Generates contrast pairs for DPO training with
#   prompt/response_accepted/response_rejected structure.
# - QAAugmentation: Specific implementation for prompt/response generation
# - TakeAugmentation: Specific implementation for opinion/take extraction
# - TextCleanAugmentation: Cleans PDF-to-text conversions by removing spurious formatting
#   using Gemini-2.5-flash-preview-05-20. Processes text in chunks and saves to "cleaned" folder.
# - AVAILABLE_AUGMENTATIONS (dict): Maps augmentation names to their classes.
# - get_augmentation_by_name (function): Instantiates an augmentation by its name.

from abc import ABC, abstractmethod
import json
import os
from typing import List, Dict, Any
import threading

from src.utils.litellm_logger import litellm_completion

# Import text cleaning functions
from ..process.text_clean import clean_full_text


class DataAugmentation(ABC):
    """
    Abstract base class for data augmentations.
    An augmentation takes raw file content and creates additional raw files.
    """

    def __init__(self, **kwargs):
        """
        Initialize the augmentation with any parameters.
        Subclasses should call super().__init__(**kwargs) and then handle their specific parameters.
        """
        self.params = kwargs  # Store all parameters for potential use by subclasses

    @abstractmethod
    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        **kwargs: Any,
    ) -> List[str]:
        """
        Augments the content of a single file by creating additional raw files.

        Args:
            file_path: The path to the original file being augmented.
            file_content: The raw string content of the file.
            raw_data_dir: The absolute path to the raw data directory.
            **kwargs: Additional arguments for specific augmentation implementations.

        Returns:
            A list of paths to the newly created augmented files.
        """
        pass

    @classmethod
    def get_required_params(cls) -> List[str]:
        """
        Returns a list of parameter names that are required for this augmentation.
        Override in subclasses to specify required parameters.
        """
        return []

    def validate_params(self) -> bool:
        """
        Validates that all required parameters are present.
        Override in subclasses for custom validation logic.
        """
        required_params = self.__class__.get_required_params()
        missing_params = [
            param for param in required_params if param not in self.params
        ]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {self.__class__.__name__}: {', '.join(missing_params)}"
            )
        return True


class BaseLLMAugmentation(DataAugmentation):
    """
    Base class for LLM-based augmentations that use Gemini via LiteLLM with function calling.

    This class handles the common logic for:
    - Accumulating results from multiple files (thread-safe)
    - Calling Gemini with tools
    - Writing results to a .jsonl file during finalization

    Subclasses must implement:
    - _get_tool_definition(): Define the tool schema for Gemini
    - _generate_prompt(): Create the prompt for the given text content
    - _parse_tool_response(): Extract results from the tool call response
    - _get_output_filename(): Specify the output filename pattern
    - _get_target_folder(): Specify the target folder (e.g., "augmentations/prompt_response_pairs")
    """

    def __init__(self, augmentation_name: str):
        super().__init__()
        self.results_accumulator = []
        self.output_file_path = None
        self._lock = threading.Lock()  # For thread-safe accumulation
        self.augmentation_name = augmentation_name

    @abstractmethod
    def _get_tool_definition(self) -> Dict[str, Any]:
        """Returns the tool definition for Gemini to use."""
        pass

    @abstractmethod
    def _generate_prompt(self, text_content: str) -> str:
        """Generates the prompt to send to Gemini for the given text content."""
        pass

    @abstractmethod
    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """
        Parses the tool call response to extract the results.

        Args:
            tool_call: A tool call object from the Gemini response

        Returns:
            List of dictionaries containing the extracted data
        """
        pass

    @abstractmethod
    def _get_output_filename(self) -> str:
        """Returns the filename for the output file (e.g., 'qa.jsonl')."""
        pass

    @abstractmethod
    def _get_target_folder(self) -> str:
        """Returns the target folder name. Must be implemented by subclasses."""
        pass

    def _call_llm(
        self, text_content: str, model_type: str = "gemini"
    ) -> List[Dict[str, Any]]:
        """
        Calls an LLM (Gemini or local) to process the given text.

        Args:
            text_content: The text content to process
            model_type: Either "gemini" or "local"

        Returns:
            List of dictionaries containing the extracted results
        """
        prompt = self._generate_prompt(text_content)
        tools = [self._get_tool_definition()]

        try:
            if model_type == "local":
                response = self._call_local_llm(prompt, tools)
            else:
                response = self._call_api_llm(prompt, tools)

            results = []

            # Parse tool calls from the response
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "tool_calls"):
                    for tool_call in choice.message.tool_calls:
                        results.extend(self._parse_tool_response(tool_call))

            return results

        except Exception:
            return []

    def _call_api_llm(self, prompt: str, tools: List[Dict[str, Any]]):
        """Call model via LiteLLM API."""
        return litellm_completion(
            # model="gemini/gemini-2.5-pro-preview-05-06",
            # model="claude-sonnet-4-20250514",
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
        )

    def _call_local_llm(self, prompt: str, tools: List[Dict[str, Any]]):
        """Call local LLM via queue manager."""
        from ...inference.local_llm_adapter import local_completion

        return local_completion(
            model="local",
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto",
        )

    # Backward compatibility
    def _call_gemini(self, text_content: str) -> List[Dict[str, Any]]:
        """Legacy method for backward compatibility."""
        return self._call_llm(text_content, model_type="gemini")

    def augment(
        self,
        file_path: str,
        file_content: str,
        raw_data_dir: str,
        model_type: str = "gemini",
        **kwargs: Any,
    ) -> List[str]:
        """Processes content from a file using an LLM (Gemini or local)."""
        if not file_content.strip():
            return []

        # Create target directory if it doesn't exist
        target_dir = os.path.join(raw_data_dir, self._get_target_folder())
        os.makedirs(target_dir, exist_ok=True)

        # Set output file path if not already set (thread-safe)
        with self._lock:
            if self.output_file_path is None:
                self.output_file_path = os.path.join(
                    target_dir, self._get_output_filename()
                )

        # Generate results using the specified LLM
        results = self._call_llm(file_content, model_type=model_type)

        if not results:
            return []

        # Add source file path, augmentation name, and model type to each result and accumulate
        results_to_add = []
        for result in results:
            if self._is_valid_result(result):
                result_with_metadata = {
                    **result,
                    "source": file_path,
                    "augmentation_name": self.augmentation_name,
                    "model_type": model_type,
                }
                results_to_add.append(result_with_metadata)

        # Add all valid results to accumulator in one atomic operation
        if results_to_add:
            with self._lock:
                self.results_accumulator.extend(results_to_add)

        return []  # We don't return files until finalize is called

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """
        Validates a result dictionary. Override in subclasses for custom validation.
        Default implementation checks that all values are non-empty strings.
        """
        return all(
            isinstance(value, str) and value.strip()
            for value in result.values()
            if isinstance(value, str)
        )

    def finalize(self) -> List[str]:
        """
        Writes all accumulated results to the .jsonl file.
        Should be called after all files have been processed.
        """
        with self._lock:
            if not self.results_accumulator or self.output_file_path is None:
                return []

            try:
                with open(self.output_file_path, "w", encoding="utf-8") as f:
                    for result in self.results_accumulator:
                        f.write(json.dumps(result) + "\n")

                print(
                    f"Saved {len(self.results_accumulator)} results to {self.output_file_path}"
                )
                return [self.output_file_path]
            except Exception as e:
                print(f"Error writing results to {self.output_file_path}: {e}")
                return []
