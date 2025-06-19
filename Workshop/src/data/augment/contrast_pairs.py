from typing import List, Dict, Any
import json

from .augmentations import BaseLLMAugmentation


class ContrastPairAugmentation(BaseLLMAugmentation):
    """
    Generates contrast pairs for DPO training with prompt/response_accepted/response_rejected structure.
    """

    def __init__(
        self,
        prompt_template: str,
        output_filename: str,
        augmentation_name: str = "ContrastPairAugmentation",
    ):
        super().__init__(augmentation_name=augmentation_name)
        self.prompt_template = prompt_template
        self.output_filename = output_filename

    def _get_tool_definition(self) -> Dict[str, Any]:
        """Returns the tool definition for Gemini to use."""
        return {
            "type": "function",
            "function": {
                "name": "log_contrast_pairs",
                "description": "Log contrast pairs for preference learning",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "prompt": {
                                        "type": "string",
                                        "description": "The prompt/question",
                                    },
                                    "response_accepted": {
                                        "type": "string",
                                        "description": "The preferred/better response",
                                    },
                                    "response_rejected": {
                                        "type": "string",
                                        "description": "The less preferred/worse response",
                                    },
                                },
                                "required": [
                                    "prompt",
                                    "response_accepted",
                                    "response_rejected",
                                ],
                            },
                        }
                    },
                    "required": ["pairs"],
                },
            },
        }

    def _generate_prompt(self, text_content: str) -> str:
        """Generates the prompt for the given text content."""
        return self.prompt_template.format(text_content=text_content)

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response to extract contrast pairs."""
        if tool_call.function.name == "log_contrast_pairs":
            try:
                function_args = json.loads(tool_call.function.arguments)
                if "pairs" in function_args:
                    return function_args["pairs"]
            except json.JSONDecodeError:
                pass
        return []

    def _get_output_filename(self) -> str:
        """Returns the output filename."""
        return self.output_filename

    def _get_target_folder(self) -> str:
        """Returns the target folder for contrast pair augmentations."""
        return "augmentations/contrast_pairs"

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that a contrast pair has all required fields."""
        prompt = result.get("prompt", "").strip()
        accepted = result.get("response_accepted", "").strip()
        rejected = result.get("response_rejected", "").strip()
        return bool(prompt and accepted and rejected)

    @staticmethod
    def extract_text_strings_from_file(file_path: str) -> List[str]:
        """
        Extracts all text strings from a contrast pair augmentation JSONL file.
        Returns a list of strings containing all prompt, response_accepted, and response_rejected content.
        """

        text_strings = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)

                        # Extract prompt if present and non-empty
                        prompt = data.get("prompt", "").strip()
                        if prompt:
                            text_strings.append(prompt)

                        # Extract response_accepted if present and non-empty
                        response_accepted = data.get("response_accepted", "").strip()
                        if response_accepted:
                            text_strings.append(response_accepted)

                        # Extract response_rejected if present and non-empty
                        response_rejected = data.get("response_rejected", "").strip()
                        if response_rejected:
                            text_strings.append(response_rejected)

                    except json.JSONDecodeError:
                        continue
        except (IOError, UnicodeDecodeError):
            pass

        return text_strings


class TakeContrastAugmentation(ContrastPairAugmentation):
    """Generates contrast pairs where the accepted response is the author's take and rejected is the opposite take."""

    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Now that you have read the above text, please generate 0-5 contrast pairs based on takes/opinions/judgements from the text.

For each pair, you should:

GUIDELINES FOR THE QUESTION:
- The question should be something the text takes a stance on
- The question should be standalone on its own, even without the context of the text
- The question should not mention or acknowledge the text in any way

GUIDELINES FOR THE PREFERRED RESPONSE:
- The preferred response should be a natural standalone answer to the question
- It should clearly be something that the author of the text would endorse
- It should be written in a way that is consistent with the author's voice and style

GUIDELINES FOR THE REJECTED RESPONSE:
- The rejected response should be the opposite take/opinion to what the author would say
- It should still be a natural standalone answer to the question
- It should represent a plausible alternative viewpoint that contradicts the author's stance
- Make it reasonable but clearly something the author would disagree with

Please use the tool calls available to you to log these 0-5 contrast pairs."""

        super().__init__(
            prompt_template=prompt_template,
            output_filename="take_contrast.jsonl",
            augmentation_name="take_contrast",
        )


class StylePairsAugmentation(ContrastPairAugmentation):
    """Generates style contrast pairs with exact quotes vs paraphrases, using empty prompts."""

    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Now that you have read the above text, please produce 5 pairs of an exact quote and a paraphrase of that quote.

GUIDELINES FOR THE EXACT QUOTATION:
- Should be a complete, meaningful passage from the text
- Must be longer than 3 sentences; 2-3 medium-sized paragraphs is a good length
- Must be word-for-word identical to what appears in the original text
- Should be substantial enough to demonstrate the author's style
- If they clearly exist, focus on passages that have particular flair or style, where the author differs from a generic style.

GUIDELINES FOR THE PARAPHRASE:
- Should convey the same factual content and meaning as the exact quotation
- Should use different vocabulary, sentence structure, and style
- Don't overdo the style difference - just try to paraphrase it as you would naturally put the same ideas in your own words

Please use the tool calls available to you to log these style contrast pairs."""

        super().__init__(
            prompt_template=prompt_template,
            output_filename="style_pairs.jsonl",
            augmentation_name="style_pairs",
        )

    def _get_tool_definition(self) -> Dict[str, Any]:
        """Returns the tool definition for Gemini to use."""
        return {
            "type": "function",
            "function": {
                "name": "log_style_pairs",
                "description": "Log style contrast pairs with exact quotes vs paraphrases",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "exact_quote": {
                                        "type": "string",
                                        "description": "Exact quotation from the text",
                                    },
                                    "paraphrase": {
                                        "type": "string",
                                        "description": "Paraphrase/rewrite of the exact quotation",
                                    },
                                },
                                "required": ["exact_quote", "paraphrase"],
                            },
                        }
                    },
                    "required": ["pairs"],
                },
            },
        }

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response to extract style pairs and automatically sets empty prompts."""
        if tool_call.function.name == "log_style_pairs":
            try:
                function_args = json.loads(tool_call.function.arguments)
                if "pairs" in function_args:
                    results = []
                    for pair in function_args["pairs"]:
                        if "exact_quote" in pair and "paraphrase" in pair:
                            results.append(
                                {
                                    "prompt": "",  # Automatically set to empty string
                                    "response_accepted": pair["exact_quote"],
                                    "response_rejected": pair["paraphrase"],
                                }
                            )
                    return results
            except json.JSONDecodeError:
                pass
        return []

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that a style pair has both accepted and rejected responses (prompt can be empty)."""
        accepted = result.get("response_accepted", "").strip()
        rejected = result.get("response_rejected", "").strip()
        return bool(accepted and rejected)
