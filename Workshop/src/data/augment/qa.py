from typing import List, Dict, Any
import json

from .augmentations import BaseLLMAugmentation


class SimpleQAAugmentation(BaseLLMAugmentation):
    """
    A flexible augmentation class that generates prompt/response pairs from text content.
    Takes prompt template and tool definition as constructor parameters to reduce boilerplate.
    """

    def __init__(
        self,
        prompt_template: str,
        tool_name: str,
        tool_description: str,
        output_filename: str,
        augmentation_name: str = "SimpleQAAugmentation",
        **kwargs,  # Accept any extra parameters
    ):
        super().__init__(augmentation_name=augmentation_name)
        self.prompt_template = prompt_template
        self.tool_name = tool_name
        self.tool_description = tool_description
        self.output_filename = output_filename
        self.params = kwargs  # Store any extra parameters

    def _get_tool_definition(self) -> Dict[str, Any]:
        """Returns the tool definition for Gemini to use."""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
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
                                        "description": "The prompt",
                                    },
                                    "response": {
                                        "type": "string",
                                        "description": "The response",
                                    },
                                },
                                "required": ["prompt", "response"],
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
        """Parses the tool call response to extract pairs."""
        if tool_call.function.name == self.tool_name:
            try:
                function_args = json.loads(tool_call.function.arguments)
                if "pairs" in function_args:
                    # Convert to standard prompt/response format
                    results = []
                    for pair in function_args["pairs"]:
                        if "prompt" in pair and "response" in pair:
                            results.append(
                                {
                                    "prompt": pair["prompt"],
                                    "response": pair["response"],
                                }
                            )
                    return results
            except json.JSONDecodeError:
                pass
        return []

    def _get_output_filename(self) -> str:
        """Returns the output filename."""
        return self.output_filename

    def _get_target_folder(self) -> str:
        """Returns the target folder for QA augmentations."""
        return "augmentations/prompt_response_pairs"

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that a pair has both prompt and response."""
        prompt = result.get("prompt", "").strip()
        response = result.get("response", "").strip()
        return bool(prompt and response)

    @staticmethod
    def extract_text_strings_from_file(file_path: str) -> List[str]:
        """
        Extracts all text strings from a QA augmentation JSONL file.
        Returns a list of strings containing all prompt and response content.
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
                        # Extract prompt if present and non-empty
                        prompt = data.get("prompt", "").strip()
                        if prompt:
                            text_strings.append(prompt)

                        # Extract response if present and non-empty
                        response = data.get("response", "").strip()
                        if response:
                            text_strings.append(response)

                    except json.JSONDecodeError:
                        continue
        except (IOError, UnicodeDecodeError):
            pass

        return text_strings


# Specific augmentation implementations using the new flexible classes


class QAAugmentation(SimpleQAAugmentation):
    """Question/Answer augmentation with standardized prompt/response format."""

    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Now that you have read the above text, please generate 0-5 prompt/response pairs.

GUIDELINES FOR THE PROMPT:
- The prompt should be answered by the text
- The prompt should be standalone on its own, even without the context of the text
- The prompt should not mention or acknowledge the text in any way

GUIDELINES FOR THE RESPONSE:
- The response should be very close to a quotation from the text
- The response should be a natural standalone answer to the prompt
- The response should not change the meaning of the original text
- The response should remain in the voice of the original author to the fullest extent grammatically possible (but with changes if e.g. an exact quote has references to something that is not in the answer)
- Make sure that the response flows naturally and grammatically as a response to the prompt; the response should not need the surrounding text to make sense.

Please use the tool calls available to you to log these 0-5 prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_qa_pairs",
            tool_description="Log prompt/response pairs extracted from the text",
            output_filename="qa.jsonl",
            augmentation_name="qa",
        )


class TakeAugmentation(SimpleQAAugmentation):
    """Take/Opinion augmentation with standardized prompt/response format."""

    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Now that you have read the above text, please generate 0-5 pairs of a prompt someone could have about the subject matter of the text, and a response that represents the take/opinion/judgement the author of the text would have about the prompt.

GUIDELINES FOR THE PROMPT:
- The prompt should be something the text takes a stance on
- The prompt should be standalone on its own, even without the context of the text
- The prompt should not mention or acknowledge the text in any way

GUIDELINES FOR THE RESPONSE:
- The response should be a natural standalone answer to the prompt
- The response should clearly be something that the author of the text would endorse
- The response should be written in a way that is consistent with the author's voice and style. Imagine you are answering on the author's behalf

Please use the tool calls available to you to log these 0-5 prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_takes",
            tool_description="Log takes/opinions/judgements extracted from the text",
            output_filename="takes.jsonl",
            augmentation_name="takes",
        )


class ConceptExplanationAugmentation(SimpleQAAugmentation):
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

For each instance in the text where a concept is explained, please generate a prompt asking for an explanation of that concept, and then quote the author's explanation of the concept for your response.

GUIDELINES FOR THE PROMPT:
- The prompt should be asking for an explanation of a concept.
- It should be concise
- It should be standalone on its own, even without the context of the text
- It should be specific enough that the quotation in the response is a natural response to it

GUIDELINES FOR THE RESPONSE:
- The response should be a natural standalone answer to the prompt
- The response should be a quotation from the text (except for any modification necessary for it to be grammatically standalone and sensible, and the addition of anything required ot make it standalone, or the deletion of any parenthetical parts that don't make sense in the context of the prompt)

Please use the tool calls available to you to log these prompt/response pairs for concept explanations within the text. If there are none, please return an empty list."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_concept_explanations",
            tool_description="Log concept explanations extracted from the text",
            output_filename="concept_explanations.jsonl",
            augmentation_name="concept_explanations",
        )


class NotesOnTopicAugmentation(SimpleQAAugmentation):
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

From this text, identify the core question or topic that it discusses, and then produce a skeleton sketch of the logical connections within it; imagine what bullet point notes the author would have made to themselves in the process of writing the text.

GUIDELINES FOR THE PROMPT:
- The prompt field is where you submit the research question / topic / concern that the text is addressing.
- Start the prompt with "Notes: "
- If the text is a broad overview of a topic, without a particular slant or concern, the prompt can just be "Notes: " followed by the name of the topic at hand . However, if it clearly takes a stance or an angle, or is all about answering some question, the prompt should have that question. If the text has a title that is very descriptive, you can just use the title.

GUIDELINES FOR THE RESPONSE:
- The response field is where you write the notes the author of the text might've made to themselves before writing.
- Use terminology that is consistent with the text; use concepts and terms that the author uses when appropriate.
- Focus on the logical flow, the underlying structure, and on associating ideas in the same way the author does.
- Depending on the complexity and length of the text, the notes can be anywhere from a few bullet points to a page or two.
- Adapt the note structure to the format. If it's an opinion piece, focus on the author's opinions and argumentation. If it's technical, note the core equations and concept definitions and ideas to understand concisely. If it's fiction, the notes might be a structure of key plot beats and events. And so on.

Please use the tool calls available to you to log these prompt/response pairs for notes on the topic of the text. If there are none, please return an empty list."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_notes_on_topic",
            tool_description="Log what notes on the topic of the text might've been used to write it",
            output_filename="notes_on_topic.jsonl",
            augmentation_name="notes_on_topic",
        )


class FillInNotesAugmentation(SimpleQAAugmentation):
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

From the above text, we are going to create some brainstorm prompt/response pairs. The prompt is going to start with "Brainstorm: " and then list a scattered subset of the ideas in the text, and the response is going to mention what's missing from the prompt but in the text.

GUIDELINES FOR THE PROMPT:
- The prompt field is where you submit the brainstorm prompt.
- Start the prompt with "Brainstorm: ". After this, write either a paragraph or two, or some bullet points, or whatever makes most sense, concisely discussing some of the main concerns and ideas in the text, but with some key pieces / insights / arguments / concepts / links missing.
- The prompt should be standalone on its own, not referring to the text in any way. It should be a natural brain-dump of thoughts someone might have, regardless of whether they're writing the text above, or any article at all.

GUIDELINES FOR THE RESPONSE:
- The response field is where you respond to the prompt with the missing pieces that the text mentions.
- As before, it can be a few paragraphs or some notes, based on what's most appropriate and on maintaining variety.
- It should be written as notes / a bunch of ideas / brainstorm; doesn't have to be a polished, conversation-style response
- The response should not refer to the text in any way; it should be a standalone response to the prompt.
- The missing pieces should be genuinely insightful ideas / links / arguments / concepts; they should have an "aha" factor to them if possible.


Please use the tool calls available to you to log 0-3 such prompt/response pairs based on the text."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_fill_in_notes",
            tool_description="Log what notes on the topic of the text might've been used to write it",
            output_filename="fill_in_notes.jsonl",
            augmentation_name="fill_in_notes",
        )


class NewsArticleAugmentation(SimpleQAAugmentation):
    """News Article augmentation that creates news-style articles about the author's content."""

    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["author_name"]

    def __init__(self, **kwargs):
        # First get the author name and validate it
        self.author_name = kwargs.get("author_name", None)
        if not self.author_name:
            raise ValueError("Author name is required for NewsArticleAugmentation")

        # Create the prompt template with the author name
        prompt_template = f"""Read this text:
{{text_content}}

The above text is by {self.author_name}.

Based on this text, write 0-2 news articles in the tone of a newspaper that explains what the author says about the topic, including any opinions, takes, and judgements, and focusing on whatever is novel / newsworthy / out of the ordinary in what the author says.

Please use the tool calls available to you to log these prompt/response pairs."""

        # Initialize the parent class with required parameters
        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_news_articles",
            tool_description="Log news-style articles based on the author's content",
            output_filename="news_articles.jsonl",
            augmentation_name="news_articles",
        )

        # Store all original kwargs including author_name in params for validation
        self.params.update(kwargs)

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response and automatically sets empty prompts."""
        results = super()._parse_tool_response(tool_call)

        # Set prompt to empty string and add author_name to each result
        for result in results:
            result["prompt"] = ""  # Set prompt to empty string
            result["author_name"] = self.author_name

        return results

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that an article has response content (prompt can be empty)."""
        response = result.get("response", "").strip()
        return bool(response)


class InterviewAugmentation(SimpleQAAugmentation):
    """Interview augmentation that creates interview transcripts with the author."""

    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["author_name"]

    def __init__(self, **kwargs):
        # First get the author name and validate it
        self.author_name = kwargs.get("author_name", None)
        if not self.author_name:
            raise ValueError("Author name is required for InterviewAugmentation")

        # Create the prompt template with the author name
        prompt_template = f"""Read this text:
{{text_content}}

The above text is by {self.author_name}.

Based on this text, write ONE interview transcript between an interviewer and the author, called {self.author_name}, in which the author is asked questions about the topics covered in the above text, and the author answers as you'd guess they would based on the text. The interviewer should focus on whatever is novel / newsworthy / out of the ordinary in what the author says, and drill into those with questions.

The interview should be written as a single long response that looks like an interview transcript, with interviewer questions/comments tagged with "INTERVIEWER: ", and the author's responses tagged with "{self.author_name.upper()}: ". There should be an empty line between each exchange.

The interview should contain at least 10-15 exchanges (back-and-forth between interviewer and author). You can be creative about the stance and opinions the interviewer takes, but make sure whatever the author responds is well-supported given the text of theirs you read above.

Important: You should generate exactly ONE prompt/response pair where:
- The prompt is empty (leave it as an empty string)
- The response contains the entire interview transcript with all the exchanges

Please use the tool calls available to you to log this single interview transcript."""

        # Initialize the parent class with required parameters
        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_interviews",
            tool_description="Log interview-style transcripts based on the author's content",
            output_filename="interviews.jsonl",
            augmentation_name="interviews",
        )

        # Store all original kwargs including author_name in params for validation
        self.params.update(kwargs)

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response and automatically sets empty prompts."""
        results = super()._parse_tool_response(tool_call)

        # Set prompt to empty string and add author_name to each result
        for result in results:
            result["prompt"] = ""  # Set prompt to empty string
            result["author_name"] = self.author_name

        return results

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that an interview has response content (prompt can be empty)."""
        response = result.get("response", "").strip()
        return bool(response)


# TODO: finish
class TwitterThreadAugmentation(SimpleQAAugmentation):
    @classmethod
    def get_required_params(cls) -> List[str]:
        return ["author_name"]

    def __init__(self, **kwargs):
        # First get the author name and validate it
        self.author_name = kwargs.get("author_name", None)
        if not self.author_name:
            raise ValueError("Author name is required for TwitterThreadAugmentation")

        # Create the prompt template with the author name
        prompt_template = f"""Read this text:
{{text_content}}

The above text is by {self.author_name}.

Now that you have read the above text, please generate 0-2 Twitter threads that explain what {self.author_name} says about the topic, including any opinions, takes, and judgements, and focusing on whatever is novel / newsworthy / out of the ordinary in what the author says.

GUIDELINES FOR THE PROMPT:
- The prompt should be empty (we'll set it to empty string automatically)

GUIDELINES FOR THE RESPONSE:
- The response should be written as a Twitter thread, with numbered tweets (1/n, 2/n, etc.)
- Each tweet should be under 280 characters when possible
- The thread should capture the key insights and novel ideas from the text
- The tone should be engaging and social media appropriate
- The thread should be standalone and not reference the original text
- The thread should be written from the author's first-person perspective (as if {self.author_name} is tweeting)
- The author should NOT be referred to in third-person within the thread itself

Please use the tool calls available to you to log these prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_twitter_threads",
            tool_description="Log Twitter thread-style content based on the text",
            output_filename="twitter_threads.jsonl",
            augmentation_name="twitter_threads",
        )

        # Store all original kwargs including author_name in params for validation
        self.params.update(kwargs)

    def _parse_tool_response(self, tool_call) -> List[Dict[str, Any]]:
        """Parses the tool call response and automatically sets empty prompts."""
        results = super()._parse_tool_response(tool_call)

        # Set prompt to empty string and add author_name to each result
        for result in results:
            result["prompt"] = ""  # Set prompt to empty string
            result["author_name"] = self.author_name

        return results

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Validates that a thread has response content (prompt can be empty)."""
        response = result.get("response", "").strip()
        return bool(response)


class BeliefsOnTopicAugmentation(SimpleQAAugmentation):
    # TODO: finish class
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Based on this text, generate 0-5 prompt/response pairs that ask the author of the text about their beliefs related to topics covered in the text, and how the author might respond based on what they wrote.

GUIDELINES FOR THE PROMPT:
- The prompt should be asking the author of the text about their beliefs related to some topic covered in the text, on which the author takes a stance.
- The prompt should be standalone on its own, not referring to the text in any way.

GUIDELINES FOR THE RESPONSE:
- The response should be a natural standalone answer to the prompt.
- The response should describe beliefs, value judgements, or opinions.
- The response should clearly be in agreement with something the author of the text would say.
- The response should be written in a way that is consistent with the author's voice and style, though it should not be an exact quote.
- If there is nuance or conditions to the author's beliefs on this question, the response should mention it - it should not oversimplify what the author thinks.

Please use the tool calls available to you to log these prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_beliefs_on_topic",
            tool_description="Log beliefs and opinions from the author about topics in the text",
            output_filename="beliefs_on_topic.jsonl",
            augmentation_name="beliefs_on_topic",
        )


class ExamplesMethodsSourcesAugmentation(SimpleQAAugmentation):
    # TODO: finish class
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Based on this text, generate 0-5 prompt/response pairs that ask the author of the text about which examples, methods, or sources they might consider when discussing this topic, and which they might use to illustrate or reason about the content of the text. For example: which sort of anecdotes, historical examples, types of analysis or reasoning, empirical data, and so forth does the author call upon?

GUIDELINES FOR THE PROMPT:
- The prompt should be a question to the author of the text about examples, methods, approaches, or sources they'd take to something discussed in the text.
- The prompt should be standalone on its own, not referring to the text in any way.

GUIDELINES FOR THE RESPONSE:
- The response should be a natural standalone answer to the prompt, without referring to the text in any way.
- The response should describe the examples, methods, approaches, or sources that the author uses in the text.
- The response should clearly be in agreement with something the author of the text would say.
- The response should be written in a way that is consistent with the author's voice and style, though it should not be an exact quote.

Some examples:
- If the text is an argument for the democratic peace theory, that mentions post-WW2 Europe as a key example, the prompt could be "What examples would you use to illustrate the democratic peace theory?" and the response could then mention those examples.
- If the text is a guide to product development that heavily emphasizes the role of customer interviews, the prompt could be "How would you approach software product development?" and the response could highlight customer interviews as the core.
- If the text has a bibliography or otherwise links extensively to other work, the prompt could be "What sources would you go to to study [topic/question]?" or "Where would you look for information on [topic/question]?" and the response could mention the sources and the types of sources the author would go to.
- If the piece is a statistical analysis of a question about which football strategies work best, the prompt might be "How would you pick between defensive and offensive strategies in football?" and the response could emphasize that a statistical analysis based on data is best (and mention any conclusions from the analysis in the text, if directly applicable to the prompt).

Please use the tool calls available to you to log these prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_examples_methods_sources",
            tool_description="Log examples and methods the author uses to illustrate or reason about topics",
            output_filename="examples_methods_sources.jsonl",
            augmentation_name="examples_methods_sources",
        )


class ChoicesAugmentation(SimpleQAAugmentation):
    # TODO: finish class
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Based on this text, generate 0-5 prompt/response pairs where the prompt asks the author of the text to choose between two things, and the response is the author's choice and reasoning based on their perspective shown in the text.

GUIDELINES FOR THE PROMPT:
- The prompt should present a choice between two options related to topics discussed in the text.
- The prompt should be standalone on its own, not referring to the text in any way.
- The choice should be one where the author would have a clear preference based on their views in the text.

GUIDELINES FOR THE RESPONSE:
- The response should state the author's choice and provide reasoning consistent with their perspective in the text.
- The response should be a natural standalone answer to the prompt.
- The response should clearly reflect the author's values, priorities, or reasoning as shown in the text.
- The response should be written in a way that is consistent with the author's voice and style.

Please use the tool calls available to you to log these prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_choices",
            tool_description="Log choices the author would make between options based on their perspective",
            output_filename="choices.jsonl",
            augmentation_name="choices",
        )


class ChainOfThoughtAugmentation(SimpleQAAugmentation):
    # TODO: finish class
    def __init__(self):
        prompt_template = """Read this text:
{text_content}

Read this text:
{text_content}

Based on this text, write a prompt chain-of-thought transcript, like a transcription of the author's thoughts as they reason through the topics and considerations in the text.

GUIDELINES FOR THE PROMPT:
- The prompt should say something like "Let's think about [topic/question/concern]" (but vary the phrasing)
- The prompt should be standalone on its own, not referring to the text in any way.

GUIDELINES FOR THE RESPONSE:
- The response should be written in a discursive, chain-of-thought style, as if the author is thinking aloud.
- The response should not quote the text directly
- The response should follow the logical structure and arguments of the text
- The response should not be polished; it should reach dead-ends, backtrack, and then rework again, like a discovery process filling out what eventually becomes the content and argument of the text.

Please use the tool calls available to you to log 0-3 such prompt/response pairs."""

        super().__init__(
            prompt_template=prompt_template,
            tool_name="log_chain_of_thought",
            tool_description="Log chain-of-thought reasoning process that might have led to the text",
            output_filename="chain_of_thought.jsonl",
            augmentation_name="chain_of_thought",
        )
