# This file defines the central registry for data augmentations, providing
# class mappings for instantiation. Output folders are now handled by the base
# classes in each augmentation file to match data structure types.
#
# AVAILABLE_AUGMENTATIONS: Maps augmentation names to their classes
# get_augmentation_by_name: Creates augmentation instances by name

from typing import Dict, Type
import inspect
import warnings

# Import all augmentation classes
from .augmentations import DataAugmentation
from .qa import (
    QAAugmentation,
    TakeAugmentation,
    ConceptExplanationAugmentation,
    NotesOnTopicAugmentation,
    FillInNotesAugmentation,
    NewsArticleAugmentation,
    InterviewAugmentation,
    TwitterThreadAugmentation,
    BeliefsOnTopicAugmentation,
    ExamplesMethodsSourcesAugmentation,
    ChoicesAugmentation,
    ChainOfThoughtAugmentation,
)
from .contrast_pairs import TakeContrastAugmentation, StylePairsAugmentation
from .util_augmentations import (
    TextCleanAugmentation,
    TextChunkAugmentation,
    PunctuationSpaceAugmentation,
    MarkdownHeaderChunkAugmentation,
)
from .conversation import TopicChatAugmentation, TranscriptConversationAugmentation


# Dictionary to map augmentation names to classes
AVAILABLE_AUGMENTATIONS: Dict[str, Type[DataAugmentation]] = {
    # QA augmentations from qa.py
    "qa": QAAugmentation,
    "takes": TakeAugmentation,
    "concept_explanations": ConceptExplanationAugmentation,
    "notes_on_topic": NotesOnTopicAugmentation,
    "fill_in_notes": FillInNotesAugmentation,
    "news_articles": NewsArticleAugmentation,
    "interviews": InterviewAugmentation,
    "twitter_threads": TwitterThreadAugmentation,
    "beliefs_on_topic": BeliefsOnTopicAugmentation,
    "examples_methods_sources": ExamplesMethodsSourcesAugmentation,
    "choices": ChoicesAugmentation,
    "chain_of_thought": ChainOfThoughtAugmentation,
    # contrast pair augmentations from contrast_pairs.py
    "take_contrast": TakeContrastAugmentation,
    "style_pairs": StylePairsAugmentation,
    # conversation augmentations from conversation.py
    "topic_chat": TopicChatAugmentation,
    "transcript_conversation": TranscriptConversationAugmentation,
    # utility augmentations from util_augmentations.py
    "text_clean": TextCleanAugmentation,
    "text_chunk": TextChunkAugmentation,
    "punctuation_space": PunctuationSpaceAugmentation,
    "markdown_header_chunk": MarkdownHeaderChunkAugmentation,
}


def get_augmentation_by_name(augmentation_name: str, **kwargs) -> DataAugmentation:
    """
    Instantiates and returns a data augmentation based on its name.

    Args:
        augmentation_name: The registered name of the augmentation.
        **kwargs: Additional arguments to pass to the augmentation's constructor.

    Returns:
        An instance of the requested DataAugmentation.

    Raises:
        ValueError: If the augmentation_name is unknown or required parameters are missing.
    """
    augmentation_class = AVAILABLE_AUGMENTATIONS.get(augmentation_name.lower())
    if augmentation_class:
        # Get the constructor signature to filter out unsupported parameters
        constructor_signature = inspect.signature(augmentation_class.__init__)

        # Filter kwargs to only include parameters the constructor accepts
        filtered_kwargs = {}
        unused_params = []

        for param_name, param_value in kwargs.items():
            if param_name in constructor_signature.parameters or any(
                param.kind == param.VAR_KEYWORD
                for param in constructor_signature.parameters.values()
            ):
                # Parameter is accepted (either explicitly or via **kwargs)
                filtered_kwargs[param_name] = param_value
            else:
                # Parameter is not accepted
                unused_params.append(param_name)

        # Warn about unused parameters
        if unused_params:
            warnings.warn(
                f"Parameters {unused_params} are not used by {augmentation_name} augmentation and will be ignored.",
                UserWarning,
            )

        # Create the augmentation instance with only supported parameters
        instance = augmentation_class(**filtered_kwargs)

        # Validate that all required parameters are present
        instance.validate_params()

        return instance
    else:
        available = list(AVAILABLE_AUGMENTATIONS.keys())
        raise ValueError(
            f"Unknown augmentation: {augmentation_name}. Available: {available}"
        )


def list_all_augmentations() -> list[str]:
    """
    Returns a list of all registered augmentation names.

    Returns:
        List of all augmentation names
    """
    return list(AVAILABLE_AUGMENTATIONS.keys())
