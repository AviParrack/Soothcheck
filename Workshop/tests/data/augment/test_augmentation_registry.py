# Test file for augmentation_registry module

import pytest
from src.data.augment.augmentation_registry import (
    get_augmentation_by_name,
    list_all_augmentations,
    AVAILABLE_AUGMENTATIONS,
)


def test_registry_completeness():
    """Test that all expected augmentations are in the registry"""
    expected_augmentations = {
        "qa",
        "takes",
        "concept_explanations",
        "notes_on_topic",
        "fill_in_notes",
        "news_articles",
        "interviews",
        "twitter_threads",
        "beliefs_on_topic",
        "examples_methods_sources",
        "choices",
        "chain_of_thought",
        "topic_chat",
        "take_contrast",
        "style_pairs",
        "text_clean",
        "text_chunk",
    }

    actual_augmentations = set(AVAILABLE_AUGMENTATIONS.keys())
    assert expected_augmentations == actual_augmentations


def test_get_augmentation_by_name():
    """Test getting augmentation instances by name"""
    # Test successful instantiation
    aug = get_augmentation_by_name("qa")
    assert aug is not None
    assert hasattr(aug, "augment")

    # Test error for unknown augmentation
    with pytest.raises(ValueError, match="Unknown augmentation: unknown"):
        get_augmentation_by_name("unknown")


def test_list_all_augmentations():
    """Test listing all augmentations"""
    augmentations = list_all_augmentations()
    assert isinstance(augmentations, list)
    assert len(augmentations) > 0
    assert "qa" in augmentations
    assert "takes" in augmentations


def test_augmentation_folder_structure():
    """Test that augmentations use the correct folder structure"""
    # Test QA augmentations
    qa_aug = get_augmentation_by_name("qa")
    assert qa_aug._get_target_folder() == "augmentations/prompt_response_pairs"

    # Test conversation augmentations
    chat_aug = get_augmentation_by_name("topic_chat", author_name="Test Author")
    assert chat_aug._get_target_folder() == "augmentations/chats"

    # Test contrast pair augmentations
    contrast_aug = get_augmentation_by_name("take_contrast")
    assert contrast_aug._get_target_folder() == "augmentations/contrast_pairs"
