# Test file for file_processor module showing how the refactored code can be tested

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import tempfile
import os

from src.data.augment.file_processor import (
    find_files_for_augmentation,
    create_augmentation_instances,
    process_single_augmentation,
)
from src.data.augment.augmentation_registry import get_augmentation_by_name
from src.config_models.data_config import DatasetConfig


@pytest.fixture
def mock_config():
    """Create a mock dataset configuration"""
    config = MagicMock(spec=DatasetConfig)
    config.get_absolute_raw_data_path.return_value = "/fake/raw/data"
    return config


@pytest.fixture
def sample_augmentation_config():
    """Sample augmentation configuration dictionary"""
    return {
        "name": "qa",
        "sources": ["text"],
    }


def test_find_files_for_augmentation(mock_config, sample_augmentation_config):
    """Test finding files for augmentation"""
    with patch("src.data.augment.file_processor.find_files_for_component") as mock_find:
        mock_find.return_value = ["/path/to/file1.txt", "/path/to/file2.txt"]

        result = find_files_for_augmentation(
            mock_config, "test_dataset", sample_augmentation_config
        )

        assert result == ["/path/to/file1.txt", "/path/to/file2.txt"]
        mock_find.assert_called_once_with(
            mock_config, "test_dataset", sample_augmentation_config, "augmentation"
        )


def test_create_augmentation_instances():
    """Test creating augmentation instances with real classes (no parameters needed)"""
    augmentation_configs = [
        {"name": "qa", "sources": ["text"]},
        {"name": "takes", "sources": ["text"]},
    ]

    result = create_augmentation_instances(augmentation_configs)

    assert len(result) == 2
    assert "qa" in result
    assert "takes" in result
    # Verify the instances are of the correct type
    from src.data.augment.qa import QAAugmentation, TakeAugmentation

    assert isinstance(result["qa"], QAAugmentation)
    assert isinstance(result["takes"], TakeAugmentation)


def test_create_augmentation_instances_with_author_name():
    """Test creating augmentation instances with author name using real classes"""
    augmentation_configs = [
        {"name": "topic_chat", "sources": ["text"]},
        {"name": "qa", "sources": ["text"]},
    ]

    result = create_augmentation_instances(
        augmentation_configs, author_name="Test Author"
    )

    # topic_chat requires author_name, so it should be created successfully
    assert "topic_chat" in result
    assert hasattr(result["topic_chat"], "author_name")
    assert result["topic_chat"].author_name == "Test Author"

    # qa doesn't require author_name but should still be created
    assert "qa" in result

    # Verify the instances are of the correct type
    from src.data.augment.conversation import TopicChatAugmentation
    from src.data.augment.qa import QAAugmentation

    assert isinstance(result["topic_chat"], TopicChatAugmentation)
    assert isinstance(result["qa"], QAAugmentation)


def test_create_augmentation_instances_with_multiple_params():
    """Test that the dynamic parameter system only passes required parameters"""
    augmentation_configs = [
        {"name": "custom_aug", "sources": ["text"]},
    ]

    # Test with news_articles which requires author_name
    augmentation_configs = [
        {"name": "news_articles", "sources": ["text"]},
    ]

    result = create_augmentation_instances(
        augmentation_configs,
        author_name="Test Author",
        unused_param="ignored",  # This should be ignored
    )

    # news_articles requires author_name, so it should be created successfully
    assert "news_articles" in result
    assert hasattr(result["news_articles"], "author_name")
    assert result["news_articles"].author_name == "Test Author"

    # Verify the instance is of the correct type
    from src.data.augment.qa import NewsArticleAugmentation

    assert isinstance(result["news_articles"], NewsArticleAugmentation)


def test_process_single_augmentation():
    """Test processing a single file with a single augmentation"""
    mock_augmentation = MagicMock()
    mock_augmentation.augment.return_value = ["created_file.txt"]

    config_dict = {
        "name": "qa",
        "sources": ["text"],
        "extra_param": "test_value",
    }

    result = process_single_augmentation(
        file_path="/path/to/test.txt",
        file_content="test content",
        augmentation_name="qa",
        augmentation_instance=mock_augmentation,
        augmentation_config_item_dict=config_dict,
        raw_data_dir_abs="/raw/data",
    )

    # Check result
    assert result == ("qa", 1)

    # Check that augment was called with correct parameters
    mock_augmentation.augment.assert_called_once()
    call_args = mock_augmentation.augment.call_args[1]

    assert call_args["file_path"] == "/path/to/test.txt"
    assert call_args["file_content"] == "test content"
    assert call_args["raw_data_dir"] == "/raw/data"
    assert call_args["extra_param"] == "test_value"
    # name and sources should be filtered out
    assert "name" not in call_args
    assert "sources" not in call_args


def test_process_single_augmentation_no_files_created():
    """Test processing when no files are created"""
    mock_augmentation = MagicMock()
    mock_augmentation.augment.return_value = []  # No files created

    config_dict = {"name": "qa", "sources": ["text"]}

    result = process_single_augmentation(
        file_path="/path/to/test.txt",
        file_content="test content",
        augmentation_name="qa",
        augmentation_instance=mock_augmentation,
        augmentation_config_item_dict=config_dict,
        raw_data_dir_abs="/raw/data",
    )

    assert result == ("qa", 0)


def test_process_single_augmentation_returns_none():
    """Test processing when augmentation returns None"""
    mock_augmentation = MagicMock()
    mock_augmentation.augment.return_value = None

    config_dict = {"name": "qa", "sources": ["text"]}

    result = process_single_augmentation(
        file_path="/path/to/test.txt",
        file_content="test content",
        augmentation_name="qa",
        augmentation_instance=mock_augmentation,
        augmentation_config_item_dict=config_dict,
        raw_data_dir_abs="/raw/data",
    )

    assert result == ("qa", 0)
