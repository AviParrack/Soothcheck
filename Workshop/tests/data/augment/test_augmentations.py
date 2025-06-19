import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
from typing import List, Dict, Any
import os
from pathlib import Path

from src.data.augment.augmentations import DataAugmentation, BaseLLMAugmentation
from src.data.augment.qa import NewsArticleAugmentation, SimpleQAAugmentation
from src.data.augment.augmentation_registry import get_augmentation_by_name
from src.data.augment.conversation import TopicChatAugmentation


# Mock response for LLM calls - this needs to match the structure expected by the code
def create_mock_llm_response(tool_name="log_news_articles", pairs=None):
    if pairs is None:
        pairs = [{"prompt": "Test prompt", "response": "Test response"}]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.tool_calls = [MagicMock()]
    mock_response.choices[0].message.tool_calls[0].function = MagicMock()
    mock_response.choices[0].message.tool_calls[0].function.name = tool_name
    mock_response.choices[0].message.tool_calls[0].function.arguments = json.dumps(
        {"pairs": pairs}
    )
    return mock_response


@pytest.fixture
def mock_litellm():
    """Fixture to mock litellm_completion"""
    with patch("src.data.augment.augmentations.litellm_completion") as mock:
        mock.return_value = create_mock_llm_response()
        yield mock


@pytest.fixture
def sample_text():
    """Sample text for testing augmentations"""
    return "This is a sample text for testing augmentations."


@pytest.fixture
def sample_file_path(tmp_path):
    """Create a sample file for testing"""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample content")
    return str(file_path)


@pytest.fixture
def mock_file_operations():
    """Mock file operations to allow content verification"""
    written_content = []

    def mock_write(content):
        written_content.append(content)
        return len(content)

    mock_file_handle = MagicMock()
    mock_file_handle.write = mock_write
    mock_file_handle.__enter__ = lambda x: mock_file_handle
    mock_file_handle.__exit__ = lambda x, y, z, w: None

    with patch("builtins.open", return_value=mock_file_handle) as mock_open_func:
        with patch("os.makedirs") as mock_makedirs:
            with patch("os.path.exists", return_value=False):
                yield mock_open_func, mock_makedirs, written_content


def test_news_articles_augmentation_requires_author_name():
    """Test that NewsArticleAugmentation requires author_name"""
    with pytest.raises(
        ValueError, match="Author name is required for NewsArticleAugmentation"
    ):
        NewsArticleAugmentation()


def test_news_articles_augmentation_with_author_name():
    """Test that NewsArticleAugmentation works with author_name"""
    aug = NewsArticleAugmentation(author_name="Test Author")
    assert aug.author_name == "Test Author"


def test_news_articles_augmentation_removes_author_name_from_kwargs():
    """Test that author_name is properly handled and not passed to parent unnecessarily"""
    # This tests that NewsArticleAugmentation properly handles author_name
    # and doesn't cause issues with the parent class
    aug = NewsArticleAugmentation(author_name="Test Author")
    assert aug.author_name == "Test Author"

    # Test that SimpleQAAugmentation doesn't require author_name
    simple_aug = SimpleQAAugmentation(
        prompt_template="test",
        tool_name="test",
        tool_description="test",
        output_filename="test.jsonl",
    )
    assert simple_aug is not None


def test_get_augmentation_by_name_with_params():
    """Test that get_augmentation_by_name properly handles parameters"""
    # Should work with author_name
    aug = get_augmentation_by_name("news_articles", author_name="Test Author")
    assert isinstance(aug, NewsArticleAugmentation)
    assert aug.author_name == "Test Author"

    # Should fail without author_name
    with pytest.raises(
        ValueError, match="Author name is required for NewsArticleAugmentation"
    ):
        get_augmentation_by_name("news_articles")


def test_news_articles_augmentation_process(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test the full augmentation process for news articles"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Test augment method
    result = aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Should return empty list as files are only created during finalize
    assert result == []

    # Test finalize method
    files = aug.finalize()
    assert len(files) == 1
    assert files[0].endswith("news_articles.jsonl")


def test_topic_chat_augmentation_requires_author_name():
    """Test that TopicChatAugmentation requires author_name"""
    with pytest.raises(
        ValueError, match="Author name is required for TopicChatAugmentation"
    ):
        TopicChatAugmentation()


def test_topic_chat_augmentation_with_author_name():
    """Test that TopicChatAugmentation works with author_name"""
    aug = TopicChatAugmentation(author_name="Test Author")
    assert aug.author_name == "Test Author"


def test_augmentation_parameter_passing():
    """Test that parameters are properly passed through the augmentation chain"""
    # Test with extra parameters that should be stored in params
    aug = NewsArticleAugmentation(
        author_name="Test Author", extra_param="should be ignored"
    )
    assert aug.author_name == "Test Author"
    assert aug.params.get("extra_param") == "should be ignored"


def test_augmentation_metadata_in_output(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test that augmentation metadata is included in output"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Run augmentation
    aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Get output file
    output_files = aug.finalize()
    assert len(output_files) == 1

    # Verify file operations
    mock_makedirs.assert_called_once()
    mock_open_func.assert_called()

    # Get the written content
    assert len(written_content) > 0
    result = json.loads(written_content[0])

    # Check metadata
    assert result["author_name"] == "Test Author"
    assert result["source"] == sample_file_path
    assert result["augmentation_name"] == "news_articles"
    assert "prompt" in result
    assert "response" in result


def test_augmentation_error_handling(
    mock_litellm, sample_text, sample_file_path, tmp_path
):
    """Test error handling in augmentations"""
    # Make mock raise an exception
    mock_litellm.side_effect = Exception("Test error")

    aug = NewsArticleAugmentation(author_name="Test Author")

    # Should handle error gracefully
    result = aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    assert result == []  # Should return empty list on error


def test_augmentation_empty_content(mock_litellm, sample_file_path, tmp_path):
    """Test handling of empty content"""
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Should handle empty content gracefully
    result = aug.augment(
        file_path=sample_file_path, file_content="", raw_data_dir=str(tmp_path)
    )

    assert result == []  # Should return empty list for empty content


def test_augmentation_output_directory_creation(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test that augmentation creates output directory if it doesn't exist"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Run augmentation
    aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Finalize to trigger file creation
    aug.finalize()

    # Verify directory creation
    mock_makedirs.assert_called_once()
    expected_dir = os.path.join(str(tmp_path), "augmentations/prompt_response_pairs")
    mock_makedirs.assert_called_with(expected_dir, exist_ok=True)


def test_augmentation_output_format(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test that augmentation output follows the correct format"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Run augmentation
    aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Finalize to trigger file creation
    aug.finalize()

    # Get the written content
    assert len(written_content) > 0
    result = json.loads(written_content[0])

    # Verify required fields
    required_fields = [
        "prompt",
        "response",
        "source",
        "augmentation_name",
        "author_name",
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Verify field types
    assert isinstance(result["prompt"], str)
    assert isinstance(result["response"], str)
    assert isinstance(result["source"], str)
    assert isinstance(result["augmentation_name"], str)
    assert isinstance(result["author_name"], str)


def test_augmentation_multiple_articles(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test handling of multiple articles in a single augmentation"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations

    # Modify mock response to include multiple articles
    mock_litellm.return_value = create_mock_llm_response(
        "log_news_articles",
        [
            {"prompt": "Test prompt 1", "response": "Test response 1"},
            {"prompt": "Test prompt 2", "response": "Test response 2"},
        ],
    )

    aug = NewsArticleAugmentation(author_name="Test Author")

    # Run augmentation
    aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Finalize to trigger file creation
    aug.finalize()

    # Get the written content - should be two separate JSONL lines
    assert len(written_content) == 2
    result1 = json.loads(written_content[0])
    result2 = json.loads(written_content[1])

    # Verify both articles are included with empty prompts
    assert result1["prompt"] == ""
    assert result2["prompt"] == ""
    assert result1["response"] == "Test response 1"
    assert result2["response"] == "Test response 2"


def test_augmentation_file_naming(
    mock_litellm, sample_text, sample_file_path, tmp_path, mock_file_operations
):
    """Test that augmentation creates files with correct naming pattern"""
    mock_open_func, mock_makedirs, written_content = mock_file_operations
    aug = NewsArticleAugmentation(author_name="Test Author")

    # Run augmentation
    aug.augment(
        file_path=sample_file_path, file_content=sample_text, raw_data_dir=str(tmp_path)
    )

    # Get output file
    output_files = aug.finalize()
    assert len(output_files) == 1
    output_file = output_files[0]

    # Verify file naming pattern
    assert output_file.endswith(".jsonl")
    assert "news_articles" in output_file
    assert "augmentations/prompt_response_pairs" in output_file
