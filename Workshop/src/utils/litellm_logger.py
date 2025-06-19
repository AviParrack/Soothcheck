# This file provides centralized error logging for LiteLLM calls across the project.
# It implements selective error-only logging that writes to log files instead of cluttering stdout.
#
# Key functions:
# - setup_litellm_error_logging: Sets up the logging configuration and creates log directory
# - litellm_error_logger: Logger function that only logs errors and failures
# - litellm_completion: Wrapper around litellm.completion with error logging and intelligent retry

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
import random

import litellm

# Suppress LiteLLM's built-in "Give Feedback" messages for retried errors
litellm.suppress_debug_info = True


def setup_litellm_error_logging(log_dir: str = "logs") -> None:
    """
    Sets up error logging for LiteLLM.
    Creates the log directory if it doesn't exist and configures the logger.

    Args:
        log_dir: Directory to store log files (default: "logs")
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Configure logging
    log_file = log_path / f"litellm_errors_{datetime.now().strftime('%Y%m%d')}.log"

    # Set up the logger
    logger = logging.getLogger("litellm_errors")
    logger.setLevel(logging.INFO)  # Capture INFO level to see retries

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)  # Log INFO and above

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(file_handler)

    print(f"LiteLLM error logging configured. Errors will be logged to: {log_file}")


def litellm_error_logger(model_call_dict: Dict[str, Any]) -> None:
    """
    Custom logger function for LiteLLM that logs errors and retries.
    This is passed to litellm.completion() via the logger_fn parameter.

    Args:
        model_call_dict: Dictionary containing call details from LiteLLM
    """
    log_event_type = model_call_dict.get("log_event_type")

    if log_event_type == "failed_api_call":
        logger = logging.getLogger("litellm_errors")

        # Extract key information
        model = model_call_dict.get("model", "Unknown")
        exception = model_call_dict.get("exception", "Unknown error")
        traceback = model_call_dict.get("traceback_exception", "No traceback available")

        # Log the error with structured information
        error_msg = (
            f"LiteLLM API Call Failed\n"
            f"Model: {model}\n"
            f"Exception: {exception}\n"
            f"Traceback: {traceback}\n"
            f"Full call details: {json.dumps(model_call_dict, indent=2, default=str)}"
        )

        logger.error(error_msg)

        # Also print to console for immediate visibility (but not the full spam)
        print(
            f"LiteLLM Error: {exception} (Model: {model}) - Check logs/litellm_errors_*.log for details"
        )

    # Log retry attempts to understand transient issues
    elif log_event_type == "pre_api_call":
        # Check if this looks like a retry attempt
        kwargs = model_call_dict.get("kwargs", {})
        if "num_retries" in kwargs or "retry" in str(model_call_dict).lower():
            logger = logging.getLogger("litellm_errors")
            model = model_call_dict.get("model", "Unknown")
            logger.info(f"LiteLLM retry attempt for model {model}: {model_call_dict}")

        # Only log pre-call issues if there are obvious error indicators
        call_str = str(model_call_dict).lower()
        if any(
            keyword in call_str for keyword in ["error", "fail", "exception", "invalid"]
        ):
            logger = logging.getLogger("litellm_errors")
            logger.warning(f"LiteLLM Pre-call issue detected: {model_call_dict}")

    # Log successful retries to see what was being retried
    elif log_event_type == "successful_api_call":
        # Only log if this was likely a retry (has retry-related fields)
        if any(key in model_call_dict for key in ["num_retries", "retry_count"]):
            logger = logging.getLogger("litellm_errors")
            model = model_call_dict.get("model", "Unknown")
            logger.info(f"LiteLLM successful retry for model {model}")


def _is_retryable_error(exception: Exception) -> bool:
    """
    Determine if an error is worth retrying.

    Args:
        exception: The exception that occurred

    Returns:
        True if the error is retryable, False otherwise
    """
    error_str = str(exception).lower()

    # Rate limiting errors - definitely retry
    if any(
        keyword in error_str
        for keyword in [
            "rate limit",
            "rate_limit",
            "quota exceeded",
            "too many requests",
            "throttled",
            "rate exceeded",
        ]
    ):
        return True

    # Temporary server errors - retry
    if any(
        keyword in error_str
        for keyword in [
            "server error",
            "internal error",
            "service unavailable",
            "timeout",
            "connection",
            "network",
            "temporary",
        ]
    ):
        return True

    # HTTP status codes that are retryable
    if any(code in error_str for code in ["500", "502", "503", "504", "429"]):
        return True

    # Don't retry authentication, authorization, or validation errors
    if any(
        keyword in error_str
        for keyword in [
            "authentication",
            "authorization",
            "invalid api key",
            "unauthorized",
            "forbidden",
            "not found",
            "bad request",
            "invalid request",
        ]
    ):
        return False

    # Default to not retrying unknown errors
    return False


def _calculate_backoff_delay(
    attempt: int, base_delay: float = 30.0, max_delay: float = 120.0
) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds

    Returns:
        Delay in seconds
    """
    delay = base_delay * (1.5**attempt)

    # Cap at max_delay
    delay = min(delay, max_delay)

    # Add jitter (random factor between 0.5 and 1.5)
    jitter = random.uniform(0.5, 1.5)

    return delay * jitter


def litellm_completion(
    model: str,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    temperature: Optional[float] = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs,
) -> Any:
    """
    Wrapper around litellm.completion that automatically includes error logging and intelligent retry logic.
    Use this instead of calling litellm.completion directly.

    Args:
        model: Model name/identifier
        messages: List of message dictionaries
        tools: Optional tools for function calling
        tool_choice: Optional tool choice setting
        temperature: Optional temperature setting
        max_retries: Maximum number of retry attempts (default: 3)
        base_delay: Base delay for exponential backoff in seconds (default: 1.0)
        **kwargs: Additional arguments passed to litellm.completion

    Returns:
        Response from litellm.completion

    Raises:
        Exception: The last exception encountered after all retries are exhausted
    """
    logger = logging.getLogger("litellm_errors")
    last_exception = None

    for attempt in range(
        max_retries + 1
    ):  # +1 because we want max_retries actual retries
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                logger_fn=litellm_error_logger,
                **kwargs,
            )

            # If we get here, the call succeeded
            if attempt > 0:
                logger.info(
                    f"LiteLLM call succeeded after {attempt} retries (model: {model})"
                )
                print(f"✓ LiteLLM call succeeded after {attempt} retries")

            return response

        except Exception as e:
            last_exception = e
            error_msg = str(e)

            # Always log serious errors
            logger.error(
                f"LiteLLM call failed (attempt {attempt + 1}/{max_retries + 1})\n"
                f"Model: {model}\n"
                f"Error: {error_msg}\n"
                f"Error type: {type(e).__name__}"
            )

            # Always print error to console for immediate visibility
            print(
                f"⚠️  LiteLLM Error (attempt {attempt + 1}/{max_retries + 1}): {error_msg}"
            )

            # Check if this is the last attempt
            if attempt >= max_retries:
                logger.error(
                    f"LiteLLM call failed permanently after {max_retries} retries (model: {model})"
                )
                print(f"✗ LiteLLM call failed permanently after {max_retries} retries")
                break

            # Check if error is retryable
            if not _is_retryable_error(e):
                logger.warning(
                    f"LiteLLM error is not retryable, giving up (model: {model}): {error_msg}"
                )
                print(f"✗ LiteLLM error is not retryable, giving up")
                break

            # Calculate delay and wait
            delay = _calculate_backoff_delay(attempt, base_delay)
            logger.info(
                f"LiteLLM retrying in {delay:.1f} seconds (attempt {attempt + 1}/{max_retries + 1})"
            )
            print(
                f"⏳ Retrying in {delay:.1f}s (attempt {attempt + 2}/{max_retries + 1})"
            )

            time.sleep(delay)

    # If we get here, all retries failed
    raise last_exception


# Initialize error logging when this module is imported
setup_litellm_error_logging()
