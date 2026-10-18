"""
Utility Functions for arXiv Research Companion.

Responsibilities:
- Provide helper functions used across multiple modules.
- Common functionalities:
    * Logging setup: configure loggers with consistent format and level.
    * Text processing: clean and normalize text (remove extra whitespace, unicode normalization).
    * arXiv ID validation: check if a string is a valid arXiv identifier.
    * Date handling: parse and format dates for arXiv API queries and storage.
    * Error handling: decorators or functions for retrying operations with exponential backoff.
    * Progress tracking: simple progress bars or status reporting for long-running tasks.
    * File operations: safe reading/writing of JSON, pickle, or other data files.
    * Environment variable loading: wrapper around python-dotenv with default fallbacks.
    * Metrics collection: simple counters and timers for monitoring performance.
- Designed to be lightweight and dependency-light (mostly using standard library).
- Functions should be pure where possible for easy testing.
- Avoid putting complex business logic here; keep it as genuine utilities.
"""

import os
import re
import json
import pickle
import hashlib
import logging
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from functools import wraps
import time
import random


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> None:
    """
    Configure logging for the application with consistent format and level.

    Args:
        level: Logging level (e.g., logging.INFO, "INFO")
        format_str: Format string for log messages
        log_file: Optional path to log file
        console_output: Whether to output to console
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    handlers = []

    if log_file:
        # Ensure directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    if console_output:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def clean_text(text: str) -> str:
    """
    Clean and normalize text by removing extra whitespace and normalizing unicode.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def is_valid_arxiv_id(arxiv_id: str) -> bool:
    """
    Check if a string is a valid arXiv identifier.

    Args:
        arxiv_id: String to check

    Returns:
        True if valid arXiv ID, False otherwise
    """
    if not arxiv_id:
        return False

    # arXiv ID formats:
    # Old format: hep-th/9901010 (with version optional: hep-th/9901010v1 or hep-th/9901010V1)
    # New format: 1501.00001 (with version optional: 1501.00001v1 or 1501.00001V1)

    # Remove any version suffix for checking (case-insensitive for v/V)
    id_without_version = re.sub(r'[vV]\d+$', '', arxiv_id.strip())

    # Check new format: YYYY.NNNNN
    if re.match(r'^\d{4}\.\d{5}$', id_without_version):
        return True

    # Check old format: archive/class/YYMMNNN (with optional version)
    # Archive identifiers can contain letters, digits, hyphens, and dots
    # but must start with a letter and follow arXiv taxonomy conventions
    if re.match(r'^[a-zA-Z][a-zA-Z0-9\-]*(\.[a-zA-Z0-9\-]+)?/\d{7}$', id_without_version):
        return True

    return False


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse a date string into a datetime object.

    Args:
        date_str: Date string in various formats

    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not date_str:
        return None

    # Try common date formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%B %d, %Y",
        "%d %B %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def format_date(date: datetime, format_str: str = "%Y-%m-%d") -> str:
    """
    Format a datetime object as a string.

    Args:
        date: DateTime object to format
        format_str: Format string (default: YYYY-MM-DD)

    Returns:
        Formatted date string
    """
    if not isinstance(date, datetime):
        raise ValueError("Input must be a datetime object")

    return date.strftime(format_str)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for retrying operations with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        break

                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** attempt),
                        max_delay
                    )

                    # Add jitter if enabled
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # 0.5 to 1.0 multiplier

                    time.sleep(delay)

            # If we got here, all retries failed
            raise last_exception

        return wrapper
    return decorator


def show_progress(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 30,
    fill: str = "█",
    print_end: str = "\r"
) -> None:
    """
    Display a simple progress bar in the terminal.

    Args:
        current: Current progress value
        total: Total progress value
        prefix: String to display before progress bar
        suffix: String to display after progress bar
        length: Length of progress bar in characters
        fill: Character to use for filled portion
        print_end: End character for print (default: carriage return)
    """
    percent = f"{100 * (current / float(total)):.1f}"
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)

    # Print newline on completion
    if current == total:
        print()


def save_json(data: Any, file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Safely save data as JSON to a file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path to output file
        indent: JSON indentation level
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first, then rename for atomicity
    temp_path = file_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    temp_path.replace(file_path)


def load_json(file_path: Union[str, Path]) -> Any:
    """
    Safely load JSON data from a file.

    Args:
        file_path: Path to input file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_pickle(data: Any, file_path: Union[str, Path]) -> None:
    """
    Safely save data as pickle to a file.

    Args:
        data: Data to save
        file_path: Path to output file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first, then rename for atomicity
    temp_path = file_path.with_suffix('.tmp')
    with open(temp_path, 'wb') as f:
        pickle.dump(data, f)

    temp_path.replace(file_path)


def load_pickle(file_path: Union[str, Path]) -> Any:
    """
    Safely load pickle data from a file.

    Args:
        file_path: Path to input file

    Returns:
        Loaded data

    Raises:
        FileNotFoundError: If file doesn't exist
        pickle.PickleError: If file contains invalid pickle data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'rb') as f:
        return pickle.load(f)


def hash_string(text: str) -> str:
    """
    Generate a SHA256 hash of a string.

    Args:
        text: Input string

    Returns:
        Hexadecimal SHA256 hash
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def load_env_var(key: str, default: Any = None, required: bool = False) -> Any:
    """
    Load an environment variable with optional default and validation.

    Args:
        key: Environment variable name
        default: Default value if not found
        required: Whether the variable is required

    Returns:
        Environment variable value or default

    Raises:
        ValueError: If required variable is missing
    """
    value = os.getenv(key)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default

    return value


def timer(func):
    """
    Decorator to time function execution and log the duration.

    Args:
        func: Function to decorate

    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger = logging.getLogger(func.__module__)
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")

        return result

    return wrapper