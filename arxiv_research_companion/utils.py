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