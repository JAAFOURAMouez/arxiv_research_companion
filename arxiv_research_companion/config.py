"""
Configuration Management for arXiv Research Companion.

Responsibilities:
- Centralize configuration parameters for the application.
- Support loading from environment variables (for secrets) and a configuration file.
- Provide default values for all configurable parameters.
- Define sections for different components:
    * arXiv API settings: categories, rate limit, date ranges for ingestion.
    * Embedding model: model name, batch size, device (CPU/GPU).
    * Vector database: path, collection name, distance metric.
    * Language model: provider (HF API, local), model name, API token (if needed), generation parameters.
    * Ingestion scheduler: interval (e.g., daily), time of day.
    * Interface: Gradio theme, port, share flag.
    * Logging: level, format, file path.
- Use python-dotenv to load environment variables from a .env file.
- Provide a simple way to access settings throughout the code (e.g., config.get('section.key')).
- Validate configuration values where possible (e.g., check that paths exist, numbers are in range).
- Allow overriding via command-line arguments for development convenience.
- Ensure that sensitive information (like API tokens) is not hardcoded and is loaded from environment.
"""