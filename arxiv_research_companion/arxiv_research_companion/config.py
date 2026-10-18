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

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from environment variables and .env file.

    Args:
        config_path: Optional path to a .env file. If None, looks for .env in current directory.

    Returns:
        Dictionary containing all configuration parameters organized by section.
    """
    # Load environment variables from .env file
    if config_path is None:
        config_path = ".env"

    env_path = Path(config_path)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Build configuration dictionary with defaults
    config = {
        # arXiv API settings
        "arxiv": {
            "categories": os.getenv("ARXIV_CATEGORIES", "cs.AI,cs.LG,stat.ML").split(","),
            "max_results_per_request": int(os.getenv("ARXIV_MAX_RESULTS_PER_REQUEST", "100")),
            "rate_limit_seconds": float(os.getenv("ARXIV_RATE_LIMIT_SECONDS", "3.0")),
            "days_back": int(os.getenv("ARXIV_DAYS_BACK", "1")),
            # cap per ingestion run (oldest-first cursor => capped runs leave no gap)
            "max_total_per_run": int(os.getenv("ARXIV_MAX_TOTAL_PER_RUN", "300")),
        },

        # Retrieval strategy: dense | bm25 | hybrid | hybrid_rerank.
        # "dense" is simply what the app did before; pick the deployed mode from the
        # numbers produced by eval/run_retrieval_eval.py (see eval/README.md).
        "retrieval": {
            "mode": os.getenv("RETRIEVAL_MODE", "dense"),
            "candidate_depth": int(os.getenv("RETRIEVAL_CANDIDATE_DEPTH", "50")),
            "rerank_depth": int(os.getenv("RETRIEVAL_RERANK_DEPTH", "20")),
            "rrf_k": int(os.getenv("RETRIEVAL_RRF_K", "60")),
            "doc_text": os.getenv("RETRIEVAL_DOC_TEXT", "abstract"),
        },

        # Citation contract on generated answers
        "citations": {
            "enforce": os.getenv("CITATIONS_ENFORCE", "true").lower() == "true",
            "policy": os.getenv("CITATIONS_POLICY", "drop"),
            "max_repair_attempts": int(os.getenv("CITATIONS_MAX_REPAIRS", "1")),
        },

        # Embedding model settings
        "embedding": {
            "model_name": os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
            "batch_size": int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            "device": os.getenv("EMBEDDING_DEVICE", "cpu"),  # or cuda
            "normalize_embeddings": os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
        },

        # Vector database settings
        "vector_db": {
            "path": os.getenv("VECTOR_DB_PATH", "./chroma_db"),
            "collection_name": os.getenv("VECTOR_DB_COLLECTION", "arxiv_papers"),
            "distance_metric": os.getenv("VECTOR_DB_DISTANCE_METRIC", "cosine"),
        },

        # Language model settings
        "llm": {
            # "groq" is selected automatically when GROQ_API_KEY is set (unless LLM_PROVIDER is given)
            "provider": os.getenv("LLM_PROVIDER") or ("groq" if os.getenv("GROQ_API_KEY") else "huggingface_api"),
            "model_name": os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b"),
            "inference_provider": os.getenv("HF_INFERENCE_PROVIDER") or None,
            "api_token": os.getenv("HF_API_TOKEN", ""),  # for Hugging Face Inference API
            "groq_api_key": os.getenv("GROQ_API_KEY", ""),
            "timeout": float(os.getenv("LLM_TIMEOUT", "60.0")),
            "max_retries": int(os.getenv("LLM_MAX_RETRIES", "3")),  # on HTTP 429/5xx
            "retry_base_delay": float(os.getenv("LLM_RETRY_BASE_DELAY", "1.0")),
            "max_new_tokens": int(os.getenv("LLM_MAX_NEW_TOKENS", "100")),
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
            "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
            "do_sample": os.getenv("LLM_DO_SAMPLE", "true").lower() == "true",
        },

        # Ingestion scheduler settings
        "ingestion": {
            "db_path": os.getenv("INGESTION_DB_PATH", "./arxiv_papers.db"),
            "interval_hours": int(os.getenv("INGESTION_INTERVAL_HOURS", "24")),
            "time_of_day": os.getenv("INGESTION_TIME_OF_DAY", "02:00"),  # 2 AM
            "incremental": os.getenv("INGESTION_INCREMENTAL", "true").lower() == "true",
        },

        # Interface settings
        "interface": {
            "theme": os.getenv("INTERFACE_THEME", "default"),
            "port": int(os.getenv("INTERFACE_PORT", "7860")),
            "share": os.getenv("INTERFACE_SHARE", "false").lower() == "true",
            "auth_username": os.getenv("INTERFACE_AUTH_USERNAME", ""),
            "auth_password": os.getenv("INTERFACE_AUTH_PASSWORD", ""),
        },

        # Logging settings
        "logging": {
            "level": os.getenv("LOG_LEVEL", "INFO"),
            "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            "file_path": os.getenv("LOG_FILE_PATH", "arxiv_companion.log"),
            "console_output": os.getenv("LOG_CONSOLE_OUTPUT", "true").lower() == "true",
        },
    }

    # Validate configuration
    _validate_config(config)

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration values.

    Args:
        config: Configuration dictionary to validate.

    Raises:
        ValueError: If any configuration value is invalid.
    """
    # Validate arXiv settings
    if config["arxiv"]["max_results_per_request"] < 1 or config["arxiv"]["max_results_per_request"] > 1000:
        raise ValueError("arXiv max_results_per_request must be between 1 and 1000")

    if config["arxiv"]["rate_limit_seconds"] < 0:
        raise ValueError("arXiv rate_limit_seconds must be non-negative")

    if config["arxiv"]["days_back"] < 0:
        raise ValueError("arXiv days_back must be non-negative")

    if config["arxiv"]["max_total_per_run"] < 1:
        raise ValueError("arXiv max_total_per_run must be at least 1")

    # Validate retrieval / citation settings
    if config["retrieval"]["mode"] not in ("dense", "bm25", "hybrid", "hybrid_rerank"):
        raise ValueError("RETRIEVAL_MODE must be dense, bm25, hybrid or hybrid_rerank")
    if config["retrieval"]["doc_text"] not in ("abstract", "title_abstract"):
        raise ValueError("RETRIEVAL_DOC_TEXT must be abstract or title_abstract")
    if config["citations"]["policy"] not in ("drop", "flag"):
        raise ValueError("CITATIONS_POLICY must be drop or flag")
    if config["llm"]["max_retries"] < 0:
        raise ValueError("LLM max_retries must be non-negative")

    # Validate embedding settings
    if config["embedding"]["batch_size"] < 1:
        raise ValueError("Embedding batch_size must be at least 1")

    if config["embedding"]["device"] not in ["cpu", "cuda"]:
        raise ValueError("Embedding device must be either 'cpu' or 'cuda'")

    # Validate vector DB settings
    if config["vector_db"]["distance_metric"] not in ["cosine", "l2", "ip"]:
        raise ValueError("Vector DB distance_metric must be cosine, l2, or ip")

    # Validate LLM settings
    if config["llm"]["max_new_tokens"] < 1:
        raise ValueError("LLM max_new_tokens must be at least 1")

    if not 0 <= config["llm"]["temperature"] <= 2:
        raise ValueError("LLM temperature must be between 0 and 2")

    if not 0 <= config["llm"]["top_p"] <= 1:
        raise ValueError("LLM top_p must be between 0 and 1")

    # Validate ingestion settings
    if config["ingestion"]["interval_hours"] < 1:
        raise ValueError("Ingestion interval_hours must be at least 1")

    # Validate interface settings
    if config["interface"]["port"] < 1 or config["interface"]["port"] > 65535:
        raise ValueError("Interface port must be between 1 and 65535")

    # Validate logging settings
    valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if config["logging"]["level"].upper() not in valid_log_levels:
        raise ValueError(f"Logging level must be one of {valid_log_levels}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation (e.g., 'arxiv.categories').

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to the configuration value.
        default: Default value to return if key_path is not found.

    Returns:
        The configuration value or default if not found.
    """
    keys = key_path.split(".")
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default