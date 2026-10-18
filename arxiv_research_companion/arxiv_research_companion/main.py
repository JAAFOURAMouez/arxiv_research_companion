"""
Main entry point for the arXiv Research Companion application.
"""

import logging
import sys

from arxiv_research_companion.config import load_config
from arxiv_research_companion.data_ingestion import get_stored_papers, ingest_new_papers
from arxiv_research_companion.embedding_store import index_papers, initialize_vector_store
from arxiv_research_companion.interface import launch_interface
from arxiv_research_companion.utils import setup_logging as configure_logging


def setup_logging(config=None):
    """Configure application logging from the loaded settings."""
    logging_config = (config or {}).get("logging", {})
    configure_logging(
        level=logging_config.get("level", "INFO"),
        format_str=logging_config.get(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        log_file=logging_config.get("file_path"),
        console_output=logging_config.get("console_output", True),
    )


def main():
    """Load configured data, initialize retrieval, then launch Gradio."""
    config = load_config()
    setup_logging(config)
    logger = logging.getLogger(__name__)
    logger.info("Starting arXiv Research Companion...")

    try:
        vector_store = initialize_vector_store(config)
        logger.info("Vector store initialized.")

        ingestion_config = config.get("ingestion", {})
        db_path = ingestion_config.get("db_path", "./arxiv_papers.db")
        arxiv_config = config.get("arxiv", {})
        logger.info("Starting data ingestion...")
        ingest_new_papers(
            arxiv_config.get("categories", ["cs.AI", "cs.LG", "stat.ML"]),
            db_path,
            max_results_per_request=arxiv_config.get("max_results_per_request", 100),
        )

        papers = get_stored_papers(db_path)
        indexed_count = index_papers(vector_store, papers)
        logger.info("Ingestion completed; indexed %s papers.", indexed_count)

        launch_interface(config, vector_store)
    except Exception:
        logger.exception("Application failed to start")
        sys.exit(1)


if __name__ == "__main__":
    main()
