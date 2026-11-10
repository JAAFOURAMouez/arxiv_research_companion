"""
Framework-independent application state: config, embedder, vector index and retriever.

Shared by the FastAPI server (``server.py``) and the Streamlit app (``streamlit_app.py``).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from arxiv_research_companion.config import load_config
from arxiv_research_companion.data_ingestion import get_stored_papers
from arxiv_research_companion.embedders import OnnxEmbedder
from arxiv_research_companion.embedding_store import index_papers
from arxiv_research_companion.retrieval import build_retriever, fts5_available

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent

_FALLBACK_CHAIN = {"hybrid_rerank": "hybrid", "hybrid": "dense", "bm25": "dense"}


def make_retriever(config: Dict[str, Any], embedder, collection, papers: List[Dict]):
    """Build the configured retriever; degrade (with a recorded reason) instead of crashing."""
    rcfg = config["retrieval"]
    mode = rcfg["mode"]
    reason = None
    while True:
        try:
            if mode in ("bm25", "hybrid", "hybrid_rerank") and not fts5_available():
                raise RuntimeError("SQLite FTS5 is not available in this Python build")
            retriever = build_retriever(
                mode,
                embedder=embedder,
                papers=papers,
                collection=collection,
                doc_text_mode=rcfg["doc_text"],
                candidate_depth=rcfg["candidate_depth"],
                rerank_depth=rcfg["rerank_depth"],
                rrf_k=rcfg["rrf_k"],
            )
            rcfg["effective_mode"] = mode
            rcfg["fallback_reason"] = reason
            return retriever
        except (ImportError, RuntimeError) as exc:
            if mode not in _FALLBACK_CHAIN:
                raise
            logger.warning("Retrieval mode %s unavailable (%s); falling back", mode, exc)
            reason = f"{mode} unavailable: {exc}"
            mode = _FALLBACK_CHAIN[mode]


def refresh_retriever(config: Dict[str, Any], vector_store: Dict[str, Any], papers: List[Dict]) -> None:
    vector_store["papers_by_id"] = {p["arxiv_id"]: p for p in papers}
    vector_store["retriever"] = make_retriever(
        config, vector_store["model"], vector_store["collection"], papers
    )


def build_state() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load config, build the vector store and index the papers stored in SQLite.

    The host file system is ephemeral (Render free tier, Streamlit Cloud), so the index is
    rebuilt from arxiv_papers.db into a temporary Chroma directory at every start.
    index_papers() is idempotent.

    Test hooks: INGESTION_DB_PATH overrides the SQLite file, ARC_CHROMA_DIR the Chroma dir.
    """
    import chromadb

    config = load_config()
    config["ingestion"]["db_path"] = os.getenv(
        "INGESTION_DB_PATH", str(BASE_DIR / "arxiv_papers.db")
    )

    model = OnnxEmbedder()
    chroma_dir = os.getenv("ARC_CHROMA_DIR") or str(Path(tempfile.gettempdir()) / "arxiv_chroma")
    client = chromadb.PersistentClient(path=chroma_dir)
    collection = client.get_or_create_collection(
        name=config["vector_db"]["collection_name"],
        metadata={"hnsw:space": config["vector_db"]["distance_metric"]},
    )
    vector_store = {"model": model, "collection": collection, "dimension": 384}

    papers = get_stored_papers(config["ingestion"]["db_path"])
    indexed = index_papers(vector_store, papers)
    logger.info("Indexed %s new papers at startup.", indexed)
    refresh_retriever(config, vector_store, papers)
    return config, vector_store


if __name__ == "__main__":
    # Build step (render.yaml): download the model and embed every paper ahead of time, so a
    # cold start on a 0.1-CPU instance only loads the index. Set ARC_CHROMA_DIR and
    # ONNX_MODEL_DIR to folders inside the project so they ship with the deploy.
    logging.basicConfig(level=logging.INFO)
    _, store = build_state()
    print(f"Index ready: {store['collection'].count()} papers")
