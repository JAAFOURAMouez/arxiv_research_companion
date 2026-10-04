"""
Embedding and Vector Storage Layer for arXiv Research Companion.

Responsibilities:
- Generate numerical embeddings from text using a sentence transformer model.
- Default model: sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).
- Manage a vector database for storing and searching embeddings.
- Current choice: ChromaDB (embedded, persistent, zero-server setup).
- Support adding new vectors (incremental updates) with associated metadata.
- Perform similarity search: given a query vector, return top-k most similar vectors.
- Provide metadata enrichment: retrieve stored paper details for given IDs.
- Handle persistence: save vector database to disk for reuse between runs.
- Allow configuration of embedding model and vector database parameters.
- Provide fallback mechanisms for embedding generation (e.g., batch processing).
- Ensure thread-safety if used in a multi-threaded context (though designed for single process).
"""

import logging
import hashlib
from typing import List, Dict, Any, Optional
import numpy as np

# Import sentence transformers and chromadb
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
except ImportError as e:
    logging.warning(f"Optional dependencies not available: {e}")
    SentenceTransformer = None
    chromadb = None

logger = logging.getLogger(__name__)

def initialize_vector_store(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize the embedding model and vector store.

    Args:
        config: Configuration dictionary with keys:
            - embedding_model: Name of the sentence transformer model (default: 'sentence-transformers/all-MiniLM-L6-v2')
            - vector_db_path: Path to ChromaDB persistent directory (default: './chroma_db')

    Returns:
        Dictionary containing:
            - model: SentenceTransformer instance
            - collection: ChromaDB collection instance
            - dimension: Embedding dimension
    """
    if SentenceTransformer is None or chromadb is None:
        raise ImportError("Required dependencies (sentence-transformers, chromadb) not installed.")

    # Prefer the nested configuration returned by load_config; retain flat-key
    # fallbacks for callers using the earlier configuration shape.
    embedding_config = config.get('embedding') or {}
    vector_db_config = config.get('vector_db') or {}
    model_name = embedding_config.get(
        'model_name', config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
    )
    db_path = vector_db_config.get('path', config.get('vector_db_path', './chroma_db'))
    collection_name = vector_db_config.get('collection_name', 'arxiv_papers')
    distance_metric = vector_db_config.get('distance_metric', 'cosine')

    logger.info(f"Initializing embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    dimension = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {dimension}")

    logger.info(f"Initializing ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric}
    )
    logger.info(f"Collection '{collection_name}' ready with count: {collection.count()}")

    return {
        "model": model,
        "collection": collection,
        "dimension": dimension
    }

def add_embeddings(
    vector_store: Dict[str, Any],
    texts: List[str],
    metadata: List[Dict[str, Any]],
    ids: Optional[List[str]] = None,
) -> None:
    """
    Generate embeddings for texts and add them to the vector store.

    Args:
        vector_store: Dictionary from initialize_vector_store
        texts: List of text strings to embed (e.g., paper abstracts)
        metadata: List of metadata dictionaries for each text (must include 'arxiv_id')
    """
    model = vector_store["model"]
    collection = vector_store["collection"]

    if not texts:
        logger.warning("No texts provided to add_embeddings")
        return

    logger.info(f"Generating embeddings for {len(texts)} texts")
    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    # Ensure embeddings is a list of lists for ChromaDB
    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    # Generate stable IDs based on arXiv IDs or a hash of the text.
    if ids is None:
        ids = []
        for i, meta in enumerate(metadata):
            arxiv_id = meta.get('arxiv_id')
            if arxiv_id:
                ids.append(arxiv_id.split('v')[0])
            else:
                text_hash = hashlib.md5(texts[i].encode()).hexdigest()
                ids.append(f"hash_{text_hash}")

    chroma_metadata = [
        {
            key: ", ".join(str(value) for value in item) if isinstance(item, list) else item
            for key, item in paper.items()
            if item is not None
        }
        for paper in metadata
    ]
    logger.info(f"Adding {len(embeddings)} embeddings to vector store")
    collection.upsert(
        embeddings=embeddings,
        metadatas=chroma_metadata,
        documents=texts,
        ids=ids
    )
    logger.info(f"Successfully added embeddings. New collection count: {collection.count()}")

def index_papers(vector_store: Dict[str, Any], papers: List[Dict[str, Any]], batch_size: int = 500) -> int:
    """Index papers not already present in Chroma, including their abstract documents."""
    collection = vector_store["collection"]
    indexed = 0
    for start in range(0, len(papers), batch_size):
        batch = papers[start:start + batch_size]
        ids = [str(paper["arxiv_id"]).split("v")[0] for paper in batch]
        existing = collection.get(ids=ids, include=["metadatas"]).get("ids", [])
        existing_ids = set(existing)
        missing = [paper for paper, paper_id in zip(batch, ids) if paper_id not in existing_ids]
        if not missing:
            continue

        texts = [paper.get("summary") or "" for paper in missing]
        metadata = [
            {key: value for key, value in paper.items() if key not in {"summary", "arxiv_id"}}
            | {"arxiv_id": str(paper["arxiv_id"]).split("v")[0]}
            for paper in missing
        ]
        missing_ids = [str(paper["arxiv_id"]).split("v")[0] for paper in missing]
        add_embeddings(vector_store, texts, metadata, ids=missing_ids)
        indexed += len(missing)
    return indexed


def search_similar(vector_store: Dict[str, Any], query_text: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Search for similar vectors in the vector store.

    Args:
        vector_store: Dictionary from initialize_vector_store
        query_text: Query text to search for
        top_k: Number of results to return (default: 5)

    Returns:
        Dictionary with keys:
            - ids: List of paper IDs
            - distances: List of similarity distances
            - metadatas: List of metadata dictionaries
    """
    model = vector_store["model"]
    collection = vector_store["collection"]

    if not query_text:
        logger.warning("Empty query text provided to search_similar")
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    logger.info(f"Searching for similar texts to query: '{query_text[:50]}...' (top_k={top_k})")
    # Generate query embedding
    query_embedding = model.encode([query_text], show_progress_bar=True)
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    # Query the collection
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"]  # We don't need the documents (the abstracts) as we have them in metadata
    )
    logger.info(f"Found {len(results['ids'][0]) if results['ids'] else 0} similar texts")

    # ChromaDB returns lists of lists for ids, distances, metadatas
    # We want to return the same structure for consistency
    return {
        "ids": results.get("ids", [[]]),
        "distances": results.get("distances", [[]]),
        "metadatas": results.get("metadatas", [[]])
    }

def get_paper_by_id(vector_store: Dict[str, Any], paper_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a paper by its ID from the vector store.

    Args:
        vector_store: Dictionary from initialize_vector_store
        paper_id: The arXiv ID of the paper (without version)

    Returns:
        Dictionary with keys:
            - id: The paper ID
            - metadata: Metadata dictionary
            - document: The abstract text
        Or None if not found.
    """
    collection = vector_store["collection"]

    # Clean the paper ID (remove version if present)
    clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id

    logger.info(f"Retrieving paper by ID: {clean_id}")
    try:
        result = collection.get(
            ids=[clean_id],
            include=["metadatas", "documents"]
        )
        if not result["ids"]:
            logger.warning(f"Paper with ID {clean_id} not found")
            return None

        # ChromaDB returns lists for ids, metadatas, documents
        return {
            "id": result["ids"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else {},
            "document": result["documents"][0] if result["documents"] else ""
        }
    except Exception as e:
        logger.error(f"Error retrieving paper {clean_id}: {e}")
        return None