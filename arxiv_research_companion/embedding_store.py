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