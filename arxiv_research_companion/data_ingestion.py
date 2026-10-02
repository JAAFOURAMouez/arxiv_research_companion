"""
Data Ingestion Layer for arXiv Research Companion.

Responsibilities:
- Fetch new papers from arXiv API based on categories and date ranges.
- Parse XML responses from arXiv Atom feed.
- Extract metadata: arXiv ID, title, authors, abstract, categories, published date, DOI.
- Deduplicate papers using arXiv ID (and optionally DOI).
- Store paper metadata in a persistent store (SQLite) for tracking.
- Prepare abstracts for embedding by cleaning and queuing.
- Support incremental updates (only process papers since last successful run).
- Handle arXiv API rate limits (1 request per 3 seconds).
- Provide functions for manual or scheduled ingestion.
"""