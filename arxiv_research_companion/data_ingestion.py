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
import xml.etree.ElementTree as ET
import time
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import requests

# arXiv API base URL
ARXIV_API_URL = "http://export.arxiv.org/api/query"

# Rate limiting: arXiv requests should be spaced at least 3 seconds apart
MIN_REQUEST_INTERVAL = 3.0
_last_request_time = 0.0

def _respect_rate_limit():
    """Ensure we don't exceed arXiv's rate limit of 1 request per 3 seconds."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.time()

def fetch_arxiv_papers(
    categories: List[str],
    start_date: Optional[datetime] = None,
    max_results: int = 100
) -> List[Dict]:
    """
    Fetch papers from arXiv API for given categories and optional start date.

    Args:
        categories: List of arXiv category strings (e.g., ['cs.AI', 'cs.LG'])
        start_date: Only fetch papers published after this date (inclusive)
        max_results: Maximum number of results to return per request

    Returns:
        List of paper dictionaries with keys:
        ['arxiv_id', 'title', 'authors', 'summary', 'categories', 'published', 'doi']
    """
    # Build a valid arXiv query; requests encodes spaces in the query parameter.
    category_query = " OR ".join(f"cat:{category}" for category in categories)
    query = f"({category_query})" if len(categories) > 1 else category_query

    if start_date:
        # arXiv date-range syntax requires concrete UTC bounds and spaces around TO.
        date_str = start_date.strftime('%Y%m%d%H%M')
        end_date = datetime.now(timezone.utc).strftime('%Y%m%d%H%M')
        query += f" AND submittedDate:[{date_str} TO {end_date}]"

    # Parameters for API request
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending'
    }

    _respect_rate_limit()
    try:
        response = requests.get(ARXIV_API_URL, params=params, timeout=120)  # Further increased timeout for reliability
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch from arXiv API: {e}")
        return []

    return _parse_arxiv_xml(response.text)

def _parse_arxiv_xml(xml_text: str) -> List[Dict]:
    """
    Parse arXiv Atom XML response into list of paper dictionaries.

    Args:
        xml_text: Raw XML response from arXiv API

    Returns:
        List of paper dictionaries
    """
    papers = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logging.error(f"Failed to parse arXiv XML: {e}")
        return papers

    # Define namespaces
    ns = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    # Find all entry elements
    for entry in root.findall('atom:entry', ns):
        paper = {}

        # Extract arXiv ID (like http://arxiv.org/abs/2103.12345v1)
        id_elem = entry.find('atom:id', ns)
        if id_elem is not None and id_elem.text:
            # Get the ID without version (we'll store base ID)
            arxiv_id = id_elem.text.split('/')[-1]
            # Remove version if present (e.g., 2103.12345v1 -> 2103.12345)
            if 'v' in arxiv_id:
                arxiv_id = arxiv_id.split('v')[0]
            paper['arxiv_id'] = arxiv_id

        # Title
        title_elem = entry.find('atom:title', ns)
        if title_elem is not None and title_elem.text:
            paper['title'] = title_elem.text.strip()

        # Authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        paper['authors'] = authors

        # Summary/Abstract
        summary_elem = entry.find('atom:summary', ns)
        if summary_elem is not None and summary_elem.text:
            paper['summary'] = summary_elem.text.strip()

        # Categories
        categories = []
        for category in entry.findall('atom:category', ns):
            term = category.get('term')
            if term:
                categories.append(term)
        paper['categories'] = categories

        # Published date
        published_elem = entry.find('atom:published', ns)
        if published_elem is not None and published_elem.text:
            try:
                # Parse ISO format datetime
                pub_date = datetime.fromisoformat(published_elem.text.replace('Z', '+00:00'))
                paper['published'] = pub_date
            except ValueError:
                logging.warning(f"Could not parse published date: {published_elem.text}")
                paper['published'] = None

        # DOI (optional)
        doi_elem = entry.find('arxiv:doi', ns)
        if doi_elem is not None and doi_elem.text:
            paper['doi'] = doi_elem.text.strip()
        else:
            paper['doi'] = None

        papers.append(paper)

    return papers

def _init_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize SQLite database for storing paper metadata.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create papers table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            arxiv_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,  -- JSON list of authors
            summary TEXT,
            categories TEXT,  -- JSON list of categories
            published TIMESTAMP,
            doi TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Create index on published date for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_papers_published
        ON papers(published)
    ''')

    conn.commit()
    return conn

def _paper_exists(conn: sqlite3.Connection, arxiv_id: str) -> bool:
    """
    Check if a paper already exists in the database.

    Args:
        conn: SQLite connection
        arxiv_id: arXiv ID to check

    Returns:
        True if paper exists, False otherwise
    """
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM papers WHERE arxiv_id = ?', (arxiv_id,))
    return cursor.fetchone() is not None

def _store_paper(conn: sqlite3.Connection, paper: Dict):
    """
    Store a paper in the database.

    Args:
        conn: SQLite connection
        paper: Paper dictionary to store
    """
    import json

    cursor = conn.cursor()

    # Convert lists to JSON strings for storage
    authors_json = json.dumps(paper.get('authors', []))
    categories_json = json.dumps(paper.get('categories', []))

    cursor.execute('''
        INSERT OR REPLACE INTO papers
        (arxiv_id, title, authors, summary, categories, published, doi)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        paper['arxiv_id'],
        paper.get('title', ''),
        authors_json,
        paper.get('summary', ''),
        categories_json,
        paper.get('published'),
        paper.get('doi')
    ))

    conn.commit()

def get_last_update_timestamp(db_path: str) -> Optional[datetime]:
    """
    Get the timestamp of the most recent paper in the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Most recent published date, or None if database is empty
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT MAX(published) FROM papers
            WHERE published IS NOT NULL
        ''')
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            # SQLite returns TIMESTAMP as string, parse it
            return datetime.fromisoformat(result[0])
        return None
    except sqlite3.Error as e:
        logging.error(f"Error reading last update timestamp: {e}")
        return None

def get_total_papers_count(db_path: str) -> int:
    """
    Get the total number of papers in the database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Total number of papers
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM papers')
        result = cursor.fetchone()
        conn.close()

        if result and result[0]:
            return int(result[0])
        return 0
    except sqlite3.Error as e:
        logging.error(f"Error reading total papers count: {e}")
        return 0

def get_stored_papers(db_path: str) -> List[Dict]:
    """Return papers stored in SQLite in the shape used by the embedding layer."""
    import json

    conn = _init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, title, authors, summary, categories, published, doi "
            "FROM papers ORDER BY published DESC"
        ).fetchall()
    finally:
        conn.close()

    papers = []
    for arxiv_id, title, authors, summary, categories, published, doi in rows:
        papers.append({
            "arxiv_id": arxiv_id,
            "title": title or "",
            "authors": json.loads(authors) if authors else [],
            "summary": summary or "",
            "categories": json.loads(categories) if categories else [],
            "published": published,
            "doi": doi,
        })
    return papers


def ingest_new_papers(
    categories: List[str],
    db_path: str,
    max_results_per_request: int = 100
) -> int:
    """
    Fetch and store new papers from arXiv since last update.

    Args:
        categories: List of arXiv categories to monitor
        db_path: Path to SQLite database for storage
        max_results_per_request: Number of results to fetch per API request

    Returns:
        Number of new papers stored
    """
    # Initialize database
    conn = _init_db(db_path)

    # Determine start date for incremental update
    last_update = get_last_update_timestamp(db_path)
    start_date = None
    if last_update:
        # Fetch papers published after the last update
        # Add a small buffer to avoid missing papers due to clock skew
        start_date = last_update - timedelta(minutes=10)
        logging.info(f"Fetching papers since {start_date}")
    else:
        logging.info("No previous update found, fetching recent papers")
        # If no prior data, fetch papers from the last 30 days
        start_date = datetime.now() - timedelta(days=30)

    # Fetch papers from arXiv
    papers = fetch_arxiv_papers(categories, start_date=start_date, max_results=max_results_per_request)

    if not papers:
        logging.warning("No papers fetched from arXiv")
        conn.close()
        return 0

    # Store new papers (skipping duplicates)
    new_count = 0
    for paper in papers:
        if not _paper_exists(conn, paper['arxiv_id']):
            _store_paper(conn, paper)
            new_count += 1
        else:
            logging.debug(f"Paper {paper['arxiv_id']} already exists, skipping")

    conn.close()
    logging.info(f"Stored {new_count} new papers out of {len(papers)} fetched")
    return new_count

# For direct script execution (e.g., for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage
    categories = ['cs.AI', 'cs.LG']
    db_path = "arxiv_papers.db"
    count = ingest_new_papers(categories, db_path)
    print(f"Ingested {count} new papers")