"""
Presentation Layer for arXiv Research Companion.

Responsibilities:
- Provide a user interface for interacting with the system.
- Primary interface: Gradio-based web chat interface.
- Components:
    * Chatbot for user questions and system answers.
    * Sidebar controls:
        - Category selector (predefined arXiv categories: cs.AI, cs.LG, stat.ML, etc.)
        - Date range filter (for limiting papers to recent ones)
        - Number of results slider (k, for retrieval)
        - Model selection dropdown (if multiple LLMs are configured)
    * Response display:
        - Generated answer with inline citations (e.g., [1] Title)
        - Expandable cards for each source paper showing title, authors, and abstract snippet.
        - "View on arXiv" button/link for each paper.
    * Status indicators:
        - Last update timestamp (when the database was last refreshed).
        - Total number of papers in the database.
        - Activity indicator (when background ingestion is running).
- Alternative interface: FastAPI-based REST API for programmatic access (optional).
    * Endpoints:
        - POST /query: {question} -> {answer, sources}
        - GET /papers: with filters -> paginated list of papers
        - GET /stats: -> database statistics
        - POST /ingest: (admin only) -> trigger manual ingestion
- Handles user input validation and error display.
- Formats output for readability (markdown support in Gradio).
- Ensures the interface is responsive and works well on different screen sizes.
- Designed to be runnable as a standalone application (via main.py).
"""