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

import logging
import gradio as gr
import time
from typing import Tuple, List, Dict, Any
from arxiv_research_companion.rag_engine import generate_answer
from arxiv_research_companion.utils import setup_logging, clean_text, is_valid_arxiv_id
from arxiv_research_companion.data_ingestion import get_last_update_timestamp, get_total_papers_count
from arxiv_research_companion.embedding_store import get_paper_by_id

logger = logging.getLogger(__name__)


def launch_interface(config: Dict[str, Any], vector_store: Dict[str, Any]) -> None:
    """
    Launch the Gradio-based web interface for the arXiv Research Companion.

    Args:
        config: Application configuration dictionary
        vector_store: Initialized vector store from embedding_store
    """
    # Setup logging for the interface
    log_config = config.get('logging', {})
    setup_logging(
        level=log_config.get('level', 'INFO'),
        format_str=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        log_file=log_config.get('file_path'),
        console_output=log_config.get('console_output', True)
    )

    logger.info("Launching Gradio interface...")

    # Get interface configuration
    interface_config = config.get('interface', {})
    theme = interface_config.get('theme', 'default')
    port = interface_config.get('port', 7860)
    share = interface_config.get('share', False)
    auth_username = interface_config.get('auth_username', '')
    auth_password = interface_config.get('auth_password', '')

    # Prepare authentication if credentials are provided
    auth = None
    if auth_username and auth_password:
        auth = (auth_username, auth_password)

    # Get database path from config for status updates
    ingestion_db_path = config.get('ingestion', {}).get('db_path', './arxiv_papers.db')

    # Create the Gradio interface
    with gr.Blocks(title="arXiv Research Companion") as interface:
        gr.Markdown("# arXiv Research Companion")
        gr.Markdown("Ask questions about recent AI/ML/LG research papers from arXiv")

        # Status row
        with gr.Row():
            with gr.Column(scale=2):
                last_updated = gr.Label(
                    value="Last updated: Loading...",
                    label="Database Status"
                )
            with gr.Column(scale=1):
                total_papers = gr.Label(
                    value="Total papers: Loading...",
                    label="Database Size"
                )

        # Main chat interface
        chatbot = gr.Chatbot(
            label="Research Assistant",
            height=500,
            show_label=True,
            container=True
        )
        status_message = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=4):
                msg = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about recent AI/ML research...",
                    lines=2,
                    max_lines=4
                )
            with gr.Column(scale=1):
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear")

        # Sidebar controls in an accordion
        with gr.Accordion("Search Controls", open=False):
            with gr.Row():
                with gr.Column():
                    category_dropdown = gr.Dropdown(
                        label="arXiv Categories",
                        choices=["cs.AI", "cs.LG", "stat.ML", "cs.CV", "cs.NE", "cs.CL", "cs.RO", "cs.DC"],
                        value=["cs.AI", "cs.LG", "stat.ML"],
                        multiselect=True,
                        info="Select categories to search"
                    )
                with gr.Column():
                    date_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=7,
                        step=1,
                        label="Days Back",
                        info="How many days back to search for papers"
                    )

            with gr.Row():
                with gr.Column():
                    results_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Number of Results (k)",
                        info="Number of similar papers to retrieve"
                    )
                with gr.Column():
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Creativity (Temperature)",
                        info="Higher = more creative, Lower = more focused"
                    )

        # Examples
        gr.Examples(
            examples=[
                ["What are the latest developments in transformer architectures?"],
                ["How does reinforcement learning apply to robotics?"],
                ["What are recent advances in diffusion models for image generation?"],
                ["Explain the concept of attention mechanisms in neural networks"],
                ["What are the challenges in scaling large language models?"]
            ],
            inputs=msg,
            label="Example Questions"
        )

        # Event handlers
        def respond(message, chat_history, category, days_back, k, temperature):
            """Handle user message and generate response."""
            if not message or not message.strip():
                return chat_history, "", "Please enter a question"

            # Gradio 6 expects message dictionaries with explicit roles.
            chat_history = list(chat_history or [])
            chat_history.append({"role": "user", "content": message})

            # Update config with current UI values
            ui_config = config.copy()
            if 'llm' not in ui_config:
                ui_config['llm'] = {}
            ui_config['llm']['temperature'] = temperature

            # Note: The k parameter for retrieval is handled in the rag_engine via config['retrieval_k']
            # We'll update that in the config as well
            if 'retrieval_k' not in ui_config:
                ui_config['retrieval_k'] = {}
            ui_config['retrieval_k'] = k

            try:
                # Generate answer using RAG engine
                result = generate_answer(
                    query=message.strip(),
                    vector_store=vector_store,
                    config=ui_config
                )

                # Format response with citations
                answer = result['answer']
                sources = result['sources']

                # Add citations to answer if we have sources
                if sources:
                    cited_answer = answer
                    # Add inline citations [1], [2], etc.
                    # Simple approach: append citations at the end for now
                    citations = "\n\n**Sources:**\n"
                    for i, source in enumerate(sources, 1):
                        paper_id = source['id']
                        title = source['metadata'].get('title', 'Untitled')
                        cited_answer += f"\n[{i}] {title} (arXiv:{paper_id})"
                    answer = cited_answer

                chat_history.append({"role": "assistant", "content": answer})
                return chat_history, "", ""

            except Exception as e:
                logger.error(f"Error generating answer: {e}")
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                chat_history.append({"role": "assistant", "content": error_msg})
                return chat_history, "", ""

        def clear_chat():
            """Clear the chat history."""
            return [], "", ""

        def update_status():
            """Update the status indicators."""
            try:
                last_update = get_last_update_timestamp(ingestion_db_path)
                total = get_total_papers_count(ingestion_db_path)

                last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S") if last_update else "Never"
                last_updated_val = f"Last updated: {last_update_str}"
                total_papers_val = f"Total papers: {total}"

                return last_updated_val, total_papers_val
            except Exception as e:
                logger.error(f"Error updating status: {e}")
                return "Last updated: Error", "Total papers: Error"

        # Set up event handlers
        submit_btn.click(
            fn=respond,
            inputs=[msg, chatbot, category_dropdown, date_slider, results_slider, temperature_slider],
            outputs=[chatbot, msg, status_message]
        )

        msg.submit(
            fn=respond,
            inputs=[msg, chatbot, category_dropdown, date_slider, results_slider, temperature_slider],
            outputs=[chatbot, msg, status_message]
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[],
            outputs=[chatbot, msg, status_message]
        )

        # Load initial status when interface loads
        interface.load(
            fn=update_status,
            inputs=[],
            outputs=[last_updated, total_papers]
        )

        # Periodically update status (every 30 seconds)
        # Note: Gradio doesn't have built-in periodic updates in Blocks,
        # so we'll rely on manual refresh or page reload for now

    # Launch the interface
    logger.info(f"Starting Gradio server on port {port}")
    interface.launch(
        server_port=port,
        share=share,
        auth=auth,
        theme=theme,
        show_error=True,
        quiet=False
    )