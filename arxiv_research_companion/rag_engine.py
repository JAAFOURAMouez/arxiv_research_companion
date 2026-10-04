"""
Retrieval and Generation Layer (RAG Engine) for arXiv Research Companion.

Responsibilities:
- Receive a user query and process it for retrieval.
- Embed the query using the same sentence transformer model used for papers.
- Retrieve the top-k most similar paper abstracts from the vector store.
- Format the retrieved abstracts into a context string, including metadata (title, authors, etc.).
- Construct a prompt for the language model that includes the context and the user query.
- Use a language model (LLM) to generate an answer based on the prompt.
- Default LLM: Hugging Face Inference API with a model like google/flan-t5-base or similar.
- Process the LLM response to extract the answer and format citations.
- Provide grounding: ensure the answer is based only on the retrieved abstracts (basic hallucination check).
- Return a structured response containing the generated answer and list of source papers.
- Handle errors gracefully (e.g., if LLM API fails, return retrieved abstracts with a disclaimer).
- Support configuration of retrieval parameters (k) and LLM parameters (temperature, max_tokens).
"""

import logging
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


def generate_answer(query: str, vector_store: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an answer to a user query using the RAG pipeline.

    Args:
        query: The user's question.
        vector_store: The vector store dictionary from embedding_store.initialize_vector_store.
        config: Configuration dictionary with an optional nested 'llm' section containing
            model_name, max_new_tokens, temperature, and api_token. Legacy flat LLM keys
            (llm_model, llm_max_new_tokens, llm_temperature, hf_api_token) are also accepted.
            retrieval_k is read from the top level.

    Returns:
        Dictionary with keys:
            - answer: The generated answer string.
            - sources: List of source paper dictionaries, each containing:
                - id: The paper's arXiv ID.
                - metadata: The paper's metadata dictionary.
                - document: The paper's abstract.
    """
    if not query:
        logger.warning("Empty query provided to generate_answer")
        return {"answer": "Please provide a question.", "sources": []}

    # Prefer the nested configuration returned by load_config; retain flat-key
    # fallbacks for callers using the earlier configuration shape.
    llm_config = config.get('llm') or {}
    llm_model = llm_config.get('model_name', config.get('llm_model', 'openai/gpt-oss-120b'))
    llm_max_new_tokens = llm_config.get('max_new_tokens', config.get('llm_max_new_tokens', 100))
    llm_temperature = llm_config.get('temperature', config.get('llm_temperature', 0.7))
    llm_top_p = llm_config.get('top_p', config.get('llm_top_p'))
    hf_inference_provider = llm_config.get('inference_provider') or config.get('hf_inference_provider') or None
    retrieval_k = config.get('retrieval_k', 2)
    hf_api_token = llm_config.get('api_token') or config.get('hf_api_token') or None
    llm_timeout = llm_config.get('timeout', config.get('llm_timeout', 60.0))

    try:
        # Step 1: Retrieve relevant papers from the vector store
        logger.info(f"Retrieving top {retrieval_k} papers for query: '{query[:50]}...'")
        # Embed the query using the same model
        model = vector_store["model"]
        query_embedding = model.encode([query])
        # ChromaDB expects a list of embeddings, and we have one query.
        # We'll use the collection's query method with the embedding.
        results = vector_store["collection"].query(
            query_embeddings=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
            n_results=retrieval_k,
            include=["metadatas", "documents"]
        )
        ids_list = results.get('ids', [[]])[0] if results.get('ids') else []

        def first_query_values(key):
            values = results.get(key) or []
            if not values:
                return []
            # Chroma returns one list per query. Some legacy mocks modeled one
            # singleton list per result, so flatten that shape for compatibility.
            if len(values) == len(ids_list) and all(isinstance(value, list) for value in values):
                return [value[0] if value else None for value in values]
            return values[0] if isinstance(values[0], list) else values

        formatted_results = {
            'ids': [ids_list],
            'distances': [results.get('distances', [[]])[0]] if results.get('distances') else [[]],
            'metadatas': [first_query_values('metadatas')],
            'documents': [first_query_values('documents')]
        }
    except Exception as e:
        logger.error(f"Error retrieving papers from vector store: {e}")
        # If retrieval fails, we can still try to generate an answer from the query alone, but we'll return an error.
        return {"answer": f"Error retrieving information: {str(e)}", "sources": []}

    # Step 2: Format the retrieved abstracts into a context string
    context = _format_context(formatted_results)
    if context == "No relevant papers found.":
        return {
            "answer": "I couldn't find any papers to answer this question. Try again after papers have been ingested.",
            "sources": [],
        }

    # Step 3: Construct the prompt for the LLM
    prompt = _construct_prompt(query, context)

    # Step 4: Generate answer using the LLM
    try:
        # Initialize the Hugging Face Inference API client
        client_options = {'model': llm_model, 'token': hf_api_token, 'timeout': llm_timeout}
        if hf_inference_provider:
            client_options['provider'] = hf_inference_provider
        client = InferenceClient(**client_options)
        generation_options = {
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': llm_max_new_tokens,
            'temperature': llm_temperature,
        }
        if llm_top_p is not None:
            generation_options['top_p'] = llm_top_p

        response = client.chat_completion(**generation_options)
        logger.debug(f"LLM raw response: {response}")
        choices = response.get('choices') if isinstance(response, dict) else getattr(response, 'choices', None)
        logger.debug(f"LLM choices: {choices}")
        if not choices:
            raise ValueError("Chat completion returned no choices")
        message = choices[0].get('message') if isinstance(choices[0], dict) else getattr(choices[0], 'message', None)
        logger.debug(f"LLM message: {message}")
        content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
        logger.debug(f"LLM content: {content}")
        if not isinstance(content, str) or not content.strip():
            logger.warning(f"LLM returned empty content for model {llm_model}, falling back to retrieved context")
            answer = f"I retrieved relevant information but couldn't generate a summary. Here's what I found:\n\n{context}"
        else:
            answer = content.strip()
    except Exception as e:
        logger.exception("Error generating answer with LLM for model %s", llm_model)
        # If generation fails, return the retrieved abstracts with a helpful message.
        answer = (
            f"I encountered an error while trying to generate an answer with model '{llm_model}'. "
            f"Here's the relevant information I found instead:\n\n{context}"
        )
        # Return early to avoid issues with undefined variables
        return {
            "answer": answer,
            "sources": []
        }

    # Step 5: Provide grounding (basic hallucination check)
    # We'll do a very simple check: if the answer is empty or just whitespace, we'll return a disclaimer.
    if not answer or answer.isspace():
        answer = "I couldn't generate a meaningful answer based on the retrieved information."

    # Step 6: Format the sources for return
    sources = []
    try:
        # Extract the source papers from the formatted results
        ids_list = formatted_results['ids'][0] if formatted_results['ids'] else []
        metadatas_list = formatted_results['metadatas'][0] if formatted_results['metadatas'] else []
        documents_list = formatted_results['documents'][0] if formatted_results['documents'] else []
        for i, paper_id in enumerate(ids_list):
            metadata = metadatas_list[i] if i < len(metadatas_list) else {}
            document = documents_list[i] if i < len(documents_list) else ""
            sources.append({
                "id": paper_id,
                "metadata": metadata,
                "document": document
            })
    except Exception as e:
        logger.warning(f"Error formatting sources: {e}")
        sources = []

    return {
        "answer": answer,
        "sources": sources
    }


def _format_context(retrieval_results: Dict[str, Any]) -> str:
    """
    Format retrieval results into a context string for the LLM prompt.

    Args:
        retrieval_results: Dictionary with keys 'ids', 'distances', 'metadatas', 'documents'
                           (each being a list of lists, where the outer list corresponds to the query).

    Returns:
        A formatted string containing the title, authors, and abstract of each retrieved paper.
    """
    if not retrieval_results or not retrieval_results.get('ids') or not retrieval_results['ids'][0]:
        return "No relevant papers found."

    context_parts = []
    ids_list = retrieval_results['ids'][0]
    metadatas_list = retrieval_results['metadatas'][0]
    documents_list = retrieval_results['documents'][0]

    for i, paper_id in enumerate(ids_list):
        metadata = metadatas_list[i] if i < len(metadatas_list) else {}
        document = documents_list[i] if i < len(documents_list) else ""
        title = metadata.get('title', 'Untitled')
        authors = metadata.get('authors', [])
        authors_str = ", ".join(authors) if isinstance(authors, list) else str(authors)
        context_parts.append(
            f"Title: {title}\nAuthors: {authors_str}\nAbstract: {document}\n"
        )

    return "\n---\n".join(context_parts)


def _construct_prompt(query: str, context: str) -> str:
    """
    Construct a prompt for the language model that includes the context and the user query.

    Args:
        query: The user's question.
        context: The formatted context string from retrieved papers.

    Returns:
        A prompt string that instructs the LLM to answer the question based on the context.
    """
    if not context or context == "No relevant papers found.":
        return f"""Answer the question based on the context.

Context: {context}

Question: {query}

Answer:"""
    else:
        return f"""Answer the question based on the context. Use only the information from the context to formulate your answer. If the context does not contain enough information to answer the question, say that you cannot answer based on the provided information.

Context:
{context}

Question:
{query}

Answer:"""