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