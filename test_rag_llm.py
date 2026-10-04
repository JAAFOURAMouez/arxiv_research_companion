#!/usr/bin/env python3
"""
Test script to check if the RAG engine's LLM interaction is working
"""

import os
import sys
sys.path.insert(0, '/mnt/c/Users/jaafo/3a/arxiv_research_companion')

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import logging

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_rag_llm_interaction():
    """Test the LLM interaction similar to how RAG engine does it"""
    print("Testing RAG engine LLM interaction...")

    # Get configuration from environment (same as RAG engine)
    llm_model = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    llm_max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "100"))
    llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    hf_api_token = os.getenv("HF_API_TOKEN", "")
    llm_timeout = float(os.getenv("LLM_TIMEOUT", "60.0"))

    print(f"Model: {llm_model}")
    print(f"Max new tokens: {llm_max_new_tokens}")
    print(f"Temperature: {llm_temperature}")
    print(f"Token set: {bool(hf_api_token)}")
    print(f"Timeout: {llm_timeout}")

    if not hf_api_token:
        print("ERROR: HF_API_TOKEN not set in environment")
        return False

    try:
        # Initialize the Hugging Face Inference API client (same as RAG engine)
        client_options = {'model': llm_model, 'token': hf_api_token, 'timeout': llm_timeout}
        client = InferenceClient(**client_options)

        # Test with a simple prompt similar to what RAG would generate
        test_prompt = """Answer the question based on the context. Use only the information from the context to formulate your answer. If the context does not contain enough information to answer the question, say that you cannot answer based on the provided information.

Context:
Title: Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.

Question:
What is the main innovation of the Transformer model?

Answer:"""

        print(f"Sending test prompt (first 100 chars): {test_prompt[:100]}...")

        generation_options = {
            'messages': [{'role': 'user', 'content': test_prompt}],
            'max_tokens': llm_max_new_tokens,
            'temperature': llm_temperature,
        }

        logger.debug(f"Calling chat_completion with options: {generation_options}")
        response = client.chat_completion(**generation_options)
        logger.debug(f"Raw response: {response}")

        # Extract content (same logic as RAG engine)
        choices = response.get('choices') if isinstance(response, dict) else getattr(response, 'choices', None)
        logger.debug(f"Choices: {choices}")
        if not choices:
            raise ValueError("Chat completion returned no choices")
        message = choices[0].get('message') if isinstance(choices[0], dict) else getattr(choices[0], 'message', None)
        logger.debug(f"Message: {message}")
        content = message.get('content') if isinstance(message, dict) else getattr(message, 'content', None)
        logger.debug(f"Content: {content}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Chat completion returned no answer content")
        answer = content.strip()

        print(f"✅ SUCCESS: Got answer: {answer}")
        return True

    except Exception as e:
        print(f"❌ ERROR: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rag_llm_interaction()
    if success:
        print("\n✅ RAG LLM test PASSED")
    else:
        print("\n❌ RAG LLM test FAILED")