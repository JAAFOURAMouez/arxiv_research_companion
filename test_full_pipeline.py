#!/usr/bin/env python3
"""
Test script to check the full RAG pipeline
"""

import sys
sys.path.insert(0, '/mnt/c/Users/jaafo/3a/arxiv_research_companion')

from arxiv_research_companion.config import load_config
from arxiv_research_companion.embedding_store import initialize_vector_store
from arxiv_research_companion.rag_engine import generate_answer

def test_full_pipeline():
    """Test the full RAG pipeline"""
    print("Testing full RAG pipeline...")

    try:
        # Load configuration
        config = load_config()
        print("✓ Configuration loaded")

        # Initialize vector store
        vector_store = initialize_vector_store(config)
        print("✓ Vector store initialized")
        print(f"  Collection count: {vector_store['collection'].count()}")

        # Test query
        test_query = "What is machine learning?"
        print(f"✓ Testing query: '{test_query}'")

        # Generate answer
        result = generate_answer(test_query, vector_store, config)

        print("✓ Answer generated successfully")
        print(f"  Answer: {result['answer'][:100]}...")
        print(f"  Number of sources: {len(result['sources'])}")

        return True

    except Exception as e:
        print(f"❌ Full pipeline test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_pipeline()
    if not success:
        sys.exit(1)
    else:
        print("\n✅ All tests PASSED")