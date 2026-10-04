#!/usr/bin/env python3
"""
Test script to check if data ingestion is working with increased timeout
"""

import sys
sys.path.insert(0, '/mnt/c/Users/jaafo/3a/arxiv_research_companion')

from arxiv_research_companion.data_ingestion import ingest_new_papers
import tempfile
import os

def test_data_ingestion():
    """Test data ingestion with a small request"""
    print("Testing data ingestion...")

    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name

    try:
        # Test with a very small request to minimize time
        count = ingest_new_papers(
            categories=['cs.AI'],  # Just one category
            db_path=db_path,
            max_results_per_request=5  # Very small number
        )

        print(f"Ingested {count} papers")

        if count >= 0:  # Success if we didn't get an exception
            print("✅ Data ingestion test PASSED")
            return True
        else:
            print("❌ Data ingestion test FAILED")
            return False

    except Exception as e:
        print(f"❌ Data ingestion test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    success = test_data_ingestion()
    if not success:
        sys.exit(1)