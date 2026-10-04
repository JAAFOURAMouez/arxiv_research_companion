import unittest
import tempfile
import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion.utils import *


class TestUtils(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir)

    def test_clean_text(self):
        """Test text cleaning functionality"""
        # Test basic cleaning
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("\n\t\n"), "")

        # Test unicode normalization
        self.assertEqual(clean_text("café"), "café")  # Should remain the same
        self.assertEqual(clean_text("é"), "é")  # Combined characters should normalize

        # Test multiple spaces
        self.assertEqual(clean_text("a    b"), "a b")

    def test_is_valid_arxiv_id(self):
        """Test arXiv ID validation"""
        # Valid new format IDs
        self.assertTrue(is_valid_arxiv_id("1501.00001"))
        self.assertTrue(is_valid_arxiv_id("2103.12345"))
        self.assertTrue(is_valid_arxiv_id("2103.12345v1"))  # With version
        self.assertTrue(is_valid_arxiv_id("2103.12345V2"))  # Uppercase V

        # Valid old format IDs
        self.assertTrue(is_valid_arxiv_id("hep-th/9901010"))
        self.assertTrue(is_valid_arxiv_id("hep-th/9901010v1"))  # With version
        self.assertTrue(is_valid_arxiv_id("math.GT/0309136"))

        # Invalid IDs
        self.assertFalse(is_valid_arxiv_id(""))
        self.assertFalse(is_valid_arxiv_id("invalid"))
        self.assertFalse(is_valid_arxiv_id("1501.0000"))  # Too few digits
        self.assertFalse(is_valid_arxiv_id("1501.000001"))  # Too many digits
        self.assertFalse(is_valid_arxiv_id("hep-th/990101"))  # Too few digits
        self.assertFalse(is_valid_arxiv_id("hep-th/99010101"))  # Too many digits
        self.assertFalse(is_valid_arxiv_id("1501.00001 extra"))  # Extra text

    def test_parse_and_format_date(self):
        """Test date parsing and formatting"""
        # Test parsing various formats
        test_cases = [
            ("2023-01-15", "%Y-%m-%d"),
            ("2023-01-15T10:30:00", "%Y-%m-%dT%H:%M:%S"),
            ("2023-01-15T10:30:00Z", "%Y-%m-%dT%H:%M:%SZ"),
            ("15/01/2023", "%d/%m/%Y"),
            ("01/15/2023", "%m/%d/%Y"),
            ("January 15, 2023", "%B %d, %Y"),
            ("15 January 2023", "%d %B %Y"),
        ]

        for date_str, expected_format in test_cases:
            parsed = parse_date(date_str)
            self.assertIsInstance(parsed, datetime)
            # Check that we can format it back reasonably
            formatted = format_date(parsed)
            self.assertIsInstance(formatted, str)
            self.assertGreater(len(formatted), 0)

        # Test invalid date strings
        self.assertIsNone(parse_date(""))
        self.assertIsNone(parse_date("not a date"))
        self.assertIsNone(parse_date("2023-13-45"))  # Invalid month/day

    def test_save_and_load_json(self):
        """Test JSON save and load functionality"""
        test_data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {
                "key": "value"
            }
        }

        file_path = self.test_path / "test.json"

        # Test saving
        save_json(test_data, file_path)
        self.assertTrue(file_path.exists())

        # Test loading
        loaded_data = load_json(file_path)
        self.assertEqual(loaded_data, test_data)

        # Test that temporary file is cleaned up
        temp_path = file_path.with_suffix('.tmp')
        self.assertFalse(temp_path.exists())

    def test_save_and_load_pickle(self):
        """Test pickle save and load functionality"""
        test_data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {
                "key": "value"
            },
            "self_ref": None  # Will be set below
        }
        test_data["self_ref"] = test_data  # Create a circular reference

        file_path = self.test_path / "test.pkl"

        # Test saving
        save_pickle(test_data, file_path)
        self.assertTrue(file_path.exists())

        # Test loading
        loaded_data = load_pickle(file_path)
        self.assertEqual(loaded_data["name"], test_data["name"])
        self.assertEqual(loaded_data["values"], test_data["values"])
        self.assertEqual(loaded_data["nested"]["key"], test_data["nested"]["key"])
        # Note: Circular reference handling might differ, but basic structure should be there

        # Test that temporary file is cleaned up
        temp_path = file_path.with_suffix('.tmp')
        self.assertFalse(temp_path.exists())

    def test_hash_string(self):
        """Test string hashing functionality"""
        # Test that same input gives same output
        hash1 = hash_string("hello world")
        hash2 = hash_string("hello world")
        self.assertEqual(hash1, hash2)

        # Test that different inputs give different outputs
        hash3 = hash_string("hello world!")
        self.assertNotEqual(hash1, hash3)

        # Test that output is hexadecimal
        self.assertTrue(all(c in '0123456789abcdef' for c in hash1))
        self.assertEqual(len(hash1), 64)  # SHA256 produces 64 hex characters

    def test_load_env_var(self):
        """Test environment variable loading"""
        # Set a test environment variable
        os.environ["TEST_VAR"] = "test_value"

        try:
            # Test loading existing variable
            self.assertEqual(load_env_var("TEST_VAR"), "test_value")

            # Test loading with default
            self.assertEqual(load_env_var("NONEXISTENT_VAR", "default"), "default")

            # Test loading nonexistent without default
            self.assertIsNone(load_env_var("NONEXISTENT_VAR"))

            # Test required variable that exists
            self.assertEqual(load_env_var("TEST_VAR", required=True), "test_value")

            # Test required variable that doesn't exist
            del os.environ["TEST_VAR"]
            with self.assertRaises(ValueError):
                load_env_var("TEST_VAR", required=True)
        finally:
            # Clean up
            if "TEST_VAR" in os.environ:
                del os.environ["TEST_VAR"]

    def test_retry_decorator(self):
        """Test the retry_with_backoff decorator"""
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01, max_delay=0.1)
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        # Should succeed after 3 attempts
        result = failing_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)

        # Test that it fails after max retries
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01, max_delay=0.1)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with self.assertRaises(RuntimeError):
            always_failing_function()

        self.assertEqual(call_count, 3)  # Initial attempt + 2 retries

    def test_show_progress(self):
        """Test progress bar display (just check it doesn't crash)"""
        # This mainly tests that the function doesn't crash
        # We can't easily test the visual output in unit tests
        try:
            show_progress(0, 10)
            show_progress(5, 10)
            show_progress(10, 10)  # Completion
        except Exception as e:
            self.fail(f"show_progress raised an exception: {e}")

    def test_timer_decorator(self):
        """Test the timer decorator"""
        @timer
        def fast_function():
            time.sleep(0.01)
            return "done"

        # Should execute without error
        result = fast_function()
        self.assertEqual(result, "done")


if __name__ == '__main__':
    unittest.main()