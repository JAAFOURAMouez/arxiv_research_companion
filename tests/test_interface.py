import unittest
import sys
from unittest.mock import patch, MagicMock
import os

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion.interface import launch_interface
from arxiv_research_companion.utils import clean_text, is_valid_arxiv_id


class TestInterfaceUtils(unittest.TestCase):
    """Test utility functions used in the interface module"""

    def test_clean_text(self):
        """Test text cleaning function"""
        self.assertEqual(clean_text("  hello   world  "), "hello world")
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text("\n\t\n"), "")

    def test_is_valid_arxiv_id(self):
        """Test arXiv ID validation function"""
        # Valid IDs
        self.assertTrue(is_valid_arxiv_id("1501.00001"))
        self.assertTrue(is_valid_arxiv_id("2103.12345v1"))
        self.assertTrue(is_valid_arxiv_id("hep-th/9901010"))

        # Invalid IDs
        self.assertFalse(is_valid_arxiv_id(""))
        self.assertFalse(is_valid_arxiv_id("invalid"))
        self.assertFalse(is_valid_arxiv_id("1501.0000"))


class TestInterfaceLogic(unittest.TestCase):
    """Test the core logic of the interface module"""

    @patch('arxiv_research_companion.interface.gr')
    @patch('arxiv_research_companion.interface.logging')
    def test_launch_interface_basic_structure(self, mock_logging, mock_gradio):
        """Test that launch_interface sets up the basic Gradio structure"""
        # Mock config and vector_store
        config = {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': 'test.log',
                'console_output': True
            },
            'interface': {
                'theme': 'default',
                'port': 7860,
                'share': False
            }
        }
        vector_store = MagicMock()

        # Mock Gradio components
        mock_blocks = MagicMock()
        mock_gradio.Blocks.return_value.__enter__.return_value = mock_blocks

        # Call the function
        launch_interface(config, vector_store)

        # Verify that Blocks was called with the correct title
        mock_gradio.Blocks.assert_called()
        args, kwargs = mock_gradio.Blocks.call_args
        self.assertIn('title', kwargs)
        self.assertEqual(kwargs['title'], 'arXiv Research Companion')

        # Verify that launch was called
        mock_blocks.launch.assert_called()

    @patch('arxiv_research_companion.interface.gr')
    @patch('arxiv_research_companion.interface.setup_logging')
    @patch('arxiv_research_companion.interface.get_total_papers_count')
    @patch('arxiv_research_companion.interface.get_last_update_timestamp')
    def test_status_uses_configured_ingestion_database(
        self, mock_last_update, mock_total, mock_setup_logging, mock_gradio
    ):
        config = {
            'logging': {},
            'interface': {},
            'ingestion': {'db_path': '/tmp/configured-papers.sqlite'},
        }
        mock_blocks = MagicMock()
        mock_gradio.Blocks.return_value.__enter__.return_value = mock_blocks
        mock_last_update.return_value = None
        mock_total.return_value = 3

        launch_interface(config, MagicMock())
        update_status = mock_blocks.load.call_args.kwargs['fn']
        update_status()

        mock_last_update.assert_called_once_with('/tmp/configured-papers.sqlite')
        mock_total.assert_called_once_with('/tmp/configured-papers.sqlite')

    @patch('arxiv_research_companion.interface.gr')
    @patch('arxiv_research_companion.interface.logging')
    def test_launch_interface_with_auth(self, mock_logging, mock_gradio):
        """Test that launch_interface handles authentication correctly"""
        # Mock config with auth credentials
        config = {
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_path': 'test.log',
                'console_output': True
            },
            'interface': {
                'theme': 'default',
                'port': 7860,
                'share': False,
                'auth_username': 'testuser',
                'auth_password': 'testpass'
            }
        }
        vector_store = MagicMock()

        # Mock Gradio components
        mock_blocks = MagicMock()
        mock_gradio.Blocks.return_value.__enter__.return_value = mock_blocks

        # Call the function
        launch_interface(config, vector_store)

        # Verify that launch was called with auth parameter
        mock_blocks.launch.assert_called()
        args, kwargs = mock_blocks.launch.call_args
        self.assertIn('auth', kwargs)
        self.assertEqual(kwargs['auth'], ('testuser', 'testpass'))


if __name__ == '__main__':
    unittest.main()