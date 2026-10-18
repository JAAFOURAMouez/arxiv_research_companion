import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion.config import load_config, get_config_value, _validate_config


class TestConfig(unittest.TestCase):

    def setUp(self):
        # Create a temporary .env file for testing
        self.test_dir = tempfile.mkdtemp()
        self.env_path = Path(self.test_dir) / ".env"

        # Write test environment variables
        with open(self.env_path, 'w') as f:
            f.write("ARXIV_CATEGORIES=cs.AI,cs.LG\n")
            f.write("ARXIV_MAX_RESULTS_PER_REQUEST=50\n")
            f.write("ARXIV_RATE_LIMIT_SECONDS=2.5\n")
            f.write("EMBEDDING_MODEL_NAME=test-model\n")
            f.write("VECTOR_DB_PATH=./test_db\n")
            f.write("INGESTION_DB_PATH=./test_papers.db\n")
            f.write("LLM_MODEL_NAME=test-llm\n")
            f.write("HF_API_TOKEN=test-token-123\n")
            f.write("HF_INFERENCE_PROVIDER=test-provider\n")
            f.write("LLM_TIMEOUT=45.5\n")
            f.write("INTERFACE_PORT=8080\n")
            f.write("LOG_LEVEL=DEBUG\n")

        # Track which environment variables we set so we can clean them up
        self.vars_to_clear = [
            'ARXIV_CATEGORIES',
            'ARXIV_MAX_RESULTS_PER_REQUEST',
            'ARXIV_RATE_LIMIT_SECONDS',
            'EMBEDDING_MODEL_NAME',
            'VECTOR_DB_PATH',
            'INGESTION_DB_PATH',
            'LLM_MODEL_NAME',
            'HF_API_TOKEN',
            'HF_INFERENCE_PROVIDER',
            'LLM_TIMEOUT',
            'INTERFACE_PORT',
            'LOG_LEVEL'
        ]

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

        # Clear the environment variables we set in setUp
        for var in self.vars_to_clear:
            if var in os.environ:
                del os.environ[var]

    def test_load_config_from_env(self):
        """Test loading configuration from .env file"""
        config = load_config(str(self.env_path))

        # Check that values from .env are loaded
        self.assertEqual(config['arxiv']['categories'], ['cs.AI', 'cs.LG'])
        self.assertEqual(config['arxiv']['max_results_per_request'], 50)
        self.assertEqual(config['arxiv']['rate_limit_seconds'], 2.5)
        self.assertEqual(config['embedding']['model_name'], 'test-model')
        self.assertEqual(config['vector_db']['path'], './test_db')
        self.assertEqual(config['ingestion']['db_path'], './test_papers.db')
        self.assertEqual(config['llm']['model_name'], 'test-llm')
        self.assertEqual(config['llm']['api_token'], 'test-token-123')
        self.assertEqual(config['llm']['inference_provider'], 'test-provider')
        self.assertEqual(config['interface']['port'], 8080)
        self.assertEqual(config['logging']['level'], 'DEBUG')

    def test_load_config_defaults(self):
        """Test that default values are used when .env doesn't specify values"""
        # Create a minimal .env file
        minimal_env = Path(self.test_dir) / ".env_minimal"
        with open(minimal_env, 'w') as f:
            f.write("")  # Empty file

        config = load_config(str(minimal_env))

        # Check that default values are present
        self.assertEqual(config['arxiv']['categories'], ['cs.AI', 'cs.LG', 'stat.ML'])  # default
        self.assertEqual(config['arxiv']['max_results_per_request'], 100)  # default
        self.assertEqual(config['arxiv']['rate_limit_seconds'], 3.0)  # default
        self.assertEqual(config['embedding']['model_name'], 'sentence-transformers/all-MiniLM-L6-v2')  # default
        self.assertEqual(config['vector_db']['path'], './chroma_db')  # default
        self.assertEqual(config['ingestion']['db_path'], './arxiv_papers.db')
        self.assertEqual(config['llm']['model_name'], 'openai/gpt-oss-120b')  # default
        self.assertIsNone(config['llm']['inference_provider'])
        self.assertEqual(config['llm']['timeout'], 60.0)  # default

    def test_load_config_timeout_from_env(self):
        """Test that LLM timeout is loaded from .env"""
        config = load_config(str(self.env_path))
        self.assertEqual(config['llm']['timeout'], 45.5)

    def test_get_config_value(self):
        """Test getting configuration values with dot notation"""
        config = load_config(str(self.env_path))

        # Test existing values
        self.assertEqual(get_config_value(config, 'arxiv.categories'), ['cs.AI', 'cs.LG'])
        self.assertEqual(get_config_value(config, 'llm.model_name'), 'test-llm')
        self.assertEqual(get_config_value(config, 'interface.port'), 8080)

        # Test non-existing values with default
        self.assertEqual(get_config_value(config, 'nonexistent.key', 'default'), 'default')
        self.assertIsNone(get_config_value(config, 'nonexistent.key'))

    def test_validate_config_valid(self):
        """Test that valid configuration passes validation"""
        config = load_config(str(self.env_path))
        # Should not raise an exception
        try:
            _validate_config(config)
        except ValueError:
            self.fail("_validate_config() raised ValueError unexpectedly!")

    def test_validate_config_invalid_arxiv_max_results(self):
        """Test validation fails for invalid arXiv max_results_per_request"""
        config = load_config(str(self.env_path))
        config['arxiv']['max_results_per_request'] = 1500  # Too high

        with self.assertRaises(ValueError):
            _validate_config(config)

        config['arxiv']['max_results_per_request'] = 0  # Too low
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_validate_config_invalid_llm_temperature(self):
        """Test validation fails for invalid LLM temperature"""
        config = load_config(str(self.env_path))
        config['llm']['temperature'] = 3.0  # Too high

        with self.assertRaises(ValueError):
            _validate_config(config)

        config['llm']['temperature'] = -0.5  # Too low
        with self.assertRaises(ValueError):
            _validate_config(config)

    def test_validate_config_invalid_interface_port(self):
        """Test validation fails for invalid interface port"""
        config = load_config(str(self.env_path))
        config['interface']['port'] = 70000  # Too high

        with self.assertRaises(ValueError):
            _validate_config(config)

        config['interface']['port'] = 0  # Too low
        with self.assertRaises(ValueError):
            _validate_config(config)


if __name__ == '__main__':
    unittest.main()