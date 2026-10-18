import unittest
from unittest.mock import patch, MagicMock

from arxiv_research_companion import main


class TestMainStartup(unittest.TestCase):
    @patch('arxiv_research_companion.main.launch_interface')
    @patch('arxiv_research_companion.main.index_papers')
    @patch('arxiv_research_companion.main.get_stored_papers')
    @patch('arxiv_research_companion.main.ingest_new_papers')
    @patch('arxiv_research_companion.main.initialize_vector_store')
    @patch('arxiv_research_companion.main.setup_logging')
    @patch('arxiv_research_companion.main.load_config')
    def test_main_passes_config_and_indexes_before_launch(
        self,
        mock_load_config,
        mock_setup_logging,
        mock_initialize,
        mock_ingest,
        mock_get_papers,
        mock_index,
        mock_launch,
    ):
        config = {
            'arxiv': {'categories': ['cs.AI'], 'max_results_per_request': 12},
            'ingestion': {'db_path': '/tmp/papers.sqlite'},
            'logging': {'level': 'WARNING'},
        }
        vector_store = MagicMock()
        papers = [{'arxiv_id': '2401.00001', 'summary': 'Abstract'}]
        sequence = []
        mock_load_config.return_value = config
        mock_initialize.side_effect = lambda _: (sequence.append('initialize') or vector_store)
        mock_ingest.side_effect = lambda *args, **kwargs: sequence.append('ingest')
        mock_get_papers.side_effect = lambda _: (sequence.append('read_sqlite') or papers)
        mock_index.side_effect = lambda *_: (sequence.append('index') or 1)
        mock_launch.side_effect = lambda *_: sequence.append('launch')

        main.main()

        mock_setup_logging.assert_called_once_with(config)
        mock_initialize.assert_called_once_with(config)
        mock_ingest.assert_called_once_with(
            ['cs.AI'], '/tmp/papers.sqlite', max_results_per_request=12
        )
        mock_get_papers.assert_called_once_with('/tmp/papers.sqlite')
        mock_index.assert_called_once_with(vector_store, papers)
        mock_launch.assert_called_once_with(config, vector_store)
        self.assertEqual(sequence, ['initialize', 'ingest', 'read_sqlite', 'index', 'launch'])


if __name__ == '__main__':
    unittest.main()
