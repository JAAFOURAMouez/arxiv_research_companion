import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys
import numpy as np

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion import embedding_store


class TestEmbeddingStore(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for the vector store
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "chroma_db")

    def tearDown(self):
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(self.test_dir)

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_initialize_vector_store(self, mock_chroma_client, mock_sentence_transformer):
        # Mock the SentenceTransformer and ChromaDB client
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Call the function
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_db_path': self.db_path
        }
        vector_store = embedding_store.initialize_vector_store(config)

        # Assertions
        mock_sentence_transformer.assert_called_once_with('sentence-transformers/all-MiniLM-L6-v2')
        mock_chroma_client.assert_called_once_with(path=self.db_path)
        mock_client.get_or_create_collection.assert_called_once_with(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )
        self.assertEqual(vector_store['model'], mock_model)
        self.assertEqual(vector_store['collection'], mock_collection)
        self.assertEqual(vector_store['dimension'], 384)

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_add_embeddings(self, mock_chroma_client, mock_sentence_transformer):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Initialize vector store
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_db_path': self.db_path
        }
        vector_store = embedding_store.initialize_vector_store(config)

        # Prepare texts and metadata
        texts = ["First paper abstract", "Second paper abstract"]
        metadata = [
            {'arxiv_id': '2103.12345', 'title': 'Paper One'},
            {'arxiv_id': '2103.67890', 'title': 'Paper Two'}
        ]

        # Call add_embeddings
        embedding_store.add_embeddings(vector_store, texts, metadata)

        # Assertions
        mock_model.encode.assert_called_once_with(texts, show_progress_bar=True)
        mock_collection.upsert.assert_called_once()
        # Check the arguments passed to collection.upsert
        args, kwargs = mock_collection.upsert.call_args
        self.assertIn('embeddings', kwargs)
        self.assertIn('metadatas', kwargs)
        self.assertIn('documents', kwargs)
        self.assertIn('ids', kwargs)
        # Check that embeddings are a list of lists (or numpy array converted to list)
        embeddings = kwargs['embeddings']
        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3])
        self.assertEqual(embeddings[1], [0.4, 0.5, 0.6])
        self.assertEqual(kwargs['metadatas'], metadata)
        # Check that IDs are generated (we can check the format)
        self.assertEqual(len(kwargs['ids']), 2)
        self.assertTrue(all(isinstance(id_str, str) for id_str in kwargs['ids']))

    def test_index_papers_stores_abstract_and_skips_existing_ids(self):
        model = MagicMock()
        model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        collection = MagicMock()
        collection.get.side_effect = [
            {'ids': []},
            {'ids': ['2103.12345']},
        ]
        vector_store = {'model': model, 'collection': collection}
        papers = [{
            'arxiv_id': '2103.12345',
            'title': 'Paper One',
            'authors': ['Author One'],
            'categories': ['cs.AI'],
            'summary': 'Abstract one',
            'published': '2021-03-15 00:00:00',
        }]

        self.assertEqual(embedding_store.index_papers(vector_store, papers), 1)
        self.assertEqual(embedding_store.index_papers(vector_store, papers), 0)
        collection.upsert.assert_called_once()
        self.assertEqual(collection.upsert.call_args.kwargs['documents'], ['Abstract one'])
        self.assertEqual(collection.upsert.call_args.kwargs['ids'], ['2103.12345'])
        self.assertEqual(collection.upsert.call_args.kwargs['metadatas'][0]['authors'], 'Author One')

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_search_similar(self, mock_chroma_client, mock_sentence_transformer):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        # Mock the query result
        mock_collection.query.return_value = {
            'ids': [['2103.12345', '2103.67890']],
            'distances': [[0.1, 0.2]],
            'metadatas': [
                [{'arxiv_id': '2103.12345', 'title': 'Paper One'}],
                [{'arxiv_id': '2103.67890', 'title': 'Paper Two'}]
            ]
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Initialize vector store
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_db_path': self.db_path
        }
        vector_store = embedding_store.initialize_vector_store(config)

        # Call search_similar
        query_text = "query abstract"
        top_k = 2
        results = embedding_store.search_similar(vector_store, query_text, top_k)

        # Assertions
        mock_model.encode.assert_called_once_with([query_text], show_progress_bar=True)
        mock_collection.query.assert_called_once()
        # Check the arguments to query
        args, kwargs = mock_collection.query.call_args
        self.assertIn('query_embeddings', kwargs)
        self.assertIn('n_results', kwargs)
        self.assertEqual(kwargs['n_results'], top_k)
        query_embedding = kwargs['query_embeddings']
        self.assertEqual(query_embedding, [[0.1, 0.2, 0.3]])
        # Check the results structure
        self.assertIn('ids', results)
        self.assertIn('distances', results)
        self.assertIn('metadatas', results)
        self.assertEqual(len(results['ids'][0]), 2)
        self.assertEqual(results['ids'][0], ['2103.12345', '2103.67890'])
        self.assertEqual(results['distances'][0], [0.1, 0.2])
        self.assertEqual(len(results['metadatas']), 2)
        self.assertEqual(results['metadatas'][0], [{'arxiv_id': '2103.12345', 'title': 'Paper One'}])
        self.assertEqual(results['metadatas'][1], [{'arxiv_id': '2103.67890', 'title': 'Paper Two'}])

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_get_paper_by_id(self, mock_chroma_client, mock_sentence_transformer):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        # Mock the get result
        mock_collection.get.return_value = {
            'ids': ['2103.12345'],
            'metadatas': [{'arxiv_id': '2103.12345', 'title': 'Paper One', 'abstract': 'An abstract'}],
            'documents': ['An abstract']
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Initialize vector store
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_db_path': self.db_path
        }
        vector_store = embedding_store.initialize_vector_store(config)

        # Call get_paper_by_id
        paper_id = '2103.12345'
        result = embedding_store.get_paper_by_id(vector_store, paper_id)

        # Assertions
        mock_collection.get.assert_called_once()
        args, kwargs = mock_collection.get.call_args
        self.assertIn('ids', kwargs)
        self.assertEqual(kwargs['ids'], [paper_id])
        # Check the result
        self.assertEqual(result['id'], '2103.12345')
        self.assertEqual(result['metadata'], {'arxiv_id': '2103.12345', 'title': 'Paper One', 'abstract': 'An abstract'})
        self.assertEqual(result['document'], 'An abstract')

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_initialize_vector_store_uses_nested_config(self, mock_chroma_client, mock_sentence_transformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_sentence_transformer.return_value = mock_model
        mock_collection = MagicMock()
        mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection

        config = {
            'embedding': {'model_name': 'nested-model'},
            'vector_db': {
                'path': self.db_path,
                'collection_name': 'configured-papers',
                'distance_metric': 'l2',
            },
        }
        vector_store = embedding_store.initialize_vector_store(config)

        mock_sentence_transformer.assert_called_once_with('nested-model')
        mock_chroma_client.assert_called_once_with(path=self.db_path)
        mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(
            name='configured-papers', metadata={'hnsw:space': 'l2'}
        )
        self.assertEqual(vector_store['collection'], mock_collection)

    @patch('arxiv_research_companion.embedding_store.SentenceTransformer')
    @patch('arxiv_research_companion.embedding_store.chromadb.PersistentClient')
    def test_persistence(self, mock_chroma_client, mock_sentence_transformer):
        # Setup mocks
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_sentence_transformer.return_value = mock_model

        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chroma_client.return_value = mock_client

        # Initialize vector store
        config = {
            'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'vector_db_path': self.db_path
        }
        vector_store = embedding_store.initialize_vector_store(config)

        # The vector store should have been created with the persist directory
        mock_chroma_client.assert_called_once_with(path=self.db_path)

        # We can also test that the collection is the same
        self.assertEqual(vector_store['collection'], mock_collection)


if __name__ == '__main__':
    unittest.main()