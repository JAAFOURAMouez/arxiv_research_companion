import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion import rag_engine


class TestRagEngine(unittest.TestCase):

    def setUp(self):
        # Create a mock vector store dict as returned by embedding_store.initialize_vector_store
        self.vector_store = {
            "model": MagicMock(),
            "collection": MagicMock()
        }
        # Mock the model's encode method to return a fake embedding
        import numpy as np
        self.vector_store["model"].encode.return_value = np.array([[0.1, 0.2, 0.3]])
        # Mock the collection's query method to return some results
        self.vector_store["collection"].query.return_value = {
            'ids': [['2103.12345', '2103.67890']],
            'distances': [[0.1, 0.2]],
            'metadatas': [
                [{'arxiv_id': '2103.12345', 'title': 'Paper One', 'authors': ['Author One']}],
                [{'arxiv_id': '2103.67890', 'title': 'Paper Two', 'authors': ['Author Two']}]
            ],
            'documents': [
                ['Abstract one'],
                ['Abstract two']
            ]
        }

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_process_query(self, mock_inference_client):
        # Mock the InferenceClient and its chat_completion method
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': [{'message': {'content': "This is a generated answer based on the context."}}]}
        mock_inference_client.return_value = mock_client

        config = {
            'llm_model': 'Qwen/Qwen2.5-7B-Instruct',
            'llm_max_new_tokens': 100,
            'llm_temperature': 0.7,
            'retrieval_k': 2
        }
        result = rag_engine.generate_answer("test query", self.vector_store, config)

        # Assertions
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertEqual(result['answer'], "This is a generated answer based on the context.")
        self.assertEqual(len(result['sources']), 2)
        self.assertEqual(result['sources'][0]['id'], '2103.12345')
        self.assertEqual(result['sources'][0]['metadata']['title'], 'Paper One')
        self.assertEqual(result['sources'][1]['id'], '2103.67890')
        self.assertEqual(result['sources'][1]['metadata']['title'], 'Paper Two')

        # Check that the model's encode method was called with the query
        self.vector_store["model"].encode.assert_called_once_with(["test query"])
        # Check that the collection's query method was called with the embedding
        self.vector_store["collection"].query.assert_called_once()
        args, kwargs = self.vector_store["collection"].query.call_args
        self.assertIn('query_embeddings', kwargs)
        self.assertIn('n_results', kwargs)
        self.assertEqual(kwargs['n_results'], 2)
        # Check that the query_embeddings is a list of one embedding (our mock returned [[0.1,0.2,0.3]])
        query_embedding = kwargs['query_embeddings']
        self.assertEqual(len(query_embedding), 1)
        self.assertEqual(query_embedding[0], [0.1, 0.2, 0.3])

        # Check that the LLM was called with a prompt that contains the context
        mock_client.chat_completion.assert_called_once()
        args, kwargs = mock_client.chat_completion.call_args
        messages = kwargs['messages']
        self.assertEqual(messages[0]['role'], 'user')
        prompt = messages[0]['content']
        self.assertIn('test query', prompt)
        self.assertIn('Paper One', prompt)
        self.assertIn('Paper Two', prompt)
        self.assertIn('Abstract one', prompt)
        self.assertIn('Abstract two', prompt)
        self.assertEqual(kwargs['max_tokens'], 100)
        self.assertEqual(kwargs['temperature'], 0.7)

        # Check that the InferenceClient was constructed with the default timeout
        mock_inference_client.assert_called_once_with(
            model='Qwen/Qwen2.5-7B-Instruct', token=None, timeout=60.0
        )

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_empty_store_skips_llm_call(self, mock_inference_client):
        self.vector_store['collection'].query.return_value = {
            'ids': [[]], 'distances': [[]], 'metadatas': [[]], 'documents': [[]]
        }

        result = rag_engine.generate_answer('test query', self.vector_store, {'retrieval_k': 2})

        self.assertEqual(result['sources'], [])
        self.assertIn('couldn\'t find any papers', result['answer'])
        mock_inference_client.assert_not_called()

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_chroma_nested_results_keep_all_sources(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': [{'message': {'content': "Grounded response."}}]}
        mock_inference_client.return_value = mock_client
        self.vector_store['collection'].query.return_value = {
            'ids': [['2103.12345', '2103.67890']],
            'distances': [[0.1, 0.2]],
            'metadatas': [[
                {'title': 'Paper One', 'authors': ['Author One']},
                {'title': 'Paper Two', 'authors': ['Author Two']},
            ]],
            'documents': [['Abstract one', 'Abstract two']],
        }

        result = rag_engine.generate_answer('test query', self.vector_store, {'retrieval_k': 2})

        self.assertEqual([source['metadata']['title'] for source in result['sources']],
                         ['Paper One', 'Paper Two'])
        self.assertIn('Abstract two', mock_client.chat_completion.call_args.kwargs['messages'][0]['content'])

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_nested_llm_config_reaches_inference_client(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': [{'message': {'content': "Generated response."}}]}
        mock_inference_client.return_value = mock_client
        token = "test-token-never-log"
        config = {
            'llm': {
                'model_name': 'test-model',
                'api_token': token,
                'max_new_tokens': 42,
                'temperature': 0.25,
                'top_p': 0.8,
                'do_sample': False,
                'timeout': 123.0,
            },
            'retrieval_k': 2,
        }

        rag_engine.generate_answer("test query", self.vector_store, config)

        mock_inference_client.assert_called_once_with(model='test-model', token=token, timeout=123.0)
        generation_kwargs = mock_client.chat_completion.call_args.kwargs
        self.assertEqual(generation_kwargs['max_tokens'], 42)
        self.assertEqual(generation_kwargs['temperature'], 0.25)
        self.assertNotIn('do_sample', generation_kwargs)
        self.assertEqual(generation_kwargs['top_p'], 0.8)

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_configured_inference_provider_reaches_client(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': [{'message': {'content': 'Generated.'}}]}
        mock_inference_client.return_value = mock_client

        rag_engine.generate_answer('test query', self.vector_store, {
            'llm': {'model_name': 'test-model', 'inference_provider': 'test-provider'},
            'retrieval_k': 2,
        })

        mock_inference_client.assert_called_once_with(
            model='test-model', token=None, provider='test-provider', timeout=60.0
        )

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_model_not_supported_error_gives_account_guidance(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception(
            "model_not_supported: not supported by any provider enabled for account"
        )
        mock_inference_client.return_value = mock_client

        result = rag_engine.generate_answer('test query', self.vector_store, {'retrieval_k': 2})

        self.assertIn('Enable an Inference Provider', result['answer'])
        self.assertIn('ensure your token can call Inference Providers', result['answer'])
        self.assertIn('Abstract one', result['answer'])
        self.assertEqual(len(result['sources']), 2)

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_process_query_llm_failure(self, mock_inference_client):
        # Mock the InferenceClient to raise an exception
        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = Exception("LLM API failed")
        mock_inference_client.return_value = mock_client

        config = {
            'llm_model': 'Qwen/Qwen2.5-7B-Instruct',
            'llm_max_new_tokens': 100,
            'llm_temperature': 0.7,
            'retrieval_k': 2
        }
        result = rag_engine.generate_answer("test query", self.vector_store, config)

        # Even if LLM fails, we should return the retrieved abstracts with a disclaimer
        self.assertIn('answer', result)
        self.assertIn('sources', result)
        self.assertIn('failed to generate an answer', result['answer'].lower())
        self.assertEqual(len(result['sources']), 2)

        # Check that the retrieval still happened
        self.vector_store["model"].encode.assert_called_once_with(["test query"])
        self.vector_store["collection"].query.assert_called_once()

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_malformed_chat_response_returns_retrieved_context(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': []}
        mock_inference_client.return_value = mock_client

        result = rag_engine.generate_answer('test query', self.vector_store, {'retrieval_k': 2})

        self.assertIn("model 'openai/gpt-oss-120b'", result['answer'])
        self.assertIn('compatible Hugging Face chat inference provider may be unavailable', result['answer'])
        self.assertIn('Abstract one', result['answer'])
        self.assertEqual(len(result['sources']), 2)

    @patch('arxiv_research_companion.rag_engine.InferenceClient')
    def test_blank_chat_content_returns_retrieved_context(self, mock_inference_client):
        mock_client = MagicMock()
        mock_client.chat_completion.return_value = {'choices': [{'message': {'content': '  '}}]}
        mock_inference_client.return_value = mock_client

        result = rag_engine.generate_answer('test query', self.vector_store, {'retrieval_k': 2})

        self.assertIn("model 'openai/gpt-oss-120b'", result['answer'])
        self.assertIn('Abstract one', result['answer'])
        self.assertEqual(len(result['sources']), 2)

    def test_format_context(self):
        # Test the internal function that formats context from retrieved results
        retrieval_results = {
            'ids': [['2103.12345', '2103.67890']],
            'distances': [[0.1, 0.2]],
            'metadatas': [
                [{'arxiv_id': '2103.12345', 'title': 'Paper One', 'authors': ['Author One']},
                 {'arxiv_id': '2103.67890', 'title': 'Paper Two', 'authors': ['Author Two']}]
            ],
            'documents': [
                ['Abstract one', 'Abstract two']
            ]
        }
        # We'll assume there is a function _format_context in rag_engine
        context = rag_engine._format_context(retrieval_results)
        self.assertIn('Paper One', context)
        self.assertIn('Paper Two', context)
        self.assertIn('Author One', context)
        self.assertIn('Author Two', context)
        self.assertIn('Abstract one', context)
        self.assertIn('Abstract two', context)

    def test_construct_prompt(self):
        # Test the internal function that constructs the prompt for the LLM
        query = "What is the meaning of life?"
        context = "Context: Paper OneAbstract onePaper TwoAbstract two"
        prompt = rag_engine._construct_prompt(query, context)
        self.assertIn(query, prompt)
        self.assertIn(context, prompt)
        # Check that the prompt includes instructions for the LLM
        self.assertIn("Answer the question based on the context", prompt)
        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)

if __name__ == '__main__':
    unittest.main()