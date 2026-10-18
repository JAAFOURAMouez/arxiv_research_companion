import unittest
from unittest.mock import patch, MagicMock
import tempfile
import os
import sqlite3
from datetime import datetime, timedelta
import json
import sys

# Add the parent directory to sys.path to allow imports when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from arxiv_research_companion import data_ingestion


class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        # Create a temporary database for testing
        self.db_fd, self.db_path = tempfile.mkstemp()
        self.conn = sqlite3.connect(self.db_path)
        # Initialize the database schema
        data_ingestion._init_db(self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_respect_rate_limit_logic(self):
        # Test the rate limiting logic by directly manipulating the global variable
        # Reset the global variable (not ideal but for testing)
        data_ingestion._last_request_time = 0.0

        # Case 1: First call, should not sleep if we are already past the interval? Actually, if last request was long ago, no sleep.
        # We'll mock time.time to return a fixed value and then check.
        with patch('arxiv_research_companion.data_ingestion.time.time') as mock_time:
            mock_time.return_value = 10.0  # Current time is 10
            data_ingestion._last_request_time = 5.0  # Last request was at 5, so 5 seconds passed -> no sleep needed
            with patch('arxiv_research_companion.data_ingestion.time.sleep') as mock_sleep:
                data_ingestion._respect_rate_limit()
                mock_sleep.assert_not_called()

            # Case 2: Last request was very recent, need to sleep
            data_ingestion._last_request_time = 9.0  # Last request at 9, current time 10 -> 1 second passed -> need to sleep 2 seconds
            with patch('arxiv_research_companion.data_ingestion.time.sleep') as mock_sleep:
                data_ingestion._respect_rate_limit()
                mock_sleep.assert_called_with(2.0)

    @patch('arxiv_research_companion.data_ingestion.requests.get')
    def test_fetch_arxiv_papers(self, mock_get):
        # Mock the response from arXiv API
        mock_response = MagicMock()
        mock_response.text = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2103.12345v1</id>
            <title>Test Paper</title>
            <summary>This is a test abstract.</summary>
            <author><name>Test Author</name></author>
            <category term="cs.AI"/>
            <published>2021-03-15T00:00:00Z</published>
          </entry>
        </feed>'''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        # Call the function
        papers = data_ingestion.fetch_arxiv_papers(['cs.AI'], max_results=10)

        # Assertions
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper['arxiv_id'], '2103.12345')
        self.assertEqual(paper['title'], 'Test Paper')
        self.assertEqual(paper['summary'], 'This is a test abstract.')
        self.assertEqual(paper['authors'], ['Test Author'])
        self.assertEqual(paper['categories'], ['cs.AI'])
        self.assertIsNotNone(paper['published'])
        self.assertIsNone(paper['doi'])

        # Check that the rate limiting function was called (indirectly via the mock)
        # We can't easily test the internal _respect_rate_limit without patching it, but we can trust it's called.

    @patch('arxiv_research_companion.data_ingestion._respect_rate_limit')
    @patch('arxiv_research_companion.data_ingestion.requests.get')
    def test_fetch_query_uses_arxiv_date_range_syntax(self, mock_get, _mock_rate_limit):
        response = MagicMock()
        response.text = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        data_ingestion.fetch_arxiv_papers(
            ['cs.AI', 'cs.LG'],
            start_date=datetime(2026, 9, 3, 12, 30),
        )

        query = mock_get.call_args.kwargs['params']['search_query']
        self.assertRegex(
            query,
            r'^\(cat:cs\.AI OR cat:cs\.LG\) AND submittedDate:\[202609031230 TO \d{12}\]$',
        )

    def test_parse_arxiv_xml(self):
        xml_text = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
          <entry>
            <id>http://arxiv.org/abs/2103.12345v1</id>
            <title>Test Paper</title>
            <summary>This is a test abstract.</summary>
            <author><name>Test Author</name></author>
            <category term="cs.AI"/>
            <category term="cs.LG"/>
            <published>2021-03-15T00:00:00Z</published>
            <arxiv:doi>10.1234/test</arxiv:doi>
          </entry>
          <entry>
            <id>http://arxiv.org/abs/2103.67890</id>
            <title>Another Paper</title>
            <summary>Another abstract.</summary>
            <author><name>Author One</name></author>
            <author><name>Author Two</name></author>
            <category term="stat.ML"/>
            <published>2021-03-16T00:00:00Z</published>
          </entry>
        </feed>'''
        papers = data_ingestion._parse_arxiv_xml(xml_text)
        self.assertEqual(len(papers), 2)

        # First paper
        self.assertEqual(papers[0]['arxiv_id'], '2103.12345')
        self.assertEqual(papers[0]['title'], 'Test Paper')
        self.assertEqual(papers[0]['summary'], 'This is a test abstract.')
        self.assertEqual(papers[0]['authors'], ['Test Author'])
        self.assertEqual(papers[0]['categories'], ['cs.AI', 'cs.LG'])
        self.assertEqual(papers[0]['doi'], '10.1234/test')

        # Second paper
        self.assertEqual(papers[1]['arxiv_id'], '2103.67890')
        self.assertEqual(papers[1]['title'], 'Another Paper')
        self.assertEqual(papers[1]['summary'], 'Another abstract.')
        self.assertEqual(papers[1]['authors'], ['Author One', 'Author Two'])
        self.assertEqual(papers[1]['categories'], ['stat.ML'])
        self.assertIsNone(papers[1]['doi'])

    def test_init_db(self):
        # The database is initialized in setUp via _init_db, but we can test that the table exists.
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='papers';")
        table_exists = cursor.fetchone()
        self.assertIsNotNone(table_exists)

        # Check for index
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_papers_published';")
        index_exists = cursor.fetchone()
        self.assertIsNotNone(index_exists)

    def test_paper_exists(self):
        # Add a paper to the database
        paper = {
            'arxiv_id': '2103.12345',
            'title': 'Test Paper',
            'authors': ['Author One'],
            'summary': 'Test summary',
            'categories': ['cs.AI'],
            'published': datetime(2021, 3, 15),
            'doi': '10.1234/test'
        }
        data_ingestion._store_paper(self.conn, paper)

        # Check that it exists
        self.assertTrue(data_ingestion._paper_exists(self.conn, '2103.12345'))
        # Check that a non-existent paper does not exist
        self.assertFalse(data_ingestion._paper_exists(self.conn, '2103.99999'))

    def test_store_paper(self):
        paper = {
            'arxiv_id': '2103.12345',
            'title': 'Test Paper',
            'authors': ['Author One', 'Author Two'],
            'summary': 'Test summary',
            'categories': ['cs.AI', 'cs.LG'],
            'published': datetime(2021, 3, 15),
            'doi': '10.1234/test'
        }
        data_ingestion._store_paper(self.conn, paper)

        # Retrieve the paper and check the fields
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE arxiv_id = ?', (paper['arxiv_id'],))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        # The row structure: (arxiv_id, title, authors, summary, categories, published, doi, updated_at)
        self.assertEqual(row[0], paper['arxiv_id'])
        self.assertEqual(row[1], paper['title'])
        self.assertEqual(json.loads(row[2]), paper['authors'])  # authors are stored as JSON
        self.assertEqual(row[3], paper['summary'])
        self.assertEqual(json.loads(row[4]), paper['categories'])  # categories are stored as JSON
        self.assertEqual(row[4], json.dumps(paper['categories']))  # Actually, let's check the JSON string
        # We stored the JSON string, so row[4] should be the JSON string of the list
        self.assertEqual(row[4], json.dumps(paper['categories']))
        self.assertEqual(row[5], paper['published'].isoformat(sep=' '))  # SQLite stores TIMESTAMP as string in ISO format without T?
        # Actually, we stored the datetime object, and SQLite converts it to a string in ISO format.
        # Let's just check that the date part is correct.
        self.assertIn('2021-03-15', row[5])
        self.assertEqual(row[6], paper['doi'])

    def test_get_last_update_timestamp(self):
        # Initially, no papers, so should return None
        self.assertIsNone(data_ingestion.get_last_update_timestamp(self.db_path))

        # Add a paper
        paper = {
            'arxiv_id': '2103.12345',
            'title': 'Test Paper',
            'authors': ['Author One'],
            'summary': 'Test summary',
            'categories': ['cs.AI'],
            'published': datetime(2021, 3, 15),
            'doi': None
        }
        data_ingestion._store_paper(self.conn, paper)

        # Now the last update should be the paper's published date
        last_update = data_ingestion.get_last_update_timestamp(self.db_path)
        self.assertIsNotNone(last_update)
        self.assertEqual(last_update.date(), datetime(2021, 3, 15).date())

        # Add another paper with a later date
        paper2 = {
            'arxiv_id': '2103.67890',
            'title': 'Test Paper 2',
            'authors': ['Author Two'],
            'summary': 'Test summary 2',
            'categories': ['cs.LG'],
            'published': datetime(2021, 4, 15),
            'doi': None
        }
        data_ingestion._store_paper(self.conn, paper2)

        last_update = data_ingestion.get_last_update_timestamp(self.db_path)
        self.assertIsNotNone(last_update)
        self.assertEqual(last_update.date(), datetime(2021, 4, 15).date())

    def test_get_stored_papers_decodes_fields_for_indexing(self):
        data_ingestion._store_paper(self.conn, {
            'arxiv_id': '2103.12345',
            'title': 'Stored paper',
            'authors': ['Author One'],
            'summary': 'Stored abstract',
            'categories': ['cs.AI'],
            'published': datetime(2021, 3, 15),
            'doi': None,
        })

        papers = data_ingestion.get_stored_papers(self.db_path)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['authors'], ['Author One'])
        self.assertEqual(papers[0]['categories'], ['cs.AI'])
        self.assertEqual(papers[0]['summary'], 'Stored abstract')

    @patch('arxiv_research_companion.data_ingestion.fetch_arxiv_papers')
    def test_ingest_new_papers(self, mock_fetch):
        # Mock fetch_arxiv_papers to return some papers
        mock_fetch.return_value = [
            {
                'arxiv_id': '2103.12345',
                'title': 'Test Paper 1',
                'authors': ['Author One'],
                'summary': 'Summary 1',
                'categories': ['cs.AI'],
                'published': datetime(2021, 3, 15),
                'doi': None
            },
            {
                'arxiv_id': '2103.67890',
                'title': 'Test Paper 2',
                'authors': ['Author Two'],
                'summary': 'Summary 2',
                'categories': ['cs.LG'],
                'published': datetime(2021, 4, 15),
                'doi': None
            }
        ]

        # Call ingest_new_papers
        count = data_ingestion.ingest_new_papers(['cs.AI', 'cs.LG'], self.db_path, max_results_per_request=10)

        # Should have stored 2 new papers
        self.assertEqual(count, 2)

        # Check that the papers are in the database
        self.assertTrue(data_ingestion._paper_exists(self.conn, '2103.12345'))
        self.assertTrue(data_ingestion._paper_exists(self.conn, '2103.67890'))

        # Now call again with the same mock data (which will return the same papers)
        # This time, since the papers already exist, we should store 0 new papers.
        count = data_ingestion.ingest_new_papers(['cs.AI', 'cs.LG'], self.db_path, max_results_per_request=10)
        self.assertEqual(count, 0)

        # Test with one new and one existing paper
        mock_fetch.return_value = [
            {
                'arxiv_id': '2103.12345',  # Already exists
                'title': 'Test Paper 1',
                'authors': ['Author One'],
                'summary': 'Summary 1',
                'categories': ['cs.AI'],
                'published': datetime(2021, 3, 15),
                'doi': None
            },
            {
                'arxiv_id': '2103.99999',  # New paper
                'title': 'Test Paper 3',
                'authors': ['Author Three'],
                'summary': 'Summary 3',
                'categories': ['cs.AI'],
                'published': datetime(2021, 5, 15),
                'doi': None
            }
        ]

        count = data_ingestion.ingest_new_papers(['cs.AI', 'cs.LG'], self.db_path, max_results_per_request=10)
        self.assertEqual(count, 1)  # Only the new paper should be stored
        self.assertTrue(data_ingestion._paper_exists(self.conn, '2103.99999'))


if __name__ == '__main__':
    unittest.main()