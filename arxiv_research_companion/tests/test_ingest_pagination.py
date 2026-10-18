import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import requests

from arxiv_research_companion import data_ingestion as di
from arxiv_research_companion import ingest_cli

BASE = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def corpus(n):
    return [
        {"arxiv_id": f"2610.{i:05d}", "title": f"T{i}", "authors": ["A"], "summary": f"S{i}",
         "categories": ["cs.AI"], "published": BASE + timedelta(minutes=i), "doi": None}
        for i in range(n)
    ]


class FakeArxiv:
    """Serves `papers` honouring start_date / start / max_results / sort_order."""

    def __init__(self, papers):
        self.papers = papers
        self.calls = []
        self.fail = False

    def __call__(self, categories, start_date=None, max_results=100, *, start=0,
                 sort_order="descending", raise_on_error=False):
        self.calls.append({"start": start, "max": max_results, "order": sort_order, "since": start_date})
        if self.fail:
            if raise_on_error:
                raise di.IngestionError("down")
            return []
        rows = [p for p in self.papers if start_date is None or p["published"] >= start_date.replace(second=0, microsecond=0)]
        rows.sort(key=lambda p: p["published"], reverse=(sort_order == "descending"))
        return rows[start:start + max_results]


class TestPagination(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "p.db")

    def ids(self):
        return {p["arxiv_id"] for p in di.get_stored_papers(self.db)}

    def test_multi_page_fetch_stops_at_cap_and_is_oldest_first(self):
        fake = FakeArxiv(corpus(25))
        with patch.object(di, "fetch_arxiv_papers", fake):
            n = di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=10, max_total=15)
        self.assertEqual(n, 20)  # whole pages are stored; cap reached after 2 pages
        self.assertEqual([c["start"] for c in fake.calls], [0, 10])
        self.assertTrue(all(c["order"] == "ascending" for c in fake.calls))
        self.assertEqual(self.ids(), {f"2610.{i:05d}" for i in range(20)})

    def test_capped_runs_leave_no_gap(self):
        fake = FakeArxiv(corpus(25))
        with patch.object(di, "fetch_arxiv_papers", fake):
            runs = [di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=10,
                                         max_total=10, lookback_minutes=10) for _ in range(5)]
        self.assertEqual(self.ids(), {f"2610.{i:05d}" for i in range(25)})
        self.assertEqual(sum(runs), 25)

    def test_duplicates_in_lookback_window_do_not_consume_cap_or_stall(self):
        # 40 papers one minute apart; a 30-minute lookback re-scans ~30 stored ones each run
        fake = FakeArxiv(corpus(40))
        with patch.object(di, "fetch_arxiv_papers", fake):
            total = 0
            for _ in range(10):
                got = di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=10,
                                           max_total=5, lookback_minutes=30)
                total += got
                if len(self.ids()) == 40:
                    break
        self.assertEqual(len(self.ids()), 40)

    def test_late_exposed_paper_is_caught_by_lookback(self):
        papers = corpus(10)
        late = {"arxiv_id": "2610.09999", "title": "late", "authors": [], "summary": "s",
                "categories": ["cs.AI"], "published": BASE + timedelta(minutes=5), "doi": None}
        fake = FakeArxiv(papers)
        with patch.object(di, "fetch_arxiv_papers", fake):
            di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=100)
            fake.papers = papers + [late]  # appears after the run, but dated before the cursor
            n_no_lookback = di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=100, lookback_minutes=1)
            self.assertEqual(n_no_lookback, 0)
            n = di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=100, lookback_minutes=60)
        self.assertEqual(n, 1)
        self.assertIn("2610.09999", self.ids())

    def test_strict_raises_and_non_strict_returns_zero(self):
        fake = FakeArxiv(corpus(3))
        fake.fail = True
        with patch.object(di, "fetch_arxiv_papers", fake):
            self.assertEqual(di.ingest_new_papers(["cs.AI"], self.db), 0)
            with self.assertRaises(di.IngestionError):
                di.ingest_new_papers(["cs.AI"], self.db, strict=True)

    def test_max_fetch_bounds_runaway(self):
        fake = FakeArxiv(corpus(100))
        with patch.object(di, "fetch_arxiv_papers", fake):
            di.ingest_new_papers(["cs.AI"], self.db, max_results_per_request=10, max_total=1000, max_fetch=30)
        self.assertEqual(len(fake.calls), 3)


class TestRetryAndParsing(unittest.TestCase):
    def test_get_with_retry_retries_429_then_succeeds(self):
        bad = requests.HTTPError(response=type("R", (), {"status_code": 429})())
        good = type("G", (), {"text": "<feed/>", "raise_for_status": lambda self: None})()
        seq = [bad, good]

        def fake_get(url, params=None, timeout=None):
            item = seq.pop(0)
            if isinstance(item, Exception):
                resp = type("R", (), {"raise_for_status": lambda self, e=item: (_ for _ in ()).throw(e)})()
                return resp
            return item

        with patch.object(di.requests, "get", fake_get), patch.object(di, "_respect_rate_limit"), \
                patch.object(di.time, "sleep") as sleep:
            out = di._get_with_retry({"q": 1})
        self.assertIs(out, good)
        sleep.assert_called_once()

    def test_get_with_retry_does_not_retry_400(self):
        bad = requests.HTTPError(response=type("R", (), {"status_code": 400})())
        calls = []

        def fake_get(url, params=None, timeout=None):
            calls.append(1)
            return type("R", (), {"raise_for_status": lambda self: (_ for _ in ()).throw(bad)})()

        with patch.object(di.requests, "get", fake_get), patch.object(di, "_respect_rate_limit"), \
                patch.object(di.time, "sleep"):
            with self.assertRaises(requests.HTTPError):
                di._get_with_retry({})
        self.assertEqual(len(calls), 1)

    def test_total_results_parser(self):
        xml = '<feed xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults xmlns:opensearch="x">1234</opensearch:totalResults></feed>'
        self.assertEqual(di._parse_total_results(xml), 1234)
        self.assertIsNone(di._parse_total_results("<feed/>"))


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "p.db")

    def test_success_prints_json_summary(self):
        fake = FakeArxiv(corpus(5))
        with patch.object(di, "fetch_arxiv_papers", fake), patch.object(ingest_cli, "ingest_new_papers", di.ingest_new_papers):
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = ingest_cli.main(["--db", self.db, "--max-total", "100"])
        self.assertEqual(code, 0)
        summary = json.loads(buf.getvalue().strip().splitlines()[-1])
        self.assertEqual((summary["before"], summary["new"], summary["after"]), (0, 5, 5))

    def test_api_failure_is_a_failing_exit_code(self):
        fake = FakeArxiv(corpus(5))
        fake.fail = True
        with patch.object(di, "fetch_arxiv_papers", fake):
            self.assertEqual(ingest_cli.main(["--db", self.db]), 1)


if __name__ == "__main__":
    unittest.main()
