"""
Command-line ingestion used by the nightly GitHub Action.

    python -m arxiv_research_companion.ingest_cli --db arxiv_papers.db --max-total 1000

Fails loudly (non-zero exit) when the arXiv API cannot be reached, when the SQLite file is
corrupt, or when the paper count went *down*.  "No new papers" is a success.
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import List, Optional

from arxiv_research_companion.config import load_config
from arxiv_research_companion.data_ingestion import (
    IngestionError,
    get_last_update_timestamp,
    get_total_papers_count,
    ingest_new_papers,
)


def _integrity_ok(db_path: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=cfg["ingestion"]["db_path"])
    parser.add_argument("--categories", default=",".join(cfg["arxiv"]["categories"]))
    parser.add_argument("--page-size", type=int, default=cfg["arxiv"]["max_results_per_request"])
    parser.add_argument("--max-total", type=int, default=cfg["arxiv"]["max_total_per_run"],
                        help="max NEW papers stored per run")
    parser.add_argument("--lookback-hours", type=float, default=72.0,
                        help="re-scan this far behind the newest stored paper (catches papers "
                             "arXiv exposes late; duplicates are skipped)")
    parser.add_argument("--max-fetch", type=int, default=None, help="safety bound on papers fetched")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    before = get_total_papers_count(args.db) if os.path.exists(args.db) else 0
    run_started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        new = ingest_new_papers(
            categories, args.db,
            max_results_per_request=args.page_size,
            max_total=args.max_total,
            strict=True,
            lookback_minutes=int(args.lookback_hours * 60),
            max_fetch=args.max_fetch,
        )
    except IngestionError as exc:
        print(f"INGESTION FAILED: {exc}", file=sys.stderr)
        return 1

    after = get_total_papers_count(args.db)
    newest = get_last_update_timestamp(args.db)
    conn = sqlite3.connect(args.db)
    try:  # published-date span of what this run added: shows how late arXiv exposes papers
        span = conn.execute(
            "SELECT MIN(published), MAX(published) FROM papers WHERE updated_at >= ?", (run_started,)
        ).fetchone()
    finally:
        conn.close()
    summary = {
        "new_published_min": span[0], "new_published_max": span[1],
        "before": before, "new": new, "after": after,
        "newest_published": newest.isoformat() if newest else None,
        "categories": categories, "max_total": args.max_total, "lookback_hours": args.lookback_hours,
    }
    print(json.dumps(summary))
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("### Nightly ingestion\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")

    if not _integrity_ok(args.db):
        print("SQLite integrity_check failed", file=sys.stderr)
        return 1
    if after < before:
        print(f"Paper count decreased ({before} -> {after})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
