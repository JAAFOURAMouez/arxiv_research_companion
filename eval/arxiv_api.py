"""Minimal arXiv API client: fetch/resolve papers by ID (batched, rate-limited, cached)."""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests

API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


def parse_feed(xml_text: str) -> Dict[str, dict]:
    """id -> entry for real entries; arXiv's error pseudo-entries (id contains /api/errors) are skipped."""
    root = ET.fromstring(xml_text)
    found: Dict[str, dict] = {}
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", default="", namespaces=NS) or "").strip()
        if "/api/errors" in raw_id:
            continue
        aid = re.sub(r"v\d+$", "", re.sub(r"^https?://arxiv\.org/abs/", "", raw_id))
        found[aid] = {
            "arxiv_id": aid,
            "title": " ".join((e.findtext("a:title", default="", namespaces=NS) or "").split()),
            "summary": " ".join((e.findtext("a:summary", default="", namespaces=NS) or "").split()),
            "published": (e.findtext("a:published", default="", namespaces=NS) or "").strip(),
            "categories": [c.get("term") for c in e.findall("a:category", NS) if c.get("term")],
        }
    return found


def fetch_by_ids(ids: Iterable[str], *, batch: int = 40, pause: float = 3.0, max_attempts: int = 4,
                 sleep=time.sleep) -> Dict[str, Optional[dict]]:
    """Return {id: entry or None}.  None = arXiv answered but has no such paper.
    Raises RuntimeError if arXiv could not be reached (never mistaken for 'not found')."""
    ids = list(dict.fromkeys(ids))
    out: Dict[str, Optional[dict]] = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        params = {"id_list": ",".join(chunk), "max_results": len(chunk)}
        for attempt in range(1, max_attempts + 1):
            sleep(pause)  # arXiv asks for <= 1 request / 3 s
            try:
                r = requests.get(API, params=params, timeout=60)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}")
                r.raise_for_status()
                found = parse_feed(r.text)
                break
            except (requests.RequestException, ET.ParseError) as exc:
                if attempt == max_attempts:
                    raise RuntimeError(f"arXiv API unreachable for {chunk[:3]}...: {exc}") from exc
                sleep(pause * 2 ** attempt)
        for i in chunk:
            out[i] = found.get(i)
    return out


def resolve_ids(ids: Iterable[str], cache_path: Optional[Path] = None, **kw) -> Dict[str, str]:
    """{id: 'resolved' | 'not_found' | 'unknown'}.  'unknown' = API failure, not evidence either way."""
    ids = list(dict.fromkeys(ids))
    cache: Dict[str, str] = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text())
    todo = [i for i in ids if cache.get(i) not in ("resolved", "not_found")]
    if todo:
        try:
            for i, entry in fetch_by_ids(todo, **kw).items():
                cache[i] = "resolved" if entry else "not_found"
        except RuntimeError:
            for i in todo:
                cache.setdefault(i, "unknown")
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(cache, indent=1))
    return {i: cache.get(i, "unknown") for i in ids}
