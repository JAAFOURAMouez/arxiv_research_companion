"""I/O helpers, provenance metadata and embedder factory shared by the eval scripts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"

_NEW_ID = re.compile(r"^\d{4}\.\d{4,5}$")
_OLD_ID = re.compile(r"^[a-z\-]+(?:\.[A-Z]{2})?/\d{7}$")


def normalize_arxiv_id(raw: str) -> str:
    """'arXiv:2610.02207v2', 'https://arxiv.org/abs/2610.02207v2', ' 2610.02207 ' -> '2610.02207'."""
    s = (raw or "").strip()
    s = re.sub(r"^(?:https?://)?(?:export\.)?arxiv\.org/(?:abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:\s*", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    s = re.sub(r"v\d+$", "", s)
    return s.strip()


def is_valid_arxiv_id(s: str) -> bool:
    return bool(_NEW_ID.match(s) or _OLD_ID.match(s))


# --------------------------------------------------------------------------- #
# files                                                                       #
# --------------------------------------------------------------------------- #
def read_jsonl(path) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows: Iterable[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_questions(path) -> List[dict]:
    """CSV (qid,question,relevant_arxiv_ids) or JSONL -> list of dicts with a normalised ID list.

    In the CSV, IDs are separated by ';', ',' or whitespace.  Rows with an empty question are
    skipped (the template ships with 40 empty rows).
    """
    path = str(path)
    rows: List[dict]
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    else:
        rows = read_jsonl(path)
    out = []
    for r in rows:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        raw = r.get("relevant_arxiv_ids") or r.get("relevant") or []
        if isinstance(raw, str):
            raw = [x for x in re.split(r"[;,\s]+", raw) if x]
        ids: List[str] = []
        for x in raw:
            nid = normalize_arxiv_id(x)
            if nid and nid not in ids:
                ids.append(nid)
        out.append({"qid": (r.get("qid") or f"q{len(out) + 1:02d}").strip(), "question": q,
                    "relevant_arxiv_ids": ids, **{k: v for k, v in r.items()
                                                   if k in ("notes", "audited", "kind")}})
    return out


# --------------------------------------------------------------------------- #
# provenance                                                                  #
# --------------------------------------------------------------------------- #
_PKGS = ("numpy", "ranx", "sentence-transformers", "transformers", "torch", "chromadb",
         "onnxruntime", "ragas", "scipy", "requests", "streamlit")


def package_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for name in _PKGS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def run_meta(**extra) -> dict:
    import sqlite3
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "sqlite": sqlite3.sqlite_version, "git_sha": git_sha(),
        "packages": package_versions(), **extra,
    }


def rss_mb() -> float:
    """Peak resident set size of this process in MB (approximate; monotonic)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


# --------------------------------------------------------------------------- #
# embedders / cache                                                           #
# --------------------------------------------------------------------------- #
def make_embedder(kind: str):
    from arxiv_research_companion.embedders import (HashEmbedder, OnnxEmbedder,
                                                    SentenceTransformerEmbedder)
    if kind == "onnx":
        return OnnxEmbedder()
    if kind == "st":
        return SentenceTransformerEmbedder()
    if kind == "hash":
        return HashEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")


def is_smoke(embedder=None, reranker=None, engine: str = "ranx") -> bool:
    """Runs using stand-ins (hash embedder / fake reranker / no ranx) are not real results."""
    return bool(getattr(embedder, "is_smoke", False) or getattr(reranker, "is_smoke", False)
                or engine != "ranx")


def cached_embeddings(embedder, texts: List[str], cache_key: str) -> np.ndarray:
    """Embed once per (embedder, corpus hash).  Cache lives in results/cache (git-ignored)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256("\x1f".join(texts).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"emb_{cache_key}_{digest}.npy"
    if path.exists():
        return np.load(path)
    t0 = time.time()
    vecs = np.asarray(embedder.encode(texts), dtype=np.float32)
    np.save(path, vecs)
    print(f"  embedded {len(texts)} texts with {cache_key} in {time.time() - t0:.1f}s", flush=True)
    return vecs


def load_snapshot(path) -> List[dict]:
    rows = read_jsonl(path)
    for r in rows:
        r.setdefault("origin", "prod")
        r.setdefault("title", "")
        r.setdefault("summary", "")
    return rows
