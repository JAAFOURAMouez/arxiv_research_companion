"""
Cross-check the embedding pipeline on BEIR SciFact against published numbers.

    python -m eval.scifact_crosscheck --download            # or --data-dir path/to/scifact
    python -m eval.scifact_crosscheck --data-dir ... --max-queries 50     # quick slice (NOT comparable)

Default = the FULL SciFact test split (all test queries, all 5,183 documents).  A query slice
keeps the full corpus but is flagged "not comparable to published numbers" because the
published figures are over all test queries.

Rows: dense[onnx] and dense[st] (the cross-check: same model, deployed vs reference runtime),
then bm25 / hybrid / hybrid_rerank built on the ONNX embeddings (secondary evidence only: SciFact
is claim verification, not question answering over recent arXiv abstracts).
Document text = title + " " + text, as in BEIR/MTEB.  Published values come from
eval/data/published_reference.json, which you fill in (left null on purpose).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import requests

from eval import metrics as M
from eval.common import CACHE_DIR, RESULTS_DIR, cached_embeddings, is_smoke, make_embedder, read_json, run_meta, write_json
from eval.pipeline import build_retrievers, latency_summary, make_reranker, run_queries

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
KS = (1, 3, 5, 10, 100)


def download(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(BEIR_URL, timeout=120)
    r.raise_for_status()
    zipfile.ZipFile(io.BytesIO(r.content)).extractall(dest)
    return find_dir(dest)


def find_dir(root: Path) -> Path:
    for p in [root, *root.rglob("*")]:
        if (p / "corpus.jsonl").exists():
            return p
    raise FileNotFoundError(f"no corpus.jsonl under {root}")


def load_scifact(d: Path):
    corpus = [json.loads(l) for l in open(d / "corpus.jsonl", encoding="utf-8") if l.strip()]
    queries = {j["_id"]: j["text"] for j in (json.loads(l) for l in open(d / "queries.jsonl", encoding="utf-8") if l.strip())}
    qrels: dict = {}
    with open(d / "qrels" / "test.tsv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], set()).add(row["corpus-id"])
    papers = [{"arxiv_id": str(c["_id"]), "title": "",
               "summary": ((c.get("title") or "") + " " + (c.get("text") or "")).strip()} for c in corpus]
    qs = [{"qid": q, "question": queries[q]} for q in sorted(qrels) if q in queries]
    return papers, qs, {q: qrels[q] for q in sorted(qrels) if q in queries}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--embedders", default="onnx,st", type=lambda s: s.split(","))
    ap.add_argument("--configs", default="bm25,hybrid,hybrid_rerank", type=lambda s: s.split(","))
    ap.add_argument("--reranker", choices=["ce", "fake"], default="ce")
    ap.add_argument("--published", default="eval/data/published_reference.json")
    ap.add_argument("--published-key", default="all-MiniLM-L6-v2")
    ap.add_argument("--engine", choices=["ranx", "reference"], default="ranx")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS_DIR / "scifact.json"))
    args = ap.parse_args(argv)

    if args.data_dir:
        d = find_dir(Path(args.data_dir))
    elif args.download:
        d = download(CACHE_DIR / "beir" / "scifact")
    else:
        print("give --data-dir or --download", file=sys.stderr)
        return 2
    papers, queries, qrels = load_scifact(d)
    full = not args.max_queries
    if args.max_queries:
        queries = queries[: args.max_queries]
        qrels = {q["qid"]: qrels[q["qid"]] for q in queries}
    qids = [q["qid"] for q in queries]
    texts = [p["summary"] for p in papers]
    print(f"SciFact: {len(papers)} docs, {len(queries)} queries (full split: {full})")

    rows, per_query_cols, embs = {}, {}, {}
    reranker = make_reranker(args.reranker) if "hybrid_rerank" in args.configs else None

    def run(name, retriever):
        rankings, lat = run_queries(retriever, queries, 100)
        ev = M.evaluate_run(rankings, qrels, engine=args.engine, ks=KS)
        pq = ev["per_query"]
        ci = M.bootstrap_ci(M.column(pq, "ndcg@10", qids), n=args.bootstrap, seed=args.seed)
        rows[name] = {"means": ev["means"], "ndcg@10_ci95": [ci[1], ci[2]], "latency": latency_summary(lat),
                      "ranx_vs_reference_max_abs_diff": ev["ranx_vs_reference_max_abs_diff"]}
        per_query_cols[name] = M.column(pq, "ndcg@10", qids)
        print(f"  {name:<14} ndcg@10={ev['means']['ndcg@10']:.4f}  recall@100={ev['means']['recall@100']:.4f}", flush=True)

    for kind in args.embedders:
        emb = make_embedder(kind)
        vecs = cached_embeddings(emb, texts, f"scifact_{kind}")
        embs[kind] = (emb, vecs)
        r = build_retrievers(papers, vecs, emb, ["dense"], doc_text_mode="abstract")
        run(f"dense[{kind}]", r["dense"])
    emb, vecs = embs[args.embedders[0]]
    others = build_retrievers(papers, vecs, emb, args.configs, reranker=reranker, doc_text_mode="abstract")
    for name in args.configs:
        run(f"{name}[{args.embedders[0]}]", others[name])

    cross = {}
    if "dense[onnx]" in rows and "dense[st]" in rows:
        cross["onnx_minus_st_ndcg@10"] = M.paired_bootstrap_diff(
            per_query_cols["dense[onnx]"], per_query_cols["dense[st]"], n=args.bootstrap, seed=args.seed)
    pub = read_json(args.published) if Path(args.published).exists() else {}
    ref = ((pub.get("scifact") or {}).get(args.published_key) or {})
    tol = pub.get("tolerance_ndcg10", 0.02)
    checks = {}
    for k in args.embedders:
        mine = rows[f"dense[{k}]"]["means"]["ndcg@10"]
        if ref.get("ndcg@10") is None:
            checks[k] = {"status": "NOT CHECKED (no published value in published_reference.json)", "mine": mine}
        else:
            diff = mine - ref["ndcg@10"]
            status = "within tolerance" if abs(diff) <= tol and full else (
                "OUT OF TOLERANCE" if full else "not comparable (query slice)")
            checks[k] = {"status": status, "mine": mine, "published": ref["ndcg@10"], "diff": diff,
                         "tolerance": tol, "source_url": ref.get("source_url")}
    write_json(args.out, {"meta": run_meta(full_test_split=full, n_docs=len(papers), n_queries=len(queries),
                                           smoke=is_smoke(emb, reranker, args.engine), beir_url=BEIR_URL),
                          "rows": rows, "onnx_vs_st": cross, "published_check": checks})
    print(json.dumps(checks, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
