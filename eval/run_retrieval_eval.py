"""
Retrieval evaluation: dense ONNX MiniLM, SQLite-FTS5 BM25, hybrid RRF, hybrid + cross-encoder.

    python -m eval.run_retrieval_eval --snapshot eval/data/corpus_snapshot.jsonl \
        --questions eval/data/questions.csv --out results/retrieval.json

Reports recall@1/3/5/10, hit@k, MRR@10, nDCG@10 (ranx, cross-checked against a reference
implementation), bootstrap 95% CIs, latency, a random baseline, and the configuration chosen by
the pre-declared rule in eval/metrics.py::select_config.

Corpus modes
  prod       only the production papers; labelled IDs outside it are dropped from the qrels and
             questions left without any relevant paper are excluded (counts are reported)
  augmented  production papers + labelled papers fetched by build_corpus --fetch-missing
"""
from __future__ import annotations

import argparse
import sys

from arxiv_research_companion.retrieval import doc_text
from eval import metrics as M
from eval.common import (RESULTS_DIR, cached_embeddings, is_smoke, load_questions, load_snapshot,
                         make_embedder, read_json, rss_mb, run_meta, write_json)
from eval.pipeline import CONFIGS, build_retrievers, latency_summary, make_reranker, run_queries
from pathlib import Path


def evaluate_corpus(papers, vecs, questions, embedder, reranker, args, label):
    doc_ids = {p["arxiv_id"] for p in papers}
    qrels, kept, dropped = {}, [], []
    n_lab = n_lab_in = 0
    for q in questions:
        rel = [d for d in q["relevant_arxiv_ids"] if d in doc_ids]
        n_lab += len(q["relevant_arxiv_ids"])
        n_lab_in += len(rel)
        if rel:
            qrels[q["qid"]] = set(rel)
            kept.append(q)
        else:
            dropped.append(q["qid"])
    out = {"n_docs": len(papers), "n_queries": len(kept), "dropped_queries_no_relevant_in_corpus": dropped,
           "labeled_ids": n_lab, "labeled_ids_in_corpus": n_lab_in, "configs": {}}
    if not kept:
        out["error"] = "no question has a relevant paper in this corpus"
        return out
    qids = [q["qid"] for q in kept]
    retrievers = build_retrievers(
        papers, vecs, embedder, args.configs, reranker=reranker, doc_text_mode=args.doc_text,
        dense_backend=args.dense_backend, candidate_depth=args.candidate_depth,
        rerank_depth=args.rerank_depth, rrf_k=args.rrf_k)
    ndcg_cols = {}
    for name in args.configs:
        print(f"[{label}] {name} ...", flush=True)
        rankings, lat = run_queries(retrievers[name], kept, args.depth)
        ev = M.evaluate_run(rankings, qrels, engine=args.engine)
        pq = ev["per_query"]
        cis = {m: M.bootstrap_ci(M.column(pq, m, qids), n=args.bootstrap, seed=args.seed)
               for m in M.REPORT_METRICS}
        ndcg_cols[name] = M.column(pq, "ndcg@10", qids)
        out["configs"][name] = {
            "means": ev["means"], "ci95": {m: [lo, hi] for m, (_, lo, hi) in cis.items()},
            "latency": latency_summary(lat), "peak_rss_mb_after": rss_mb(),
            "ranx_vs_reference_max_abs_diff": ev["ranx_vs_reference_max_abs_diff"],
            "per_query": pq, "top20": {q: r[:20] for q, r in rankings.items()},
            "recall@10_zero_queries": [q for q in qids if pq[q]["recall@10"] == 0.0],
        }
    rb = M.random_baseline(sorted(doc_ids), qrels, seed=args.seed)
    out["random_baseline"] = {"means": {m: float(sum(rb[q][m] for q in qids) / len(qids)) for m in M.REPORT_METRICS},
                              "note": "expected value of a uniformly random ranking (200 shuffles)"}
    out["selection"] = M.select_config(ndcg_cols, n_boot=args.bootstrap, seed=args.seed)
    out["queries"] = [{"qid": q["qid"], "n_relevant_in_corpus": len(qrels[q["qid"]]),
                       "n_relevant_labeled": len(q["relevant_arxiv_ids"])} for q in kept]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--questions", default="eval/data/questions.csv")
    ap.add_argument("--out", default=str(RESULTS_DIR / "retrieval.json"))
    ap.add_argument("--corpus-mode", choices=["prod", "augmented", "both"], default="both")
    ap.add_argument("--configs", default=",".join(CONFIGS), type=lambda s: s.split(","))
    ap.add_argument("--embedder", choices=["onnx", "st", "hash"], default="onnx")
    ap.add_argument("--dense-backend", choices=["numpy", "chroma"], default="numpy")
    ap.add_argument("--reranker", choices=["ce", "fake"], default="ce")
    ap.add_argument("--cross-encoder", default=None)
    ap.add_argument("--doc-text", choices=["abstract", "title_abstract"], default="abstract")
    ap.add_argument("--depth", type=int, default=100)
    ap.add_argument("--candidate-depth", type=int, default=50)
    ap.add_argument("--rerank-depth", type=int, default=20)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--engine", choices=["ranx", "reference"], default="ranx")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    questions = load_questions(args.questions)
    if not questions:
        print("ERROR: no questions (is the CSV still the empty template?)", file=sys.stderr)
        return 1
    snapshot = load_snapshot(args.snapshot)
    meta_path = Path(args.snapshot).with_suffix(".meta.json")
    snap_meta = read_json(meta_path) if meta_path.exists() else {}

    embedder = make_embedder(args.embedder)
    vecs_all = cached_embeddings(embedder, [doc_text(p, args.doc_text) for p in snapshot],
                                 f"{args.embedder}_{args.doc_text}")
    reranker = make_reranker(args.reranker, args.cross_encoder) if "hybrid_rerank" in args.configs else None
    rss_after_models = rss_mb()

    has_extra = any(p["origin"] != "prod" for p in snapshot)
    modes = ["prod", "augmented"] if args.corpus_mode == "both" else [args.corpus_mode]
    if "augmented" in modes and not has_extra:
        print("NOTE: snapshot has no labeled_extra papers; 'augmented' == 'prod', skipping it.")
        modes = [m for m in modes if m != "augmented"]

    result = {"meta": run_meta(snapshot_sha256=snap_meta.get("snapshot_sha256"), embedder=args.embedder,
                               reranker=args.reranker, engine=args.engine, doc_text=args.doc_text,
                               dense_backend=args.dense_backend, depth=args.depth,
                               smoke=is_smoke(embedder, reranker, args.engine),
                               rss_mb_after_model_load=rss_after_models,
                               cross_encoder=getattr(reranker, "model_name", None)),
              "primary_mode": "augmented" if has_extra else "prod", "modes": {}}
    for mode in modes:
        papers = [p for p in snapshot if mode == "augmented" or p["origin"] == "prod"]
        idx = [i for i, p in enumerate(snapshot) if mode == "augmented" or p["origin"] == "prod"]
        result["modes"][mode] = evaluate_corpus(papers, vecs_all[idx], questions, embedder, reranker, args, mode)
    write_json(args.out, result)

    for mode, r in result["modes"].items():
        print(f"\n== corpus={mode}  docs={r['n_docs']}  queries={r['n_queries']} "
              f"(dropped {len(r['dropped_queries_no_relevant_in_corpus'])}) ==")
        if "error" in r:
            print(r["error"])
            continue
        cols = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
        print("config".ljust(15) + "".join(c.rjust(11) for c in cols) + "   p50_ms")
        for n, c in {**r["configs"], "random": {"means": r["random_baseline"]["means"], "latency": {"p50_ms": 0}}}.items():
            print(n.ljust(15) + "".join(f"{c['means'][m]:11.3f}" for m in cols) + f"{c['latency']['p50_ms']:9.1f}")
        s = r["selection"]
        print(f"selection: best_by_mean={s['best_by_mean']}  chosen={s['chosen']}  ({s['rule']})")
    if result["meta"]["smoke"]:
        print("\n!! SMOKE RUN (stand-in embedder/reranker/engine): numbers above are NOT results !!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
