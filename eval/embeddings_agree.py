"""
Do the ONNX embeddings (what the app deploys) agree with sentence-transformers?

For each abstract in the corpus snapshot (and each labelled question) compute the cosine between
the ONNX and the sentence-transformers embedding of the SAME text.  Pass criterion: every cosine
>= --threshold (default 0.99).  Also reports whether the two models retrieve the same neighbours.

    python -m eval.embeddings_agree --snapshot eval/data/corpus_snapshot.jsonl \
        --questions eval/data/questions.csv

Exit code 1 if the criterion fails.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from arxiv_research_companion.embedders import OnnxEmbedder, SentenceTransformerEmbedder
from eval.common import RESULTS_DIR, load_questions, load_snapshot, run_meta, write_json


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def cosine_stats(a: np.ndarray, b: np.ndarray, thr: float) -> dict:
    cos = np.sum(_norm(a) * _norm(b), axis=1)
    return {"n": int(len(cos)), "min": float(cos.min()), "p01": float(np.quantile(cos, 0.01)),
            "p05": float(np.quantile(cos, 0.05)), "median": float(np.median(cos)),
            "mean": float(cos.mean()), "frac_ge_threshold": float((cos >= thr).mean()),
            "n_below_threshold": int((cos < thr).sum()), "all_ge_threshold": bool((cos >= thr).all())}, cos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--questions", default="eval/data/questions.csv")
    ap.add_argument("--threshold", type=float, default=0.99)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default=str(RESULTS_DIR / "embeddings_agree.json"))
    args = ap.parse_args(argv)

    docs = load_snapshot(args.snapshot)
    if args.max_docs:
        docs = docs[: args.max_docs]
    texts = [d["summary"] for d in docs]  # the app embeds the abstract only
    queries = [q["question"] for q in load_questions(args.questions)]

    onnx, st = OnnxEmbedder(), SentenceTransformerEmbedder()
    a_docs, b_docs = onnx.encode(texts), st.encode(texts)
    docs_stats, docs_cos = cosine_stats(a_docs, b_docs, args.threshold)

    tok = st.model.tokenizer
    lengths = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    worst = np.argsort(docs_cos)[:5]
    docs_stats["worst5"] = [{"arxiv_id": docs[i]["arxiv_id"], "cosine": float(docs_cos[i]),
                             "n_tokens": int(lengths[i])} for i in worst]
    docs_stats["n_over_256_tokens"] = int(sum(l > 256 for l in lengths))
    over = np.array([l > 256 for l in lengths])
    if over.any() and (~over).any():
        docs_stats["mean_cos_over_256_tokens"] = float(docs_cos[over].mean())
        docs_stats["mean_cos_le_256_tokens"] = float(docs_cos[~over].mean())

    result = {"meta": run_meta(snapshot=args.snapshot, threshold=args.threshold), "docs": docs_stats}
    if queries:
        a_q, b_q = onnx.encode(queries), st.encode(queries)
        q_stats, _ = cosine_stats(a_q, b_q, args.threshold)
        k = min(10, len(docs))
        nn_a = np.argsort(-(_norm(a_q) @ _norm(a_docs).T), axis=1)[:, :k]
        nn_b = np.argsort(-(_norm(b_q) @ _norm(b_docs).T), axis=1)[:, :k]
        q_stats["top1_same_fraction"] = float(np.mean(nn_a[:, 0] == nn_b[:, 0]))
        q_stats["top10_overlap_mean"] = float(np.mean([len(set(x) & set(y)) / k for x, y in zip(nn_a, nn_b)]))
        result["queries"] = q_stats
    result["pass"] = bool(docs_stats["all_ge_threshold"] and result.get("queries", {"all_ge_threshold": True})["all_ge_threshold"])
    write_json(args.out, result)
    print(f"docs: {docs_stats}")
    if "queries" in result:
        print(f"queries: {result['queries']}")
    print("PASS" if result["pass"] else f"FAIL (some cosine < {args.threshold})")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
