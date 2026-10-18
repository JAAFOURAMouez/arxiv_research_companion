"""
Run the real generation path (rag_engine.generate_answer) over the question sets.

    GROQ_API_KEY=... python -m eval.generate_answers --questions eval/data/questions.csv \
        --ooc eval/data/ooc_questions.csv --retrieval-mode dense --out results/answers.jsonl

Uses the app's own defaults (k=3, temperature 0.7, 800 tokens, citation enforcement on, one
repair round) unless overridden, and records everything needed by the checkers: raw and final
answer, retrieved IDs in order, top-1 dense cosine, citation-enforcement info, errors.
Resumable (--resume): already answered qids are skipped, so a Groq rate limit costs nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from arxiv_research_companion.config import load_config
from arxiv_research_companion.rag_engine import generate_answer
from arxiv_research_companion.retrieval import doc_text
from eval.common import (RESULTS_DIR, cached_embeddings, is_smoke, load_questions, load_snapshot,
                         make_embedder, read_json, read_jsonl, run_meta, write_json)
from eval.pipeline import build_retrievers, make_reranker


def main(argv=None) -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="eval/data/questions.csv")
    ap.add_argument("--ooc", default=None)
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--corpus-mode", choices=["prod", "augmented"], default=None,
                    help="default: augmented if the snapshot has labeled_extra papers, else prod")
    ap.add_argument("--retrieval-mode", default=cfg["retrieval"]["mode"],
                    choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    ap.add_argument("--embedder", choices=["onnx", "st", "hash"], default="onnx")
    ap.add_argument("--reranker", choices=["ce", "fake"], default="ce")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--no-enforce", action="store_true", help="ablation: disable citation enforcement")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between questions")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default=str(RESULTS_DIR / "answers.jsonl"))
    args = ap.parse_args(argv)

    snapshot = load_snapshot(args.snapshot)
    mode = args.corpus_mode or ("augmented" if any(p["origin"] != "prod" for p in snapshot) else "prod")
    papers = [p for p in snapshot if mode == "augmented" or p["origin"] == "prod"]
    embedder = make_embedder(args.embedder)
    vecs = cached_embeddings(embedder, [doc_text(p) for p in snapshot], f"{args.embedder}_abstract")
    keep = [i for i, p in enumerate(snapshot) if mode == "augmented" or p["origin"] == "prod"]
    reranker = make_reranker(args.reranker) if args.retrieval_mode == "hybrid_rerank" else None
    retriever = build_retrievers(papers, vecs[keep], embedder, [args.retrieval_mode], reranker=reranker)[args.retrieval_mode]
    dense_for_score = build_retrievers(papers, vecs[keep], embedder, ["dense"])["dense"]
    store = {"model": embedder, "collection": None, "retriever": retriever,
             "papers_by_id": {p["arxiv_id"]: p for p in papers}}

    config = {**cfg, "retrieval_k": args.k,
              "llm": {**cfg["llm"], "temperature": args.temperature, "max_new_tokens": args.max_tokens},
              "citations": {**cfg["citations"], "enforce": not args.no_enforce}}
    if not config["llm"]["groq_api_key"] and not config["llm"]["api_token"]:
        print("ERROR: set GROQ_API_KEY (or HF_API_TOKEN)", file=sys.stderr)
        return 1

    items = [{**q, "kind": "in_scope"} for q in load_questions(args.questions)]
    if args.ooc:
        items += [{**q, "kind": "ooc"} for q in load_questions(args.ooc)]
    out_path = Path(args.out)
    done = {r["qid"] for r in read_jsonl(out_path)} if args.resume and out_path.exists() else set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode_w = "a" if args.resume and out_path.exists() else "w"
    with open(out_path, mode_w, encoding="utf-8") as fh:
        for n, item in enumerate(items, 1):
            if item["qid"] in done:
                continue
            res = generate_answer(item["question"], store, config)
            retrieved = [s["id"] for s in res["sources"]]
            rel = set(item.get("relevant_arxiv_ids", []))
            rec = {"qid": item["qid"], "kind": item["kind"], "question": item["question"],
                   "answer": res["answer"], "raw_answer": res.get("raw_answer"),
                   "refused": bool(res.get("refused")), "degraded": bool(res.get("degraded")),
                   "llm_error": res.get("llm_error"), "retrieved_ids": retrieved,
                   "relevant_ids": sorted(rel), "relevant_in_topk": bool(rel & set(retrieved)) if rel else None,
                   "top1_dense_cosine": dense_for_score.dense_top1_score(item["question"]),
                   "citations": res.get("citations"), "audited": item.get("audited")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{n}/{len(items)}] {item['qid']} refused={rec['refused']} degraded={rec['degraded']}", flush=True)
            time.sleep(args.sleep)
    write_json(Path(args.out).with_suffix(".meta.json"),
               run_meta(retrieval_mode=args.retrieval_mode, corpus_mode=mode, k=args.k,
                        temperature=args.temperature, max_tokens=args.max_tokens,
                        enforce=not args.no_enforce, llm_model=config["llm"]["model_name"],
                        llm_provider=config["llm"]["provider"], embedder=args.embedder,
                        smoke=is_smoke(embedder, reranker)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
