"""
RAGAS faithfulness on the generated answers (in-scope, answered, non-refused only).

    GROQ_API_KEY=... python -m eval.run_ragas --answers results/answers.jsonl \
        --snapshot eval/data/corpus_snapshot.jsonl

The judge is an LLM of a different family from the generator (default Llama on Groq, JUDGE_MODEL
to change).  `[n]` markers are stripped from the answer first.  Contexts = the retrieved abstracts
exactly as shown to the generator.

Written against the ragas >= 0.2 API (SingleTurnSample / EvaluationDataset / Faithfulness /
LangchainLLMWrapper).  ragas changes quickly; this file could not be executed by its author, so
if an import fails adjust the three lines marked `# ragas-api`.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from arxiv_research_companion.citations import strip_citation_markers
from eval import metrics as M
from eval.common import RESULTS_DIR, load_snapshot, read_jsonl, run_meta, write_json
from eval.judges import DEFAULT_JUDGE_MODEL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", default=str(RESULTS_DIR / "answers.jsonl"))
    ap.add_argument("--snapshot", default="eval/data/corpus_snapshot.jsonl")
    ap.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max-workers", type=int, default=1, help="keep low: Groq free-tier rate limits")
    ap.add_argument("--out", default=str(RESULTS_DIR / "ragas.json"))
    args = ap.parse_args(argv)

    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("ERROR: GROQ_API_KEY not set", file=sys.stderr)
        return 1
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate  # ragas-api
    from ragas.llms import LangchainLLMWrapper  # ragas-api
    from ragas.metrics import Faithfulness  # ragas-api
    from ragas.run_config import RunConfig

    snap = {p["arxiv_id"]: p for p in load_snapshot(args.snapshot)}
    recs = [r for r in read_jsonl(args.answers)
            if r["kind"] == "in_scope" and not r["refused"] and not r["degraded"]]
    samples = []
    for r in recs:
        ctx = [f"{snap[i]['title']}. {snap[i]['summary']}" for i in r["retrieved_ids"]]
        samples.append(SingleTurnSample(user_input=r["question"], response=strip_citation_markers(r["answer"]),
                                        retrieved_contexts=ctx))
    llm = LangchainLLMWrapper(ChatOpenAI(model=args.judge_model, api_key=key, temperature=0,
                                         base_url="https://api.groq.com/openai/v1"))
    res = evaluate(EvaluationDataset(samples=samples), metrics=[Faithfulness()], llm=llm,
                   run_config=RunConfig(max_workers=args.max_workers, timeout=120, max_retries=6))
    df = res.to_pandas()
    col = "faithfulness"
    scores = [float(x) if x == x else None for x in df[col]]
    valid = [s for s in scores if s is not None]
    mean, lo, hi = M.bootstrap_ci(valid, n=10000)
    out = {"meta": run_meta(judge_model=args.judge_model, n_answers=len(recs)),
           "faithfulness_mean": mean, "ci95": [lo, hi], "n_scored": len(valid), "n_nan": len(scores) - len(valid),
           "per_question": {r["qid"]: s for r, s in zip(recs, scores)}}
    write_json(args.out, out)
    print({k: v for k, v in out.items() if k != "per_question"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
