# Evaluation

Everything here is meant to be run **by you** (network + models needed). Nothing in the repo contains
a measured result: `results/RESULTS.md` is generated from the JSON your runs produce, and
`results/RESULTS_TEMPLATE.md` shows its shape with blanks.

```bash
uv lock                      # first time only: creates uv.lock (commit it) and requirements.txt
uv sync --group eval         # ranx, sentence-transformers (torch), transformers, ragas ...
export GROQ_API_KEY=...      # generation + LLM judge + RAGAS
bash eval/run_all.sh         # or run the steps below one by one
```

## 0. Inputs you provide
| file | what |
|---|---|
| `eval/data/questions.csv` | 40 rows: `qid, question, relevant_arxiv_ids` (IDs separated by `;`). Label from arxiv.org **without running the app**. The shipped file is an empty template. |
| `eval/data/ooc_questions.csv` | 14 **draft** out-of-corpus questions (6 off-topic, 6 in-domain-but-absent, 2 false-premise). I only checked by keyword that the in-domain topics have no hit in the 300 stored abstracts. `validate_labels --ooc` prints the closest titles: read them, delete any question a stored paper could answer, then set `audited=true`. Unaudited rows are ignored by `refusal_eval`. |
| `eval/data/published_reference.json` | the published SciFact nDCG@10 for all-MiniLM-L6-v2, **left null on purpose**: copy it from MTEB/BEIR with the URL. |
| `eval/data/calibration_to_label.csv` | produced by `make_calibration_set`; you fill the `label` column. |

## 1. Order of steps
1. `python -m eval.validate_labels --questions eval/data/questions.csv --db arxiv_papers.db --check-arxiv --ooc eval/data/ooc_questions.csv`
2. `python -m eval.build_corpus ... --fetch-missing` freezes the corpus (`corpus_snapshot.jsonl` + sha256). **Commit it**: the nightly job keeps changing `arxiv_papers.db`, a moving corpus makes numbers incomparable.
3. `python -m eval.scifact_crosscheck --download` (full SciFact test split).
4. `python -m eval.embeddings_agree` (ONNX vs sentence-transformers, cosine >= 0.99 on every abstract and question; exit 1 if not).
5. `python -m eval.run_retrieval_eval` -> table + selected config.
6. Set `RETRIEVAL_MODE` to the chosen mode, then `python -m eval.generate_answers --ooc ... --resume`.
7. `python -m eval.refusal_eval`.
8. `python -m eval.make_calibration_set` -> label 30 rows -> `python -m eval.calibrate_judge`.
9. `python -m eval.check_citations --judge both --resolve`, `python -m eval.run_ragas`.
10. `python -m eval.report` -> `results/RESULTS.md`; paste it back.

## 2. Design decisions you should know about
* **Evaluate the shipped code.** `eval/` imports `arxiv_research_companion.retrieval` and `rag_engine`; there is no separate "eval implementation".
* **Corpus size is the main threat to validity.** The stored corpus is 300 abstracts from 1-15 Oct 2026. Hand-labelled papers found on arxiv.org will mostly not be in it. Hence two corpus modes: `prod` (labels outside the corpus dropped, questions with none left excluded, counts shown) and `augmented` (labelled papers fetched from arXiv and added; the 300 production papers act as distractors). `augmented` is the primary mode when it exists. Both are small: read recall@10 next to the `random` row.
* **Metrics.** recall@k is `|relevant in top-k| / |relevant|` (so recall@1 cannot reach 1 for a query with 2+ relevant papers; `hit@k` is reported beside it). Headline numbers come from `ranx`; a 30-line reference implementation recomputes them and the run aborts if they differ by more than 1e-6. Per-query scores come from the reference implementation and feed bootstrap CIs (10,000 resamples, seed 0).
* **Choosing the configuration (rule fixed before any number exists).** Best mean nDCG@10; every other config is compared to it with a paired bootstrap (Bonferroni-adjusted); the **cheapest** config whose CI contains 0 wins (cost order bm25 < dense < hybrid < hybrid+rerank). With 40 queries this will often return the cheapest; that reflects statistical power, not equivalence. SciFact numbers are secondary evidence only. Latency and RSS are reported so you can weigh cost: `hybrid_rerank` needs torch, which the Streamlit-Cloud runtime deliberately avoids (`requirements.txt` comment).
* **What is indexed.** Abstract only (what the app embeds). `--doc-text title_abstract` is an ablation, applied to dense, BM25 and rerank alike.
* **BM25.** SQLite FTS5, `porter unicode61`, OR-query over lower-cased tokens minus a short stop-word list; those choices are part of the configuration being evaluated.
* **Hybrid.** RRF, k=60, top-50 from each list. **Rerank.** top-20 of the fused list re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`; deeper ranks keep RRF order below them. Depth, k and model are flags; changing them after seeing results is tuning on 40 queries.
* **Dense backend.** `numpy` (exact) by default; `--dense-backend chroma` evaluates the HNSW index the app actually queries. Run both once.
* **Generation.** Prompt numbers the retrieved papers `[1]..[k]`; each factual sentence must end with `[n]`; insufficient context => the exact token `NOT_ENOUGH_INFORMATION`. `rag_engine` validates, asks the model to repair once, then drops sentences that still have no valid citation (`CITATIONS_POLICY=flag` keeps and marks them). The checker also reports first-pass compliance of the raw output, so you see what enforcement had to fix.
* **Support judging.** NLI = `cross-encoder/nli-deberta-v3-base` with max over 3-sentence windows of the abstract; LLM judge = a Llama model on Groq (different family from the gpt-oss generator). Both are scored against your 30 labels; the NLI threshold is chosen on them (leave-one-out estimate reported). Partial counts as not supported.
* **Refusal.** Reports OOC refusal rate and in-scope false-refusal rate together (an always-refuse bot would get 100% on the first). LLM failures are excluded from both and counted. No retrieval-score gate is implemented; the AUROC of the top-1 cosine tells you whether one could help.

## 3. Smoke runs
`--embedder hash`, `--reranker fake`, `--engine reference` exist so the pipeline can be exercised without models. Output from such runs is stamped `smoke: true` and printed with a banner; never quote it.
