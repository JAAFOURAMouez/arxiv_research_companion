#!/usr/bin/env bash
# Full evaluation, in dependency order.  Run from the repo root after `uv sync --group eval`.
# Steps that need a hand-made input stop with a clear message; re-run after providing it.
set -euo pipefail
Q=eval/data/questions.csv
OOC=eval/data/ooc_questions.csv
SNAP=eval/data/corpus_snapshot.jsonl
RUN="uv run --group eval python -m"

$RUN eval.validate_labels --questions $Q --db arxiv_papers.db --check-arxiv --ooc $OOC
echo ">>> Audit the OOC titles printed above, set audited=true in $OOC, then continue (Ctrl-C to stop)."; read -r _

$RUN eval.build_corpus --db arxiv_papers.db --questions $Q --out $SNAP --fetch-missing
$RUN eval.scifact_crosscheck --download
$RUN eval.embeddings_agree --snapshot $SNAP --questions $Q || echo "!! embeddings disagree (see results/embeddings_agree.json)"
$RUN eval.run_retrieval_eval --snapshot $SNAP --questions $Q

# Put the chosen mode in RETRIEVAL_MODE (the retrieval table prints it), then:
MODE="${RETRIEVAL_MODE:-dense}"
$RUN eval.generate_answers --questions $Q --ooc $OOC --snapshot $SNAP --retrieval-mode "$MODE" --resume
$RUN eval.refusal_eval

if [ ! -f eval/data/calibration_to_label.csv ]; then
  $RUN eval.make_calibration_set
  echo ">>> Label eval/data/calibration_to_label.csv, then re-run this script."; exit 0
fi
$RUN eval.calibrate_judge
$RUN eval.check_citations --judge both --resolve
$RUN eval.run_ragas
$RUN eval.report
