"""
Label the calibration pairs in the terminal, one at a time.

    python -m eval.label_calibration [--file eval/data/calibration_to_label.csv]

Shows the sentence and the cited abstract (no judge score, so nothing can bias you) and asks:
    s = supported    the abstract states or clearly implies the sentence
    p = partial      only part of the sentence is backed (partial counts as NOT supported)
    u = unsupported  the abstract does not back it
    b = back one pair    k = skip for now    q = save and quit
The CSV is rewritten after every answer, so you can stop and resume at any time; already
labelled rows are skipped (use --relabel to go through everything again).
"""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from typing import Dict, List

KEYS = {"s": "supported", "p": "partial", "u": "unsupported"}
VALID = set(KEYS.values())


def load(path: str):
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def save(path: str, fields: List[str], rows: List[Dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def show(row: Dict[str, str], pos: int, total: int, done: int) -> None:
    wrap = lambda t: textwrap.fill(t, width=100)  # noqa: E731
    print("\n" + "=" * 100)
    print(f"[{pos}/{total}]  labelled so far: {done}/{total}   pair {row['pair_id']}   cited paper {row['cited_id']}")
    print("-" * 100)
    print("SENTENCE:\n" + wrap(row["sentence"]))
    print("\nCITED PAPER: " + row["title"])
    print("ABSTRACT:\n" + wrap(row["abstract"]))
    print("-" * 100)


def run(path: str, relabel: bool, ask=input) -> int:
    fields, rows = load(path)
    if "label" not in fields:
        print("ERROR: no `label` column in", path, file=sys.stderr)
        return 1
    order = [i for i, r in enumerate(rows) if relabel or (r.get("label") or "").strip().lower() not in VALID]
    i = 0
    while i < len(order):
        idx = order[i]
        done = sum((r.get("label") or "").strip().lower() in VALID for r in rows)
        show(rows[idx], idx + 1, len(rows), done)
        try:
            ans = ask("[s]upported  [p]artial  [u]nsupported  [b]ack  s[k]ip  [q]uit > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = "q"
        if ans in KEYS:
            rows[idx]["label"] = KEYS[ans]
            save(path, fields, rows)
            i += 1
        elif ans == "b":
            i = max(0, i - 1)
        elif ans == "k":
            i += 1
        elif ans == "q":
            break
        else:
            print("  type s, p, u, b, k or q")
    left = sum((r.get("label") or "").strip().lower() not in VALID for r in rows)
    save(path, fields, rows)
    if left:
        print(f"\nSaved. {left} pair(s) still unlabelled; run the same command to continue.")
    else:
        print("\nAll pairs labelled. Re-run: bash eval/run_all.sh")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="eval/data/calibration_to_label.csv")
    ap.add_argument("--relabel", action="store_true", help="go through already-labelled rows too")
    args = ap.parse_args(argv)
    return run(args.file, args.relabel)


if __name__ == "__main__":
    sys.exit(main())
