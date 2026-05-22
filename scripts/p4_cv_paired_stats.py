"""Paired stats for the P4-revised CV joint eval.

Reads results/p4_cv/joint_*.json (the new CV result) and
results/p3_matrix/{none,retrieval,oracle}_*.json (existing), filters
the latter to the eval_qids in data/dev_cv_split.json, then runs
McNemar + Newcombe MOVER paired-diff CI for joint vs each baseline.

Usage:
  .venv/bin/python scripts/p4_cv_paired_stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.paired_stats import (  # noqa: E402
    format_paired_table,
    mcnemar_with_correction,
    newcombe_mover_diff_ci,
)


def _latest(dir_path: Path, setting: str) -> dict:
    paths = sorted(dir_path.glob(f"{setting}_*.json"))
    if not paths:
        raise FileNotFoundError(f"No {setting}_*.json in {dir_path}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _correct_by_qid(doc: dict, allow_qids: set[int]) -> dict[int, bool]:
    return {r["question_id"]: bool(r.get("correct"))
            for r in doc["results"]
            if r["question_id"] in allow_qids}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired McNemar + MOVER for P4-revised CV joint vs P3 baselines."
    )
    parser.add_argument("--cv-dir", default="results/p4_cv")
    parser.add_argument("--p3-dir", default="results/p3_matrix")
    parser.add_argument("--split", default="data/dev_cv_split.json")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    eval_qids = set(split["eval_qids"])

    joint_doc = _latest(Path(args.cv_dir), "joint")
    docs = {"joint": joint_doc}
    for s in ("none", "retrieval", "oracle"):
        docs[s] = _latest(Path(args.p3_dir), s)

    correct = {s: _correct_by_qid(d, eval_qids) for s, d in docs.items()}

    rows = []
    n_final = 0
    for other in ("none", "retrieval", "oracle"):
        a_corr = correct[other]
        b_corr = correct["joint"]
        qids = sorted(set(a_corr) & set(b_corr))
        n = len(qids)
        n_final = n
        a_correct = sum(1 for q in qids if a_corr[q])
        b_correct = sum(1 for q in qids if b_corr[q])
        b_count = sum(1 for q in qids if a_corr[q] and not b_corr[q])
        c_count = sum(1 for q in qids if not a_corr[q] and b_corr[q])
        chi2, p = mcnemar_with_correction(b_count, c_count)
        lo, hi = newcombe_mover_diff_ci(
            a_correct=a_correct, a_total=n,
            b_correct=b_correct, b_total=n,
            b_count=b_count, c_count=c_count,
        )
        diff = (a_correct - b_correct) / max(n, 1)
        rows.append({
            "pair_name": f"{other} vs joint",
            "a_pct": a_correct / max(n, 1) * 100,
            "b_pct": b_correct / max(n, 1) * 100,
            "diff_pct": diff * 100,
            "chi2": chi2,
            "p_value": p,
            "lo": lo,
            "hi": hi,
        })

    print(f"Paired analysis on n={n_final} held-out dev qids "
          f"(P4r CV eval intersection with P3 results).")
    print()
    print(format_paired_table(rows, n=n_final))
    return 0


if __name__ == "__main__":
    sys.exit(main())
