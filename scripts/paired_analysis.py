"""CLI: compute paired McNemar + Newcombe MOVER CI on the P3 matrix.

Usage:
  .venv/bin/python scripts/paired_analysis.py [matrix_dir]
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


def _latest(matrix: Path, setting: str) -> dict:
    paths = sorted(matrix.glob(f"{setting}_*.json"))
    if not paths:
        raise FileNotFoundError(f"No {setting}_*.json in {matrix}")
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _correct_by_qid(doc: dict) -> dict[int, bool]:
    return {r["question_id"]: bool(r.get("correct")) for r in doc["results"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired McNemar + MOVER analysis.")
    parser.add_argument("matrix_dir", nargs="?", default="results/p3_matrix")
    args = parser.parse_args()

    matrix = Path(args.matrix_dir)
    docs = {s: _latest(matrix, s) for s in
            ("none", "retrieval", "oracle", "joint")}
    correct = {s: _correct_by_qid(d) for s, d in docs.items()}

    rows = []
    # Joint vs each of {none, retrieval, oracle}: B = joint, A = other.
    for other in ("none", "retrieval", "oracle"):
        a_corr = correct[other]
        b_corr = correct["joint"]
        qids = sorted(set(a_corr) & set(b_corr))
        n = len(qids)
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

    print(format_paired_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
