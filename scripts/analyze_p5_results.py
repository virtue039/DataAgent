"""Consolidated P5 analysis: D2 ablation + composition subset + audit histogram.

Reads:
  - results/p3_matrix/{none,retrieval,oracle}_*.json (filtered to 675 eval qids)
  - results/p4_cv/joint_*.json (the P4r CV joint result)
  - results/p5_d2_ablation/joint_no_bfs_*.json (the new D2 ablation)
  - results/p5_d3_composition/joint_*.json (joint with atoms-only graphs on 30 comp qids)
  - data/p5_composition_qids.json (the 30 composition qids)
  - results/p5_audit/judgments.jsonl (LLM-judge classifications)

Outputs (stdout):
  - Section A: overall 5-way EX table on 675 held-out qids
  - Section B: paired stats joint vs joint_no_bfs (D2 ablation) + joint vs others
  - Section C: composition subset (n=30) EX table — all four settings + joint atoms-only
  - Section D: audit category histogram

Usage:
  .venv/bin/python scripts/analyze_p5_results.py
"""
from __future__ import annotations

import argparse
import collections
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


def _latest(dir_path: Path, setting: str) -> dict | None:
    paths = sorted(dir_path.glob(f"{setting}_*.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def _correct_by_qid(doc: dict | None, allow: set[int] | None = None) -> dict[int, bool]:
    if doc is None:
        return {}
    return {r["question_id"]: bool(r.get("correct"))
            for r in doc["results"]
            if allow is None or r["question_id"] in allow}


def _ex_pct(correct: dict[int, bool], qids: set[int]) -> tuple[float, int, int]:
    rel = [c for q, c in correct.items() if q in qids]
    if not rel:
        return 0.0, 0, 0
    n = len(rel)
    n_correct = sum(1 for c in rel if c)
    return 100.0 * n_correct / n, n_correct, n


def _paired_row(name: str, a: dict[int, bool], b: dict[int, bool],
                qids: set[int]) -> dict:
    qs = sorted(set(a) & set(b) & qids)
    n = len(qs)
    a_correct = sum(1 for q in qs if a[q])
    b_correct = sum(1 for q in qs if b[q])
    b_count = sum(1 for q in qs if a[q] and not b[q])
    c_count = sum(1 for q in qs if not a[q] and b[q])
    chi2, p = mcnemar_with_correction(b_count, c_count)
    lo, hi = newcombe_mover_diff_ci(
        a_correct=a_correct, a_total=n,
        b_correct=b_correct, b_total=n,
        b_count=b_count, c_count=c_count,
    )
    diff = (a_correct - b_correct) / max(n, 1)
    return {
        "pair_name": name,
        "a_pct": a_correct / max(n, 1) * 100,
        "b_pct": b_correct / max(n, 1) * 100,
        "diff_pct": diff * 100,
        "chi2": chi2,
        "p_value": p,
        "lo": lo,
        "hi": hi,
        "n": n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P5 consolidated analysis.")
    parser.add_argument("--p3-dir", default="results/p3_matrix")
    parser.add_argument("--cv-dir", default="results/p4_cv")
    parser.add_argument("--d2-dir", default="results/p5_d2_ablation")
    parser.add_argument("--d3-dir", default="results/p5_d3_composition")
    parser.add_argument("--split", default="data/dev_cv_split.json")
    parser.add_argument("--comp-qids", default="data/p5_composition_qids.json")
    parser.add_argument("--audit-judgments", default="results/p5_audit/judgments.jsonl")
    args = parser.parse_args()

    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    eval_qids = set(split["eval_qids"])

    comp_doc = json.loads(Path(args.comp_qids).read_text(encoding="utf-8"))
    comp_qids = {it["question_id"] for it in comp_doc["items"]}

    docs = {
        "none": _latest(Path(args.p3_dir), "none"),
        "retrieval": _latest(Path(args.p3_dir), "retrieval"),
        "oracle": _latest(Path(args.p3_dir), "oracle"),
        "joint": _latest(Path(args.cv_dir), "joint"),
        "joint_no_bfs": _latest(Path(args.d2_dir), "joint_no_bfs"),
        "joint_atoms": _latest(Path(args.d3_dir), "joint"),
    }
    correct = {k: _correct_by_qid(v) for k, v in docs.items()}

    # ===== Section A: Overall 5-way on 675 held-out =====
    print("=" * 60)
    print("SECTION A — Overall 5-way EX on 675 held-out CV qids")
    print("=" * 60)
    print(f"{'Setting':<16} | {'EX%':>6} | {'Correct/Total':>14}")
    print("-" * 50)
    for s in ("none", "retrieval", "joint_no_bfs", "joint", "oracle"):
        if not correct[s]:
            print(f"{s:<16} | {'--':>6} | {'--':>14}  (no results yet)")
            continue
        pct, nc, n = _ex_pct(correct[s], eval_qids)
        print(f"{s:<16} | {pct:>5.2f}% | {nc:>5}/{n:<8}")

    # ===== Section B: Paired stats (D2 isolation focus) =====
    print()
    print("=" * 60)
    print("SECTION B — Paired stats on 675 held-out CV qids")
    print("=" * 60)

    rows = []
    pairs_under_test = [
        ("none vs joint", "none", "joint"),
        ("retrieval vs joint", "retrieval", "joint"),
        ("joint_no_bfs vs joint", "joint_no_bfs", "joint"),  # D2 isolation
        ("retrieval vs joint_no_bfs", "retrieval", "joint_no_bfs"),  # D1 isolation
        ("oracle vs joint", "oracle", "joint"),
    ]
    for name, a, b in pairs_under_test:
        if not correct[a] or not correct[b]:
            continue
        rows.append(_paired_row(name, correct[a], correct[b], eval_qids))
    if rows:
        print(format_paired_table(rows, n=rows[0]["n"]))
    else:
        print("(no comparable pairs yet)")

    # ===== Section C: Composition subset (n=30, atoms-only KB) =====
    print()
    print("=" * 60)
    print(f"SECTION C — Composition subset (n={len(comp_qids)}, atoms-only KB)")
    print("=" * 60)
    print(f"{'Setting':<16} | {'EX%':>6} | {'Correct/Total':>14}")
    print("-" * 50)
    for s, label in [
        ("none", "none"),
        ("retrieval", "retrieval"),
        ("joint_atoms", "joint (atoms KB)"),
        ("oracle", "oracle"),
    ]:
        if not correct[s]:
            print(f"{label:<16} | {'--':>6} | {'--':>14}")
            continue
        pct, nc, n = _ex_pct(correct[s], comp_qids)
        print(f"{label:<16} | {pct:>5.2f}% | {nc:>5}/{n:<8}")

    # ===== Section D: Audit histogram =====
    print()
    print("=" * 60)
    print("SECTION D — Error attribution histogram (n=40 audit targets)")
    print("=" * 60)
    judgments_path = Path(args.audit_judgments)
    if not judgments_path.exists():
        print(f"(no judgments at {judgments_path} yet)")
    else:
        cats = collections.Counter()
        per_db_cats = collections.defaultdict(collections.Counter)
        rows = []
        for line in judgments_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats[obj.get("category", "other")] += 1
            per_db_cats[obj.get("db_id", "?")][obj.get("category", "other")] += 1
            rows.append(obj)
        total = sum(cats.values())
        print(f"Total judgments: {total}")
        for c in ("specificity_loss", "retrieval_irrelevance",
                  "format_dilution", "other"):
            n = cats.get(c, 0)
            pct = 100 * n / max(total, 1)
            bar = "█" * int(pct / 2)
            print(f"  {c:<22} {n:>3} ({pct:5.1f}%) {bar}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
