"""Extract the audit target set for oracle-correct / joint-wrong qids.

For each qid where oracle.correct=True AND joint.correct=False, assembles
a comparison record containing the question, gold SQL, both evidence
strings, and both predicted SQLs.

Defaults preserve the original P4 behavior (both result dirs = P3 matrix).
P5 retargets joint to results/p4_cv via --joint-results and adds
non-empty evidence filtering plus reproducible sampling.

Usage (P4 legacy):
  .venv/bin/python scripts/audit_oracle_vs_joint.py \
    --matrix-dir results/p3_matrix \
    --output results/p4_audit/audit_targets.json

Usage (P5 retargeted):
  .venv/bin/python scripts/audit_oracle_vs_joint.py \
    --oracle-results results/p3_matrix \
    --joint-results results/p4_cv \
    --out results/p5_audit/audit_targets.json \
    --sample-size 40 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.audit_analysis import build_audit_targets  # noqa: E402


def _latest_result(results_dir: Path, setting: str) -> tuple[Path, dict]:
    paths = sorted(results_dir.glob(f"{setting}_*.json"))
    if not paths:
        print(f"ERROR: no {setting}_*.json in {results_dir}", file=sys.stderr)
        sys.exit(1)
    return paths[-1], json.loads(paths[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract oracle-correct/joint-wrong qids for audit."
    )
    # Legacy single-dir flag (P4 default).
    parser.add_argument("--matrix-dir", default=None,
                        help=("[legacy] Directory with both oracle_*.json and "
                              "joint_*.json. When given, used for both unless "
                              "--oracle-results / --joint-results override."))
    # P5 split flags.
    parser.add_argument("--oracle-results", default="results/p3_matrix",
                        help="Directory containing oracle_*.json.")
    parser.add_argument("--joint-results", default="results/p3_matrix",
                        help=("Directory containing joint_*.json. "
                              "P5 passes results/p4_cv here."))
    # Output: both --output (legacy) and --out (P5 spec) accepted.
    parser.add_argument("--output", default=None,
                        help="[legacy] Output path for the target set.")
    parser.add_argument("--out", default=None,
                        help="Output path for the target set (P5 alias).")
    parser.add_argument("--sample-size", type=int, default=40,
                        help="Random subsample size (0 = keep all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducible sampling.")
    parser.add_argument("--filter-nonempty-evidence",
                        dest="filter_nonempty_evidence",
                        action="store_true", default=True,
                        help="Drop targets where joint_evidence is empty.")
    parser.add_argument("--no-filter-nonempty-evidence",
                        dest="filter_nonempty_evidence",
                        action="store_false",
                        help="Keep targets with empty joint evidence "
                             "(restores original P4 behavior).")
    args = parser.parse_args()

    # Resolve oracle/joint dirs: --matrix-dir overrides both when given.
    oracle_dir = Path(args.matrix_dir or args.oracle_results)
    joint_dir = Path(args.matrix_dir or args.joint_results)

    # Resolve output path: --out wins, else --output, else default.
    out_str = args.out or args.output or "results/p4_audit/audit_targets.json"
    output = Path(out_str)
    output.parent.mkdir(parents=True, exist_ok=True)

    oracle_path, oracle = _latest_result(oracle_dir, "oracle")
    joint_path, joint = _latest_result(joint_dir, "joint")

    targets = build_audit_targets(
        oracle["results"], joint["results"],
        filter_nonempty_evidence=args.filter_nonempty_evidence,
    )
    n_before_sample = len(targets)

    sampled = False
    if args.sample_size and args.sample_size > 0 and len(targets) > args.sample_size:
        rng = random.Random(args.seed)
        targets = rng.sample(targets, args.sample_size)
        targets.sort(key=lambda t: t["question_id"])
        sampled = True

    meta = {
        "source_oracle": oracle_path.name,
        "source_joint": joint_path.name,
        "oracle_dir": str(oracle_dir),
        "joint_dir": str(joint_dir),
        "filter_nonempty_evidence": args.filter_nonempty_evidence,
        "n_eligible": n_before_sample,
        "n_targets": len(targets),
        "sampled": sampled,
        "sample_size": args.sample_size,
        "seed": args.seed,
    }

    output.write_text(json.dumps(
        {"meta": meta, "targets": targets}, indent=2), encoding="utf-8")
    print(f"Wrote {len(targets)} target qids "
          f"(eligible={n_before_sample}, sampled={sampled}) to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
