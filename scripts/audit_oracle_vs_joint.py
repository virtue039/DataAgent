"""Extract the P4 audit target set from the P3 matrix.

For each qid where oracle.correct=True AND joint.correct=False, assembles
a comparison record containing the question, gold SQL, both evidence
strings, and both predicted SQLs. Writes results/p4_audit/audit_targets.json.

Usage:
  .venv/bin/python scripts/audit_oracle_vs_joint.py \
    --matrix-dir results/p3_matrix \
    --output results/p4_audit/audit_targets.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _latest_result(matrix_dir: Path, setting: str) -> dict:
    paths = sorted(matrix_dir.glob(f"{setting}_*.json"))
    if not paths:
        print(f"ERROR: no {setting}_*.json in {matrix_dir}", file=sys.stderr)
        sys.exit(1)
    return json.loads(paths[-1].read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract oracle-correct/joint-wrong qids for P4 audit."
    )
    parser.add_argument("--matrix-dir", default="results/p3_matrix",
                        help="Directory with the P3 4-setting result JSONs.")
    parser.add_argument("--output", default="results/p4_audit/audit_targets.json",
                        help="Output path for the target set.")
    args = parser.parse_args()

    matrix = Path(args.matrix_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    oracle = _latest_result(matrix, "oracle")
    joint = _latest_result(matrix, "joint")

    o_by_qid = {r["question_id"]: r for r in oracle["results"]}
    j_by_qid = {r["question_id"]: r for r in joint["results"]}

    targets = []
    for qid in sorted(j_by_qid):
        j = j_by_qid[qid]
        o = o_by_qid.get(qid)
        if o is None:
            continue
        if not o.get("correct") or j.get("correct"):
            continue
        targets.append({
            "question_id": qid,
            "db_id": j["db_id"],
            "difficulty": j["difficulty"],
            "question": j["question"],
            "gold_sql": j["gold_sql"],
            "oracle_evidence": o.get("evidence", ""),
            "oracle_predicted_sql": o.get("predicted_sql", ""),
            "joint_evidence": j.get("evidence", ""),
            "joint_predicted_sql": j.get("predicted_sql", ""),
        })

    output.write_text(json.dumps(
        {"meta": {"source_oracle": str(sorted(matrix.glob('oracle_*.json'))[-1].name),
                  "source_joint": str(sorted(matrix.glob('joint_*.json'))[-1].name),
                  "n_targets": len(targets)},
         "targets": targets}, indent=2), encoding="utf-8")
    print(f"Wrote {len(targets)} target qids to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
