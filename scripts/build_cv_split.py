"""Build the within-db CV split from a v2 annotation file.

Reads `annotation/dev_extracted.json` (or a smoke version), runs the
stratified split, writes:
  - data/dev_cv_split.json — {kb_qids: [...], eval_qids: [...]} for downstream
  - data/dev_cv_eval_qids.json — {"items": [{question_id: ...}, ...]} for run_eval.py
  - data/dev_cv_kb_qids.json   — same shape but with KB qids (for symmetry)

Usage:
  .venv/bin/python scripts/build_cv_split.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.cv_split import make_within_db_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build within-db CV split file.")
    parser.add_argument("--annotation", default="annotation/dev_extracted.json")
    parser.add_argument("--split-out", default="data/dev_cv_split.json")
    parser.add_argument("--eval-qids-out", default="data/dev_cv_eval_qids.json")
    parser.add_argument("--kb-qids-out", default="data/dev_cv_kb_qids.json")
    parser.add_argument("--kb-fraction", type=float, default=0.5)
    args = parser.parse_args()

    data = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    items = data["items"]
    kb, eval_set = make_within_db_split(items, kb_fraction=args.kb_fraction)
    print(f"From {len(items)} items in {args.annotation}:")
    print(f"  KB qids:   {len(kb):>5d}")
    print(f"  Eval qids: {len(eval_set):>5d}")

    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.split_out).write_text(json.dumps({
        "meta": {"source": args.annotation,
                 "kb_fraction": args.kb_fraction,
                 "n_kb": len(kb), "n_eval": len(eval_set)},
        "kb_qids": sorted(kb),
        "eval_qids": sorted(eval_set),
    }, indent=2), encoding="utf-8")

    Path(args.eval_qids_out).write_text(json.dumps({
        "items": [{"question_id": q} for q in sorted(eval_set)],
    }, indent=2), encoding="utf-8")
    Path(args.kb_qids_out).write_text(json.dumps({
        "items": [{"question_id": q} for q in sorted(kb)],
    }, indent=2), encoding="utf-8")

    print(f"Wrote {args.split_out}, {args.eval_qids_out}, {args.kb_qids_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
