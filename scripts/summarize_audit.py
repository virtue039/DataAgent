"""CLI: read judgments.jsonl, print the 5 audit summary sections.

Usage:
  .venv/bin/python scripts/summarize_audit.py [judgments_path]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.audit_analysis import (  # noqa: E402
    category_histogram,
    format_case_studies,
    format_headline,
    format_histogram,
    format_per_db,
    format_per_difficulty,
    per_db_breakdown,
    per_difficulty_breakdown,
    pick_case_studies,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize P4 LLM judgments.")
    parser.add_argument("judgments", nargs="?",
                        default="results/p4_audit/judgments.jsonl")
    args = parser.parse_args()

    path = Path(args.judgments)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return 1
    judgments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            judgments.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    print(f"Loaded {len(judgments)} judgments from {path}", file=sys.stderr)

    histogram = category_histogram(judgments)
    print(format_histogram(histogram))
    print()
    print(format_per_db(per_db_breakdown(judgments)))
    print()
    print(format_per_difficulty(per_difficulty_breakdown(judgments)))
    print()
    print(format_case_studies(pick_case_studies(judgments, k_per_category=2)))
    print()
    print(format_headline(histogram))
    return 0


if __name__ == "__main__":
    sys.exit(main())
