"""CLI entry point for the P3 matrix analysis.

Reads the four setting result JSONs from a results directory, prints the
four analysis sections (overall, per-difficulty, per-db, per-db delta)
to stdout. The user pastes the output into Report.md / paper draft.

Usage:
  .venv/bin/python scripts/analyze_p3_matrix.py results/p3_matrix/

See docs/superpowers/specs/2026-05-21-p3-scale-eval-design.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from the scripts/ dir or the repo root.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.analysis import (  # noqa: E402
    format_overall,
    format_per_db,
    format_per_db_delta,
    format_per_difficulty,
    per_db_delta,
    per_db_table,
    per_difficulty_table,
    per_setting_overall,
)


_SETTINGS = ("none", "retrieval", "oracle", "joint")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze the P3 4-way matrix (overall + per-difficulty + per-db)."
    )
    parser.add_argument("matrix_dir",
                        help="Directory containing <setting>_*.json result files.")
    args = parser.parse_args()

    matrix_dir = Path(args.matrix_dir)
    if not matrix_dir.is_dir():
        print(f"ERROR: {matrix_dir} is not a directory", file=sys.stderr)
        return 1

    # For each setting, pick the most-recent timestamped file.
    docs: dict[str, dict] = {}
    for s in _SETTINGS:
        paths = sorted(matrix_dir.glob(f"{s}_*.json"))
        if not paths:
            print(f"WARN: no {s}_*.json in {matrix_dir}", file=sys.stderr)
            continue
        # Most-recent by lex order on the timestamp suffix.
        chosen = paths[-1]
        docs[s] = json.loads(chosen.read_text(encoding="utf-8"))
        print(f"loaded {s}: {chosen.name} "
              f"(n={docs[s].get('summary', {}).get('n_scored', '?')})",
              file=sys.stderr)

    if not docs:
        print("ERROR: no result files found", file=sys.stderr)
        return 1

    # Print the four sections.
    print(format_overall(per_setting_overall(docs)))
    print()
    print(format_per_difficulty(per_difficulty_table(docs)))
    print()
    print(format_per_db(per_db_table(docs)))
    print()
    print(format_per_db_delta(per_db_delta(docs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
