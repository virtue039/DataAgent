"""P5-D3 composition-query candidate selector.

Mechanically filters the 675 held-out dev qids using three heuristics:

- C1 (linguistic): the question contains a conjunction/intersection marker.
- C2 (SQL structural): the gold SQL has 2+ ANDs (outside string literals),
  HAVING+WHERE together, or a subquery `SELECT ... (SELECT`.
- C3 (evidence-decomposable): the extracted v2 nodes contain 2+ Formula
  or Rule entries (each is one definitional clause).

For each qid, writes a record with the three boolean flags and a 0-3
`score = c1 + c2 + c3`. Output is sorted desc by score.

Usage:
  .venv/bin/python scripts/select_composition_qids.py \\
      --eval-qids data/dev_cv_eval_qids.json \\
      --bird-dev data/bird_dev/dev_20240627/dev.json \\
      --extracted annotation/dev_extracted.json \\
      --output data/p5_composition_candidates.json

See docs/superpowers/specs/2026-05-22-p5-d3-composition-specialty-design.md
§3.1 for the C1/C2/C3 definitions and the manual-review workflow that
operates on this candidate list.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- C1: linguistic conjunction markers ---------------------------------

# Substrings checked case-insensitively against the question text.
_C1_SUBSTRINGS = (
    " and ",
    " who are also ",
    " but ",
    "intersection",
    "both",
    " & ",
)

# Compiled regexes for "paired" patterns (binary composition cues).
_C1_PATTERNS = (
    re.compile(r"\bwho\b.*\band\b", re.IGNORECASE),
    re.compile(r"\bthe\s+\S+\s+with\s+the\s+\S+", re.IGNORECASE),
)


def _check_c1(question: str) -> bool:
    if not question:
        return False
    q = question.casefold()
    if any(s in q for s in _C1_SUBSTRINGS):
        return True
    return any(p.search(question) for p in _C1_PATTERNS)


# --- C2: SQL structural markers -----------------------------------------

_AND_TOKEN = re.compile(r"\bAND\b", re.IGNORECASE)
_HAVING = re.compile(r"\bHAVING\b", re.IGNORECASE)
_WHERE = re.compile(r"\bWHERE\b", re.IGNORECASE)
_SUBQUERY = re.compile(r"SELECT\b.*\(\s*SELECT\b", re.IGNORECASE | re.DOTALL)


def _strip_string_literals(sql: str) -> str:
    """Remove SQLite string literals so AND inside 'foo and bar' doesn't count."""
    # Single-quoted strings (SQLite uses '' as an escape); replace contents.
    return re.sub(r"'(?:[^']|'')*'", "''", sql)


def _check_c2(sql: str) -> bool:
    if not sql:
        return False
    no_strings = _strip_string_literals(sql)
    and_count = len(_AND_TOKEN.findall(no_strings))
    if and_count >= 2:
        return True
    if _HAVING.search(no_strings) and _WHERE.search(no_strings):
        return True
    if _SUBQUERY.search(no_strings):
        return True
    return False


# --- C3: evidence-decomposable via extracted v2 nodes -------------------

_DEFINITIONAL_TYPES = frozenset({"Formula", "Rule"})


def _check_c3(nodes: list[dict] | None) -> bool:
    if not nodes:
        return False
    count = sum(
        1 for n in nodes
        if isinstance(n, dict) and n.get("type") in _DEFINITIONAL_TYPES
    )
    return count >= 2


# --- Main ----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select composition-query candidates from the held-out 675 dev qids.",
    )
    parser.add_argument("--eval-qids", default="data/dev_cv_eval_qids.json",
                        help="held-out qids file with shape {items: [{question_id}, ...]}")
    parser.add_argument("--bird-dev", default="data/bird_dev/dev_20240627/dev.json",
                        help="full BIRD dev json (list of {question_id, db_id, question, evidence, SQL})")
    parser.add_argument("--extracted", default="annotation/dev_extracted.json",
                        help="annotation/dev_extracted.json with v2 typed nodes")
    parser.add_argument("--output", default="data/p5_composition_candidates.json",
                        help="output JSON with {meta, candidates}")
    args = parser.parse_args()

    eval_qids_doc = json.loads(Path(args.eval_qids).read_text(encoding="utf-8"))
    eval_qids = {it["question_id"] for it in eval_qids_doc["items"]}

    bird_dev = json.loads(Path(args.bird_dev).read_text(encoding="utf-8"))
    bird_by_qid = {row["question_id"]: row for row in bird_dev}

    extracted = json.loads(Path(args.extracted).read_text(encoding="utf-8"))
    nodes_by_qid: dict[int, list[dict]] = {
        it["question_id"]: it.get("nodes", []) or []
        for it in extracted.get("items", [])
    }

    candidates: list[dict] = []
    missing_bird = 0
    for qid in sorted(eval_qids):
        row = bird_by_qid.get(qid)
        if row is None:
            missing_bird += 1
            continue
        question = row.get("question", "") or ""
        gold_sql = row.get("SQL", "") or ""
        evidence = row.get("evidence", "") or ""
        db_id = row.get("db_id", "")
        v2_nodes = nodes_by_qid.get(qid, [])

        c1 = _check_c1(question)
        c2 = _check_c2(gold_sql)
        c3 = _check_c3(v2_nodes)
        score = int(c1) + int(c2) + int(c3)

        candidates.append({
            "question_id": qid,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "evidence": evidence,
            "score": score,
            "c1": c1,
            "c2": c2,
            "c3": c3,
        })

    # Sort desc by score, then ascending by qid for stability.
    candidates.sort(key=lambda r: (-r["score"], r["question_id"]))

    n_ge2 = sum(1 for c in candidates if c["score"] >= 2)
    n_eq3 = sum(1 for c in candidates if c["score"] == 3)

    payload = {
        "meta": {
            "eval_qids_source": args.eval_qids,
            "bird_dev_source": args.bird_dev,
            "extracted_source": args.extracted,
            "n_total_eval_qids": len(eval_qids),
            "n_with_bird_row": len(candidates),
            "n_missing_bird": missing_bird,
            "n_score_ge_2": n_ge2,
            "n_score_eq_3": n_eq3,
            "c1_substrings": list(_C1_SUBSTRINGS),
            "c1_patterns": [p.pattern for p in _C1_PATTERNS],
        },
        "candidates": candidates,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    print(f"Wrote {len(candidates)} candidates to {out_path}.")
    print(f"Found {n_ge2} candidates with score>=2; {n_eq3} with score==3.")
    if missing_bird:
        print(f"Note: {missing_bird} eval qids had no row in {args.bird_dev}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
