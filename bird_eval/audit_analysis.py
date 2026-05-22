"""Pure-function aggregation for P4 audit judgments.

Inputs are lists of judgment dicts emitted by scripts/llm_judge_failures.py:
  {"question_id": int, "db_id": str, "difficulty": str,
   "category": str, "reason": str, "evidence_quote": str}

Outputs are list-of-dict tables ready for printing.

See docs/superpowers/specs/2026-05-22-p4-audit-and-stats-design.md.
"""
from __future__ import annotations

import math
from collections import defaultdict

_CATEGORIES = ("specificity_loss", "retrieval_irrelevance",
               "format_dilution", "other")
_Z_95 = 1.959963984540054


def _wilson_ci(correct: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    z2 = _Z_95 * _Z_95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (_Z_95 * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def build_audit_targets(
    oracle_results: list[dict],
    joint_results: list[dict],
    *,
    filter_nonempty_evidence: bool = True,
) -> list[dict]:
    """Build the (oracle-correct, joint-wrong) audit target list.

    Both inputs are lists of per-qid result dicts as produced by the
    settings runner (with keys: question_id, db_id, difficulty, question,
    gold_sql, evidence, predicted_sql, correct, ...).

    When `filter_nonempty_evidence` is True (the P5 default), targets whose
    joint `evidence` field is empty/missing are dropped — those cases
    cannot be classified by the LLM judge because there is no joint
    rendering to compare against the oracle. Set False to preserve the
    original P4 behavior.
    """
    o_by_qid = {r["question_id"]: r for r in oracle_results}
    j_by_qid = {r["question_id"]: r for r in joint_results}
    targets: list[dict] = []
    for qid in sorted(j_by_qid):
        j = j_by_qid[qid]
        o = o_by_qid.get(qid)
        if o is None:
            continue
        if not o.get("correct") or j.get("correct"):
            continue
        joint_evidence = j.get("evidence", "") or ""
        if filter_nonempty_evidence and not joint_evidence.strip():
            continue
        targets.append({
            "question_id": qid,
            "db_id": j["db_id"],
            "difficulty": j["difficulty"],
            "question": j["question"],
            "gold_sql": j["gold_sql"],
            "oracle_evidence": o.get("evidence", ""),
            "oracle_predicted_sql": o.get("predicted_sql", ""),
            "joint_evidence": joint_evidence,
            "joint_predicted_sql": j.get("predicted_sql", ""),
        })
    return targets


def category_histogram(judgments: list[dict]) -> list[dict]:
    """Return one row per category with count, proportion, Wilson CI."""
    n = len(judgments)
    counts = defaultdict(int)
    for j in judgments:
        counts[j.get("category", "other")] += 1
    rows = []
    for cat in _CATEGORIES:
        c = counts.get(cat, 0)
        lo, hi = _wilson_ci(c, n)
        rows.append({"category": cat, "count": c,
                     "proportion": c / max(n, 1),
                     "ci_lo": lo, "ci_hi": hi})
    # Sort descending by count for display.
    rows.sort(key=lambda r: -r["count"])
    return rows


def per_db_breakdown(judgments: list[dict]) -> list[dict]:
    """Return one row per db with n and per-category counts."""
    by_db: dict[str, dict] = defaultdict(lambda: {c: 0 for c in _CATEGORIES})
    db_n = defaultdict(int)
    for j in judgments:
        db = j.get("db_id", "?")
        cat = j.get("category", "other")
        by_db[db][cat] += 1
        db_n[db] += 1
    rows = []
    for db in sorted(by_db):
        row = {"db_id": db, "n": db_n[db]}
        row.update(by_db[db])
        rows.append(row)
    return rows


def per_difficulty_breakdown(judgments: list[dict]) -> list[dict]:
    """Return one row per difficulty with n and per-category counts."""
    by_diff: dict[str, dict] = defaultdict(lambda: {c: 0 for c in _CATEGORIES})
    diff_n = defaultdict(int)
    for j in judgments:
        diff = j.get("difficulty", "?")
        cat = j.get("category", "other")
        by_diff[diff][cat] += 1
        diff_n[diff] += 1
    rows = []
    for diff in ("simple", "moderate", "challenging"):
        if diff in by_diff:
            row = {"difficulty": diff, "n": diff_n[diff]}
            row.update(by_diff[diff])
            rows.append(row)
    return rows


def pick_case_studies(judgments: list[dict], k_per_category: int = 2) -> list[dict]:
    """Pick up to k_per_category representative judgments per non-empty category.

    Picks in input order (stable). Empty categories get 0 picks.
    """
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for j in judgments:
        by_cat[j.get("category", "other")].append(j)
    out = []
    for cat in _CATEGORIES:
        out.extend(by_cat.get(cat, [])[:k_per_category])
    return out


# ------------------ Formatting -------------------


def format_histogram(rows: list[dict]) -> str:
    n = sum(r["count"] for r in rows)
    lines = [f"==== P4 SECTION 1: Category histogram (n={n}) ====", "",
             f"{'Category':<24} | {'count':>6} | {'prop.':>7} | {'95% Wilson CI':<18}"]
    lines.append("-" * 70)
    for r in rows:
        ci = f"[{r['ci_lo']*100:5.2f}, {r['ci_hi']*100:5.2f}]"
        lines.append(f"{r['category']:<24} | {r['count']:>6} | "
                     f"{r['proportion']*100:>6.2f}% | {ci:<18}")
    return "\n".join(lines)


def format_per_db(rows: list[dict]) -> str:
    lines = ["==== P4 SECTION 2: Per-db category breakdown ====", "",
             f"{'db_id':<26} | {'n':>4} | "
             + " | ".join(f"{c[:9]:>9}" for c in _CATEGORIES)]
    lines.append("-" * 90)
    for r in rows:
        cells = [f"{r.get(c, 0):>9}" for c in _CATEGORIES]
        lines.append(f"{r['db_id']:<26} | {r['n']:>4} | " + " | ".join(cells))
    return "\n".join(lines)


def format_per_difficulty(rows: list[dict]) -> str:
    lines = ["==== P4 SECTION 3: Per-difficulty category breakdown ====", "",
             f"{'difficulty':<14} | {'n':>4} | "
             + " | ".join(f"{c[:9]:>9}" for c in _CATEGORIES)]
    lines.append("-" * 80)
    for r in rows:
        cells = [f"{r.get(c, 0):>9}" for c in _CATEGORIES]
        lines.append(f"{r['difficulty']:<14} | {r['n']:>4} | " + " | ".join(cells))
    return "\n".join(lines)


def format_case_studies(cases: list[dict]) -> str:
    lines = ["==== P4 SECTION 4: Case studies (2 per category) ====", ""]
    for c in cases:
        lines.append(f"--- qid={c.get('question_id')} db={c.get('db_id')} "
                     f"diff={c.get('difficulty')} cat={c.get('category')} ---")
        lines.append(f"Reason: {c.get('reason', '')}")
        if c.get("evidence_quote"):
            lines.append(f"Quote:  {c['evidence_quote']!r}")
        lines.append("")
    return "\n".join(lines)


def format_headline(histogram_rows: list[dict]) -> str:
    """Format the rule-based P5 recommendation."""
    total = sum(r["count"] for r in histogram_rows)
    if total == 0:
        return "==== P4 SECTION 5: Headline ====\n\nNo judgments — cannot recommend."

    top = histogram_rows[0]
    top_pct = top["proportion"] * 100

    lines = ["==== P4 SECTION 5: Headline ====", ""]
    lines.append(f"Dominant failure mode: {top['category']} "
                 f"({top_pct:.2f}% of {total} failures, "
                 f"95% CI [{top['ci_lo']*100:.2f}, {top['ci_hi']*100:.2f}]).")
    lines.append("")
    if top_pct >= 40.0:
        strength = "strong"
    elif top_pct >= 30.0:
        strength = "moderate"
    else:
        strength = "weak"
    lines.append(f"Signal strength: {strength}.")

    recs = {
        "specificity_loss": ("improve extraction to capture exact constants and "
                             "column names in Formula expressions and Rule "
                             "conditions (require value-literal grounding for "
                             "any predicate)."),
        "retrieval_irrelevance": ("improve retrieval: question-conditioned query "
                                  "generation or re-ranking before subgraph "
                                  "rendering."),
        "format_dilution": ("simplify the renderer: fewer bullets, shorter form, "
                            "or compress the rendered evidence."),
        "other": ("mixed / non-content-causal failures dominate; pivot to a "
                  "competing baseline (dbt semantic layer B5) rather than "
                  "iterating on this method."),
    }
    if strength == "weak":
        lines.append("Recommended P5: no single fix dominates; pivot to "
                     "dbt semantic layer B5 as a competing baseline.")
    else:
        lines.append(f"Recommended P5: {recs.get(top['category'], recs['other'])}")
    return "\n".join(lines)
