"""Pure-function analysis helpers for the P3 4-way eval matrix.

Inputs are result-JSON-shaped dicts (the structure emitted by run_eval.py:
{config, summary, results: [{db_id, difficulty, correct, ...}, ...]}).

Outputs are list-of-dict tables ready for tabular printing.

See docs/superpowers/specs/2026-05-21-p3-scale-eval-design.md.
"""
from __future__ import annotations

import math
from collections import defaultdict

_Z_95 = 1.959963984540054  # 1.96 to 16 digits


def wilson_ci(correct: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial proportion.

    Edge cases:
    - n == 0: returns (0.0, 1.0). The honest "we have no data" CI.
    - correct in {0, n}: Wilson handles cleanly without going negative or > 1.
    """
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))) / denom
    lo = max(0.0, center - margin)
    hi = min(1.0, center + margin)
    return lo, hi


# ------------------ Aggregation -------------------

_SETTINGS_ORDER = ("none", "retrieval", "oracle", "joint")
_DIFFICULTIES_ORDER = ("simple", "moderate", "challenging")


def _results_of(doc: dict) -> list[dict]:
    return list(doc.get("results", []))


def per_setting_overall(docs: dict[str, dict]) -> list[dict]:
    """Returns one row per setting: {setting, n, correct, ex, ci_lo, ci_hi}."""
    rows = []
    for s in _SETTINGS_ORDER:
        if s not in docs:
            continue
        results = _results_of(docs[s])
        n = len(results)
        c = sum(1 for r in results if r.get("correct"))
        lo, hi = wilson_ci(c, n)
        rows.append({"setting": s, "n": n, "correct": c,
                     "ex": (c / max(n, 1)), "ci_lo": lo, "ci_hi": hi})
    return rows


def per_difficulty_table(docs: dict[str, dict]) -> list[dict]:
    """Returns one row per difficulty.

    Each row: {difficulty, n_<setting>, <setting>_ex} for each setting present.
    For brevity we use the setting name as a key holding the EX%, and
    "n_<setting>" for the per-cell count.
    """
    out = []
    for diff in _DIFFICULTIES_ORDER:
        row: dict = {"difficulty": diff}
        for s in _SETTINGS_ORDER:
            if s not in docs:
                continue
            sub = [r for r in _results_of(docs[s]) if r.get("difficulty") == diff]
            n = len(sub)
            c = sum(1 for r in sub if r.get("correct"))
            row[s] = c / max(n, 1)
            row[f"n_{s}"] = n
        out.append(row)
    return out


def per_db_table(docs: dict[str, dict]) -> list[dict]:
    """Returns one row per db_id (sorted alphabetically).

    Each row: {db_id, n, <setting>_ex for each setting}. The "n" is computed
    from whichever setting is in docs (they should all have the same qid set
    per the run_eval.py contract).
    """
    # Collect all db_ids across the docs.
    all_dbs = set()
    for doc in docs.values():
        for r in _results_of(doc):
            db = r.get("db_id")
            if db is not None:
                all_dbs.add(db)

    # Use the first setting's results to compute the per-db count.
    first_setting = next(iter(docs))
    db_to_count = defaultdict(int)
    for r in _results_of(docs[first_setting]):
        db_to_count[r.get("db_id")] += 1

    out = []
    for db in sorted(all_dbs):
        row: dict = {"db_id": db, "n": db_to_count[db]}
        for s in _SETTINGS_ORDER:
            if s not in docs:
                continue
            sub = [r for r in _results_of(docs[s]) if r.get("db_id") == db]
            n = len(sub)
            c = sum(1 for r in sub if r.get("correct"))
            row[s] = c / max(n, 1)
        out.append(row)
    return out


def per_db_delta(
    docs: dict[str, dict],
    tie_tol: float = 0.01,
) -> list[dict]:
    """Per-db Δ(joint − retrieval), sorted descending by Δ.

    Requires both "joint" and "retrieval" docs in `docs`.

    Each row: {db_id, n, joint_ex, retr_ex, delta, direction}
    direction ∈ {"joint wins", "retr wins", "tie"} based on |Δ| vs tie_tol.
    """
    if "joint" not in docs or "retrieval" not in docs:
        return []
    # Per-db EX% for joint and retrieval.
    j_results = _results_of(docs["joint"])
    r_results = _results_of(docs["retrieval"])
    j_by_db: dict[str, list[bool]] = defaultdict(list)
    r_by_db: dict[str, list[bool]] = defaultdict(list)
    for x in j_results:
        j_by_db[x.get("db_id")].append(bool(x.get("correct")))
    for x in r_results:
        r_by_db[x.get("db_id")].append(bool(x.get("correct")))

    rows = []
    for db in sorted(set(j_by_db) | set(r_by_db)):
        jx = j_by_db[db]
        rx = r_by_db[db]
        j_ex = sum(jx) / max(len(jx), 1)
        r_ex = sum(rx) / max(len(rx), 1)
        delta = j_ex - r_ex
        if abs(delta) <= tie_tol:
            direction = "tie"
        elif delta > 0:
            direction = "joint wins"
        else:
            direction = "retr wins"
        rows.append({"db_id": db, "n": len(jx),
                     "joint_ex": j_ex, "retr_ex": r_ex,
                     "delta": delta, "direction": direction})

    rows.sort(key=lambda r: -r["delta"])
    return rows


# ------------------ Formatting -------------------


def format_overall(rows: list[dict]) -> str:
    lines = ["==== SECTION 1: Overall EX% with 95% Wilson CI ====",
             "",
             f"{'Setting':<10} | {'EX%':>6} | {'95% CI':<16} | {'Scored / Total':>14}",
             "-" * 60]
    for r in rows:
        ci = f"[{r['ci_lo']*100:5.2f}, {r['ci_hi']*100:5.2f}]"
        lines.append(f"{r['setting']:<10} | {r['ex']*100:6.2f} | {ci:<16} | "
                     f"{r['correct']:>6} / {r['n']:<5}")
    return "\n".join(lines)


def format_per_difficulty(rows: list[dict]) -> str:
    lines = ["==== SECTION 2: Per-difficulty × setting EX% ====",
             "",
             f"{'Difficulty':<22} | "
             + " | ".join(f"{s:>10}" for s in _SETTINGS_ORDER)]
    lines.append("-" * 80)
    for r in rows:
        # Use the first setting's n_ as the row's representative n.
        n_key = next((f"n_{s}" for s in _SETTINGS_ORDER if f"n_{s}" in r), None)
        n = r.get(n_key, 0) if n_key else 0
        diff_label = f"{r['difficulty']} (n={n})"
        cells = [f"{r[s]*100:>9.2f}%" if s in r else f"{'  --':>10}"
                 for s in _SETTINGS_ORDER]
        lines.append(f"{diff_label:<22} | " + " | ".join(cells))
    return "\n".join(lines)


def format_per_db(rows: list[dict]) -> str:
    lines = ["==== SECTION 3: Per-db × setting EX% ====",
             "",
             f"{'db_id':<26} | {'n':>4} | "
             + " | ".join(f"{s:>9}" for s in _SETTINGS_ORDER)]
    lines.append("-" * 90)
    for r in rows:
        cells = [f"{r[s]*100:>8.2f}%" if s in r else f"{'   --':>9}"
                 for s in _SETTINGS_ORDER]
        lines.append(f"{r['db_id']:<26} | {r['n']:>4} | " + " | ".join(cells))
    return "\n".join(lines)


def format_per_db_delta(rows: list[dict]) -> str:
    lines = ["==== SECTION 4: Per-db Δ(joint − retrieval), sorted descending ====",
             "",
             f"{'db_id':<26} | {'n':>4} | {'joint%':>7} | {'retr%':>7} | "
             f"{'Δ':>7} | direction"]
    lines.append("-" * 80)
    j_wins = r_wins = ties = 0
    j_deltas: list[float] = []
    r_deltas: list[float] = []
    for r in rows:
        lines.append(f"{r['db_id']:<26} | {r['n']:>4} | "
                     f"{r['joint_ex']*100:>6.2f}% | {r['retr_ex']*100:>6.2f}% | "
                     f"{r['delta']*100:+7.2f} | {r['direction']}")
        if r["direction"] == "joint wins":
            j_wins += 1; j_deltas.append(r["delta"])
        elif r["direction"] == "retr wins":
            r_wins += 1; r_deltas.append(r["delta"])
        else:
            ties += 1
    lines.append("")
    avg_j = (sum(j_deltas) / len(j_deltas) * 100) if j_deltas else 0
    avg_r = (sum(r_deltas) / len(r_deltas) * 100) if r_deltas else 0
    lines.append(f"Joint wins on {j_wins} dbs (avg Δ = +{avg_j:.2f}pp); "
                 f"retrieval wins on {r_wins} (avg Δ = {avg_r:.2f}pp); "
                 f"{ties} ties.")
    return "\n".join(lines)
