"""Compare extractor output to the v2 gold standard. Pure functions, no I/O.

Public surface:
- match(a, b) -> bool   : node equivalence relation.
- match_pairs(predicted, gold) -> list[tuple[int, int]]
                            : 1-1 assignment between predicted and gold lists.
- compare_items(predicted, gold) -> dict   : per-item counts (per-type tp/fp/fn,
                                             grounding intersection, edge accuracy).
- aggregate(per_item_results) -> dict     : the report's `summary` block.
"""
from __future__ import annotations


_TYPES = ("Concept", "Formula", "ValueMap", "Rule", "ColumnAlias")


def _norm(s: str | None) -> str:
    return (s or "").strip().casefold()


def _bindings_signature(node: dict) -> frozenset:
    return frozenset(
        (_norm(b.get("table")), _norm(b.get("column")))
        for b in (node.get("bindings") or [])
    )


def match(a: dict, b: dict) -> bool:
    """Return True iff a and b are the "same" node for scoring purposes."""
    if a.get("type") != b.get("type"):
        return False
    t = a.get("type")
    if t == "Concept":
        return _norm(a.get("name")) == _norm(b.get("name"))
    if t == "ColumnAlias":
        return _bindings_signature(a) == _bindings_signature(b)
    if t == "ValueMap":
        return (
            _norm(a.get("table")) == _norm(b.get("table"))
            and _norm(a.get("column")) == _norm(b.get("column"))
            and _norm(a.get("value")) == _norm(b.get("value"))
        )
    if t == "Formula":
        # 'expression' is the deterministic SQL/math expression; prefer it
        # as primary match key. 'name' is a paraphrase-prone label and is
        # used only when 'expression' is missing on either side.
        # Same pattern as Rule (commit 79c7e3d, condition-primary).
        ea = _norm(a.get("expression"))
        eb = _norm(b.get("expression"))
        if ea and eb:
            return ea == eb
        ka = _norm(a.get("name"))
        kb = _norm(b.get("name"))
        return bool(ka) and ka == kb
    if t == "Rule":
        # 'condition' is a deterministic SQL expression; prefer it as primary
        # match key. 'name' is a paraphrase-prone label and is used only when
        # 'condition' is missing on either side.
        ca = _norm(a.get("condition"))
        cb = _norm(b.get("condition"))
        if ca and cb:
            return ca == cb
        ka = _norm(a.get("name"))
        kb = _norm(b.get("name"))
        return bool(ka) and ka == kb
    return False


def _grounding_set(node: dict) -> set:
    """Set of (term, table, column, value) tuples on a grounded node.
    Used for Jaccard similarity between matched Formula/Rule pairs."""
    out = set()
    for g in node.get("grounding", []) or []:
        out.add(
            (_norm(g.get("term")), _norm(g.get("table")),
             _norm(g.get("column")), _norm(g.get("value")))
        )
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def match_pairs(predicted: list[dict], gold: list[dict]) -> list[tuple[int, int]]:
    """One-to-one assignment: for each predicted node in order, take the first
    not-yet-matched gold node it matches. Returns list of (pred_idx, gold_idx)."""
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, p in enumerate(predicted):
        for j, g in enumerate(gold):
            if j in used:
                continue
            if match(p, g):
                pairs.append((i, j))
                used.add(j)
                break
    return pairs


def _empty_type_counts() -> dict[str, dict[str, int]]:
    return {t: {"tp": 0, "fp": 0, "fn": 0} for t in _TYPES}


def compare_items(predicted: list[dict], gold: list[dict]) -> dict:
    """Per-item counts. Used by `aggregate` and per-item report rows.

    Returned dict shape:
      {
        "matched": int,
        "fp": int,                      # predicted nodes unmatched
        "fn": int,                      # gold nodes unmatched
        "per_type": {<type>: {"tp": int, "fp": int, "fn": int}},
        "grounding_jaccards": list[float]   # one entry per matched
                                            # Formula/Rule pair
      }
    """
    pairs = match_pairs(predicted, gold)
    matched_pred = {i for i, _ in pairs}
    matched_gold = {j for _, j in pairs}

    per_type = _empty_type_counts()
    g_jaccards: list[float] = []

    for i, j in pairs:
        ntype = predicted[i].get("type")
        if ntype in per_type:
            per_type[ntype]["tp"] += 1
        if ntype in ("Formula", "Rule"):
            g_jaccards.append(_jaccard(_grounding_set(predicted[i]), _grounding_set(gold[j])))

    for i, p in enumerate(predicted):
        if i in matched_pred:
            continue
        ntype = p.get("type")
        if ntype in per_type:
            per_type[ntype]["fp"] += 1

    for j, g in enumerate(gold):
        if j in matched_gold:
            continue
        ntype = g.get("type")
        if ntype in per_type:
            per_type[ntype]["fn"] += 1

    return {
        "matched": len(pairs),
        "fp": len(predicted) - len(pairs),
        "fn": len(gold) - len(pairs),
        "per_type": per_type,
        "grounding_jaccards": g_jaccards,
    }


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _f1(tp: int, fp: int, fn: int) -> dict:
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "support": tp + fn}


def aggregate(per_item: list[dict]) -> dict:
    """Roll per-item counts up into a summary block (the JSON shape spec §3.4)."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    by_type_acc = _empty_type_counts()
    all_jaccards: list[float] = []

    for row in per_item:
        total_tp += row["matched"]
        total_fp += row["fp"]
        total_fn += row["fn"]
        for t, counts in row["per_type"].items():
            by_type_acc[t]["tp"] += counts["tp"]
            by_type_acc[t]["fp"] += counts["fp"]
            by_type_acc[t]["fn"] += counts["fn"]
        all_jaccards.extend(row.get("grounding_jaccards", []))

    overall = _f1(total_tp, total_fp, total_fn)
    by_type = {t: _f1(c["tp"], c["fp"], c["fn"]) for t, c in by_type_acc.items()}
    grounding_acc = (sum(all_jaccards) / len(all_jaccards)) if all_jaccards else 0.0

    return {
        "n_items": len(per_item),
        "overall": {"precision": overall["precision"],
                    "recall": overall["recall"],
                    "f1": overall["f1"]},
        "by_type": by_type,
        "grounding_acc_on_matched": grounding_acc,
    }
