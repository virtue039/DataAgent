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


def _norm(s) -> str:
    """Normalize a comparison key. Tolerates None and non-string values
    (e.g. an int ValueMap.value emitted by the LLM); coerces to str first.
    """
    if s is None:
        return ""
    return str(s).strip().casefold()


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


def _resolve_through_pairs(
    source_list: list[dict],
    target_list: list[dict],
    ref_id: str,
    pairs: list[tuple[int, int]],
    source_is_predicted: bool,
) -> int | None:
    """Look up a local node id in source_list, then find the matched
    counterpart's index in target_list via pairs.

    Returns the target-side index, or None if the ref doesn't resolve.
    """
    for src_idx, n in enumerate(source_list):
        if n.get("id") == ref_id:
            for p_idx, g_idx in pairs:
                src_match, tgt_match = (p_idx, g_idx) if source_is_predicted else (g_idx, p_idx)
                if src_match == src_idx:
                    return tgt_match
            return None
    return None


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
        "matched_pairs": list[list[int]]    # per-pair [pred_idx, gold_idx]
        "edge_pair_correct": dict          # {"defined_by": list[bool]}
        "edge_pair_jaccard": dict          # {"depends_on": list[float]}
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

    # F3b: per-pair edge accuracy data
    defined_by_correct: list[bool] = []
    depends_on_jaccards: list[float] = []

    for i, j in pairs:
        ntype = predicted[i].get("type")
        if ntype == "Concept":
            pred_def = predicted[i].get("defined_by")
            gold_def = gold[j].get("defined_by")
            if pred_def is None and gold_def is None:
                defined_by_correct.append(True)
            elif pred_def is None or gold_def is None:
                defined_by_correct.append(False)
            else:
                pred_target_in_gold = _resolve_through_pairs(
                    predicted, gold, pred_def, pairs, source_is_predicted=True,
                )
                gold_target_idx = next(
                    (gi for gi, n in enumerate(gold) if n.get("id") == gold_def),
                    None,
                )
                defined_by_correct.append(pred_target_in_gold == gold_target_idx
                                          and pred_target_in_gold is not None)
        if ntype == "Formula":
            pred_deps = predicted[i].get("depends_on", []) or []
            gold_deps = set(gold[j].get("depends_on", []) or [])
            pred_deps_remapped: set = set()
            for d in pred_deps:
                t_idx = _resolve_through_pairs(
                    predicted, gold, d, pairs, source_is_predicted=True,
                )
                if t_idx is not None:
                    gid = gold[t_idx].get("id")
                    if gid is not None:
                        pred_deps_remapped.add(gid)
            depends_on_jaccards.append(_jaccard(pred_deps_remapped, gold_deps))

    return {
        "matched": len(pairs),
        "fp": len(predicted) - len(pairs),
        "fn": len(gold) - len(pairs),
        "per_type": per_type,
        "grounding_jaccards": g_jaccards,
        "matched_pairs": [[i, j] for i, j in pairs],
        "edge_pair_correct": {"defined_by": defined_by_correct},
        "edge_pair_jaccard": {"depends_on": depends_on_jaccards},
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

    # F3c: edge accuracy across all items
    all_defined_by_correct: list[bool] = []
    all_depends_on_jaccards: list[float] = []

    # F3c: sanitization counters
    items_with_dropped_g = 0
    items_with_fully_dropped_n = 0

    for row in per_item:
        total_tp += row["matched"]
        total_fp += row["fp"]
        total_fn += row["fn"]
        for t, counts in row["per_type"].items():
            by_type_acc[t]["tp"] += counts["tp"]
            by_type_acc[t]["fp"] += counts["fp"]
            by_type_acc[t]["fn"] += counts["fn"]
        all_jaccards.extend(row.get("grounding_jaccards", []))

        # F3c: collect edge info
        edge_correct = row.get("edge_pair_correct", {})
        all_defined_by_correct.extend(edge_correct.get("defined_by", []))
        edge_jaccard = row.get("edge_pair_jaccard", {})
        all_depends_on_jaccards.extend(edge_jaccard.get("depends_on", []))

        # F3c: count items with sanitization side-effects
        s_stats = row.get("sanitize_stats", {})
        if s_stats.get("dropped_groundings", 0) > 0:
            items_with_dropped_g += 1
        if s_stats.get("dropped_nodes", 0) > 0:
            items_with_fully_dropped_n += 1

    overall = _f1(total_tp, total_fp, total_fn)
    by_type = {t: _f1(c["tp"], c["fp"], c["fn"]) for t, c in by_type_acc.items()}
    grounding_acc = (sum(all_jaccards) / len(all_jaccards)) if all_jaccards else 0.0

    # F3c: edge accuracy aggregates
    defined_by_acc = (sum(all_defined_by_correct) / len(all_defined_by_correct)
                     if all_defined_by_correct else 0.0)
    depends_on_acc = (sum(all_depends_on_jaccards) / len(all_depends_on_jaccards)
                     if all_depends_on_jaccards else 0.0)

    return {
        "n_items": len(per_item),
        "overall": {"precision": overall["precision"],
                    "recall": overall["recall"],
                    "f1": overall["f1"]},
        "by_type": by_type,
        "grounding_acc_on_matched": grounding_acc,
        "edge_acc": {
            "defined_by": defined_by_acc,
            "depends_on": depends_on_acc,
        },
        "sanitization": {
            "items_with_dropped_groundings": items_with_dropped_g,
            "items_with_fully_dropped_nodes": items_with_fully_dropped_n,
        },
    }
