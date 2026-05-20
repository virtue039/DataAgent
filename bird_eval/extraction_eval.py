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
        return _norm(a.get("name")) == _norm(b.get("name"))
    if t == "Rule":
        ka = _norm(a.get("name")) or _norm(a.get("condition"))
        kb = _norm(b.get("name")) or _norm(b.get("condition"))
        return bool(ka) and ka == kb
    return False
