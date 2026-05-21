"""P1b joint-graph builder: v2 typed nodes + Schema/FK -> per-DB JointGraph.

Public surface:
- JointGraph        : the frozen dataclass holding L1 + L2 + indexes + provenance.
- build_graph(items, schemas) -> {db_id: JointGraph}
- dump_graph(graph, path)
- load_graph(path) -> JointGraph

See docs/superpowers/specs/2026-05-21-p1b-joint-graph-builder-design.md for the
contract and the rationale behind each component (dedup, sanitize, indexes).
"""
from __future__ import annotations

from dataclasses import dataclass

from .extraction_eval import match as _node_match


# Hard-coded English stopword set used by the by_name_token index. See spec §3.5.
_STOPWORDS = frozenset({
    "a", "the", "of", "and", "or", "to", "in", "for",
    "by", "is", "are", "with", "at",
})

# Canonical id prefix per node type. See spec §3.2.
_TYPE_LETTER = {
    "Concept": "c",
    "Formula": "f",
    "ValueMap": "v",
    "Rule": "r",
    "ColumnAlias": "a",
}


@dataclass(frozen=True)
class JointGraph:
    db_id: str

    # L1 layer: deduplicated v2 typed-node dicts, keyed by canonical id.
    nodes: dict[str, dict]

    # L2 layer.
    columns: dict[tuple[str, str], dict]
    fk_edges: tuple[tuple[tuple[str, str], tuple[str, str]], ...]

    # Pre-computed indexes.
    by_type: dict[str, tuple[str, ...]]
    by_column: dict[tuple[str, str], tuple[str, ...]]
    by_name_token: dict[str, tuple[str, ...]]

    # Provenance: each canonical id -> the (question_id, original_id) it absorbed.
    provenance: dict[str, tuple[tuple[int, str], ...]]


def _collect_nodes(items: list[dict]) -> list[tuple[int, dict]]:
    """Flatten (item.question_id, node) pairs across all items, ordered by
    (question_id asc, original index within record asc) for deterministic ids.
    """
    pairs: list[tuple[int, dict]] = []
    for it in sorted(items, key=lambda x: x.get("question_id", 0)):
        qid = it.get("question_id")
        for n in it.get("nodes", []) or []:
            if isinstance(n, dict) and n.get("type") in _TYPE_LETTER:
                pairs.append((qid, n))
    return pairs


def _merge_class(members: list[tuple[int, dict]], canonical_id: str) -> dict:
    """Merge nodes in one equivalence class into one canonical node.

    - grounding: union by (term, table, column, value)
    - bindings (ColumnAlias): union by (table, column)
    - depends_on: union of ids, preserving order of first appearance
    - name/definition/expression/condition/table/column/value: first non-empty
    - defined_by: first defined_by encountered (rewritten later)
    """
    ntype = members[0][1]["type"]
    merged: dict = {"id": canonical_id, "type": ntype}

    seen_g: set = set()
    grounding: list[dict] = []
    for _, n in members:
        for g in n.get("grounding", []) or []:
            key = (g.get("term"), g.get("table"), g.get("column"), g.get("value"))
            if key not in seen_g:
                seen_g.add(key)
                grounding.append(dict(g))
    if grounding:
        merged["grounding"] = grounding

    seen_b: set = set()
    bindings: list[dict] = []
    for _, n in members:
        for b in n.get("bindings", []) or []:
            key = (b.get("table"), b.get("column"))
            if key not in seen_b:
                seen_b.add(key)
                bindings.append(dict(b))
    if bindings:
        merged["bindings"] = bindings

    seen_d: set = set()
    deps: list[str] = []
    for _, n in members:
        for d in n.get("depends_on", []) or []:
            if d not in seen_d:
                seen_d.add(d)
                deps.append(d)
    if deps:
        merged["depends_on"] = deps

    for field in ("name", "definition", "expression", "condition",
                  "table", "column", "value"):
        for _, n in members:
            v = n.get(field)
            if v:
                merged[field] = v
                break

    for _, n in members:
        if "defined_by" in n:
            merged["defined_by"] = n["defined_by"]
            break

    return merged


def _dedup_l1(
    items: list[dict],
) -> tuple[dict[str, dict],
           dict[str, tuple[tuple[int, str], ...]],
           dict[tuple[int, str], str]]:
    """Group L1 nodes by match() equivalence and assign canonical ids.

    Returns (canonical_nodes, provenance, id_remap):
    - canonical_nodes: {canonical_id: merged_node_dict}
    - provenance:      {canonical_id: ((qid, original_id), ...)}
    - id_remap:        {(qid, original_id): canonical_id} for ref rewriting
    """
    pairs = _collect_nodes(items)

    # Bucket by type, then walk and group by match() equivalence.
    by_type: dict[str, list[list[tuple[int, dict]]]] = {}
    for qid, node in pairs:
        ntype = node["type"]
        buckets = by_type.setdefault(ntype, [])
        placed = False
        for bucket in buckets:
            if _node_match(node, bucket[0][1]):
                bucket.append((qid, node))
                placed = True
                break
        if not placed:
            buckets.append([(qid, node)])

    canonical_nodes: dict[str, dict] = {}
    provenance: dict[str, tuple[tuple[int, str], ...]] = {}
    id_remap: dict[tuple[int, str], str] = {}

    # Deterministic order: iterate types in a fixed order so canonical ids are stable.
    for ntype in ("Concept", "Formula", "ValueMap", "Rule", "ColumnAlias"):
        counter = 0
        for cls_members in by_type.get(ntype, []):
            counter += 1
            canonical_id = f"{_TYPE_LETTER[ntype]}{counter}"
            canonical_nodes[canonical_id] = _merge_class(cls_members, canonical_id)
            provenance[canonical_id] = tuple(
                (qid, n["id"]) for qid, n in cls_members
            )
            for qid, n in cls_members:
                id_remap[(qid, n["id"])] = canonical_id

    return canonical_nodes, provenance, id_remap
