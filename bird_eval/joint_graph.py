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
