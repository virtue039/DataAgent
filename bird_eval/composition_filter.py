"""P5-D3 composition-query KB ablation: strip "composite" nodes.

A composite node is any L1 node whose `depends_on` field exists and has
length >= 2 — these are Formulas (or Rules) that pre-bundle two or more
atom concepts into a single canonical record. Removing them forces the
retriever to *assemble* the atoms itself across `depends_on` edges, which
is the open-world composition capability (D3 in the positioning note)
under test.

`strip_composites` preserves all dependency targets (the atoms), cleans
inbound `defined_by` / `depends_on` references that pointed at removed
nodes, rebuilds the three node indexes, and trims provenance — same
referential-closure logic as `joint_graph._drop_hallucinated`.

See docs/superpowers/specs/2026-05-22-p5-d3-composition-specialty-design.md
§3.2 for design rationale.
"""
from __future__ import annotations

from .joint_graph import JointGraph, _build_indexes


def _is_composite(node: dict) -> bool:
    """A node is a composite iff `depends_on` exists and has length >= 2."""
    deps = node.get("depends_on")
    return isinstance(deps, list) and len(deps) >= 2


def strip_composites(graph: JointGraph) -> JointGraph:
    """Return a new JointGraph with all composite nodes removed.

    A composite is any node whose `depends_on` list has length >= 2.
    Dependency targets are preserved; inbound `defined_by` / `depends_on`
    references that now point at removed nodes are cleaned up; the
    `by_type`, `by_column`, `by_name_token` indexes are rebuilt from the
    surviving nodes; `provenance` is restricted to surviving ids.

    `db_id`, `columns`, `fk_edges`, and `match_signature_hash` are passed
    through unchanged — the L2 layer is a property of the DDL, not the L1
    extraction.
    """
    removed: set[str] = {
        nid for nid, node in graph.nodes.items() if _is_composite(node)
    }
    if not removed:
        # Nothing to strip — return a structurally identical graph.
        return JointGraph(
            db_id=graph.db_id,
            nodes=dict(graph.nodes),
            columns=graph.columns,
            fk_edges=graph.fk_edges,
            by_type=graph.by_type,
            by_column=graph.by_column,
            by_name_token=graph.by_name_token,
            provenance=graph.provenance,
            match_signature_hash=graph.match_signature_hash,
        )

    surviving: dict[str, dict] = {}
    for nid, node in graph.nodes.items():
        if nid in removed:
            continue
        # Copy the node so we don't mutate the input.
        nc = dict(node)

        # Clean stale defined_by referencing a removed node.
        if "defined_by" in nc and nc["defined_by"] in removed:
            del nc["defined_by"]

        # Clean stale depends_on entries referencing removed nodes.
        if "depends_on" in nc:
            kept = [d for d in nc["depends_on"] if d not in removed]
            if kept != nc.get("depends_on"):
                if kept:
                    nc["depends_on"] = kept
                else:
                    del nc["depends_on"]

        surviving[nid] = nc

    # Rebuild indexes from scratch over the surviving nodes.
    idx = _build_indexes(surviving)

    # Restrict provenance to surviving ids.
    provenance = {
        nid: entries for nid, entries in graph.provenance.items()
        if nid in surviving
    }

    return JointGraph(
        db_id=graph.db_id,
        nodes=surviving,
        columns=graph.columns,
        fk_edges=graph.fk_edges,
        by_type=idx["by_type"],
        by_column=idx["by_column"],
        by_name_token=idx["by_name_token"],
        provenance=provenance,
        match_signature_hash=graph.match_signature_hash,
    )
