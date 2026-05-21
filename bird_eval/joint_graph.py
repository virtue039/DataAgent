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

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .ddl import parse_ddl_typed, parse_fks
from .extraction import sanitize_grounding
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


def _rewrite_refs(
    canonical_nodes: dict[str, dict],
    provenance: dict[str, tuple[tuple[int, str], ...]],
    id_remap: dict[tuple[int, str], str],
) -> dict[str, dict]:
    """Rewrite each node's defined_by and depends_on entries to canonical ids.

    For a canonical node that absorbed (qid, original_id) entries, each local
    ref like defined_by='r1' is resolved against (qid, 'r1') in id_remap.
    Unresolvable refs are dropped from that node's record.
    """
    out: dict[str, dict] = {}
    for canonical_id, node in canonical_nodes.items():
        nc = dict(node)
        prov = provenance[canonical_id]

        if "defined_by" in nc:
            old = nc["defined_by"]
            new = None
            for qid, _orig in prov:
                if (qid, old) in id_remap:
                    new = id_remap[(qid, old)]
                    break
            if new is not None:
                nc["defined_by"] = new
            else:
                del nc["defined_by"]

        if "depends_on" in nc:
            new_deps: list[str] = []
            for old in nc["depends_on"]:
                resolved = None
                for qid, _orig in prov:
                    if (qid, old) in id_remap:
                        resolved = id_remap[(qid, old)]
                        break
                if resolved is not None and resolved not in new_deps:
                    new_deps.append(resolved)
            if new_deps:
                nc["depends_on"] = new_deps
            else:
                del nc["depends_on"]

        out[canonical_id] = nc
    return out


def _drop_hallucinated(nodes: dict[str, dict], ddl: str) -> dict[str, dict]:
    """Drop hallucinated (table, column) refs from L1 nodes using P1a's sanitizer.

    Nodes that lose all grounding/bindings are removed entirely. Any incoming
    `defined_by`/`depends_on` refs to removed nodes are then cleaned up so the
    remaining nodes are referentially closed.
    """
    # P1a's sanitizer takes a list and returns (list, stats); we route via list.
    as_list = list(nodes.values())
    cleaned_list, _stats = sanitize_grounding(as_list, ddl)
    surviving_ids = {n["id"] for n in cleaned_list}
    cleaned: dict[str, dict] = {n["id"]: n for n in cleaned_list}

    # Clean up incoming refs that now point at removed nodes.
    for nid, n in list(cleaned.items()):
        if "defined_by" in n and n["defined_by"] not in surviving_ids:
            n = dict(n)
            del n["defined_by"]
            cleaned[nid] = n
        if "depends_on" in n:
            kept = [d for d in n["depends_on"] if d in surviving_ids]
            if kept != n.get("depends_on"):
                n = dict(n)
                if kept:
                    n["depends_on"] = kept
                else:
                    del n["depends_on"]
                cleaned[nid] = n

    return cleaned


_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def _tokenize(text: str | None) -> list[str]:
    """Lowercase, split on non-word characters, drop tokens < 2 chars and stopwords."""
    if not text:
        return []
    tokens: list[str] = []
    for t in _TOKEN_SPLIT_RE.split(text.casefold()):
        if len(t) >= 2 and t not in _STOPWORDS:
            tokens.append(t)
    return tokens


def _build_indexes(
    nodes: dict[str, dict],
) -> dict[str, dict]:
    """Build by_type, by_column, by_name_token indexes from canonical nodes.

    Returns a dict with three sub-dicts (each value tuple-of-ids for immutability):
      - by_type[<type>]              -> (id, ...)
      - by_column[(table, column)]   -> (id, ...)
      - by_name_token[<token>]       -> (id, ...)
    """
    by_type_lst: dict[str, list[str]] = {t: [] for t in _TYPE_LETTER}
    by_column_lst: dict[tuple[str, str], list[str]] = {}
    by_name_token_lst: dict[str, list[str]] = {}

    for nid, n in nodes.items():
        ntype = n.get("type")
        if ntype in by_type_lst:
            by_type_lst[ntype].append(nid)

        # by_column: ValueMap (table, column); ColumnAlias bindings; grounding entries
        column_refs: list[tuple[str, str]] = []
        if ntype == "ValueMap":
            t, c = n.get("table"), n.get("column")
            if t and c:
                column_refs.append((t, c))
        for b in n.get("bindings", []) or []:
            t, c = b.get("table"), b.get("column")
            if t and c:
                column_refs.append((t, c))
        for g in n.get("grounding", []) or []:
            t, c = g.get("table"), g.get("column")
            if t and c:
                column_refs.append((t, c))
        # deduplicate column_refs first so the same nid doesn't appear
        # multiple times for one (table, column) when a node has multiple
        # grounding entries for the same column (e.g. bare + value-bearing).
        for ref in dict.fromkeys(column_refs):
            by_column_lst.setdefault(ref, []).append(nid)

        # by_name_token: tokenize name + condition (Rule) + expression (Formula).
        tokens: set[str] = set()
        for field in ("name", "condition", "expression"):
            tokens.update(_tokenize(n.get(field)))
        for t in tokens:
            by_name_token_lst.setdefault(t, []).append(nid)

    return {
        "by_type": {t: tuple(ids) for t, ids in by_type_lst.items()},
        "by_column": {k: tuple(ids) for k, ids in by_column_lst.items()},
        "by_name_token": {k: tuple(ids) for k, ids in by_name_token_lst.items()},
    }


def build_graph(
    items: list[dict],
    schemas: dict[str, str],
) -> dict[str, JointGraph]:
    """Build per-DB JointGraphs from v2 records + DDL.

    items: each dict has keys 'question_id', 'db_id', 'nodes' (v2 typed-node dicts).
    schemas: {db_id: ddl_string}.

    Returns {db_id: JointGraph}.

    Raises KeyError when items reference a db_id that's not in schemas.
    """
    # group items by db_id
    per_db: dict[str, list[dict]] = {}
    for it in items:
        per_db.setdefault(it["db_id"], []).append(it)

    out: dict[str, JointGraph] = {}
    for db_id, db_items in per_db.items():
        if db_id not in schemas:
            raise KeyError(f"missing schema for db_id={db_id!r}")
        ddl = schemas[db_id]

        # L1
        canonical, prov, remap = _dedup_l1(db_items)
        rewritten = _rewrite_refs(canonical, prov, remap)
        cleaned = _drop_hallucinated(rewritten, ddl)
        # Restrict provenance to nodes that survived the sanity drop.
        prov_clean = {nid: prov[nid] for nid in cleaned if nid in prov}

        # L2
        columns = parse_ddl_typed(ddl)
        fk_edges = parse_fks(ddl)

        # Indexes
        idx = _build_indexes(cleaned)

        out[db_id] = JointGraph(
            db_id=db_id,
            nodes=cleaned,
            columns=columns,
            fk_edges=fk_edges,
            by_type=idx["by_type"],
            by_column=idx["by_column"],
            by_name_token=idx["by_name_token"],
            provenance=prov_clean,
        )
    return out


_SERIALIZATION_VERSION = "p1b.1"


def _col_key(table: str, column: str) -> str:
    """JSON-safe key for (table, column) used in by_column.

    Uses '.' as the separator -- adequate for SQLite identifiers that don't
    contain '.'. See spec §6 R4. If a table name DOES contain a '.', we raise
    a clear error rather than silently corrupt the index.
    """
    if "." in table:
        raise ValueError(f"table name contains a dot, not supported: {table!r}")
    return f"{table}.{column}"


def _col_key_inv(key: str) -> tuple[str, str]:
    table, _, column = key.partition(".")
    return (table, column)


def dump_graph(graph: JointGraph, path) -> None:
    """Serialize a JointGraph to a JSON file. Idempotent / overwrites the file."""
    payload = {
        "version": _SERIALIZATION_VERSION,
        "db_id": graph.db_id,
        "nodes": graph.nodes,
        "columns": [
            {"table": t, "column": c, "type": v.get("type")}
            for (t, c), v in graph.columns.items()
        ],
        "fk_edges": [
            {"from": {"table": e[0][0], "column": e[0][1]},
             "to":   {"table": e[1][0], "column": e[1][1]}}
            for e in graph.fk_edges
        ],
        "indexes": {
            "by_type": {t: list(ids) for t, ids in graph.by_type.items()},
            "by_column": {_col_key(t, c): list(ids) for (t, c), ids in graph.by_column.items()},
            "by_name_token": {k: list(ids) for k, ids in graph.by_name_token.items()},
        },
        "provenance": {
            nid: [list(p) for p in entries]
            for nid, entries in graph.provenance.items()
        },
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def load_graph(path) -> JointGraph:
    """Inverse of dump_graph: reconstruct a JointGraph from a JSON file."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if d.get("version") != _SERIALIZATION_VERSION:
        raise ValueError(
            f"unsupported joint-graph serialization version {d.get('version')!r}; "
            f"expected {_SERIALIZATION_VERSION!r}"
        )
    columns: dict[tuple[str, str], dict] = {
        (c["table"], c["column"]): {"type": c.get("type")} for c in d["columns"]
    }
    fk_edges = tuple(
        ((e["from"]["table"], e["from"]["column"]),
         (e["to"]["table"], e["to"]["column"]))
        for e in d["fk_edges"]
    )
    by_type = {t: tuple(ids) for t, ids in d["indexes"]["by_type"].items()}
    by_column = {_col_key_inv(k): tuple(ids) for k, ids in d["indexes"]["by_column"].items()}
    by_name_token = {k: tuple(ids) for k, ids in d["indexes"]["by_name_token"].items()}
    provenance = {
        nid: tuple(tuple(p) for p in entries)
        for nid, entries in d["provenance"].items()
    }
    return JointGraph(
        db_id=d["db_id"],
        nodes=d["nodes"],
        columns=columns,
        fk_edges=fk_edges,
        by_type=by_type,
        by_column=by_column,
        by_name_token=by_name_token,
        provenance=provenance,
    )
