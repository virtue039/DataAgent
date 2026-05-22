"""Per-qid case assembly for the P5 demo.

Pure helper used by `demo/app.py` and unit-tested in
`tests/test_case_loader.py`. No Streamlit imports — kept side-effect free
so it can be re-used from notebooks and tests.

See `docs/superpowers/specs/2026-05-22-p5-demo-design.md` §3.3 for the
return-dict contract.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default on-disk locations.
DEFAULT_P3_DIR = Path("results/p3_matrix")
DEFAULT_P4_DIR = Path("results/p4_cv")
DEFAULT_JOINT_GRAPHS_DIR = Path("data/joint_graphs_dev_cv")
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"


def load_case(
    qid: int,
    *,
    p3_dir: Path | str = DEFAULT_P3_DIR,
    p4_dir: Path | str = DEFAULT_P4_DIR,
    joint_graphs_dir: Path | str = DEFAULT_JOINT_GRAPHS_DIR,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    top_k: int = 5,
) -> dict[str, Any]:
    """Load a single BIRD dev qid + its 4-setting predictions + joint subgraph.

    Reads the latest (most-recent-timestamp) result JSONs for each setting
    under `p3_dir` (none/retrieval/oracle) and `p4_dir` (joint), then
    re-runs joint retrieval in-process to recover the actual subgraph node
    ids for visualization.

    Raises:
        KeyError: if `qid` is not present in any of the 4 result JSONs.
    """
    p3_dir = Path(p3_dir)
    p4_dir = Path(p4_dir)
    joint_graphs_dir = Path(joint_graphs_dir)

    none_row = _row_for_qid(_latest(p3_dir, "none_*.json"), qid)
    retrieval_row = _row_for_qid(_latest(p3_dir, "retrieval_*.json"), qid)
    oracle_row = _row_for_qid(_latest(p3_dir, "oracle_*.json"), qid)
    joint_row = _row_for_qid(_latest(p4_dir, "joint_*.json"), qid)

    rows = {
        "none": none_row,
        "retrieval": retrieval_row,
        "oracle": oracle_row,
        "joint": joint_row,
    }
    if all(r is None for r in rows.values()):
        raise KeyError(
            f"qid={qid} not found in any of "
            f"{p3_dir}/{{none,retrieval,oracle}}_*.json or {p4_dir}/joint_*.json"
        )

    # Pick the first non-None row for the question metadata.
    canonical = next(r for r in rows.values() if r is not None)
    db_id = canonical["db_id"]
    question = canonical["question"]
    difficulty = canonical.get("difficulty", "unknown")
    gold_sql = canonical.get("gold_sql", "")

    settings = {
        name: _settings_entry(row) for name, row in rows.items()
    }

    subgraph = _load_subgraph(
        qid=qid,
        question=question,
        db_id=db_id,
        joint_graphs_dir=joint_graphs_dir,
        embedding_model=embedding_model,
        top_k=top_k,
    )

    return {
        "qid": qid,
        "db_id": db_id,
        "difficulty": difficulty,
        "question": question,
        "gold_sql": gold_sql,
        "settings": settings,
        "subgraph": subgraph,
    }


def _settings_entry(row: dict | None) -> dict[str, Any]:
    if row is None:
        return {"evidence": "", "predicted_sql": "", "correct": False,
                "missing": True}
    return {
        "evidence": row.get("evidence", "") or "",
        "predicted_sql": row.get("predicted_sql", "") or "",
        "correct": bool(row.get("correct", False)),
        "missing": False,
    }


def _latest(dir_: Path, pattern: str) -> Path | None:
    """Return the most-recent file matching `pattern` (lex sort suffices for
    `_YYYYMMDD_HHMMSS.json` style names). Returns None if no match."""
    if not dir_.exists():
        return None
    matches = sorted(dir_.glob(pattern))
    return matches[-1] if matches else None


def _row_for_qid(path: Path | None, qid: int) -> dict | None:
    if path is None or not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("results", []) or []:
        if row.get("question_id") == qid:
            return row
    return None


def _load_subgraph(
    *,
    qid: int,
    question: str,
    db_id: str,
    joint_graphs_dir: Path,
    embedding_model: str,
    top_k: int,
) -> dict[str, Any]:
    """Re-run joint retrieval for `question` against `db_id`'s graph and
    pack the result into the {nodes, edges} demo shape (incl. L2 column
    nodes for any grounded refs).
    """
    graph_path = Path(joint_graphs_dir) / f"{db_id}.json"
    if not graph_path.exists():
        return {"nodes": [], "edges": []}

    graph = _load_joint_graph(graph_path)

    retriever = _make_retriever(
        {db_id: graph},
        embedding_model_name=embedding_model,
        cache_dir=Path(joint_graphs_dir),
    )
    retrieved = retriever.retrieve(question, db_id, top_k=top_k)

    return _assemble_subgraph(retrieved)


def _assemble_subgraph(retrieved: list[dict]) -> dict[str, Any]:
    """Flatten retrieved L1 nodes into {nodes, edges}, adding synthetic L2
    Column nodes for any grounded (table, column) pairs."""
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()

    for node in retrieved:
        nid = node.get("id")
        if nid is None or nid in seen_ids:
            continue
        seen_ids.add(nid)
        nodes.append({
            "id": nid,
            "type": node.get("type"),
            "name": node.get("name") or nid,
            "expression": node.get("expression"),
            "condition": node.get("condition"),
            "value": node.get("value"),
            "table": node.get("table"),
            "column": node.get("column"),
        })

    # Subgraph edges between L1 nodes already present in the retrieved set.
    by_id = {n["id"]: n for n in retrieved if n.get("id")}

    for node in retrieved:
        src = node.get("id")
        if not src:
            continue
        # defined_by edge
        target = node.get("defined_by")
        if target and target in by_id:
            edges.append({"src": src, "dst": target, "kind": "defined_by"})
        # depends_on edges
        for dep in node.get("depends_on", []) or []:
            if dep in by_id:
                edges.append({"src": src, "dst": dep, "kind": "depends_on"})

    # Synthetic L2 column nodes + 'grounds' edges for any grounded refs.
    added_col_ids: set[str] = set()
    for node in retrieved:
        src = node.get("id")
        if not src:
            continue
        grounding_refs: list[tuple[str, str]] = []
        # `grounding` entries
        for g in node.get("grounding", []) or []:
            t, c = g.get("table"), g.get("column")
            if t and c:
                grounding_refs.append((t, c))
        # ColumnAlias `bindings`
        for b in node.get("bindings", []) or []:
            t, c = b.get("table"), b.get("column")
            if t and c:
                grounding_refs.append((t, c))
        # ValueMap (table, column)
        if node.get("type") == "ValueMap":
            t, c = node.get("table"), node.get("column")
            if t and c:
                grounding_refs.append((t, c))

        for (t, c) in dict.fromkeys(grounding_refs):
            col_id = f"col:{t}.{c}"
            if col_id not in seen_ids:
                seen_ids.add(col_id)
                added_col_ids.add(col_id)
                nodes.append({
                    "id": col_id,
                    "type": "Column",
                    "name": f"{t}.{c}",
                    "table": t,
                    "column": c,
                })
            edges.append({"src": src, "dst": col_id, "kind": "grounds"})

    return {"nodes": nodes, "edges": edges}


def _load_joint_graph(path: Path):
    """Load a JointGraph JSON file. Indirection lets tests monkeypatch."""
    from bird_eval.joint_graph import load_graph
    return load_graph(path)


def _make_retriever(graphs, *, embedding_model_name: str, cache_dir: Path):
    """Instantiate a JointRetriever. Indirection lets tests monkeypatch
    so the heavy sentence-transformers import isn't triggered in unit tests."""
    from bird_eval.joint_retrieval import JointRetriever
    return JointRetriever(
        graphs,
        embedding_model_name=embedding_model_name,
        cache_dir=cache_dir,
    )
