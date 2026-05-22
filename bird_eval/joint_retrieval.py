"""P1d joint subgraph retrieval over per-DB JointGraphs.

Public surface:
- JointRetriever        : dense-seed retrieval + 1-hop edge expansion.

See docs/superpowers/specs/2026-05-21-p1d-joint-retrieval-design.md for the
contract and the rationale.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .joint_graph import JointGraph


def _search_text_for(node: dict, graph: JointGraph, _visited: frozenset = frozenset()) -> str:
    """Build the dense-embedding "search text" for an L1 node.

    Composition per type (see spec §3.1):
      Concept     -> name + " | " + defined_by_target.search_text (if set)
      Formula     -> name + " | " + expression
      Rule        -> (name if name else "") + " | " + condition
      ValueMap    -> name + " | " + table + "." + column + " = " + value
      ColumnAlias -> name + " | " + bindings[0].table + "." + bindings[0].column

    `_visited` is used internally to break Concept->Concept cycles.
    """
    ntype = node.get("type")
    name = (node.get("name") or "").strip()

    if ntype == "Formula":
        expr = (node.get("expression") or "").strip()
        return f"{name} | {expr}" if expr else name

    if ntype == "Rule":
        cond = (node.get("condition") or "").strip()
        return f"{name} | {cond}" if cond else (name or cond)

    if ntype == "ValueMap":
        t = node.get("table") or ""
        c = node.get("column") or ""
        v = node.get("value") or ""
        return f"{name} | {t}.{c} = {v}"

    if ntype == "ColumnAlias":
        bindings = node.get("bindings") or []
        if bindings:
            b = bindings[0]
            return f"{name} | {b.get('table', '')}.{b.get('column', '')}"
        return name

    if ntype == "Concept":
        targ = node.get("defined_by")
        nid = node.get("id")
        if targ and targ in graph.nodes and nid not in _visited:
            inner = _search_text_for(
                graph.nodes[targ], graph, _visited | {nid} if nid else _visited
            )
            return f"{name} | {inner}"
        return name

    # Unknown type: return name
    return name


class JointRetriever:
    """Joint subgraph retrieval over per-DB JointGraphs (P1d).

    Construction-time work (one-shot per graph):
      For each db_id graph:
        - Build a search text per L1 node via _search_text_for.
        - Encode with sentence-transformers MPNet, normalize_embeddings=True.
        - Optionally cache the resulting (N, dim) array to disk for reuse.

    Query-time work:
      cosine-similarity (dot, since normalized) → top_k seeds → BFS expand
      via defined_by + depends_on → return v2 node dicts in deterministic
      order.
    """

    def __init__(
        self,
        graphs: dict[str, JointGraph],
        embedding_model_name: str,
        cache_dir: Path | None = None,
    ) -> None:
        self._graphs = graphs
        self._encode_lock = threading.Lock()  # ST models aren't thread-safe

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(embedding_model_name)
        self._model_name = embedding_model_name

        # Per-graph: ordered list of node_ids matching the row order of _embeddings.
        self._node_ids: dict[str, list[str]] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        for db_id, g in graphs.items():
            ids = list(g.nodes.keys())
            self._node_ids[db_id] = ids
            self._embeddings[db_id] = self._load_or_compute(db_id, g, ids, cache_dir)

    def _load_or_compute(
        self,
        db_id: str,
        graph: JointGraph,
        ids: list[str],
        cache_dir: Path | None,
    ) -> np.ndarray:
        cache_path = self._cache_path(
            db_id, len(ids), graph.match_signature_hash, cache_dir
        )
        if cache_path is not None and cache_path.exists():
            return np.load(cache_path)

        if not ids:
            dim = self._model.get_sentence_embedding_dimension()
            arr = np.zeros((0, dim), dtype=np.float32)
        else:
            texts = [_search_text_for(graph.nodes[nid], graph) for nid in ids]
            arr = self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
            )
            arr = np.asarray(arr, dtype=np.float32)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, arr)
        return arr

    def retrieve(self, query: str, db_id: str, top_k: int = 5,
                 expand: bool = True) -> list[dict]:
        """Return the joint-retrieval subgraph for one (query, db_id) pair.

        Pipeline:
        - Encode the query.
        - Cosine top-k against this db's pre-encoded L1 nodes.
        - If `expand=True` (default): BFS expand each seed via defined_by +
          depends_on edges. The walk is unbounded but cycle-safe (seen-set
          blocks revisits), and the joint graph's edge structure naturally
          bounds subgraph size to a handful of nodes per query in practice.
        - If `expand=False` (P5-D2 ablation, see
          docs/superpowers/specs/2026-05-22-p5-d2-no-bfs-ablation-design.md):
          skip the BFS entirely and return only the top-k seed nodes by
          similarity. Used to isolate D1 (typed-node representation) from
          D2 (BFS edge expansion) contribution to the +5.93pp joint gain.
        - Return v2 node dicts in deterministic order (seeds first by
          similarity, expansion targets next in BFS order when expand=True).

        Returns [] if top_k <= 0 or db_id is not in the KB.
        """
        if db_id not in self._graphs:
            return []
        ids = self._node_ids.get(db_id, [])
        if not ids:
            return []
        if top_k <= 0:
            return []

        with self._encode_lock:
            q = self._model.encode([query], normalize_embeddings=True)[0]
        q = np.asarray(q, dtype=np.float32)

        scores = self._embeddings[db_id] @ q  # cosine since normalized
        k = min(top_k, len(ids))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        seed_ids = [ids[i] for i in top_idx]
        graph = self._graphs[db_id]

        if not expand:
            # No-BFS ablation: seeds only, no defined_by / depends_on closure.
            return [graph.nodes[nid] for nid in seed_ids if nid in graph.nodes]

        # BFS expansion via defined_by + depends_on.
        ordered: list[str] = []
        seen: set[str] = set()
        work = list(seed_ids)
        while work:
            nid = work.pop(0)
            if nid in seen or nid not in graph.nodes:
                continue
            seen.add(nid)
            ordered.append(nid)
            node = graph.nodes[nid]
            targ = node.get("defined_by")
            if targ and targ not in seen:
                work.append(targ)
            for dep in node.get("depends_on", []) or []:
                if dep not in seen:
                    work.append(dep)

        return [graph.nodes[nid] for nid in ordered]

    def _cache_path(self, db_id: str, n: int, sig_hash: str, cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None
        safe_model = self._model_name.replace("/", "_")
        sig = (sig_hash or "nosig")[:12]
        return Path(cache_dir) / f"{db_id}.{safe_model}.{sig}.{n}.npy"


def _render_subgraph_as_evidence(nodes: list[dict]) -> str:
    """Render a list of v2 typed-node dicts as a bullet text block for the
    LLM evidence prompt slot.

    Empty input returns the empty string. Otherwise the output begins with
    a header line that the eval prompt instruction can reference.
    """
    if not nodes:
        return ""
    # Build an id->node lookup so we can render cross-refs (defined_by /
    # depends_on) by name when the target is also in the subgraph.
    by_id = {n.get("id"): n for n in nodes if n.get("id") is not None}

    lines: list[str] = ["Relevant knowledge for this question:"]
    for n in nodes:
        ntype = n.get("type")
        name = n.get("name") or n.get("id") or "?"
        if ntype == "ColumnAlias":
            bindings = n.get("bindings", []) or []
            if bindings:
                cols = ", ".join(f"{b.get('table') or '?'}.{b.get('column') or '?'}" for b in bindings)
                lines.append(f"- ColumnAlias: {name} -> ({cols})")
            else:
                lines.append(f"- ColumnAlias: {name}")
        elif ntype == "ValueMap":
            t = n.get("table") or "?"
            c = n.get("column") or "?"
            v = n.get("value") or "?"
            lines.append(f"- ValueMap: {name} -> {t}.{c} = '{v}'")
        elif ntype == "Concept":
            targ = n.get("defined_by")
            if targ and targ in by_id:
                tname = by_id[targ].get("name") or targ
                lines.append(f"- Concept: {name} (defined by {by_id[targ].get('type')} \"{tname}\")")
            else:
                lines.append(f"- Concept: {name}")
        elif ntype == "Rule":
            cond = n.get("condition") or ""
            grounding = ", ".join(
                f"{g.get('table') or '?'}.{g.get('column') or '?'}"
                for g in (n.get("grounding") or [])
            )
            tail = f" [grounded: {grounding}]" if grounding else ""
            if cond:
                lines.append(f"- Rule: {name} -> {cond}{tail}")
            else:
                lines.append(f"- Rule: {name}{tail}")
        elif ntype == "Formula":
            expr = n.get("expression") or ""
            grounding = ", ".join(
                f"{g.get('table') or '?'}.{g.get('column') or '?'}"
                for g in (n.get("grounding") or [])
            )
            tail = f" [grounded: {grounding}]" if grounding else ""
            deps = n.get("depends_on") or []
            dep_tail = ""
            if deps:
                dep_names = []
                for d in deps:
                    if d in by_id:
                        dn = by_id[d].get("name") or d
                        dep_names.append(f"{by_id[d].get('type')} \"{dn}\"")
                    else:
                        dep_names.append(d)
                dep_tail = f" [depends on: {', '.join(dep_names)}]"
            expr_part = f" = {expr}" if expr else ""
            lines.append(f"- Formula: {name}{expr_part}{tail}{dep_tail}")
        else:
            lines.append(f"- {ntype or '?'}: {name}")

    return "\n".join(lines)
