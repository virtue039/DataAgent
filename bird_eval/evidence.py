"""Evidence providers for the three evaluation settings.

Research_Plan.md section six:
  none      -> lower bound (no business knowledge supplied)
  oracle    -> upper bound (BIRD's per-question gold evidence)
  retrieval -> flat-RAG baseline: top-k retrieval over a KB built from BIRD
               train evidence (the KAT-SQL-style baseline, Positioning_Note.md D1/D2)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .data import BirdExample

if TYPE_CHECKING:
    from .config import Config


class NoEvidence:
    name = "none"

    def get(self, example: BirdExample) -> str:
        return ""


class OracleEvidence:
    name = "oracle"

    def get(self, example: BirdExample) -> str:
        return example.evidence


class RetrievalEvidence:
    """Flat-RAG: retrieve top-k knowledge entries by embedding similarity.

    The KB is built from BIRD *train* evidence only, so it never overlaps with
    dev databases (the non-overlap setting, Research_Plan.md OQ2). Retrieved
    entries may reference other databases' columns -- that noise is exactly
    what the joint-graph method is meant to beat.
    """

    name = "retrieval"

    def __init__(self, retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def get(self, example: BirdExample) -> str:
        hits = self._retriever.retrieve(example.question, self._top_k)
        return "\n".join(f"- {hit.text}" for hit in hits)


class JointRetrievalEvidence:
    """P1d joint-graph evidence provider.

    Uses a JointRetriever to fetch a minimal subgraph (top-k dense seeds +
    BFS edge expansion) and renders it as a bullet text block that drops
    directly into the evidence slot of the eval prompt.

    See docs/superpowers/specs/2026-05-21-p1d-joint-retrieval-design.md.
    """

    name = "joint"

    def __init__(self, retriever, top_k: int) -> None:
        self._retriever = retriever
        self._top_k = top_k

    def get(self, example: BirdExample) -> str:
        from .joint_retrieval import _render_subgraph_as_evidence
        nodes = self._retriever.retrieve(example.question, example.db_id, self._top_k)
        return _render_subgraph_as_evidence(nodes)


def make_evidence_provider(config: "Config"):
    """Build the evidence provider for the configured setting."""
    if config.setting == "none":
        return NoEvidence()
    if config.setting == "oracle":
        return OracleEvidence()
    if config.setting == "retrieval":
        # imported lazily so the none/oracle paths need no ML dependencies
        from .knowledge_base import load_kb
        from .retrieval import EmbeddingRetriever

        entries = load_kb(config.kb_path)
        retriever = EmbeddingRetriever(entries, config.embedding_model, kb_path=config.kb_path)
        return RetrievalEvidence(retriever, config.retrieval_top_k)
    if config.setting == "joint":
        # imported lazily so the none/oracle/retrieval paths need no ML/graph deps
        from .joint_graph import JointGraph, load_graph
        from .joint_retrieval import JointRetriever

        graph_paths = sorted(config.joint_graphs_dir.glob("*.json"))
        if not graph_paths:
            raise FileNotFoundError(
                f"No joint graphs at {config.joint_graphs_dir}. "
                f"Build them first with: .venv/bin/python build_joint_graph.py "
                f"--output-dir {config.joint_graphs_dir}"
            )
        graphs: dict[str, JointGraph] = {}
        for p in graph_paths:
            g = load_graph(p)
            if g.db_id in graphs:
                raise ValueError(
                    f"Duplicate db_id {g.db_id!r}: {p} and a previous file "
                    f"both claim it. Each joint graph file must hold a unique db_id."
                )
            graphs[g.db_id] = g
        retriever = JointRetriever(
            graphs, config.embedding_model, cache_dir=config.joint_graphs_dir
        )
        return JointRetrievalEvidence(retriever, config.retrieval_top_k)
    raise ValueError(f"Unknown setting: {config.setting}")
