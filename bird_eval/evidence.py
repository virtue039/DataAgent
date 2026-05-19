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
    raise ValueError(f"Unknown setting: {config.setting}")
