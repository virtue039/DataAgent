"""Dense-embedding retrieval over the flat-RAG knowledge base.

This is the KAT-SQL-style baseline (Positioning_Note.md D1/D2): knowledge is a
flat set of text entries, retrieved purely by embedding similarity with no
structure linking knowledge to schema. The joint-graph method is meant to beat
this baseline -- so the baseline must itself be reasonably strong, hence a
sentence-transformers model (KAT-SQL used MPNet) rather than lexical matching.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .knowledge_base import KBEntry


class EmbeddingRetriever:
    """Embed the KB once, then retrieve top-k entries by cosine similarity."""

    def __init__(
        self, entries: list[KBEntry], model_name: str, kb_path: Path | None = None
    ) -> None:
        if not entries:
            raise ValueError("Knowledge base is empty -- nothing to retrieve from.")
        self.entries = entries
        self._encode_lock = threading.Lock()  # ST models are not encode-thread-safe

        # imported here so importing this module stays cheap until a retriever is built
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._embeddings = self._load_or_compute(model_name, kb_path)

    def _cache_path(self, model_name: str, kb_path: Path | None) -> Path | None:
        if kb_path is None:
            return None
        kb_path = Path(kb_path)
        safe_model = model_name.replace("/", "_")
        return kb_path.with_name(f"{kb_path.stem}.{safe_model}.{len(self.entries)}.npy")

    def _load_or_compute(self, model_name: str, kb_path: Path | None) -> np.ndarray:
        cache = self._cache_path(model_name, kb_path)
        if cache is not None and cache.exists():
            print(f"Loading cached KB embeddings from {cache}")
            return np.load(cache)

        print(f"Embedding {len(self.entries)} KB entries with '{model_name}' ...")
        texts = [e.text for e in self.entries]
        embeddings = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, embeddings)
            print(f"Cached KB embeddings to {cache}")
        return embeddings

    def retrieve(self, query: str, top_k: int) -> list[KBEntry]:
        with self._encode_lock:
            query_vec = self._model.encode([query], normalize_embeddings=True)[0]
        query_vec = np.asarray(query_vec, dtype=np.float32)

        scores = self._embeddings @ query_vec  # cosine sim (vectors are normalized)
        k = min(top_k, len(self.entries))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self.entries[i] for i in top]
