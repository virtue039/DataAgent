"""Unit tests for bird_eval/joint_retrieval.py (P1d).

Run with:
  .venv/bin/python -m unittest tests.test_joint_retrieval -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.joint_graph import JointGraph  # noqa: E402
from bird_eval.joint_retrieval import (  # noqa: E402
    JointRetriever,
    _search_text_for,
)


def _make_simple_graph(db_id: str = "shop") -> JointGraph:
    """A tiny JointGraph fixture with one of each useful node type."""
    nodes = {
        "f1": {"id": "f1", "type": "Formula",
               "name": "average product price",
               "expression": "AVG(price)",
               "grounding": [{"term": "price", "table": "products", "column": "price"}]},
        "c1": {"id": "c1", "type": "Concept",
               "name": "cheap product", "defined_by": "r1"},
        "r1": {"id": "r1", "type": "Rule",
               "name": "low-price predicate", "condition": "price < 10",
               "grounding": [{"term": "price", "table": "products", "column": "price"}]},
        "v1": {"id": "v1", "type": "ValueMap",
               "name": "in-stock products",
               "table": "products", "column": "status", "value": "in_stock"},
        "a1": {"id": "a1", "type": "ColumnAlias",
               "name": "product identifier",
               "bindings": [{"table": "products", "column": "product_id"}]},
    }
    columns = {
        ("products", "price"): {"type": "REAL"},
        ("products", "status"): {"type": "TEXT"},
        ("products", "product_id"): {"type": "INTEGER"},
    }
    return JointGraph(
        db_id=db_id,
        nodes=nodes,
        columns=columns,
        fk_edges=(),
        by_type={"Formula": ("f1",), "Concept": ("c1",), "Rule": ("r1",),
                 "ValueMap": ("v1",), "ColumnAlias": ("a1",)},
        by_column={("products", "price"): ("f1", "r1"),
                   ("products", "status"): ("v1",),
                   ("products", "product_id"): ("a1",)},
        by_name_token={"product": ("f1", "c1", "v1", "a1"),
                       "price": ("f1",),
                       "cheap": ("c1",),
                       "stock": ("v1",)},
        provenance={"f1": ((1, "f1"),), "c1": ((1, "c1"),),
                    "r1": ((1, "r1"),), "v1": ((1, "v1"),), "a1": ((1, "a1"),)},
        match_signature_hash="",
    )


class SearchTextHelper(unittest.TestCase):
    def test_formula_uses_name_plus_expression(self):
        g = _make_simple_graph()
        text = _search_text_for(g.nodes["f1"], g)
        self.assertIn("average product price", text)
        self.assertIn("AVG(price)", text)

    def test_concept_walks_defined_by(self):
        g = _make_simple_graph()
        text = _search_text_for(g.nodes["c1"], g)
        # Concept's own name + the Rule's search text (name + condition)
        self.assertIn("cheap product", text)
        self.assertIn("low-price predicate", text)
        self.assertIn("price < 10", text)


class RetrieverConstruction(unittest.TestCase):
    def test_constructor_embeds_all_l1_nodes(self):
        # We expect the retriever to hold a per-graph embedding matrix
        # with one row per L1 node. We don't make a real LLM/embedding call
        # here -- the JointRetriever uses sentence-transformers eagerly.
        # If the import or embedding pipeline is broken, this test surfaces it.
        g = _make_simple_graph()
        with tempfile.TemporaryDirectory() as cache:
            r = JointRetriever(
                {g.db_id: g},
                embedding_model_name="all-mpnet-base-v2",
                cache_dir=Path(cache),
            )
            self.assertIn("shop", r._embeddings)
            arr = r._embeddings["shop"]
            self.assertEqual(arr.shape[0], 5)  # 5 L1 nodes
            self.assertEqual(arr.shape[1], 768)  # MPNet dim


class RetrieveQuery(unittest.TestCase):
    def _retriever(self, cache: Path) -> JointRetriever:
        g = _make_simple_graph()
        return JointRetriever(
            {g.db_id: g},
            embedding_model_name="all-mpnet-base-v2",
            cache_dir=cache,
        )

    def test_retrieve_returns_seed_nodes_for_matching_query(self):
        with tempfile.TemporaryDirectory() as cache:
            r = self._retriever(Path(cache))
            # The query "cheap product" should match the Concept "cheap product"
            # at the top, and expansion should pull in the Rule "r1" via defined_by.
            nodes = r.retrieve("cheap product", "shop", top_k=1)
            types = {n.get("type") for n in nodes}
            self.assertIn("Concept", types,
                          f"Concept c1 should be a seed; got {types}")
            self.assertIn("Rule", types,
                          f"Rule r1 should be pulled in via defined_by; got {types}")
            ids = [n.get("id") for n in nodes]
            self.assertEqual(ids[0], "c1",
                             f"Concept c1 should be first (seed); got {ids}")

    def test_retrieve_returns_empty_for_unknown_db(self):
        with tempfile.TemporaryDirectory() as cache:
            r = self._retriever(Path(cache))
            self.assertEqual([], r.retrieve("anything", "no_such_db", top_k=3))


class RenderEvidence(unittest.TestCase):
    def test_render_produces_bullets_per_node(self):
        from bird_eval.joint_retrieval import _render_subgraph_as_evidence

        # Use a mix of types to exercise the per-type rendering.
        nodes = [
            {"id": "f1", "type": "Formula",
             "name": "race completion rate",
             "expression": "COUNT(...) / COUNT(...) * 100",
             "grounding": [{"term": "time", "table": "results", "column": "time"}],
             "depends_on": ["r1"]},
            {"id": "r1", "type": "Rule",
             "name": "completed", "condition": "time IS NOT NULL",
             "grounding": [{"term": "time", "table": "results", "column": "time"}]},
            {"id": "c1", "type": "Concept",
             "name": "race completion", "defined_by": "r1"},
            {"id": "v1", "type": "ValueMap",
             "name": "monthly issuance",
             "table": "account", "column": "frequency", "value": "POPLATEK MESICNE"},
            {"id": "a1", "type": "ColumnAlias",
             "name": "hometown",
             "bindings": [{"table": "zip_code", "column": "city"},
                          {"table": "zip_code", "column": "county"}]},
        ]
        text = _render_subgraph_as_evidence(nodes)
        # Header line must be present.
        self.assertIn("Relevant knowledge for this question:", text)
        # Each node's natural-language identifier shows up.
        self.assertIn("race completion rate", text)
        self.assertIn("monthly issuance", text)
        self.assertIn("hometown", text)
        # Cross-references surfaced inline.
        self.assertIn("defined by", text.lower())  # Concept c1 -> r1
        self.assertIn("depends on", text.lower())  # Formula f1 depends_on r1
        # Branch-specific content assertions (each type's distinctive field).
        self.assertIn("time IS NOT NULL", text)        # Rule.condition
        self.assertIn("COUNT(...)", text)              # Formula.expression
        self.assertIn("POPLATEK MESICNE", text)        # ValueMap.value
        self.assertIn("zip_code.city", text)           # ColumnAlias.bindings
        self.assertIn("results.time", text)            # grounding
