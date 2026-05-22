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


class RetrieveExpandFlag(unittest.TestCase):
    """P5-D2 ablation: `expand=False` must skip BFS and return seeds only.

    See docs/superpowers/specs/2026-05-22-p5-d2-no-bfs-ablation-design.md.
    """

    def _retriever(self, cache: Path) -> JointRetriever:
        g = _make_simple_graph()
        return JointRetriever(
            {g.db_id: g},
            embedding_model_name="all-mpnet-base-v2",
            cache_dir=cache,
        )

    def test_retrieve_expand_true_matches_existing_behaviour(self):
        # Default behaviour: BFS expansion pulls Rule r1 in via Concept c1's
        # defined_by edge. Same assertions as the original
        # test_retrieve_returns_seed_nodes_for_matching_query but with the
        # explicit `expand=True` kwarg.
        with tempfile.TemporaryDirectory() as cache:
            r = self._retriever(Path(cache))
            nodes = r.retrieve("cheap product", "shop", top_k=1, expand=True)
            types = {n.get("type") for n in nodes}
            self.assertIn("Concept", types,
                          f"Concept c1 should be a seed; got {types}")
            self.assertIn("Rule", types,
                          f"Rule r1 should be pulled in via defined_by; got {types}")
            ids = [n.get("id") for n in nodes]
            self.assertEqual(ids[0], "c1",
                             f"Concept c1 should be first (seed); got {ids}")

    def test_retrieve_expand_false_returns_seeds_only(self):
        # With expand=False, BFS is skipped. top_k=1 -> exactly one seed
        # node (the Concept "cheap product"), and r1 must NOT be included
        # even though c1.defined_by points at it.
        with tempfile.TemporaryDirectory() as cache:
            r = self._retriever(Path(cache))
            nodes = r.retrieve("cheap product", "shop", top_k=1, expand=False)
            ids = [n.get("id") for n in nodes]
            self.assertEqual(ids, ["c1"],
                             f"expand=False should return only the seed; got {ids}")
            self.assertNotIn("r1", ids,
                             "expand=False must skip defined_by expansion")

    def test_retrieve_expand_false_top_k_2(self):
        # top_k=2, expand=False: exactly two seeds by similarity, no edge
        # targets. Concretely, the seeds must not include r1 (the defined_by
        # target of c1) unless r1 itself is one of the top-2 by similarity
        # to the query. We pick a query that puts c1 + v1 (in-stock products)
        # on top; r1's score for this query is lower, so r1 should NOT
        # appear in the result.
        with tempfile.TemporaryDirectory() as cache:
            r = self._retriever(Path(cache))
            nodes = r.retrieve("cheap product", "shop", top_k=2, expand=False)
            ids = [n.get("id") for n in nodes]
            self.assertEqual(len(ids), 2,
                             f"top_k=2 + expand=False should return 2 nodes; got {ids}")
            # All returned nodes must be among the graph's L1 nodes; none
            # should be pure BFS-expansion artifacts.
            self.assertEqual(len(set(ids)), 2, f"duplicate ids: {ids}")
            self.assertIn("c1", ids,
                          f"Concept c1 should be a top-2 seed for 'cheap product'; got {ids}")


class MakeEvidenceProviderJointNoBFS(unittest.TestCase):
    """P5-D2: make_evidence_provider must wire `joint_no_bfs` to
    JointRetrievalEvidence with expand=False."""

    def test_make_provider_joint_no_bfs(self):
        from bird_eval.config import Config
        from bird_eval.evidence import JointRetrievalEvidence, make_evidence_provider
        from bird_eval.joint_graph import dump_graph

        # Build a tiny joint-graph file on disk so make_evidence_provider's
        # graph-loading path is exercised.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            graphs_dir = tmp_path / "joint_graphs"
            graphs_dir.mkdir()
            dump_graph(_make_simple_graph("shop"), graphs_dir / "shop.json")

            bird_dir = tmp_path / "bird"
            bird_dir.mkdir()

            config = Config(
                bird_dir=bird_dir,
                setting="joint_no_bfs",
                joint_graphs_dir=graphs_dir,
            )
            provider = make_evidence_provider(config)
            self.assertIsInstance(provider, JointRetrievalEvidence)
            self.assertFalse(provider._expand,
                             "joint_no_bfs provider must have expand=False")


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


class ProviderIntegration(unittest.TestCase):
    def test_provider_returns_rendered_evidence(self):
        # Mock retriever returns a known node list; provider should call
        # the renderer and surface its output.
        from bird_eval.data import BirdExample
        from bird_eval.evidence import JointRetrievalEvidence

        class _MockRetriever:
            def __init__(self): self.calls = []
            def retrieve(self, query, db_id, top_k, expand=True):
                self.calls.append({"query": query, "db_id": db_id,
                                   "top_k": top_k, "expand": expand})
                return [{"id": "v1", "type": "ValueMap", "name": "monthly",
                         "table": "account", "column": "frequency",
                         "value": "POPLATEK MESICNE"}]

        retriever = _MockRetriever()
        provider = JointRetrievalEvidence(retriever, top_k=3)
        ex = BirdExample(
            question_id=1, db_id="financial",
            question="How many accounts have monthly issuance?",
            evidence="", gold_sql="SELECT ...", difficulty="simple",
        )
        out = provider.get(ex)
        # Provider should have called the retriever with the question + db_id.
        self.assertEqual(retriever.calls[0]["query"], ex.question)
        self.assertEqual(retriever.calls[0]["db_id"], ex.db_id)
        self.assertEqual(retriever.calls[0]["top_k"], 3)
        # Default expand=True is preserved when constructor doesn't override.
        self.assertTrue(retriever.calls[0]["expand"])
        # Output should include the rendered evidence header + the ValueMap.
        self.assertIn("Relevant knowledge for this question:", out)
        self.assertIn("POPLATEK MESICNE", out)
