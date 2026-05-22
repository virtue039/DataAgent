"""Unit tests for demo/case_loader.py (P5 demo).

Run with:
  .venv/bin/python -m unittest tests.test_case_loader -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from demo import case_loader  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result_row(qid: int, setting: str, *, evidence: str = "",
                     predicted_sql: str = "SELECT 1", correct: bool = False,
                     db_id: str = "shop", difficulty: str = "simple",
                     question: str = "What's the cheapest item?",
                     gold_sql: str = "SELECT MIN(price) FROM products") -> dict:
    return {
        "question_id": qid,
        "db_id": db_id,
        "difficulty": difficulty,
        "question": question,
        "setting": setting,
        "evidence": evidence,
        "predicted_sql": predicted_sql,
        "gold_sql": gold_sql,
        "correct": correct,
        "error": None,
    }


def _write_result_json(path: Path, setting: str, rows: list[dict]) -> None:
    path.write_text(json.dumps({
        "summary": {"setting": setting, "n_examples": len(rows)},
        "config": {"setting": setting},
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_fake_graph(db_id: str = "shop"):
    """A tiny stand-in for JointGraph carrying just .nodes (what we read)."""
    class _G:
        pass
    g = _G()
    g.db_id = db_id
    g.nodes = {
        "f1": {"id": "f1", "type": "Formula",
               "name": "average price", "expression": "AVG(price)",
               "depends_on": ["a1"],
               "grounding": [{"term": "price", "table": "products", "column": "price"}]},
        "c1": {"id": "c1", "type": "Concept",
               "name": "cheap product", "defined_by": "r1"},
        "r1": {"id": "r1", "type": "Rule",
               "name": "low-price predicate", "condition": "price < 10"},
        "a1": {"id": "a1", "type": "ColumnAlias",
               "name": "product id",
               "bindings": [{"table": "products", "column": "product_id"}]},
    }
    return g


class _FakeRetriever:
    """Stand-in for JointRetriever that returns a fixed list of node dicts."""

    def __init__(self, graphs, embedding_model_name=None, cache_dir=None):
        self._graphs = graphs

    def retrieve(self, question: str, db_id: str, top_k: int = 5) -> list[dict]:
        g = self._graphs[db_id]
        # Return 3 nodes deterministically.
        return [g.nodes["f1"], g.nodes["c1"], g.nodes["r1"]]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class LoadCaseAssemblesFourSettings(unittest.TestCase):
    def test_load_case_assembles_four_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p3 = tmp / "results" / "p3_matrix"
            p4 = tmp / "results" / "p4_cv"
            p3.mkdir(parents=True)
            p4.mkdir(parents=True)
            qid = 42
            _write_result_json(p3 / "none_x_20260101_000000.json", "none",
                               [_make_result_row(qid, "none",
                                                 predicted_sql="SELECT a",
                                                 correct=False)])
            _write_result_json(p3 / "retrieval_x_20260101_000000.json", "retrieval",
                               [_make_result_row(qid, "retrieval",
                                                 evidence="some evidence",
                                                 predicted_sql="SELECT b",
                                                 correct=True)])
            _write_result_json(p3 / "oracle_x_20260101_000000.json", "oracle",
                               [_make_result_row(qid, "oracle",
                                                 evidence="orcl",
                                                 predicted_sql="SELECT c",
                                                 correct=True)])
            _write_result_json(p4 / "joint_x_20260101_000000.json", "joint",
                               [_make_result_row(qid, "joint",
                                                 evidence="joint stuff",
                                                 predicted_sql="SELECT d",
                                                 correct=True)])

            with mock.patch.object(case_loader, "_load_subgraph") as mock_sg:
                mock_sg.return_value = {"nodes": [], "edges": []}
                case = case_loader.load_case(
                    qid,
                    p3_dir=p3, p4_dir=p4, joint_graphs_dir=tmp / "jg",
                )
            self.assertEqual(case["qid"], qid)
            self.assertEqual(case["db_id"], "shop")
            self.assertEqual(case["question"], "What's the cheapest item?")
            for s in ("none", "retrieval", "joint", "oracle"):
                self.assertIn(s, case["settings"])
            self.assertEqual(case["settings"]["none"]["predicted_sql"], "SELECT a")
            self.assertFalse(case["settings"]["none"]["correct"])
            self.assertTrue(case["settings"]["joint"]["correct"])
            self.assertEqual(case["settings"]["joint"]["evidence"], "joint stuff")


class LoadCaseExtractsSubgraphFromJointGraph(unittest.TestCase):
    def test_load_case_extracts_subgraph_from_joint_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p3 = tmp / "results" / "p3_matrix"
            p4 = tmp / "results" / "p4_cv"
            jg = tmp / "jg"
            p3.mkdir(parents=True)
            p4.mkdir(parents=True)
            jg.mkdir(parents=True)
            qid = 7
            for s, dir_, fname in [
                ("none", p3, "none_x.json"),
                ("retrieval", p3, "retrieval_x.json"),
                ("oracle", p3, "oracle_x.json"),
                ("joint", p4, "joint_x.json"),
            ]:
                _write_result_json(dir_ / fname, s,
                                   [_make_result_row(qid, s, db_id="shop")])

            # JointGraph JSON sentinel; loader patches load_graph anyway.
            (jg / "shop.json").write_text("{}", encoding="utf-8")
            fake_g = _make_fake_graph("shop")

            with mock.patch.object(case_loader, "_load_joint_graph",
                                   return_value=fake_g), \
                 mock.patch.object(case_loader, "_make_retriever",
                                   side_effect=lambda graphs, **kw: _FakeRetriever(graphs)):
                case = case_loader.load_case(
                    qid, p3_dir=p3, p4_dir=p4, joint_graphs_dir=jg,
                )
            nodes = case["subgraph"]["nodes"]
            edges = case["subgraph"]["edges"]
            # The fake retriever returns 3 nodes.
            # The loader may add L2 column nodes from groundings; assert at least 3 L1 nodes.
            l1_nodes = [n for n in nodes if n.get("type") != "Column"]
            self.assertEqual(len(l1_nodes), 3)
            ids = {n["id"] for n in l1_nodes}
            self.assertSetEqual(ids, {"f1", "c1", "r1"})
            # The Concept c1 -> Rule r1 defined_by edge should appear (both are present).
            kinds = {(e["src"], e["dst"], e["kind"]) for e in edges}
            self.assertIn(("c1", "r1", "defined_by"), kinds)


class LoadCaseUnknownQidRaises(unittest.TestCase):
    def test_load_case_unknown_qid_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p3 = tmp / "results" / "p3_matrix"
            p4 = tmp / "results" / "p4_cv"
            p3.mkdir(parents=True)
            p4.mkdir(parents=True)
            _write_result_json(p3 / "none_x.json", "none",
                               [_make_result_row(1, "none")])
            _write_result_json(p3 / "retrieval_x.json", "retrieval",
                               [_make_result_row(1, "retrieval")])
            _write_result_json(p3 / "oracle_x.json", "oracle",
                               [_make_result_row(1, "oracle")])
            _write_result_json(p4 / "joint_x.json", "joint",
                               [_make_result_row(1, "joint")])

            with self.assertRaises(KeyError):
                case_loader.load_case(
                    99999, p3_dir=p3, p4_dir=p4, joint_graphs_dir=tmp / "jg",
                )


class SubgraphIncludesL2ColumnNodesForGroundedTerms(unittest.TestCase):
    def test_subgraph_includes_l2_column_nodes_for_grounded_terms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            p3 = tmp / "results" / "p3_matrix"
            p4 = tmp / "results" / "p4_cv"
            jg = tmp / "jg"
            p3.mkdir(parents=True)
            p4.mkdir(parents=True)
            jg.mkdir(parents=True)
            qid = 3
            for s, dir_, fname in [
                ("none", p3, "none_x.json"),
                ("retrieval", p3, "retrieval_x.json"),
                ("oracle", p3, "oracle_x.json"),
                ("joint", p4, "joint_x.json"),
            ]:
                _write_result_json(dir_ / fname, s,
                                   [_make_result_row(qid, s, db_id="shop")])
            (jg / "shop.json").write_text("{}", encoding="utf-8")
            fake_g = _make_fake_graph("shop")

            with mock.patch.object(case_loader, "_load_joint_graph",
                                   return_value=fake_g), \
                 mock.patch.object(case_loader, "_make_retriever",
                                   side_effect=lambda graphs, **kw: _FakeRetriever(graphs)):
                case = case_loader.load_case(
                    qid, p3_dir=p3, p4_dir=p4, joint_graphs_dir=jg,
                )
            nodes = case["subgraph"]["nodes"]
            edges = case["subgraph"]["edges"]
            # Formula f1 has grounding to (products, price) -> expect a synthetic
            # Column L2 node with id 'col:products.price' and a 'grounds' edge.
            col_id = "col:products.price"
            ids = {n["id"] for n in nodes}
            self.assertIn(col_id, ids)
            col_node = next(n for n in nodes if n["id"] == col_id)
            self.assertEqual(col_node["type"], "Column")
            kinds = {(e["src"], e["dst"], e["kind"]) for e in edges}
            self.assertIn(("f1", col_id, "grounds"), kinds)


if __name__ == "__main__":
    unittest.main()
