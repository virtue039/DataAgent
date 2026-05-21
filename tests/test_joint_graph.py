"""Unit tests for bird_eval/joint_graph.py.

Run with:
  .venv/bin/python -m unittest tests.test_joint_graph -v
"""
from __future__ import annotations

import dataclasses
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.joint_graph import (  # noqa: E402
    JointGraph,
    _dedup_l1,
    _rewrite_refs,
    _drop_hallucinated,
)


def _empty_graph() -> JointGraph:
    return JointGraph(
        db_id="example",
        nodes={},
        columns={},
        fk_edges=(),
        by_type={},
        by_column={},
        by_name_token={},
        provenance={},
    )


class Dataclass(unittest.TestCase):
    def test_is_frozen(self):
        g = _empty_graph()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            g.db_id = "other"  # type: ignore[misc]

    def test_equality_is_structural(self):
        a = _empty_graph()
        b = _empty_graph()
        self.assertEqual(a, b)

    def test_carries_all_required_fields(self):
        g = _empty_graph()
        for field in (
            "db_id", "nodes", "columns", "fk_edges",
            "by_type", "by_column", "by_name_token", "provenance",
        ):
            self.assertTrue(hasattr(g, field), f"missing field {field!r}")


# Shared fixture items used in dedup + sanitize + build tests below.

_ITEM_RULE_A = {
    "question_id": 1, "db_id": "shop", "difficulty": "simple",
    "question": "low-priced products",
    "raw_evidence": "low-priced products refers to price < 10",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "low-priced",
         "condition": "price < 10",
         "grounding": [{"term": "price", "table": "products", "column": "price"}]},
    ],
    "notes": "",
}

_ITEM_RULE_B_SAME_CONDITION = {
    # Same condition as ITEM_RULE_A; different name; same db.
    # Post the 2026-05-20 Rule fix, match() uses condition-primary => dedup.
    "question_id": 7, "db_id": "shop", "difficulty": "simple",
    "question": "cheap items",
    "raw_evidence": "cheap items: price < 10",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "cheap items",
         "condition": "price < 10",
         "grounding": [
             {"term": "price", "table": "products", "column": "price"},
             # extra (term, col, val) tuple that B has but A doesn't
             {"term": "price", "table": "products", "column": "price", "value": "10"},
         ]},
    ],
    "notes": "",
}

_ITEM_RULE_DISTINCT = {
    # Different condition => no dedup with A or B.
    "question_id": 9, "db_id": "shop", "difficulty": "simple",
    "question": "expensive items",
    "raw_evidence": "expensive items: price >= 100",
    "nodes": [
        {"id": "r1", "type": "Rule", "name": "expensive",
         "condition": "price >= 100",
         "grounding": [{"term": "price", "table": "products", "column": "price"}]},
    ],
    "notes": "",
}


class L1Dedup(unittest.TestCase):
    def test_collapses_same_condition_rules(self):
        canonical_nodes, provenance, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        # exactly one Rule node survives
        rules = [n for n in canonical_nodes.values() if n["type"] == "Rule"]
        self.assertEqual(1, len(rules))
        self.assertTrue(rules[0]["id"].startswith("r"))
        self.assertIn(rules[0]["id"], canonical_nodes)

    def test_provenance_records_both_origins(self):
        canonical_nodes, provenance, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        rule_id = next(k for k, n in canonical_nodes.items() if n["type"] == "Rule")
        prov = provenance[rule_id]
        self.assertEqual({(1, "r1"), (7, "r1")}, set(prov))

    def test_grounding_union_across_records(self):
        canonical_nodes, _, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_B_SAME_CONDITION]
        )
        rule = next(n for n in canonical_nodes.values() if n["type"] == "Rule")
        # union by (term, table, column, value)
        keys = {(g.get("term"), g.get("table"), g.get("column"), g.get("value"))
                for g in rule["grounding"]}
        self.assertEqual(
            {("price", "products", "price", None),
             ("price", "products", "price", "10")},
            keys,
        )

    def test_distinct_rules_kept_separate(self):
        canonical_nodes, _, _ = _dedup_l1(
            [_ITEM_RULE_A, _ITEM_RULE_DISTINCT]
        )
        rules = [n for n in canonical_nodes.values() if n["type"] == "Rule"]
        self.assertEqual(2, len(rules))
        # canonical ids are deterministic: r1, r2
        rule_ids = sorted(n["id"] for n in rules)
        self.assertEqual(["r1", "r2"], rule_ids)


_TEST_DDL = """
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    price REAL
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products (product_id),
    quantity INTEGER
);
"""


class RewriteRefs(unittest.TestCase):
    def test_concept_defined_by_rewritten_to_canonical_id(self):
        items = [{
            "question_id": 1, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "c1", "type": "Concept", "name": "cheap product",
                 "defined_by": "r1"},
                {"id": "r1", "type": "Rule", "name": "cheap",
                 "condition": "price < 10",
                 "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            ],
        }]
        canon, prov, remap = _dedup_l1(items)
        rewritten = _rewrite_refs(canon, prov, remap)
        c = next(n for n in rewritten.values() if n["type"] == "Concept")
        self.assertIn(c["defined_by"], rewritten)  # resolved to a present canonical id
        self.assertEqual("Rule", rewritten[c["defined_by"]]["type"])

    def test_formula_depends_on_rewritten(self):
        items = [{
            "question_id": 5, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "r1", "type": "Rule", "name": "y", "condition": "year > 2020",
                 "grounding": [{"term": "year", "table": "orders", "column": "quantity"}]},
                {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
                 "grounding": [{"term": "x", "table": "products", "column": "price"}],
                 "depends_on": ["r1"]},
            ],
        }]
        canon, prov, remap = _dedup_l1(items)
        rewritten = _rewrite_refs(canon, prov, remap)
        f = next(n for n in rewritten.values() if n["type"] == "Formula")
        self.assertEqual(1, len(f["depends_on"]))
        target = rewritten[f["depends_on"][0]]
        self.assertEqual("Rule", target["type"])


class SanityDrop(unittest.TestCase):
    def test_node_with_hallucinated_column_loses_that_grounding(self):
        items = [{
            "question_id": 1, "db_id": "shop", "difficulty": "simple",
            "question": "", "raw_evidence": "", "notes": "",
            "nodes": [
                {"id": "f1", "type": "Formula", "name": "x", "expression": "x",
                 "grounding": [
                     {"term": "price", "table": "products", "column": "price"},
                     {"term": "bogus", "table": "products", "column": "no_such_col"},
                 ]},
            ],
        }]
        canon, _, _ = _dedup_l1(items)
        cleaned = _drop_hallucinated(canon, _TEST_DDL)
        f = next(n for n in cleaned.values() if n["type"] == "Formula")
        self.assertEqual(1, len(f["grounding"]))
        self.assertEqual("price", f["grounding"][0]["column"])


if __name__ == "__main__":
    unittest.main()
