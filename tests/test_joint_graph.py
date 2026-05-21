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

from bird_eval.joint_graph import JointGraph, _dedup_l1  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
