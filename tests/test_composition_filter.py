"""Unit tests for bird_eval/composition_filter.py.

Run with:
  .venv/bin/python -m unittest tests.test_composition_filter -v

See docs/superpowers/specs/2026-05-22-p5-d3-composition-specialty-design.md
for the design that motivates `strip_composites`.
"""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.composition_filter import strip_composites  # noqa: E402
from bird_eval.joint_graph import JointGraph  # noqa: E402


def _make_graph(nodes: dict[str, dict], provenance: dict | None = None) -> JointGraph:
    """Build an in-memory JointGraph from a {id: node} dict — indexes are
    intentionally empty so we can assert `strip_composites` rebuilds them.
    """
    return JointGraph(
        db_id="example",
        nodes=nodes,
        columns={},
        fk_edges=(),
        by_type={},
        by_column={},
        by_name_token={},
        provenance=provenance or {nid: ((0, nid),) for nid in nodes},
    )


class StripCompositesRemovesHighDependsOn(unittest.TestCase):
    def test_strip_composites_removes_high_depends_on_node(self):
        # f1 is a composite (depends_on length 2); r1 and r2 are its atom targets.
        nodes = {
            "r1": {"id": "r1", "type": "Rule", "name": "cheap",
                   "condition": "price < 10",
                   "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            "r2": {"id": "r2", "type": "Rule", "name": "recent",
                   "condition": "year > 2020",
                   "grounding": [{"term": "year", "table": "orders", "column": "quantity"}]},
            "f1": {"id": "f1", "type": "Formula", "name": "cheap recent",
                   "expression": "x",
                   "grounding": [{"term": "x", "table": "products", "column": "price"}],
                   "depends_on": ["r1", "r2"]},
        }
        g = _make_graph(nodes)
        result = strip_composites(g)
        self.assertNotIn("f1", result.nodes,
                         "composite f1 (depends_on length 2) should be removed")
        self.assertIn("r1", result.nodes, "atom r1 should survive")
        self.assertIn("r2", result.nodes, "atom r2 should survive")
        # Indexes rebuilt: by_type for Rule should list r1 and r2 (no Formula present).
        self.assertEqual(set(result.by_type.get("Rule", ())), {"r1", "r2"})
        self.assertEqual(result.by_type.get("Formula", ()), ())
        # Provenance trimmed.
        self.assertNotIn("f1", result.provenance)
        self.assertIn("r1", result.provenance)


class StripCompositesKeepsAtoms(unittest.TestCase):
    def test_strip_composites_keeps_atoms(self):
        # Three atomic nodes: depends_on absent, empty, and length 1 are all atoms.
        nodes = {
            "r1": {"id": "r1", "type": "Rule", "name": "cheap",
                   "condition": "price < 10",
                   "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            "f1": {"id": "f1", "type": "Formula", "name": "avg",
                   "expression": "AVG(price)",
                   "grounding": [{"term": "price", "table": "products", "column": "price"}],
                   "depends_on": []},
            "f2": {"id": "f2", "type": "Formula", "name": "single-dep",
                   "expression": "x",
                   "grounding": [{"term": "x", "table": "products", "column": "price"}],
                   "depends_on": ["r1"]},
        }
        g = _make_graph(nodes)
        result = strip_composites(g)
        self.assertIn("r1", result.nodes, "atom (no depends_on) should survive")
        self.assertIn("f1", result.nodes, "atom (empty depends_on) should survive")
        self.assertIn("f2", result.nodes, "atom (depends_on length 1) should survive")
        # f2's single dep should still point at r1.
        self.assertEqual(result.nodes["f2"]["depends_on"], ["r1"])


class StripCompositesCleansInboundRefs(unittest.TestCase):
    def test_strip_composites_cleans_inbound_refs(self):
        # c1 is a Concept whose defined_by points at f1, which is a composite
        # and gets removed. After stripping, c1 should have no `defined_by` key.
        # Additionally, a Formula f2 whose depends_on contains removed f1 plus
        # surviving r1 — after stripping, depends_on should be ["r1"].
        nodes = {
            "r1": {"id": "r1", "type": "Rule", "name": "cheap",
                   "condition": "price < 10",
                   "grounding": [{"term": "price", "table": "products", "column": "price"}]},
            "r2": {"id": "r2", "type": "Rule", "name": "recent",
                   "condition": "year > 2020",
                   "grounding": [{"term": "year", "table": "orders", "column": "quantity"}]},
            "f1": {"id": "f1", "type": "Formula", "name": "composite",
                   "expression": "x",
                   "grounding": [{"term": "x", "table": "products", "column": "price"}],
                   "depends_on": ["r1", "r2"]},
            "c1": {"id": "c1", "type": "Concept",
                   "name": "cheap recent product",
                   "defined_by": "f1"},
            "f2": {"id": "f2", "type": "Formula", "name": "atom-with-mixed-deps",
                   "expression": "y",
                   "grounding": [{"term": "y", "table": "products", "column": "price"}],
                   "depends_on": ["f1"]},
        }
        g = _make_graph(nodes)
        result = strip_composites(g)
        # f1 removed.
        self.assertNotIn("f1", result.nodes)
        # c1's defined_by pointed at f1; after cleanup it should be absent.
        self.assertIn("c1", result.nodes)
        self.assertNotIn(
            "defined_by", result.nodes["c1"],
            "stale defined_by pointing at removed f1 should be dropped",
        )
        # f2's depends_on contained only f1; after cleanup, depends_on absent.
        self.assertIn("f2", result.nodes)
        self.assertNotIn(
            "depends_on", result.nodes["f2"],
            "stale depends_on pointing only at removed f1 should be dropped entirely",
        )
        # Referential closure: every surviving defined_by/depends_on target exists.
        surviving = set(result.nodes)
        for n in result.nodes.values():
            if "defined_by" in n:
                self.assertIn(n["defined_by"], surviving)
            for d in n.get("depends_on", []) or []:
                self.assertIn(d, surviving)


if __name__ == "__main__":
    unittest.main()
