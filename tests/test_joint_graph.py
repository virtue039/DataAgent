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

from bird_eval.joint_graph import JointGraph  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
