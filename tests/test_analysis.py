"""Unit tests for bird_eval/analysis.py (P3)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.analysis import (  # noqa: E402
    per_db_delta,
    per_db_table,
    per_difficulty_table,
    per_setting_overall,
    wilson_ci,
)


def _result_doc(setting: str, results: list[dict]) -> dict:
    """Build a minimal result-JSON-shaped dict (matches run_eval.py output)."""
    n_correct = sum(1 for r in results if r.get("correct"))
    n_scored = len(results)
    return {
        "config": {"setting": setting},
        "summary": {"execution_accuracy": n_correct / max(n_scored, 1),
                    "n_correct": n_correct, "n_scored": n_scored,
                    "n_examples": n_scored},
        "results": results,
    }


class WilsonCI(unittest.TestCase):
    def test_canonical_zero_of_zero(self):
        # Degenerate: no data → wide [0, 1] is the safe default.
        lo, hi = wilson_ci(0, 0)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, 1.0)

    def test_canonical_n40_14correct(self):
        # 14/40 = 0.35; Wilson 95% CI ≈ [0.221, 0.508] per Newcombe (1998).
        lo, hi = wilson_ci(14, 40)
        self.assertAlmostEqual(lo, 0.221, places=2)
        self.assertAlmostEqual(hi, 0.508, places=2)

    def test_canonical_all_correct(self):
        # 10/10 → upper bound = 1.0 (Wilson clamps cleanly), lower > 0.7
        lo, hi = wilson_ci(10, 10)
        self.assertGreater(lo, 0.69)
        self.assertLessEqual(hi, 1.0)


class PerSettingOverall(unittest.TestCase):
    def test_aggregates_each_setting(self):
        docs = {
            "none": _result_doc("none", [
                {"db_id": "a", "difficulty": "simple", "correct": True},
                {"db_id": "a", "difficulty": "moderate", "correct": False},
            ]),
            "joint": _result_doc("joint", [
                {"db_id": "a", "difficulty": "simple", "correct": True},
                {"db_id": "a", "difficulty": "moderate", "correct": True},
            ]),
        }
        rows = per_setting_overall(docs)
        d = {r["setting"]: r for r in rows}
        self.assertAlmostEqual(d["none"]["ex"], 0.5)
        self.assertAlmostEqual(d["joint"]["ex"], 1.0)
        self.assertEqual(d["none"]["n"], 2)


class PerDifficultyTable(unittest.TestCase):
    def test_groups_by_difficulty(self):
        # Two simple correct, one moderate wrong.
        doc = _result_doc("none", [
            {"db_id": "a", "difficulty": "simple", "correct": True},
            {"db_id": "a", "difficulty": "simple", "correct": True},
            {"db_id": "a", "difficulty": "moderate", "correct": False},
        ])
        table = per_difficulty_table({"none": doc})
        # Expect rows with the per-difficulty EX% for "none".
        simple = next(r for r in table if r["difficulty"] == "simple")
        moderate = next(r for r in table if r["difficulty"] == "moderate")
        self.assertAlmostEqual(simple["none"], 1.0)
        self.assertAlmostEqual(moderate["none"], 0.0)


class PerDbTable(unittest.TestCase):
    def test_groups_by_db(self):
        doc = _result_doc("none", [
            {"db_id": "alpha", "difficulty": "simple", "correct": True},
            {"db_id": "alpha", "difficulty": "simple", "correct": False},
            {"db_id": "beta", "difficulty": "simple", "correct": True},
        ])
        table = per_db_table({"none": doc})
        a = next(r for r in table if r["db_id"] == "alpha")
        b = next(r for r in table if r["db_id"] == "beta")
        self.assertEqual(a["n"], 2)
        self.assertAlmostEqual(a["none"], 0.5)
        self.assertEqual(b["n"], 1)
        self.assertAlmostEqual(b["none"], 1.0)


class PerDbDelta(unittest.TestCase):
    def test_sorts_descending_by_delta(self):
        joint_doc = _result_doc("joint", [
            # alpha: joint 2/2 = 100%, retr 0/2 = 0%, Δ = +1.0
            {"db_id": "alpha", "difficulty": "simple", "correct": True},
            {"db_id": "alpha", "difficulty": "simple", "correct": True},
            # beta: joint 0/2 = 0%, retr 1/2 = 50%, Δ = -0.5
            {"db_id": "beta", "difficulty": "simple", "correct": False},
            {"db_id": "beta", "difficulty": "simple", "correct": False},
        ])
        retr_doc = _result_doc("retrieval", [
            {"db_id": "alpha", "difficulty": "simple", "correct": False},
            {"db_id": "alpha", "difficulty": "simple", "correct": False},
            {"db_id": "beta", "difficulty": "simple", "correct": True},
            {"db_id": "beta", "difficulty": "simple", "correct": False},
        ])
        rows = per_db_delta({"joint": joint_doc, "retrieval": retr_doc})
        self.assertEqual(rows[0]["db_id"], "alpha")
        self.assertAlmostEqual(rows[0]["delta"], 1.0)
        self.assertEqual(rows[0]["direction"], "joint wins")
        self.assertEqual(rows[1]["db_id"], "beta")
        self.assertAlmostEqual(rows[1]["delta"], -0.5)
        self.assertEqual(rows[1]["direction"], "retr wins")


class PerDbDeltaEdgeCases(unittest.TestCase):
    def test_tie_within_1pp_marked_as_tie(self):
        # 50% vs 50% → tie. (We allow tie threshold = 0.01.)
        joint_doc = _result_doc("joint", [
            {"db_id": "alpha", "difficulty": "simple", "correct": True},
            {"db_id": "alpha", "difficulty": "simple", "correct": False},
        ])
        retr_doc = _result_doc("retrieval", [
            {"db_id": "alpha", "difficulty": "simple", "correct": True},
            {"db_id": "alpha", "difficulty": "simple", "correct": False},
        ])
        rows = per_db_delta({"joint": joint_doc, "retrieval": retr_doc})
        self.assertEqual(rows[0]["direction"], "tie")
