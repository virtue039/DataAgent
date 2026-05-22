"""Unit tests for bird_eval/cv_split.py (P4-revised)."""
from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.cv_split import make_within_db_split  # noqa: E402


class WithinDbSplit(unittest.TestCase):
    def test_two_dbs_balanced_split(self):
        # db_a has 4 qids, db_b has 4 qids. 50/50: 2 KB + 2 eval per db.
        items = [
            {"question_id": 1, "db_id": "a"},
            {"question_id": 2, "db_id": "a"},
            {"question_id": 3, "db_id": "a"},
            {"question_id": 4, "db_id": "a"},
            {"question_id": 5, "db_id": "b"},
            {"question_id": 6, "db_id": "b"},
            {"question_id": 7, "db_id": "b"},
            {"question_id": 8, "db_id": "b"},
        ]
        kb, eval_set = make_within_db_split(items, kb_fraction=0.5)
        # No overlap.
        self.assertEqual(set(kb) & set(eval_set), set())
        # Both sets cover both dbs (50/50 deterministic).
        self.assertEqual(len(kb), 4)
        self.assertEqual(len(eval_set), 4)
        # Deterministic: KB = first floor(n/2) sorted qids per db.
        self.assertEqual(set(kb), {1, 2, 5, 6})
        self.assertEqual(set(eval_set), {3, 4, 7, 8})

    def test_odd_count_rounds_kb_down(self):
        # 5 qids → KB gets 2 (floor(5/2)), eval gets 3.
        items = [{"question_id": i, "db_id": "a"} for i in range(1, 6)]
        kb, eval_set = make_within_db_split(items, kb_fraction=0.5)
        self.assertEqual(len(kb), 2)
        self.assertEqual(len(eval_set), 3)
        # KB = first 2 qids by sort order.
        self.assertEqual(set(kb), {1, 2})

    def test_single_qid_db_goes_to_eval_only(self):
        # If a db has 1 qid, floor(1*0.5) = 0 → KB empty for that db, eval gets it.
        items = [{"question_id": 1, "db_id": "a"},
                 {"question_id": 2, "db_id": "b"},
                 {"question_id": 3, "db_id": "b"}]
        kb, eval_set = make_within_db_split(items, kb_fraction=0.5)
        self.assertEqual(set(kb), {2})       # b's first qid → KB
        self.assertEqual(set(eval_set), {1, 3})  # a's only qid + b's second → eval


if __name__ == "__main__":
    unittest.main()
