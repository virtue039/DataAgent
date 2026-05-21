"""Unit tests for bird_eval/batch_extract.py (P2).

Run with:
  .venv/bin/python -m unittest tests.test_batch_extract -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.batch_extract import (  # noqa: E402
    STAGE_EXTRACT_FAILED,
    STAGE_GROUNDING_HALLUCINATED,
    STAGE_LLM_ERROR,
    _classify_failure,
)


class ClassifyFailure(unittest.TestCase):
    def test_exception_means_llm_error(self):
        label = _classify_failure(RuntimeError("boom"), nodes=[], stats={})
        self.assertEqual(label, STAGE_LLM_ERROR)

    def test_empty_nodes_means_extract_failed(self):
        label = _classify_failure(
            None, nodes=[], stats={"dropped_groundings": 0, "dropped_nodes": 0}
        )
        self.assertEqual(label, STAGE_EXTRACT_FAILED)

    def test_dropped_nodes_means_grounding_hallucinated(self):
        label = _classify_failure(
            None, nodes=[{"id": "f1", "type": "Formula"}],
            stats={"dropped_groundings": 2, "dropped_nodes": 1},
        )
        self.assertEqual(label, STAGE_GROUNDING_HALLUCINATED)

    def test_happy_path_returns_none(self):
        label = _classify_failure(
            None, nodes=[{"id": "f1", "type": "Formula"}],
            stats={"dropped_groundings": 0, "dropped_nodes": 0},
        )
        self.assertIsNone(label)
