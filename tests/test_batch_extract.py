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


class LoadCheckpoint(unittest.TestCase):
    def test_missing_files_return_empty(self):
        from bird_eval.batch_extract import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "x.partial.json"
            sidecar = Path(tmp) / "x.dropped.jsonl"
            payload, skip = _load_checkpoint(partial, sidecar)

            self.assertEqual(payload, {"meta": {}, "schemas": {}, "items": []})
            self.assertEqual(skip, set())

    def test_reads_partial_and_sidecar(self):
        from bird_eval.batch_extract import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "x.partial.json"
            sidecar = Path(tmp) / "x.dropped.jsonl"

            partial.write_text(json.dumps({
                "meta": {"source": "test"},
                "schemas": {"shop": "..."},
                "items": [{"question_id": 1, "db_id": "shop", "nodes": []},
                          {"question_id": 7, "db_id": "shop", "nodes": []}],
            }))
            sidecar.write_text(
                '{"question_id": 3, "db_id": "shop", "stage": "extract_failed",'
                ' "error": "x", "raw_response_chars": 0}\n'
                '{"question_id": 5, "db_id": "shop", "stage": "llm_error",'
                ' "error": "y", "raw_response_chars": 0}\n'
            )

            payload, skip = _load_checkpoint(partial, sidecar)

            self.assertEqual(payload["meta"], {"source": "test"})
            self.assertEqual(payload["schemas"], {"shop": "..."})
            self.assertEqual(len(payload["items"]), 2)
            # Skip set includes both successful and dropped qids.
            self.assertEqual(skip, {1, 7, 3, 5})

    def test_corrupted_partial_returns_empty_and_does_not_raise(self):
        from bird_eval.batch_extract import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "x.partial.json"
            sidecar = Path(tmp) / "x.dropped.jsonl"

            partial.write_text("{not valid json")

            # Should not raise even though partial is garbage.
            payload, skip = _load_checkpoint(partial, sidecar)

            self.assertEqual(payload, {"meta": {}, "schemas": {}, "items": []})
            self.assertEqual(skip, set())

    def test_non_dict_partial_returns_empty_and_does_not_raise(self):
        from bird_eval.batch_extract import _load_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "x.partial.json"
            sidecar = Path(tmp) / "x.dropped.jsonl"

            # Valid JSON, but not a dict — would crash payload.get(...) without
            # the isinstance guard.
            partial.write_text('"hello"')

            payload, skip = _load_checkpoint(partial, sidecar)

            self.assertEqual(payload, {"meta": {}, "schemas": {}, "items": []})
            self.assertEqual(skip, set())
