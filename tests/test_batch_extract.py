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


class AtomicWritePartial(unittest.TestCase):
    def test_writes_json_via_tmp_then_renames(self):
        from bird_eval.batch_extract import _atomic_write_partial

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x.partial.json"
            payload = {"meta": {"k": "v"}, "schemas": {"s": "ddl"}, "items": [
                {"question_id": 1, "db_id": "s", "nodes": []},
            ]}

            _atomic_write_partial(target, payload)

            # Target exists; tmp sibling does NOT (must have been renamed away).
            self.assertTrue(target.exists())
            self.assertFalse((Path(tmp) / "x.partial.json.tmp").exists())

            # Content round-trips.
            loaded = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(loaded, payload)


class _MockLLM:
    """Replays a queue of (string or Exception) responses for llm.complete()."""

    def __init__(self, responses: list):
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        idx = self.calls
        self.calls += 1
        if idx >= len(self._responses):
            raise IndexError(f"_MockLLM out of responses at call {idx}")
        r = self._responses[idx]
        if isinstance(r, Exception):
            raise r
        return r


_DDL_SHOP = (
    "CREATE TABLE products (\n"
    "  product_id INTEGER PRIMARY KEY,\n"
    "  price REAL,\n"
    "  status TEXT\n"
    ");"
)


class ExtractOne(unittest.TestCase):
    def test_happy_path_returns_item_dict_and_no_sidecar(self):
        from bird_eval.batch_extract import _extract_one

        # LLM returns a valid v2 node referencing a real column.
        llm_response = """```json
[{"id": "f1", "type": "Formula", "name": "avg price",
  "expression": "AVG(price)",
  "grounding": [{"term": "price", "table": "products", "column": "price"}]}]
```"""
        llm = _MockLLM([llm_response])

        item = {
            "question_id": 42, "db_id": "shop", "difficulty": "simple",
            "question": "What is the average price?",
            "raw_evidence": "average price refers to AVG(price)",
        }
        schemas = {"shop": _DDL_SHOP}

        result_item, sidecar = _extract_one(item, schemas, llm)

        self.assertIsNotNone(result_item)
        self.assertIsNone(sidecar)
        self.assertEqual(result_item["question_id"], 42)
        self.assertEqual(result_item["db_id"], "shop")
        self.assertEqual(result_item["raw_evidence"],
                         "average price refers to AVG(price)")
        self.assertEqual(len(result_item["nodes"]), 1)
        self.assertEqual(result_item["nodes"][0]["type"], "Formula")

    def test_grounding_hallucination_returns_sidecar_entry(self):
        from bird_eval.batch_extract import _extract_one

        # LLM returns a Formula whose grounding column doesn't exist in the
        # schema; sanitize_grounding will drop the Formula entirely.
        llm_response = """```json
[{"id": "f1", "type": "Formula", "name": "bad",
  "expression": "AVG(nope)",
  "grounding": [{"term": "nope", "table": "products", "column": "nope"}]}]
```"""
        llm = _MockLLM([llm_response])

        item = {
            "question_id": 99, "db_id": "shop", "difficulty": "simple",
            "question": "...",
            "raw_evidence": "bad refers to AVG(nope)",
        }
        schemas = {"shop": _DDL_SHOP}

        result_item, sidecar = _extract_one(item, schemas, llm)

        self.assertIsNone(result_item)
        self.assertIsNotNone(sidecar)
        self.assertEqual(sidecar["question_id"], 99)
        self.assertEqual(sidecar["db_id"], "shop")
        self.assertEqual(sidecar["stage"], "grounding_hallucinated")
        self.assertIn("error", sidecar)

    def test_missing_db_id_returns_schema_missing_sidecar(self):
        from bird_eval.batch_extract import (
            STAGE_SCHEMA_MISSING,
            _extract_one,
        )

        # The item references a db_id that is NOT in the schemas dict;
        # extract_one should bail before calling the LLM.
        llm = _MockLLM([])  # would IndexError if called
        item = {
            "question_id": 7, "db_id": "absent_db", "difficulty": "simple",
            "question": "?", "raw_evidence": "some evidence",
        }
        schemas = {"shop": _DDL_SHOP}

        result_item, sidecar = _extract_one(item, schemas, llm)

        self.assertIsNone(result_item)
        self.assertIsNotNone(sidecar)
        self.assertEqual(sidecar["question_id"], 7)
        self.assertEqual(sidecar["db_id"], "absent_db")
        self.assertEqual(sidecar["stage"], STAGE_SCHEMA_MISSING)
        self.assertIn("absent_db", sidecar["error"])
        # Confirm the LLM was never called.
        self.assertEqual(llm.calls, 0)
