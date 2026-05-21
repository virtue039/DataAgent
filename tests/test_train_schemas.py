"""Unit tests for bird_eval/train_schemas.py (P2)."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bird_eval.train_schemas import build_train_schemas  # noqa: E402


def _make_mini_db(parent: Path, db_id: str) -> Path:
    """Create a tiny sqlite db under parent/<db_id>/<db_id>.sqlite."""
    db_dir = parent / db_id
    db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = db_dir / f"{db_id}.sqlite"
    conn = sqlite3.connect(sqlite_path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()
    return sqlite_path


class BuildTrainSchemas(unittest.TestCase):
    def test_extracts_ddl_from_minimal_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dbs"
            _make_mini_db(root, "shop")
            cache = Path(tmp) / "schemas.json"

            schemas = build_train_schemas(
                train_dbs_root=root, cache_path=cache
            )

            self.assertIn("shop", schemas)
            self.assertIn("CREATE TABLE", schemas["shop"])
            self.assertIn("users", schemas["shop"])
            # Cache file should be written.
            self.assertTrue(cache.exists())

    def test_second_call_hits_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "dbs"
            _make_mini_db(root, "shop")
            cache = Path(tmp) / "schemas.json"

            build_train_schemas(train_dbs_root=root, cache_path=cache)
            # Sabotage the live DDL extractor path: delete the source db.
            # If the cache is honored, this still succeeds.
            (root / "shop" / "shop.sqlite").unlink()

            schemas = build_train_schemas(
                train_dbs_root=root, cache_path=cache
            )
            self.assertIn("shop", schemas)
            self.assertIn("CREATE TABLE", schemas["shop"])
