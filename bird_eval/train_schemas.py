"""Build the train-side {db_id: DDL} dictionary for P2 batch extraction.

The dev pipeline reads schemas on demand from one `bird_dir`; here we
prebuild the dict once for 69 train dbs and cache to disk. The batch
extractor loads the cache and never touches SQLite again, so the long
job's per-item path is pure Python/HTTP.
"""
from __future__ import annotations

import json
from pathlib import Path

from .data import get_schema_ddl


def build_train_schemas(
    train_dbs_root: Path = Path("data/bird_train/train/train_databases"),
    cache_path: Path = Path("data/train_schemas.json"),
) -> dict[str, str]:
    """Walk train_dbs_root, build a {db_id: ddl} dict, cache to disk.

    If `cache_path` exists, returns it directly (no SQLite work).
    Otherwise iterates each `<db_id>/<db_id>.sqlite` and calls
    `bird_eval.data.get_schema_ddl(...)` for the DDL string.

    Skips directories that don't contain the expected sqlite file (e.g.
    a stray README dir).
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    train_dbs_root = Path(train_dbs_root)
    if not train_dbs_root.is_dir():
        raise FileNotFoundError(
            f"train_dbs_root not found or not a directory: {train_dbs_root}"
        )
    schemas: dict[str, str] = {}
    for db_dir in sorted(p for p in train_dbs_root.iterdir() if p.is_dir()):
        sqlite_path = db_dir / f"{db_dir.name}.sqlite"
        if not sqlite_path.exists():
            continue
        schemas[db_dir.name] = get_schema_ddl(sqlite_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(schemas, indent=2), encoding="utf-8")
    return schemas
