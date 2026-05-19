"""Loading BIRD dev examples and extracting database schema DDL."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass
class BirdExample:
    question_id: int
    db_id: str
    question: str
    evidence: str
    gold_sql: str
    difficulty: str


def locate_bird_files(bird_dir: Path) -> tuple[Path, Path]:
    """Find dev.json and the database root inside an extracted BIRD dev folder.

    BIRD release layouts differ between versions, so search a few levels deep.
    Returns (dev_json_path, db_root) where db_root contains <db_id>/<db_id>.sqlite.
    """
    bird_dir = Path(bird_dir)
    if not bird_dir.exists():
        raise FileNotFoundError(f"BIRD directory not found: {bird_dir}")

    dev_json = next(iter(bird_dir.rglob("dev.json")), None)
    if dev_json is None:
        raise FileNotFoundError(
            f"Could not find dev.json under {bird_dir}. Download the BIRD dev set "
            "and point --bird-dir at the extracted folder."
        )

    first_db = next(iter(bird_dir.rglob("*.sqlite")), None)
    if first_db is None:
        raise FileNotFoundError(f"Could not find any .sqlite database under {bird_dir}.")

    return dev_json, first_db.parent.parent


def load_examples(dev_json: Path, limit: int | None = None) -> list[BirdExample]:
    records = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    examples: list[BirdExample] = []
    for i, rec in enumerate(records):
        examples.append(
            BirdExample(
                question_id=rec.get("question_id", i),
                db_id=rec["db_id"],
                question=rec["question"],
                evidence=(rec.get("evidence") or "").strip(),
                gold_sql=(rec.get("SQL") or rec.get("query") or "").strip(),
                difficulty=rec.get("difficulty", "unknown"),
            )
        )
    if limit is not None:
        examples = examples[:limit]
    return examples


def db_path(db_root: Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def _read_only_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


@lru_cache(maxsize=256)
def _schema_ddl(sqlite_path_str: str, sample_rows: int) -> str:
    path = Path(sqlite_path_str)
    if not path.exists():
        raise FileNotFoundError(f"SQLite database not found: {path}")
    conn = sqlite3.connect(_read_only_uri(path), uri=True)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        blocks: list[str] = []
        for name, sql in tables:
            block = sql.strip().rstrip(";") + ";"
            if sample_rows > 0:
                block += _sample_block(conn, name, sample_rows)
            blocks.append(block)
        return "\n\n".join(blocks)
    finally:
        conn.close()


def _sample_block(conn: sqlite3.Connection, table: str, n: int) -> str:
    try:
        cur = conn.execute(f'SELECT * FROM "{table}" LIMIT {n}')
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    except sqlite3.Error:
        return ""
    if not rows:
        return ""
    lines = [f"\n/* sample rows ({', '.join(cols)}): */"]
    lines += [f"/*   {row} */" for row in rows]
    return "\n" + "\n".join(lines)


def get_schema_ddl(sqlite_path: Path, sample_rows: int = 0) -> str:
    """Return concatenated CREATE TABLE statements for a database (cached)."""
    return _schema_ddl(str(sqlite_path), sample_rows)
