"""SQL execution and BIRD-style execution-accuracy comparison."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from .data import _read_only_uri


def execute_sql(sqlite_path: Path, sql: str, timeout: float = 30.0):
    """Run a query read-only against a SQLite file.

    Returns (rows, error_message); exactly one is non-None.
    """
    path = Path(sqlite_path)
    if not path.exists():
        return None, f"database not found: {path}"
    if not sql.strip():
        return None, "empty SQL"

    conn = sqlite3.connect(_read_only_uri(path), uri=True)
    timer = threading.Timer(timeout, conn.interrupt)
    timer.start()
    try:
        rows = conn.execute(sql).fetchall()
        return rows, None
    except Exception as e:  # noqa: BLE001 - any failure means the query is wrong
        return None, str(e)
    finally:
        timer.cancel()
        conn.close()


def execution_match(
    sqlite_path: Path, pred_sql: str, gold_sql: str, timeout: float = 30.0
) -> tuple[bool, str | None]:
    """BIRD execution accuracy: predicted result set must equal the gold one.

    Returns (is_correct, error_message). error_message is set when the predicted
    query fails to execute, or when the gold query itself errors (a data issue).
    """
    gold_rows, gold_err = execute_sql(sqlite_path, gold_sql, timeout)
    if gold_err is not None:
        return False, f"gold SQL error: {gold_err}"
    pred_rows, pred_err = execute_sql(sqlite_path, pred_sql, timeout)
    if pred_err is not None:
        return False, pred_err
    return set(gold_rows) == set(pred_rows), None
