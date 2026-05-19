"""Prompt construction and SQL extraction."""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You are an expert data analyst. Given a SQLite database schema and a "
    "natural-language question, write a single correct SQLite SQL query that "
    "answers it. Use only tables and columns that exist in the schema."
)


def build_user_prompt(schema_ddl: str, question: str, evidence: str) -> str:
    parts = ["### Database schema (SQLite DDL)", schema_ddl, ""]
    if evidence.strip():
        parts += ["### External knowledge", evidence.strip(), ""]
    parts += [
        "### Question",
        question.strip(),
        "",
        "### Instructions",
        "Write ONE SQLite SQL query that answers the question.",
        "Output ONLY the query inside a ```sql code block, with no explanation.",
    ]
    return "\n".join(parts)


_SQL_BLOCK = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_BLOCK = re.compile(r"```\s*(.*?)```", re.DOTALL)
_FROM_KEYWORD = re.compile(r"\b(?:WITH|SELECT)\b.*", re.DOTALL | re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Pull a single SQL statement out of an LLM response."""
    for pattern in (_SQL_BLOCK, _ANY_BLOCK):
        m = pattern.search(text)
        if m:
            return _clean(m.group(1))
    m = _FROM_KEYWORD.search(text)
    if m:
        return _clean(m.group(0))
    return _clean(text)


def _clean(sql: str) -> str:
    sql = sql.strip()
    # keep only the first statement if the model emitted trailing prose/queries
    if ";" in sql:
        sql = sql[: sql.index(";")]
    return sql.strip()
