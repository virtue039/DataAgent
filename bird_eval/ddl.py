"""DDL parsing for the embedded `schemas` block in annotation/to_annotate.json.

Used by tests/check_grounding.py (acceptance gate) and bird_eval/extraction.py
(grounding sanitizer). Lifted here so production code doesn't import from
tests/.
"""
from __future__ import annotations

import re


def parse_ddl(ddl: str) -> dict[str, set[str]]:
    """Return a dict mapping table name -> set of column names.

    Best-effort SQLite DDL parser: handles backticks, double-quotes, bare
    identifiers, and column names containing spaces / parentheses / slashes /
    hyphens / dots / percent signs. Designed for the BIRD dev DDL embedded in
    annotation/to_annotate.json under the `schemas` block.
    """
    tables: dict[str, set[str]] = {}
    parts = re.split(r'(?im)^CREATE\s+TABLE\s+', ddl)
    for p in parts[1:]:
        m = re.match(r'["`]?(\w+)["`]?\s*\(', p, re.S)
        if not m:
            continue
        tname = m.group(1)
        depth = 0
        body = ""
        start = p.find('(')
        for i, ch in enumerate(p[start:], start):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    body = p[start + 1:i]
                    break
        cols: set[str] = set()
        for line in body.split(','):
            line = line.strip()
            mm = re.match(
                r'(?:["`])?([A-Za-z][\w \(\)/\-\.%]*?)(?:["`])?\s+'
                r'(?:INTEGER|REAL|TEXT|NUMERIC|BLOB|DATE|TIMESTAMP|BOOLEAN|VARCHAR|CHAR|DATETIME)',
                line, re.I,
            )
            if mm:
                cols.add(mm.group(1).strip())
        tables[tname] = cols
    return tables
