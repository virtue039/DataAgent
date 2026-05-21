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


_TYPE_KEYWORDS = (
    "INTEGER", "REAL", "TEXT", "NUMERIC", "BLOB",
    "DATE", "TIMESTAMP", "BOOLEAN", "VARCHAR", "CHAR", "DATETIME",
)


def parse_ddl_typed(ddl: str) -> dict[tuple[str, str], dict]:
    """Return {(table, column): {"type": <SQLite type keyword>}} for the DDL.

    Same parser as `parse_ddl`, but preserves the type keyword of each column.
    Used by bird_eval.joint_graph to build the L2 layer of a JointGraph.
    """
    out: dict[tuple[str, str], dict] = {}
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
        for line in body.split(','):
            line = line.strip()
            mm = re.match(
                r'(?:["`])?([A-Za-z][\w \(\)/\-\.%]*?)(?:["`])?\s+'
                r'(' + "|".join(_TYPE_KEYWORDS) + r')',
                line, re.I,
            )
            if mm:
                col = mm.group(1).strip()
                col_type = mm.group(2).upper()
                out[(tname, col)] = {"type": col_type}
    return out


_FK_CLAUSAL_RE = re.compile(
    r'FOREIGN\s+KEY\s*\(\s*["`]?(\w+)["`]?\s*\)\s+'
    r'REFERENCES\s+["`]?(\w+)["`]?\s*\(\s*["`]?(\w+)["`]?\s*\)',
    re.IGNORECASE,
)


def parse_fks(ddl: str) -> tuple[tuple[tuple[str, str], tuple[str, str]], ...]:
    """Return ((from_table, from_col), (to_table, to_col)) tuples.

    Recognized syntax:
    - Inline:   `col TYPE REFERENCES other (other_col)`
    - Clausal:  `FOREIGN KEY (col) REFERENCES other (other_col)`

    `REFERENCES table` (implicit-PK form, no target column) is intentionally
    SKIPPED -- the joint-graph contract requires both endpoints. Documented
    in the spec §3.4 and as a known limitation in §6 R-implicit-PK.
    """
    edges: list[tuple[tuple[str, str], tuple[str, str]]] = []
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
        for line in body.split(','):
            line = line.strip()
            # Clausal: FOREIGN KEY (a) REFERENCES B (c)
            cm = _FK_CLAUSAL_RE.search(line)
            if cm:
                edges.append(((tname, cm.group(1)), (cm.group(2), cm.group(3))))
                continue
            # Inline: col TYPE [...] REFERENCES B (c)
            im = re.match(
                r'(?:["`])?([A-Za-z][\w]*?)(?:["`])?\s+\w+'   # col TYPE
                r'(?:\s+[A-Za-z_]+(?:\s+\w+)?)*\s+'           # optional modifiers
                r'REFERENCES\s+["`]?(\w+)["`]?\s*\(\s*["`]?(\w+)["`]?\s*\)',
                line, re.IGNORECASE,
            )
            if im:
                edges.append(((tname, im.group(1)), (im.group(2), im.group(3))))
    return tuple(edges)
