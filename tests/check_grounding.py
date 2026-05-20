"""Acceptance gate (spec §8 item 2): every (table, column) reference across
grounding[] and ColumnAlias.bindings must exist in the embedded DDL."""
from __future__ import annotations
import json
import re
import sys


def parse_schema(ddl: str) -> dict[str, set[str]]:
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
                r'(?:["`])?([A-Za-z][\w \(\)/\-\.]*?)(?:["`])?\s+'
                r'(?:INTEGER|REAL|TEXT|NUMERIC|BLOB|DATE|TIMESTAMP|BOOLEAN|VARCHAR|CHAR|DATETIME)',
                line, re.I,
            )
            if mm:
                cols.add(mm.group(1).strip())
        tables[tname] = cols
    return tables


def main() -> int:
    d = json.load(open('annotation/to_annotate.json'))
    dbs = {db: parse_schema(ddl) for db, ddl in d['schemas'].items()}
    errors: list[str] = []
    checks = 0
    for it in d['items']:
        tabs = dbs.get(it['db_id'], {})
        for node in it['nodes']:
            if node.get('type') == 'ValueMap':
                checks += 1
                t, c = node['table'], node['column']
                if t not in tabs:
                    errors.append(f"qid={it['question_id']} {node['id']}: table {t!r} unknown")
                elif c not in tabs[t]:
                    errors.append(f"qid={it['question_id']} {node['id']}: {t}.{c!r} unknown")
            if node.get('type') == 'ColumnAlias':
                for b in node.get('bindings', []):
                    checks += 1
                    t, c = b['table'], b['column']
                    if t not in tabs:
                        errors.append(f"qid={it['question_id']} {node['id']}: alias table {t!r} unknown")
                    elif c not in tabs[t]:
                        errors.append(f"qid={it['question_id']} {node['id']}: alias {t}.{c!r} unknown")
            for g in node.get('grounding', []) or []:
                checks += 1
                t, c = g.get('table'), g.get('column')
                if t not in tabs:
                    errors.append(f"qid={it['question_id']} {node['id']} grounding term={g.get('term')!r}: table {t!r} unknown")
                elif c not in tabs[t]:
                    errors.append(f"qid={it['question_id']} {node['id']} grounding term={g.get('term')!r}: {t}.{c!r} unknown")
    print(f"Checked {checks} (table, column) bindings across 40 items")
    if errors:
        print(f"\n{len(errors)} mismatch(es):")
        for e in errors:
            print(' ', e)
        return 1
    print('OK: every (table, column) reference matches a real DDL column.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
