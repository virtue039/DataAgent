"""Acceptance gate (spec §8 item 2): every (table, column) reference across
grounding[] and ColumnAlias.bindings must exist in the embedded DDL."""
from __future__ import annotations
import json
import sys
import os

# Add parent directory to path so bird_eval module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bird_eval.ddl import parse_ddl


def main() -> int:
    d = json.load(open('annotation/to_annotate.json'))
    dbs = {db: parse_ddl(ddl) for db, ddl in d['schemas'].items()}
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
