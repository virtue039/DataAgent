"""Within-DB stratified train/eval split for P4-revised cross-validation.

Pure function. Deterministic on input order (after a stable sort by
question_id within each db). KB gets floor(n_db * kb_fraction) qids per db;
eval gets the rest.

See docs/superpowers/specs/2026-05-22-p4r-within-db-cv-design.md.
"""
from __future__ import annotations

from collections import defaultdict


def make_within_db_split(
    items: list[dict],
    kb_fraction: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Stratified within-db 50/50 split returning (kb_qids, eval_qids).

    Within each db (grouped by `db_id`), qids are sorted ascending and the
    first `floor(n_db * kb_fraction)` go to KB; the rest go to eval. So:
    - n_db = 4, kb_fraction=0.5 → 2 KB, 2 eval.
    - n_db = 5, kb_fraction=0.5 → 2 KB, 3 eval (rounding favors larger eval).
    - n_db = 1 → 0 KB, 1 eval (single-qid dbs contribute only to eval).
    """
    by_db: dict[str, list[int]] = defaultdict(list)
    for it in items:
        qid = it.get("question_id")
        db = it.get("db_id")
        if qid is None or db is None:
            continue
        by_db[db].append(qid)

    kb: list[int] = []
    eval_set: list[int] = []
    for db in sorted(by_db):
        qids_sorted = sorted(by_db[db])
        cut = int(len(qids_sorted) * kb_fraction)  # floor
        kb.extend(qids_sorted[:cut])
        eval_set.extend(qids_sorted[cut:])

    return kb, eval_set
