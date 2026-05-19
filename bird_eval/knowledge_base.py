"""Building and loading the flat-RAG knowledge base.

The KB is constructed from BIRD *train* evidence only, so it never overlaps
with dev databases -- the non-overlap setting from Research_Plan.md (OQ2).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class KBEntry:
    text: str
    db_id: str
    question_id: int


def build_kb_from_bird(train_json: Path) -> list[KBEntry]:
    """Collect unique, non-empty evidence strings from a BIRD train.json."""
    records = json.loads(Path(train_json).read_text(encoding="utf-8"))
    entries: list[KBEntry] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        text = (rec.get("evidence") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            KBEntry(
                text=text,
                db_id=rec.get("db_id", ""),
                question_id=rec.get("question_id", i),
            )
        )
    return entries


def save_kb(entries: list[KBEntry], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(e) for e in entries]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_kb(path: Path) -> list[KBEntry]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {path}. Build it first with build_kb.py "
            "(or scripts/build_kb.sh)."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [KBEntry(**entry) for entry in data]
