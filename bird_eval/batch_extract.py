"""P2 batch extraction core.

This is the testable core; the CLI shim is `extract_bird_train.py`.
The public surface is `run_batch(...)`; everything else is module-private
but exported with leading underscore for unit tests.

Stage labels for the sidecar JSONL:
- llm_error              : underlying LLM call raised (timeout/network/5xx).
- extract_failed         : extract() returned [] after one repair shot.
                           (Combines parse_error + validator_error; see
                           the plan's "Implementation note on stage labels".)
- grounding_hallucinated : sanitize_grounding dropped at least one node.

See docs/superpowers/specs/2026-05-21-p2-bird-train-extraction-design.md.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

_LOG = logging.getLogger(__name__)

# Stage labels (sidecar JSONL "stage" field).
STAGE_LLM_ERROR = "llm_error"
STAGE_EXTRACT_FAILED = "extract_failed"
STAGE_GROUNDING_HALLUCINATED = "grounding_hallucinated"


class CatastrophicFailure(RuntimeError):
    """Raised by run_batch when systemic failure is detected (e.g. endpoint down)."""


def _classify_failure(
    exc: Exception | None,
    nodes: list[dict],
    stats: dict[str, int] | None,
) -> str | None:
    """Map a single item's extract() outcome to a stage label.

    Returns one of the STAGE_* constants, or None on success.

    Priority order:
    1. If exc is not None  -> STAGE_LLM_ERROR.
    2. Else nodes is empty -> STAGE_EXTRACT_FAILED.
    3. Else dropped_nodes>0 -> STAGE_GROUNDING_HALLUCINATED.
    4. Else                -> None (success).
    """
    if exc is not None:
        return STAGE_LLM_ERROR
    if not nodes:
        return STAGE_EXTRACT_FAILED
    if (stats or {}).get("dropped_nodes", 0) > 0:
        return STAGE_GROUNDING_HALLUCINATED
    return None


def _load_checkpoint(
    partial_path: Path,
    sidecar_path: Path,
) -> tuple[dict, set[int]]:
    """Read existing partial.json + sidecar.jsonl into (payload, skip_qids).

    On startup the batch loop uses skip_qids to avoid reprocessing items
    already attempted in a previous run (success OR drop).

    Robust to:
    - partial missing: returns empty payload + empty set.
    - sidecar missing: contributes zero qids to the set.
    - partial corrupted (invalid JSON): logs a warning, returns empty payload.
    """
    payload: dict = {"meta": {}, "schemas": {}, "items": []}
    skip: set[int] = set()

    if Path(partial_path).exists():
        try:
            payload = json.loads(Path(partial_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            _LOG.warning(
                "Could not read partial %s (%s); starting fresh.",
                partial_path, e,
            )
            payload = {"meta": {}, "schemas": {}, "items": []}
        else:
            if not isinstance(payload, dict):
                _LOG.warning(
                    "Partial %s is valid JSON but not an object (got %s); starting fresh.",
                    partial_path, type(payload).__name__,
                )
                payload = {"meta": {}, "schemas": {}, "items": []}
            else:
                for it in payload.get("items", []):
                    qid = it.get("question_id")
                    if isinstance(qid, int):
                        skip.add(qid)

    if Path(sidecar_path).exists():
        try:
            for line in Path(sidecar_path).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                qid = entry.get("question_id")
                if isinstance(qid, int):
                    skip.add(qid)
        except OSError as e:
            _LOG.warning("Could not read sidecar %s (%s); continuing.",
                         sidecar_path, e)

    return payload, skip
