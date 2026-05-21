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
- schema_missing         : the item's db_id had no entry in the schemas dict
                           (typically a misconfigured --train-dbs-root or a
                           stale --schemas-cache).

See docs/superpowers/specs/2026-05-21-p2-bird-train-extraction-design.md.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .extraction import extract

_LOG = logging.getLogger(__name__)

# Stage labels (sidecar JSONL "stage" field).
STAGE_LLM_ERROR = "llm_error"
STAGE_EXTRACT_FAILED = "extract_failed"
STAGE_GROUNDING_HALLUCINATED = "grounding_hallucinated"
STAGE_SCHEMA_MISSING = "schema_missing"


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
    2. Else dropped_nodes>0 -> STAGE_GROUNDING_HALLUCINATED (even if nodes=[]).
    3. Else nodes is empty -> STAGE_EXTRACT_FAILED.
    4. Else                -> None (success).
    """
    if exc is not None:
        return STAGE_LLM_ERROR
    if (stats or {}).get("dropped_nodes", 0) > 0:
        return STAGE_GROUNDING_HALLUCINATED
    if not nodes:
        return STAGE_EXTRACT_FAILED
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


def _count_dropped_by_stage(sidecar_path: Path) -> dict[str, int]:
    """Tally existing sidecar entries by stage label. Returns empty dict if
    the file doesn't exist or is unreadable.
    """
    counts: dict[str, int] = {}
    if not Path(sidecar_path).exists():
        return counts
    try:
        for line in Path(sidecar_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            stage = entry.get("stage")
            if isinstance(stage, str):
                counts[stage] = counts.get(stage, 0) + 1
    except OSError:
        pass
    return counts


def _atomic_write_partial(target_path: Path, payload: dict) -> None:
    """Write payload to `<target>.tmp` then atomically rename to `target`.

    Atomicity matters because we re-write on every checkpoint; a crash
    mid-write must NEVER leave a corrupted partial.json that breaks
    resume logic on the next run.
    """
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    tmp_path.replace(target_path)  # rename is atomic on POSIX


def _extract_one(
    item: dict,
    schemas: dict[str, str],
    llm,
) -> tuple[dict | None, dict | None]:
    """Run the full per-item P2 pipeline.

    Returns exactly one of:
    - (extracted_item, None): pipeline kept the item.
    - (None, sidecar_entry): pipeline dropped the item, sidecar has reason.

    Pipeline:
    1. Call extract(evidence, db_id, ddl, llm). Catches any exception as
       STAGE_LLM_ERROR.
    2. Classify via _classify_failure(exc, nodes, stats).
    3. On success, assemble the v2 item record (preserving the source
       question_id / db_id / difficulty / question / raw_evidence / notes
       fields, replacing 'nodes' with the cleaned list).
    """
    qid = item.get("question_id")
    db_id = item.get("db_id")
    evidence = item.get("raw_evidence") or item.get("evidence") or ""

    if db_id not in schemas:
        return None, {
            "question_id": qid, "db_id": db_id,
            "stage": STAGE_SCHEMA_MISSING,
            "error": f"db_id {db_id!r} not in schemas",
            "raw_response_chars": 0,
        }
    ddl = schemas[db_id]

    exc: Exception | None = None
    nodes: list[dict] = []
    stats: dict[str, int] = {"dropped_groundings": 0, "dropped_nodes": 0}
    try:
        nodes, stats = extract(evidence, db_id, ddl, llm)
    except Exception as e:  # noqa: BLE001 - we classify, not re-raise
        exc = e

    stage = _classify_failure(exc, nodes, stats)
    if stage is None:
        # Success: assemble the kept v2 record.
        return {
            "question_id": qid,
            "db_id": db_id,
            "difficulty": item.get("difficulty"),
            "question": item.get("question"),
            "raw_evidence": evidence,
            "nodes": nodes,
            "notes": item.get("notes", ""),
        }, None

    # Drop: build a sidecar entry.
    if stage == STAGE_LLM_ERROR:
        err_msg = f"{type(exc).__name__}: {exc}"
    elif stage == STAGE_EXTRACT_FAILED:
        err_msg = "extract() returned [] after repair shot (parse or v2 validation failed)"
    else:  # STAGE_GROUNDING_HALLUCINATED
        err_msg = (
            f"sanitize_grounding dropped {stats.get('dropped_nodes', 0)} node(s) and "
            f"{stats.get('dropped_groundings', 0)} grounding entries"
        )
    return None, {
        "question_id": qid,
        "db_id": db_id,
        "stage": stage,
        "error": err_msg,
        "raw_response_chars": 0,
    }


def run_batch(
    items: list[dict],
    schemas: dict[str, str],
    llm,
    output_path: Path,
    sidecar_path: Path,
    *,
    meta: dict | None = None,
    concurrency: int = 8,
    checkpoint_every: int = 200,
    consecutive_llm_error_threshold: int = 100,
    progress_cb=None,
) -> dict:
    """Top-level batch driver.

    Loads any existing partial+sidecar and skips already-processed qids.
    Then dispatches the remaining items across a ThreadPool, classifies
    each outcome via _extract_one, and writes an atomic partial checkpoint
    every `checkpoint_every` items.

    Aborts (raises CatastrophicFailure) if `consecutive_llm_error_threshold`
    items in a row hit STAGE_LLM_ERROR, in arrival order from
    `as_completed`. With `concurrency > 1` futures complete in non-
    deterministic order, so a sporadic success in the middle of a wave of
    LLM errors resets the counter. The guard is designed to catch a fully-
    dead endpoint (where ALL inflight items error); intermittent failures
    are NOT meant to abort the batch -- they pile up in the sidecar.
    The checkpoint and sidecar are intact on abort, so a follow-up run
    resumes from where the abort happened.

    On clean completion, writes the final output to `output_path` (the
    `.partial.json` sibling stays in place and the user can delete it).

    Returns a stats dict:
      {"kept": int, "dropped_llm_error": int, "dropped_extract_failed": int,
       "dropped_grounding_hallucinated": int, "dropped_schema_missing": int,
       "total": int}
    """
    output_path = Path(output_path)
    sidecar_path = Path(sidecar_path)
    partial_path = output_path.with_suffix(".partial.json")

    payload, skip = _load_checkpoint(partial_path, sidecar_path)
    if meta is not None:
        payload["meta"] = {**payload.get("meta", {}), **meta}
    payload["schemas"] = schemas  # always write current schemas dict

    accumulated_items: list[dict] = list(payload.get("items", []))
    todo = [it for it in items if it.get("question_id") not in skip]

    # Initialize drop counters from prior sidecar entries so the final
    # stats reflect all-time totals (consistent with how `kept` is
    # initialized from the prior partial's items[]).
    prior_dropped = _count_dropped_by_stage(sidecar_path)
    stats = {
        "kept": len(accumulated_items),
        "dropped_llm_error": prior_dropped.get(STAGE_LLM_ERROR, 0),
        "dropped_extract_failed": prior_dropped.get(STAGE_EXTRACT_FAILED, 0),
        "dropped_grounding_hallucinated": prior_dropped.get(STAGE_GROUNDING_HALLUCINATED, 0),
        "dropped_schema_missing": prior_dropped.get(STAGE_SCHEMA_MISSING, 0),
        "total": len(items),
    }
    consecutive_llm_errors = 0
    completed_since_checkpoint = 0

    # Open sidecar in append mode so we don't clobber prior drops.
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_fh = sidecar_path.open("a", encoding="utf-8")

    def _checkpoint() -> None:
        payload["items"] = accumulated_items
        _atomic_write_partial(partial_path, payload)

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = {
                pool.submit(_extract_one, it, schemas, llm): it
                for it in todo
            }
            for fut in as_completed(futs):
                kept_item, sidecar_entry = fut.result()
                if kept_item is not None:
                    accumulated_items.append(kept_item)
                    stats["kept"] += 1
                    consecutive_llm_errors = 0
                else:
                    stage = sidecar_entry["stage"]
                    sidecar_fh.write(json.dumps(sidecar_entry, ensure_ascii=False))
                    sidecar_fh.write("\n")
                    sidecar_fh.flush()
                    if stage == STAGE_LLM_ERROR:
                        stats["dropped_llm_error"] += 1
                        consecutive_llm_errors += 1
                    elif stage == STAGE_EXTRACT_FAILED:
                        stats["dropped_extract_failed"] += 1
                        consecutive_llm_errors = 0
                    elif stage == STAGE_GROUNDING_HALLUCINATED:
                        stats["dropped_grounding_hallucinated"] += 1
                        consecutive_llm_errors = 0
                    elif stage == STAGE_SCHEMA_MISSING:
                        stats["dropped_schema_missing"] += 1
                        consecutive_llm_errors = 0

                completed_since_checkpoint += 1
                if completed_since_checkpoint >= checkpoint_every:
                    _checkpoint()
                    completed_since_checkpoint = 0

                if progress_cb is not None:
                    progress_cb()

                if consecutive_llm_errors >= consecutive_llm_error_threshold:
                    _checkpoint()
                    raise CatastrophicFailure(
                        f"{consecutive_llm_errors} consecutive LLM errors; aborting batch."
                    )

        # Final checkpoint, then promote to the final output path atomically.
        _checkpoint()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_partial(output_path, payload)
    finally:
        sidecar_fh.close()

    return stats
