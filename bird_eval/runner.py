"""Orchestrates a BIRD evaluation run end to end."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .data import BirdExample, db_path, get_schema_ddl, load_examples, locate_bird_files
from .evidence import make_evidence_provider
from .execution import execution_match
from .llm import LLMClient
from .prompts import SYSTEM_PROMPT, build_user_prompt, extract_sql


def _process(example: BirdExample, *, config: Config, db_root: Path, provider, llm) -> dict:
    sqlite_path = db_path(db_root, example.db_id)
    record: dict = {
        "question_id": example.question_id,
        "db_id": example.db_id,
        "difficulty": example.difficulty,
        "question": example.question,
        "setting": config.setting,
    }
    try:
        schema = get_schema_ddl(sqlite_path, config.schema_sample_rows)
        evidence = provider.get(example)
        record["evidence"] = evidence
        user_prompt = build_user_prompt(schema, example.question, evidence)
    except Exception as e:  # noqa: BLE001
        record.update(error=f"prep failed: {e}", correct=False)
        return record

    if config.dry_run:
        record.update(prompt=user_prompt, correct=None, error=None)
        return record

    try:
        raw = llm.complete(SYSTEM_PROMPT, user_prompt)
    except Exception as e:  # noqa: BLE001
        record.update(error=f"llm failed: {e}", correct=False)
        return record

    pred_sql = extract_sql(raw)
    correct, exec_err = execution_match(
        sqlite_path, pred_sql, example.gold_sql, config.sql_timeout
    )
    record.update(
        predicted_sql=pred_sql,
        gold_sql=example.gold_sql,
        raw_response=raw,
        correct=correct,
        error=exec_err,
    )
    return record


def run(config: Config) -> dict:
    dev_json, db_root = locate_bird_files(config.bird_dir)
    # Load all examples; apply qid filter (if any), then the --limit slice.
    examples = load_examples(dev_json, limit=None)
    if config.question_ids_from is not None:
        target = json.loads(Path(config.question_ids_from).read_text(encoding="utf-8"))
        if isinstance(target, dict):
            items = target.get("items")
            if items is None:
                raise ValueError(
                    f"--question-ids-from JSON dict must contain an 'items' list; "
                    f"got keys: {sorted(target.keys())}"
                )
        else:
            items = target  # assume bare list of {question_id: ...} dicts
        qid_set = {int(it["question_id"]) for it in items}
        examples = [e for e in examples if e.question_id in qid_set]
    if config.limit is not None:
        examples = examples[: config.limit]
    provider = make_evidence_provider(config)
    llm = None if config.dry_run else LLMClient(config)

    print(
        f"Loaded {len(examples)} BIRD examples | setting={config.setting} "
        f"| model={config.model} | dry_run={config.dry_run}"
    )

    def worker(ex: BirdExample) -> dict:
        return _process(ex, config=config, db_root=db_root, provider=provider, llm=llm)

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
        for rec in tqdm(pool.map(worker, examples), total=len(examples), desc=config.setting):
            results.append(rec)

    summary = _summarize(results, config)
    _print_summary(summary)
    out_path = _save(results, summary, config)
    print(f"\nSaved {len(results)} results to {out_path}")
    return summary


def _summarize(results: list[dict], config: Config) -> dict:
    scored = [r for r in results if r.get("correct") is not None]
    total = len(scored)
    correct = sum(1 for r in scored if r["correct"])
    by_diff: dict[str, dict] = {}
    for r in scored:
        bucket = by_diff.setdefault(r["difficulty"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(bool(r["correct"]))
    for bucket in by_diff.values():
        bucket["ex"] = round(bucket["correct"] / bucket["total"], 4) if bucket["total"] else 0.0
    return {
        "setting": config.setting,
        "model": config.model,
        "n_examples": len(results),
        "n_scored": total,
        "ex": round(correct / total, 4) if total else 0.0,
        "exec_errors": sum(1 for r in scored if r.get("error")),
        "by_difficulty": by_diff,
    }


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 50)
    print(f"  Setting        : {summary['setting']}")
    print(f"  Model          : {summary['model']}")
    print(f"  Scored / total : {summary['n_scored']} / {summary['n_examples']}")
    print(f"  Execution Acc. : {summary['ex'] * 100:.2f}%")
    print(f"  Exec errors    : {summary['exec_errors']}")
    for diff, bucket in sorted(summary["by_difficulty"].items()):
        print(
            f"    - {diff:<12}: {bucket['ex'] * 100:6.2f}%  "
            f"({bucket['correct']}/{bucket['total']})"
        )
    print("=" * 50)


def _save(results: list[dict], summary: dict, config: Config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = config.model.replace("/", "-")
    out_path = config.output_dir / f"{config.setting}_{safe_model}_{ts}.json"
    payload = {
        "summary": summary,
        "config": {
            "setting": config.setting,
            "model": config.model,
            "limit": config.limit,
            "schema_sample_rows": config.schema_sample_rows,
            "temperature": config.temperature,
            "retrieval_top_k": config.retrieval_top_k,
            "embedding_model": config.embedding_model,
            "question_ids_from": str(config.question_ids_from) if config.question_ids_from else None,
            "dry_run": config.dry_run,
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
