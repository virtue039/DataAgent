"""LLM-as-judge runner for P4: classifies each oracle-correct/joint-wrong
qid into one of {specificity_loss, retrieval_irrelevance, format_dilution,
other}.

Reads results/p4_audit/audit_targets.json (from T3), calls the local
DeepSeek endpoint per qid, writes one JSONL line per judgment to
results/p4_audit/judgments.jsonl.

Resume-safe: on startup, scan judgments.jsonl, skip any qid already
present.

Usage:
  .venv/bin/python scripts/llm_judge_failures.py \
    [--targets results/p4_audit/audit_targets.json] \
    [--output results/p4_audit/judgments.jsonl] \
    [--concurrency 8]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bird_eval.config import Config  # noqa: E402
from bird_eval.llm import LLMClient  # noqa: E402


_VALID_CATEGORIES = {"specificity_loss", "retrieval_irrelevance",
                     "format_dilution", "other"}

_SYSTEM_PROMPT = """You are evaluating why a text-to-SQL system failed. You will see one question and two SQL-generation attempts:

- Attempt A used the BIRD gold "evidence" string and produced the correct SQL.
- Attempt B used structured knowledge retrieved from a typed-node KB and produced the WRONG SQL.

Your job: identify the PRIMARY cause of B's failure by comparing the two evidence strings (NOT the SQLs). Choose exactly one category:

- specificity_loss: A contains specific constants, exact column names, or exact values that B's bullets lack or abstract away. Example: A says "frequency = 'POPLATEK MESICNE'"; B says "ValueMap: monthly issuance" without the literal value.

- retrieval_irrelevance: B retrieves knowledge about columns/values that are off-topic for this question. The right knowledge wasn't fetched. Example: question is about California schools; B's bullets describe enrollment formulas for a different table.

- format_dilution: B's evidence is technically correct but verbose or scattered across many bullets in a way that distracts the LLM. A is one concise sentence; B is 5+ bullets including irrelevant siblings.

- other: none of the above clearly applies. The failure is likely non-content-causal (model stochasticity, prompt artifact, etc.). Use sparingly — only when you genuinely cannot attribute the failure to evidence content. Explain in `reason`.

Output JSON inside a ```json fence:
{"category": "specificity_loss|retrieval_irrelevance|format_dilution|other",
 "reason": "<one sentence>",
 "evidence_quote": "<one short quote from A or B supporting the choice>"}
"""


_FENCE_RE = re.compile(r"```json\s*(.*?)\s*```", re.DOTALL)


def _build_user(target: dict) -> str:
    return (
        f"Question (db={target['db_id']}, difficulty={target['difficulty']}):\n"
        f"{target['question']}\n\n"
        f"Gold SQL: {target['gold_sql']}\n\n"
        f"==== Attempt A (oracle, CORRECT) ====\n"
        f"Evidence:\n{target['oracle_evidence']}\n\n"
        f"Predicted SQL: {target['oracle_predicted_sql']}\n\n"
        f"==== Attempt B (joint, WRONG) ====\n"
        f"Evidence:\n{target['joint_evidence']}\n\n"
        f"Predicted SQL: {target['joint_predicted_sql']}\n\n"
        "Output JSON inside a ```json fence with category, reason, evidence_quote."
    )


def _parse_judgment(raw: str) -> dict:
    """Parse the LLM's JSON fence into a {category, reason, evidence_quote} dict.

    Falls back to category='other' with the raw text in reason if parsing fails.
    """
    m = _FENCE_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return {"category": "other", "reason": f"parse error: {raw[:200]}",
                    "evidence_quote": ""}
    else:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return {"category": "other", "reason": f"no fence: {raw[:200]}",
                    "evidence_quote": ""}
    cat = obj.get("category", "other")
    if cat not in _VALID_CATEGORIES:
        return {"category": "other",
                "reason": f"invalid category {cat!r}: {obj.get('reason', '')}",
                "evidence_quote": obj.get("evidence_quote", "")}
    return {"category": cat,
            "reason": str(obj.get("reason", ""))[:500],
            "evidence_quote": str(obj.get("evidence_quote", ""))[:500]}


def _load_done(output_path: Path) -> set[int]:
    """Read existing judgments.jsonl and return the set of completed qids."""
    if not output_path.exists():
        return set()
    done = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        qid = obj.get("question_id")
        if isinstance(qid, int):
            done.add(qid)
    return done


def _judge_one(target: dict, llm) -> dict:
    """Run the LLM judge on one target. Returns the judgment dict."""
    try:
        raw = llm.complete(_SYSTEM_PROMPT, _build_user(target))
    except Exception as e:  # noqa: BLE001 — log to sidecar, keep going
        return {"question_id": target["question_id"],
                "db_id": target["db_id"],
                "difficulty": target["difficulty"],
                "category": "other",
                "reason": f"llm_error: {type(e).__name__}: {e}",
                "evidence_quote": ""}
    parsed = _parse_judgment(raw)
    parsed["question_id"] = target["question_id"]
    parsed["db_id"] = target["db_id"]
    parsed["difficulty"] = target["difficulty"]
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge classification of P4 audit targets."
    )
    parser.add_argument("--targets", default="results/p4_audit/audit_targets.json")
    parser.add_argument("--output", default="results/p4_audit/judgments.jsonl")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    targets_path = Path(args.targets)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(targets_path.read_text(encoding="utf-8"))
    all_targets = data["targets"]

    done = _load_done(output_path)
    todo = [t for t in all_targets if t["question_id"] not in done]
    print(f"Loaded {len(all_targets)} targets; {len(done)} already done; "
          f"{len(todo)} to judge.")

    if not todo:
        print("Nothing to do. Output is up to date.")
        return 0

    # Build the LLM client.
    cfg = Config(bird_dir=Path("data/bird_dev"), setting="none")
    llm = LLMClient(cfg)

    # Open sidecar in append mode.
    fh = output_path.open("a", encoding="utf-8")
    pbar = tqdm(total=len(todo), desc="judge", unit="qid")
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futs = {pool.submit(_judge_one, t, llm): t for t in todo}
            for fut in as_completed(futs):
                judg = fut.result()
                fh.write(json.dumps(judg, ensure_ascii=False) + "\n")
                fh.flush()
                pbar.update(1)
    finally:
        pbar.close()
        fh.close()

    # Sanity recount.
    final_done = _load_done(output_path)
    print(f"Wrote {len(final_done)} judgments to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
