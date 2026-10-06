"""Runs the evaluation dataset against the agent as a Langfuse experiment.

The dataset lives in Langfuse (Datasets -> company-helper-eval); edit items there.

    uv run python -m evals.run                      # full run; results appear as a dataset run
    uv run python -m evals.run --ids q001,q031 --no-judge
    uv run python -m evals.run --ci                 # exit 1 when a threshold is breached
"""
import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

from langfuse import Evaluation, get_client

from agent.graph import agent_config, build_context, graph, initial_state
from evals.judge import JUDGE_MODEL, judge
from evals.metrics import citation_precision_recall, recall_at_k, reciprocal_rank, unique_in_order
from ingestion.vector_store import async_pool

LANGFUSE_DATASET = "company-helper-eval"
EXPERIMENT_NAME = "company-helper-eval"
RESULTS_DIR = Path(__file__).parent / "results"

# Initial investigation thresholds. A breach fails --ci; tighten them as the agent improves.
THRESHOLDS = {
    "recall@5": (">=", 0.90),
    "mrr": (">=", 0.80),
    "context_recall": (">=", 0.90),
    "citation_precision": (">=", 0.90),
    "answer_correctness": (">=", 4.0),
    "faithfulness": (">=", 4.5),
    "hallucination_rate": ("<=", 0.05),
    "abstention_accuracy": (">=", 0.90),
}

# Item-level score name -> aggregate metric name (aggregate = mean over items that have the score).
AGGREGATES = {
    "recall@1": "recall@1",
    "recall@3": "recall@3",
    "recall@5": "recall@5",
    "context_recall": "context_recall",
    "reciprocal_rank": "mrr",
    "citation_precision": "citation_precision",
    "citation_recall": "citation_recall",
    "answer_correctness": "answer_correctness",
    "faithfulness": "faithfulness",
    "completeness": "completeness",
    "citation_accuracy": "citation_accuracy",
    "relevance": "relevance",
    "hallucination": "hallucination_rate",
    "abstention_correct": "abstention_accuracy",
}


# --- task ---------------------------------------------------------------------

_pool_lock: Optional[asyncio.Lock] = None


async def _ensure_pool() -> None:
    # run_experiment runs its own event loop, so the async pool is opened lazily inside it.
    global _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    async with _pool_lock:
        if async_pool.closed:
            await async_pool.open(wait=True)


async def agent_task(*, item: Any, **kwargs: Any) -> Dict[str, Any]:
    await _ensure_pool()
    state = await graph.ainvoke(initial_state(item.input["question"]))
    _, context = build_context(state["chunks"])
    return {
        "answer": state["answer"],
        "error": state["error"],
        "cited": [{"page_id": s["page_id"], "title": s["title"]} for s in state["sources"]],
        "retrieved": [
            {"page_id": c["page_id"], "title": c["page_title"], "score": round(c["score"], 3)}
            for c in state["retrieved"]
        ],
        # Pages that passed MIN_SCORE, i.e. what the LLM actually saw.
        "context_pages": unique_in_order(c["page_id"] for c in state["chunks"]),
        "context": context,
    }


# --- item-level evaluators ----------------------------------------------------

def retrieval_evaluator(*, output: Any, expected_output: Any, **kwargs: Any) -> List[Evaluation]:
    # For abstain items, expected_sources only lists pages that are acceptable to cite,
    # so there is nothing the retriever must find.
    if not output or expected_output["should_abstain"]:
        return []
    ranked = unique_in_order(r["page_id"] for r in output["retrieved"])
    expected = [s["page_id"] for s in expected_output["sources"]]
    if not expected:
        return []
    evaluations = [
        Evaluation(name=f"recall@{k}", value=recall_at_k(ranked, expected, k)) for k in (1, 3, 5)
    ]
    evaluations.append(Evaluation(name="reciprocal_rank", value=reciprocal_rank(ranked, expected)))
    # Recall after the MIN_SCORE cutoff: a relevant page ranked in the top 5 but filtered
    # out never reaches the LLM. A gap between recall@5 and this points at the threshold.
    evaluations.append(Evaluation(
        name="context_recall", value=recall_at_k(output["context_pages"], expected, len(output["context_pages"])),
    ))
    return evaluations


def citation_evaluator(*, output: Any, expected_output: Any, **kwargs: Any) -> List[Evaluation]:
    if not output:
        return []
    precision, recall = citation_precision_recall(
        [c["page_id"] for c in output["cited"]],
        [s["page_id"] for s in expected_output["sources"]],
        expected_output["should_abstain"],
    )
    evaluations = [Evaluation(name="citation_precision", value=precision)]
    if recall is not None:
        evaluations.append(Evaluation(name="citation_recall", value=recall))
    return evaluations


async def judge_evaluator(
    *, input: Any, output: Any, expected_output: Any, metadata: Any, **kwargs: Any
) -> List[Evaluation]:
    if not output:
        return []
    try:
        verdict = await judge(
            question=input["question"],
            expected_answer=expected_output["answer"],
            evaluation_notes=(metadata or {}).get("evaluation_notes", ""),
            should_abstain=expected_output["should_abstain"],
            expected_sources=[s["title"] for s in expected_output["sources"]],
            context=output["context"],
            answer=output["answer"],
            cited_sources=[c["title"] for c in output["cited"]],
        )
    except Exception as exc:
        print(f"[judge] failed for {(metadata or {}).get('id')}: {exc}")
        return []
    reason = verdict["reason"]
    return [
        *(
            Evaluation(name=k, value=verdict[k], comment=reason)
            for k in ("answer_correctness", "faithfulness", "completeness", "citation_accuracy", "relevance")
        ),
        Evaluation(name="hallucination", value=float(verdict["hallucination"]), data_type="BOOLEAN", comment=reason),
        Evaluation(name="abstention_correct", value=float(verdict["abstention_correct"]), data_type="BOOLEAN", comment=reason),
        Evaluation(name="failure_mode", value=verdict["failure_mode"], data_type="CATEGORICAL", comment=reason),
    ]


# --- run-level evaluators -----------------------------------------------------

def aggregate(item_results: List[Any]) -> Dict[str, float]:
    values: Dict[str, List[float]] = defaultdict(list)
    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name in AGGREGATES and evaluation.value is not None:
                values[AGGREGATES[evaluation.name]].append(float(evaluation.value))
    return {metric: mean(v) for metric, v in values.items()}


def summary_evaluator(*, item_results: List[Any], **kwargs: Any) -> List[Evaluation]:
    """Run-level scores, attached to the dataset run in Langfuse."""
    return [Evaluation(name=metric, value=round(value, 4)) for metric, value in aggregate(item_results).items()]


async def close_pool_evaluator(**kwargs: Any) -> List[Evaluation]:
    # Not a metric: the last hook that runs inside run_experiment's event loop, where the pool lives.
    await async_pool.close()
    return []


# --- reporting ----------------------------------------------------------------

def breached(metrics: Dict[str, float]) -> List[str]:
    out = []
    for metric, (op, threshold) in THRESHOLDS.items():
        if metric not in metrics:
            continue
        value = metrics[metric]
        if (op == ">=" and value < threshold) or (op == "<=" and value > threshold):
            out.append(metric)
    return out


def item_rows(item_results: List[Any]) -> List[Dict[str, Any]]:
    rows = []
    for result in item_results:
        metadata = result.item.metadata or {}
        scores = {e.name: e.value for e in result.evaluations}
        reason = next((e.comment for e in result.evaluations if e.name == "failure_mode"), None)
        rows.append({
            "id": metadata.get("id"),
            "question_type": metadata.get("question_type"),
            "question": result.item.input["question"],
            "answer": result.output["answer"] if result.output else None,
            "cited": [c["title"] for c in result.output["cited"]] if result.output else [],
            "retrieved": [r["title"] for r in result.output["retrieved"]] if result.output else [],
            "scores": scores,
            "reason": reason,
            "trace_id": result.trace_id,
        })
    return sorted(rows, key=lambda r: r["id"] or "")


def is_problem(row: Dict[str, Any]) -> bool:
    s = row["scores"]
    return (
        s.get("failure_mode", "none") != "none"
        or s.get("hallucination") == 1
        or s.get("abstention_correct") == 0
        or s.get("recall@5", 1) < 1
        or s.get("context_recall", 1) < 1
        or s.get("citation_precision", 1) < 1
    )


def print_report(
    run_name: str, config: Dict[str, str], metrics: Dict[str, float], rows: List[Dict[str, Any]], total: int
) -> None:
    print(f"\n=== {run_name}  ({len(rows)}/{total} items evaluated)")
    print(f"config: {config}  judge: {JUDGE_MODEL}\n")
    bad = set(breached(metrics))
    for metric in AGGREGATES.values():
        if metric not in metrics:
            continue
        target = THRESHOLDS.get(metric)
        flag = "  ✗ BREACH" if metric in bad else ("  ✓" if target else "")
        target_txt = f"(target {target[0]} {target[1]})" if target else ""
        print(f"  {metric:<20} {metrics[metric]:>6.3f}  {target_txt}{flag}")

    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row)
    print(f"\n  {'question_type':<20} {'n':>3} {'correct':>8} {'faithful':>9} {'halluc.':>8} {'recall@5':>9}")
    for qtype, group in sorted(by_type.items()):
        def avg(key):
            vals = [r["scores"][key] for r in group if key in r["scores"]]
            return f"{mean(vals):.2f}" if vals else "-"
        hallucinations = sum(1 for r in group if r["scores"].get("hallucination") == 1)
        print(f"  {qtype:<20} {len(group):>3} {avg('answer_correctness'):>8} {avg('faithfulness'):>9} {hallucinations:>8} {avg('recall@5'):>9}")

    problems = [r for r in rows if is_problem(r)]
    print(f"\n  Items to review ({len(problems)}):")
    for row in problems:
        s = row["scores"]
        print(f"  - {row['id']} [{row['question_type']}] {s.get('failure_mode', '?')}"
              f"  correct={s.get('answer_correctness', '-')} faithful={s.get('faithfulness', '-')}"
              f" recall@5={s.get('recall@5', '-')} ctx_recall={s.get('context_recall', '-')}"
              f" cite_prec={s.get('citation_precision', '-')}")
        print(f"      Q: {row['question']}")
        print(f"      A: {(row['answer'] or '').replace(chr(10), ' ')[:220]}")
        if row["reason"]:
            print(f"      Judge: {row['reason']}")


# --- main ---------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Company Helper agent.")
    parser.add_argument("--name", help="Run name (default: <model>-<prompt_version>-<timestamp>)")
    parser.add_argument("--ids", help="Comma-separated item ids to run, e.g. q001,q031")
    parser.add_argument("--limit", type=int, help="Only run the first N items")
    parser.add_argument("--no-judge", action="store_true", help="Deterministic metrics only (no LLM judge cost)")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--ci", action="store_true", help="Exit 1 if a threshold is breached or an item fails")
    args = parser.parse_args()

    langfuse = get_client()
    data = list(langfuse.get_dataset(LANGFUSE_DATASET).items)
    if args.ids:
        wanted = set(args.ids.split(","))
        data = [item for item in data if (item.metadata or {}).get("id") in wanted]
    if args.limit:
        data = data[: args.limit]
    if not data:
        raise SystemExit("No items selected.")

    config = agent_config()
    run_name = args.name or (
        f"{config['model']}-{config['prompt_version']}-{datetime.now():%Y%m%d-%H%M%S}"
    )
    evaluators = [retrieval_evaluator, citation_evaluator] + ([] if args.no_judge else [judge_evaluator])
    result = langfuse.run_experiment(
        name=EXPERIMENT_NAME,
        run_name=run_name,
        description=f"Agent config {config}",
        data=data,
        task=agent_task,
        evaluators=evaluators,
        run_evaluators=[summary_evaluator, close_pool_evaluator],
        max_concurrency=args.concurrency,
        metadata={**config, "judge_model": JUDGE_MODEL},
    )
    langfuse.flush()

    metrics = aggregate(result.item_results)
    rows = item_rows(result.item_results)
    failed_items = len(data) - len(result.item_results)
    print_report(run_name, config, metrics, rows, len(data))

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{run_name}.json"
    out_path.write_text(json.dumps({
        "run_name": run_name,
        "config": config,
        "judge_model": JUDGE_MODEL,
        "metrics": metrics,
        "breached": breached(metrics),
        "failed_items": failed_items,
        "items": rows,
    }, indent=2, ensure_ascii=False, default=str))
    print(f"\nResults: {out_path}")
    if result.dataset_run_url:
        print(f"Langfuse: {result.dataset_run_url}")

    if failed_items:
        print(f"\n{failed_items} item(s) failed to run; see the log above.")
    if args.ci and (breached(metrics) or failed_items):
        sys.exit(1)


if __name__ == "__main__":
    main()
