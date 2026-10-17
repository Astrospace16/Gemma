"""
evaluate.py: Computes True-Fix Rate (TFR), Silent-Failure Rate (SFR),
Abstention Rate, AUROC, ECE, and Best-of-N selection for SciSilentBench.
"""
import argparse
import asyncio
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from mutator import run_code_check
from agent_adk import run_agent_on_task


def compute_ece(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across n_bins."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def compute_auroc(confidences: np.ndarray, labels: np.ndarray) -> float:
    """Computes AUROC using Mann-Whitney U statistic (no sklearn dependency required)."""
    pos = confidences[labels == 1]
    neg = confidences[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u_stat = sum(
        (p > n) + 0.5 * (p == n)
        for p in pos
        for n in neg
    )
    return float(u_stat / (len(pos) * len(neg)))


def grade_trajectory(task: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates a finished trajectory against both the visible test suite and hidden oracle."""
    code = result["final_code"]
    vis_pass = run_code_check(code, task["visible_test"])
    oracle_pass = run_code_check(code, task["hidden_oracle"])

    submitted = (result["decision"] == "SUBMIT")
    true_fix = submitted and vis_pass and oracle_pass
    silent_failure = submitted and vis_pass and (not oracle_pass)
    abstained = (result["decision"] == "ABSTAIN")

    result.update({
        "bug_type": task.get("bug_type", "UNKNOWN"),
        "vis_pass": vis_pass,
        "oracle_pass": oracle_pass,
        "true_fix": true_fix,
        "silent_failure": silent_failure,
        "abstained": abstained,
    })
    return result


def select_best_of_n(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Best-of-N selector under Condition B:
    Filters for trajectories that chose SUBMIT, sorted by highest self-reported confidence.
    If all candidates abstained, returns the highest-confidence abstention.
    """
    submitted = [c for c in candidates if c["decision"] == "SUBMIT"]
    if submitted:
        best = max(submitted, key=lambda x: x["confidence"])
    else:
        best = max(candidates, key=lambda x: x["confidence"])

    # Aggregate compute costs across all N rollouts
    best = dict(best)
    best["tokens"] = sum(c["tokens"] for c in candidates)
    best["wall_time_s"] = sum(c["wall_time_s"] for c in candidates)
    return best


def summarize_metrics(graded_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregates paper metrics across all tasks."""
    n = len(graded_results)
    if n == 0:
        return {}
    tfr = 100.0 * sum(r["true_fix"] for r in graded_results) / n
    sfr = 100.0 * sum(r["silent_failure"] for r in graded_results) / n
    abs_rate = 100.0 * sum(r["abstained"] for r in graded_results) / n

    confs = np.array([r["confidence"] for r in graded_results], dtype=float)
    labels = np.array([1 if (r["vis_pass"] and r["oracle_pass"]) else 0 for r in graded_results], dtype=int)

    auroc = compute_auroc(confs, labels)
    ece = compute_ece(confs, labels)
    mean_tokens = float(np.mean([r["tokens"] for r in graded_results]))
    mean_time = float(np.mean([r["wall_time_s"] for r in graded_results]))

    return {
        "TFR (%)": round(tfr, 2),
        "SFR (%)": round(sfr, 2),
        "Abstention (%)": round(abs_rate, 2),
        "AUROC": round(auroc, 3),
        "ECE": round(ece, 3),
        "Mean Tokens": round(mean_tokens, 1),
        "Mean Time (s)": round(mean_time, 2),
    }


async def run_evaluation(args):
    """Async evaluation loop."""
    tasks = json.loads(Path(args.tasks).read_text())
    if args.task_ids:
        requested = set(args.task_ids)
        tasks = [task for task in tasks if task["task_id"] in requested]
        missing = requested - {task["task_id"] for task in tasks}
        if missing:
            raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    final_results = []

    for task in tasks:
        temp = 0.7 if args.best_of_n > 1 else 0.2
        rollouts = []
        for _ in range(args.best_of_n):
            try:
                rollout = await asyncio.wait_for(
                    run_agent_on_task(
                        task,
                        model_name=args.model,
                        condition=args.condition,
                        temperature=temp,
                    ),
                    timeout=args.task_timeout,
                )
            except asyncio.TimeoutError:
                rollout = {
                    "task_id": task["task_id"],
                    "final_code": "",
                    "decision": "ABSTAIN",
                    "confidence": 0.0,
                    "reason": f"agent timed out after {args.task_timeout}s",
                    "tokens": 0,
                    "wall_time_s": float(args.task_timeout),
                    "turns": 0,
                    "raw_agent_output": "",
                    "tool_calls": [],
                }
            rollouts.append(rollout)

        chosen = select_best_of_n(rollouts) if args.best_of_n > 1 else rollouts[0]
        graded = grade_trajectory(task, chosen)
        final_results.append(graded)
        # Persist progress after each task so long local runs can be audited.
        output_path = Path(args.output)
        by_bug_type = {
            bug_type: summarize_metrics([r for r in final_results if r["bug_type"] == bug_type])
            for bug_type in sorted({r["bug_type"] for r in final_results})
        }
        output_path.write_text(
            json.dumps({"summary": summarize_metrics(final_results), "by_bug_type": by_bug_type, "details": final_results}, indent=2)
        )

    summary = summarize_metrics(final_results)
    by_bug_type = {
        bug_type: summarize_metrics([r for r in final_results if r["bug_type"] == bug_type])
        for bug_type in sorted({r["bug_type"] for r in final_results})
    }
    print("\n=== SCISILENTBENCH SUMMARY ===")
    print(json.dumps(summary, indent=2))

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps({"summary": summary, "by_bug_type": by_bug_type, "details": final_results}, indent=2)
    )
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SciSilentBench Evaluation Harness")
    parser.add_argument("--tasks", type=str, default="sample_tasks.json")
    parser.add_argument("--model", type=str, required=True, help="Model ID for ADK/LiteLLM")
    parser.add_argument("--condition", type=str, choices=["A", "B"], default="B")
    parser.add_argument("--best-of-n", type=int, default=1, help="Number of samples per task")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--task-timeout", type=float, default=180.0, help="Seconds allowed per agent rollout")
    parser.add_argument("--task-ids", nargs="*", default=[], help="Optional task ID subset for incremental evaluation")
    args = parser.parse_args()

    asyncio.run(run_evaluation(args))