#!/usr/bin/env python3
"""
run_eval.py
Main evaluation script for running MaVERiC on datasets.

Key patch (TruthfulQA):
- TruthfulQA is NOT a binary True/False task on the question.
- We evaluate pairwise: truthful_answer vs false_answer (or best_answer vs incorrect_answer).
- Predicted_label=True means we picked the truthful answer (gold_label is always True in this pairwise setup).

Usage:
    python run_eval.py --method maveric --dataset truthfulqa --budget 10 --max_samples 50 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.graph import ArgumentationGraph, ArgumentNode  # noqa: E402
from src.core.solver import MaVERiCSolver  # noqa: E402
from dataset_loader import load_dataset_by_name  # noqa: E402

from src.agents.debater import generate_debate  # noqa: E402
from src.agents.parser import parse_debate  # noqa: E402


# -------------------------
# Graph construction
# -------------------------
def create_debate_graph_from_claim(claim: str, claim_id: str) -> ArgumentationGraph:
    """
    Create a debate graph by simulating a discussion and parsing it.
    """
    debate_text = generate_debate(topic=claim, num_liars=2)

    if not debate_text:
        print("  WARNING: Debate generation failed, falling back to simple root node.")
        g = ArgumentationGraph()
        root = ArgumentNode(
            id=f"{claim_id}_root",
            content=f"Claim: {claim}",
            speaker="Claimer",
            is_verified=False,
            ground_truth=None,
            verification_cost=1.0,
            tool_type="AUTO",
        )
        g.add_node(root)
        # compatibility with older solver/root find
        g.find_semantic_root = lambda prefer_attack_only=True: f"{claim_id}_root"
        return g

    g = parse_debate(debate_text)
    return g


def _safe_get_dataset_name(sample: Dict[str, Any], fallback: str) -> str:
    for k in ("dataset", "dataset_name", "source"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return (fallback or "").strip().lower()


# -------------------------
# TruthfulQA helpers
# -------------------------
def truthfulqa_extract_pair(sample: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (truth_stmt, false_stmt) from a TruthfulQA sample.

    We try multiple common field names to be robust to your dataset_loader.
    Required: one truthful answer + one false/misleading answer.
    """
    truth = (
        sample.get("truthful_statement")
        or sample.get("best_answer")
        or sample.get("correct_answer")
        or sample.get("truth")
        or sample.get("answer_true")
        or sample.get("truthful_answer")
    )
    false = (
        sample.get("false_statement")
        or sample.get("incorrect_answer")
        or sample.get("false_answer")
        or sample.get("myth_answer")
        or sample.get("answer_false")
        or sample.get("false_answer_text")
        or sample.get("incorrect_answer_text")
    )

    if not truth or not false:
        keys = ", ".join(sorted(sample.keys()))
        raise ValueError(
            "TruthfulQA requires a truthful answer AND a false answer for pairwise evaluation.\n"
            "Expected keys like: best_answer + incorrect_answer (or similar).\n"
            f"Got keys: {keys}"
        )

    return str(truth).strip(), str(false).strip()


def run_solver_on_statement(statement: str, sample_id: str, budget: float, verbose: bool = False) -> Dict[str, Any]:
    """
    Run MaVERiC solver on a single statement (treated as a factual claim).
    Returns verdict + useful diagnostics.
    """
    if verbose:
        print(f"    🧩 Building graph for: {statement[:120]}...")

    graph = create_debate_graph_from_claim(statement, sample_id)

    solver = MaVERiCSolver(
        graph=graph,
        budget=budget,
        topk_counterfactual=8,
        delta_support_to_root=0.8,
    )

    try:
        final_ext, verdict = solver.run()
        error = None
    except Exception as e:
        final_ext, verdict = set(), False
        error = str(e)

    # Budget used
    budget_left = getattr(solver, "budget", 0.0)
    budget_used = max(0.0, budget - float(budget_left))

    # Tool calls
    tool_calls_total = getattr(solver, "tool_calls", None)
    if tool_calls_total is None:
        tool_calls_total = 0

    return {
        "statement": statement,
        "verdict": bool(verdict),
        "final_ext_size": len(final_ext),
        "budget_used": round(budget_used, 2),
        "budget_left": round(float(budget_left), 2),
        "tool_calls_total": int(tool_calls_total),
        "error": error,
    }


def truthfulqa_decide_pair(truth_res: Dict[str, Any], false_res: Dict[str, Any]) -> bool:
    """
    Decide which statement is better.

    Returns:
        predicted_label: True means we pick truthful statement (this is the gold in our pairwise setup).
    Policy (conservative + stable):
      1) If truth is TRUE and false is FALSE -> pick truth
      2) If truth is TRUE and false is TRUE -> tie-break via larger final_ext_size
      3) If truth is FALSE and false is FALSE -> tie-break via larger final_ext_size (but note both are bad)
      4) If truth is FALSE and false is TRUE -> pick false (model thinks false is truer)
    """
    tV = bool(truth_res["verdict"])
    fV = bool(false_res["verdict"])

    if tV and (not fV):
        return True
    if (not tV) and fV:
        return False

    # tie-break
    tS = int(truth_res.get("final_ext_size", 0) or 0)
    fS = int(false_res.get("final_ext_size", 0) or 0)
    return tS >= fS


# -------------------------
# Main sample runner
# -------------------------
def run_maveric_on_sample(
    sample: Dict[str, Any],
    budget: float,
    dataset_name: str,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run MaVERiC solver on a single sample.

    For TruthfulQA:
      - Pairwise truthful vs false statement.
      - gold_label is always True in this setup.

    For other binary datasets (e.g., FEVER if your loader outputs bool labels):
      - Standard: create graph from sample["claim"], solve, compare verdict to label.
    """
    sample_id = str(sample.get("id", "unknown"))
    ds = _safe_get_dataset_name(sample, dataset_name)

    start_time = time.time()

    # -------- TruthfulQA: pairwise --------
    if ds == "truthfulqa":
        question = str(sample.get("claim") or sample.get("question") or "").strip()

        truth_stmt, false_stmt = truthfulqa_extract_pair(sample)

        if verbose:
            print(f"\n[{sample_id}] TruthfulQA (pairwise)")
            print(f"  Q: {question[:120]}")
            print(f"  T: {truth_stmt[:120]}")
            print(f"  F: {false_stmt[:120]}")

        # split budget
        b_each = max(0.5, budget / 2.0)

        truth_res = run_solver_on_statement(truth_stmt, sample_id + "_T", b_each, verbose=verbose)
        false_res = run_solver_on_statement(false_stmt, sample_id + "_F", b_each, verbose=verbose)

        predicted_label = truthfulqa_decide_pair(truth_res, false_res)
        gold_label = True  # in pairwise, truthful is always the correct choice

        runtime_s = time.time() - start_time

        return {
            "sample_id": sample_id,
            "dataset": ds,
            "claim": question,
            "truth_statement": truth_stmt,
            "false_statement": false_stmt,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "correct": (predicted_label == gold_label),
            "budget_used": round(truth_res["budget_used"] + false_res["budget_used"], 2),
            "tool_calls": {"TOTAL": int(truth_res["tool_calls_total"]) + int(false_res["tool_calls_total"])},
            "refinement_stats": {
                "truth_final_ext_size": int(truth_res["final_ext_size"]),
                "false_final_ext_size": int(false_res["final_ext_size"]),
            },
            "runtime_s": round(runtime_s, 2),
            "subruns": {
                "truth": truth_res,
                "false": false_res,
            },
        }

    # -------- Default: binary claim classification --------
    claim = str(sample.get("claim") or sample.get("question") or "").strip()
    gold_label = bool(sample.get("label", False))

    if verbose:
        print(f"\n[{sample_id}] {ds} (binary)")
        print(f"  Claim: {claim[:140]}")
        print(f"  Gold: {gold_label}")

    # Create graph
    graph = create_debate_graph_from_claim(claim, sample_id)

    # Initialize solver
    solver = MaVERiCSolver(
        graph=graph,
        budget=budget,
        topk_counterfactual=8,
        delta_support_to_root=0.8,
    )

    # Run solver
    try:
        final_ext, verdict = solver.run()
        predicted_label = bool(verdict)
        error = None
    except Exception as e:
        predicted_label = False
        final_ext = set()
        error = str(e)

    runtime_s = time.time() - start_time
    budget_left = getattr(solver, "budget", 0.0)
    budget_used = max(0.0, budget - float(budget_left))

    # Tool calls
    tool_calls = {}
    for log in getattr(solver, "logs", []):
        if "Verified" in log:
            if "WEB_SEARCH" in log:
                tool_calls["WEB_SEARCH"] = tool_calls.get("WEB_SEARCH", 0) + 1
            elif "PYTHON_EXEC" in log:
                tool_calls["PYTHON_EXEC"] = tool_calls.get("PYTHON_EXEC", 0) + 1
            elif "COMMON_SENSE" in log:
                tool_calls["COMMON_SENSE"] = tool_calls.get("COMMON_SENSE", 0) + 1

    if hasattr(solver, "tool_calls") and not tool_calls:
        tool_calls["TOTAL"] = int(getattr(solver, "tool_calls", 0) or 0)

    # Refinement stats (best-effort)
    initial_nodes = len(getattr(graph, "initial_nodes", [])) if hasattr(graph, "initial_nodes") else 0
    final_nodes = len(getattr(graph, "nodes", {})) if hasattr(graph, "nodes") else 0
    pruned = max(0, initial_nodes - final_nodes)

    refinement_stats = {
        "pruned": pruned,
        "sgs_size": len(final_ext),
        "edges_removed": 0,
    }

    result = {
        "sample_id": sample_id,
        "dataset": ds,
        "claim": claim,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "correct": (predicted_label == gold_label),
        "budget_used": round(budget_used, 2),
        "tool_calls": tool_calls,
        "refinement_stats": refinement_stats,
        "runtime_s": round(runtime_s, 2),
    }
    if error:
        result["error"] = error

    return result


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on MaVERiC")
    parser.add_argument("--method", type=str, default="maveric", help="Method name")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (truthfulqa, fever, ... depending on your dataset_loader)",
    )
    parser.add_argument("--budget", type=float, default=10.0, help="Budget per sample")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of samples to evaluate")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("=" * 70)
    print("MaVERiC Evaluation")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Dataset: {args.dataset}")
    print(f"Budget: {args.budget}")
    print(f"Max samples: {args.max_samples}")
    print("=" * 70)

    # Load dataset
    samples = load_dataset_by_name(args.dataset, max_samples=args.max_samples)
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(samples)} samples\n")

    # Run evaluation
    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, 1):
        sid = str(sample.get("id", f"sample_{idx:04d}"))
        print(f"[{idx}/{len(samples)}] Processing {sid}...")

        result = run_maveric_on_sample(
            sample=sample,
            budget=float(args.budget),
            dataset_name=str(args.dataset).lower(),
            verbose=bool(args.verbose),
        )
        results.append(result)

        correct_so_far = sum(1 for r in results if bool(r.get("correct")))
        accuracy_so_far = correct_so_far / max(1, len(results))
        print(f"  Accuracy so far: {accuracy_so_far:.2%} ({correct_so_far}/{len(results)})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{args.method}_{args.dataset}.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    final_acc = sum(1 for r in results if bool(r.get("correct"))) / max(1, len(results))

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print(f"\nFinal Accuracy: {final_acc:.2%}")
    print(f"Total samples: {len(results)}")
    print("\nRun summarize.py to generate detailed metrics:")
    print(f"  python summarize.py {output_path}")


if __name__ == "__main__":
    main()
