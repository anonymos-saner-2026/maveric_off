#!/usr/bin/env python3
"""
run_eval.py
Main evaluation script for running MaVERiC on datasets.

Key patch (TruthfulQA):
- TruthfulQA is NOT a binary True/False task on the question.
- We evaluate pairwise: truthful_answer vs false_answer (or best_answer vs incorrect_answer).
- Predicted_label=True means we picked the truthful answer (gold_label is always True in this pairwise setup).

This patched version:
- Fixes KeyError: pruned/edges_removed
- Collects real refinement stats from solver:
    solver.pruned_nodes
    solver.edges_removed

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
from typing import Dict, Any, List, Tuple

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


# -------------------------
# Solver runner for a statement
# -------------------------
def run_solver_on_statement(statement: str, sample_id: str, budget: float, verbose: bool = False) -> Dict[str, Any]:
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

        # --- Collect robust signals ---
        y_direct = getattr(solver, "y_direct", None)  # True/False/None
        root_id = getattr(solver, "root_id", None)
        root_in_ext = bool(root_id and root_id in final_ext)

        vt = len(getattr(solver, "verified_true_ids", set()) or set())
        vf = len(getattr(solver, "verified_false_ids", set()) or set())

        # Pairwise score (optional, not decisive)
        score_fn = getattr(solver, "pairwise_score", None)
        score = float(score_fn(final_ext)) if callable(score_fn) else 0.0

        # --- Real refinement stats from solver (PATCH) ---
        pruned = int(getattr(solver, "pruned_nodes", 0) or 0)
        edges_removed = int(getattr(solver, "edges_removed", 0) or 0)

        error = None

    except Exception as e:
        final_ext, verdict = set(), False
        y_direct, root_in_ext, vt, vf, score = None, False, 0, 0, -1e9
        pruned, edges_removed = 0, 0
        error = str(e)

    budget_left = float(getattr(solver, "budget", 0.0) or 0.0)
    budget_used = max(0.0, float(budget) - budget_left)

    tool_calls_total = int(getattr(solver, "tool_calls", 0) or 0)

    return {
        "statement": statement,
        "verdict": bool(verdict),
        "y_direct": y_direct,  # ✅ strong signal
        "root_in_ext": bool(root_in_ext),  # ✅ weak signal
        "verified_true": int(vt),  # ✅ evidence
        "verified_false": int(vf),  # ✅ evidence
        "score": float(score),  # optional
        "final_ext_size": int(len(final_ext)),
        "budget_used": round(budget_used, 2),
        "budget_left": round(budget_left, 2),
        "tool_calls_total": tool_calls_total,
        # PATCH: real refinement stats
        "pruned": int(pruned),
        "edges_removed": int(edges_removed),
        "error": error,
    }


# -------------------------
# Pairwise robust chooser (your logic)
# -------------------------
import math


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _tanh_scale(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.tanh(float(x) / float(scale))


def _robust_conf_proxy(res: Dict[str, Any]) -> float:
    """
    Returns a soft confidence-like score in [0, 1].
    Designed to be:
      - bounded
      - not dominated by final_ext_size or tool spam
      - usable even if some fields missing
    """
    if res.get("error"):
        return 0.0

    y_direct = res.get("y_direct", None)
    verdict = bool(res.get("verdict", False))

    vt = int(res.get("verified_true", 0) or 0)
    vf = int(res.get("verified_false", 0) or 0)
    net = vt - vf
    cov = vt + vf

    root_in_ext = bool(res.get("root_in_ext", False))

    tool_calls = float(res.get("tool_calls_total", 0) or 0)
    budget_used = float(res.get("budget_used", 0.0) or 0.0)

    conf = 0.50

    if y_direct is True:
        conf += 0.25
    elif y_direct is False:
        conf -= 0.35

    conf += 0.10 if verdict else -0.10

    conf += 0.18 * _tanh_scale(net, scale=3.0)
    conf += 0.10 * _tanh_scale(cov, scale=6.0)

    if root_in_ext:
        conf += 0.05

    conf -= 0.06 * _tanh_scale(tool_calls, scale=10.0)
    conf -= 0.06 * _tanh_scale(budget_used, scale=10.0)

    return _clamp(conf, 0.0, 1.0)


def _rank_tuple(res: Dict[str, Any]) -> tuple:
    """
    Lexicographic ranking. Higher is better.
    """
    has_error = 1 if res.get("error") else 0

    y_direct = res.get("y_direct", None)
    if y_direct is True:
        ycat = 2
    elif y_direct is False:
        ycat = 0
    else:
        ycat = 1

    conf = _robust_conf_proxy(res)

    vt = int(res.get("verified_true", 0) or 0)
    vf = int(res.get("verified_false", 0) or 0)
    net = vt - vf

    tool_calls = float(res.get("tool_calls_total", 0) or 0)
    budget_used = float(res.get("budget_used", 0.0) or 0.0)

    ext = int(res.get("final_ext_size", 0) or 0)

    return (
        -has_error,    # 0 error better than 1 error
        ycat,          # prefer y_direct True
        conf,          # confidence proxy
        net,           # more net evidence
        -tool_calls,   # fewer tool calls
        -budget_used,  # less budget
        -ext,          # very weak tie-break
    )


def truthfulqa_decide_pair_robust(truth_res: Dict[str, Any], false_res: Dict[str, Any]) -> bool:
    """
    Returns predicted_label:
      True  -> pick truth statement
      False -> pick false statement
    """
    rt = _rank_tuple(truth_res)
    rf = _rank_tuple(false_res)

    if rt > rf:
        return True
    if rf > rt:
        return False

    t_calls = int(truth_res.get("tool_calls_total", 0) or 0)
    f_calls = int(false_res.get("tool_calls_total", 0) or 0)
    if t_calls != f_calls:
        return t_calls < f_calls
    return True


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

    For other binary datasets:
      - Standard: solve claim, compare verdict to label.
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

        b_each = max(0.5, budget / 2.0)

        truth_res = run_solver_on_statement(truth_stmt, sample_id + "_T", b_each, verbose=verbose)
        false_res = run_solver_on_statement(false_stmt, sample_id + "_F", b_each, verbose=verbose)

        predicted_label = truthfulqa_decide_pair_robust(truth_res, false_res)
        gold_label = True

        runtime_s = time.time() - start_time

        truth_ext = int(truth_res.get("final_ext_size", 0) or 0)
        false_ext = int(false_res.get("final_ext_size", 0) or 0)

        # PATCH: aggregate real refinement stats
        pruned_total = int(truth_res.get("pruned", 0) or 0) + int(false_res.get("pruned", 0) or 0)
        edges_removed_total = int(truth_res.get("edges_removed", 0) or 0) + int(false_res.get("edges_removed", 0) or 0)

        return {
            "sample_id": sample_id,
            "dataset": ds,
            "claim": question,
            "truth_statement": truth_stmt,
            "false_statement": false_stmt,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "correct": (predicted_label == gold_label),
            "budget_used": round(float(truth_res["budget_used"]) + float(false_res["budget_used"]), 2),
            "tool_calls": {
                "TOTAL": int(truth_res["tool_calls_total"]) + int(false_res["tool_calls_total"])
            },
            "refinement_stats": {
                "pruned": int(pruned_total),
                "edges_removed": int(edges_removed_total),
                "sgs_size": max(truth_ext, false_ext),
                "truth_final_ext_size": truth_ext,
                "false_final_ext_size": false_ext,
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

    graph = create_debate_graph_from_claim(claim, sample_id)

    solver = MaVERiCSolver(
        graph=graph,
        budget=budget,
        topk_counterfactual=8,
        delta_support_to_root=0.8,
    )

    final_ext, verdict = set(), False
    error = None

    try:
        final_ext, verdict = solver.run()
        predicted_label = bool(verdict)
    except Exception as e:
        predicted_label = False
        final_ext = set()
        error = str(e)

    runtime_s = time.time() - start_time
    budget_left = getattr(solver, "budget", 0.0)
    budget_used = max(0.0, budget - float(budget_left))

    tool_calls = {"TOTAL": int(getattr(solver, "tool_calls", 0) or 0)}

    # PATCH: pull real stats from solver
    refinement_stats = {
        "pruned": int(getattr(solver, "pruned_nodes", 0) or 0),
        "sgs_size": int(len(final_ext)),
        "edges_removed": int(getattr(solver, "edges_removed", 0) or 0),
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

    samples = load_dataset_by_name(args.dataset, max_samples=args.max_samples)
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(samples)} samples\n")

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
