#!/usr/bin/env python3
"""
run_cost_budget_sweep.py
Run MaVERiC across multiple budgets and tool-cost configs and plot results.

Usage:
  python run_cost_budget_sweep.py --dataset truthfulqa --max_samples 20
  python run_cost_budget_sweep.py --dataset truthfulqa --max_samples 20 --fast

Optional overrides:
  --budgets "1,3,5,10,20"
  --tool_costs_file experiments/p02_evaluation_harness/tool_costs.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Any, List

import matplotlib.pyplot as plt


def _parse_csv_numbers(raw: str) -> List[float]:
    out = []
    for x in (raw or "").split(","):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            continue
    return out


def _load_tool_costs(path: str) -> Dict[str, Dict[str, float]]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _run_eval(dataset: str, max_samples: int, budget: float, tool_costs: Dict[str, float], output_dir: str, fast: bool) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    run_eval_path = os.path.join(script_dir, "run_eval.py")
    cmd = [
        "python",
        run_eval_path,
        "--dataset",
        dataset,
        "--max_samples",
        str(max_samples),
        "--budget",
        str(budget),
        "--output_dir",
        output_dir,
    ]
    if tool_costs:
        cmd.extend(["--tool_costs", json.dumps(tool_costs)])
    env = os.environ.copy()
    if fast:
        env["MAVERIC_FAST_MODE"] = "1"
    subprocess.run(cmd, check=True, env=env)

    # Find the newest results file
    files = [f for f in os.listdir(output_dir) if f.endswith(f"_{dataset}.jsonl")]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(output_dir, x)), reverse=True)
    if not files:
        raise RuntimeError("No results files found")
    return os.path.join(output_dir, files[0])


def _summarize(jsonl_path: str) -> Dict[str, Any]:
    from summarize import load_results
    from metrics import calculate_accuracy, calculate_precision_recall_f1, aggregate_budget_stats, aggregate_tool_calls

    results = load_results(jsonl_path)
    predicted = [r.get("predicted_label") for r in results]
    gold = [r.get("gold_label") for r in results]
    verified = [r for r in results if not r.get("unverified")]

    accuracy = calculate_accuracy(predicted, gold)
    prf = calculate_precision_recall_f1(predicted, gold)
    budget_stats = aggregate_budget_stats(results)
    tool_stats = aggregate_tool_calls(results)

    return {
        "accuracy": accuracy,
        "precision": prf.get("precision", 0.0),
        "recall": prf.get("recall", 0.0),
        "f1": prf.get("f1", 0.0),
        "verified": len(verified),
        "samples": len(results),
        "avg_budget": budget_stats.get("avg", 0.0),
        "avg_tool_calls": tool_stats.get("avg", 0.0),
        "total_tool_calls": tool_stats.get("total", 0),
        "jsonl": jsonl_path,
    }


def _plot_metric(ax, xvals, series, title, ylabel):
    for label, yvals in series.items():
        ax.plot(xvals, yvals, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Budget")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget + tool cost sweep for MaVERiC")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--budgets", type=str, default="1,3,5,10,20,30")
    parser.add_argument("--tool_costs_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="experiments/p02_evaluation_harness/sweep_results")
    parser.add_argument("--fast", action="store_true", help="Enable MAVERIC_FAST_MODE")

    args = parser.parse_args()

    budgets = _parse_csv_numbers(args.budgets)
    if not budgets:
        raise ValueError("No budgets provided")

    cost_configs = _load_tool_costs(args.tool_costs_file)
    if not cost_configs:
        cost_configs = {"default": {}}

    os.makedirs(args.output_dir, exist_ok=True)

    summary_rows = []
    for cfg_name, costs in cost_configs.items():
        for budget in budgets:
            print(f"\n=== Running config={cfg_name} budget={budget} ===")
            result_path = _run_eval(
                dataset=args.dataset,
                max_samples=args.max_samples,
                budget=budget,
                tool_costs=costs,
                output_dir=args.output_dir,
                fast=args.fast,
            )
            summary = _summarize(result_path)
            summary.update({"config": cfg_name, "budget": budget})
            summary_rows.append(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(args.output_dir, f"sweep_summary_{timestamp}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    # Plot
    budgets_sorted = sorted(set(budgets))
    configs = sorted(set(r["config"] for r in summary_rows))

    metrics = {
        "Accuracy": "accuracy",
        "F1": "f1",
        "Avg Tool Calls": "avg_tool_calls",
        "Avg Budget": "avg_budget",
        "Verified Samples": "verified",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, (title, key) in zip(axes, metrics.items()):
        series = {}
        for cfg in configs:
            cfg_rows = [r for r in summary_rows if r["config"] == cfg]
            cfg_map = {r["budget"]: r.get(key, 0.0) for r in cfg_rows}
            series[cfg] = [cfg_map.get(b, 0.0) for b in budgets_sorted]
        _plot_metric(ax, budgets_sorted, series, title, key)

    fig.tight_layout()
    plot_path = os.path.join(args.output_dir, f"sweep_plots_{timestamp}.png")
    fig.savefig(plot_path, dpi=200)
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
