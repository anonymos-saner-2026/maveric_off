#!/usr/bin/env python3
"""
summarize.py
Summarize evaluation results from jsonl files.

Usage:
    python summarize.py results/20260116_143022_maveric_truthfulqa.jsonl
"""

import argparse
import json
import sys
from typing import List, Dict, Any
from metrics import (
    calculate_accuracy,
    calculate_precision_recall_f1,
    aggregate_tool_calls,
    aggregate_budget_stats,
    aggregate_refinement_stats,
)


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load results from jsonl file"""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def print_summary(results: List[Dict[str, Any]], filepath: str):
    """Print comprehensive summary of results"""
    
    # Extract metadata
    if results:
        method = "maveric"  # Could parse from filename
        dataset = results[0].get("sample_id", "").split("_")[0] if results else "unknown"
    else:
        method = "unknown"
        dataset = "unknown"
    
    n_samples = len(results)
    
    # Extract predictions and labels
    predicted = [r["predicted_label"] for r in results]
    gold = [r["gold_label"] for r in results]
    
    # Calculate metrics
    accuracy = calculate_accuracy(predicted, gold)
    prf_metrics = calculate_precision_recall_f1(predicted, gold)
    
    # Budget stats
    budget_stats = aggregate_budget_stats(results)
    
    # Tool call stats
    tool_stats = aggregate_tool_calls(results)
    
    # Refinement stats
    refine_stats = aggregate_refinement_stats(results)
    
    # Runtime stats
    runtimes = [r.get("runtime_s", 0.0) for r in results]
    avg_runtime = sum(runtimes) / len(runtimes) if runtimes else 0.0
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(f"Results file: {filepath}")
    print(f"Dataset: {dataset}")
    print(f"Method: {method}")
    print(f"Samples: {n_samples}")
    
    print("\n" + "-"*70)
    print("CLASSIFICATION METRICS")
    print("-"*70)
    correct = sum(1 for r in results if r["correct"])
    print(f"Accuracy:  {accuracy:.4f} ({correct}/{n_samples})")
    print(f"Precision: {prf_metrics['precision']:.4f}")
    print(f"Recall:    {prf_metrics['recall']:.4f}")
    print(f"F1:        {prf_metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {prf_metrics['tp']:3d}  FP: {prf_metrics['fp']:3d}")
    print(f"  FN: {prf_metrics['fn']:3d}  TN: {prf_metrics['tn']:3d}")
    
    print("\n" + "-"*70)
    print("BUDGET & EFFICIENCY")
    print("-"*70)
    print(f"Avg Budget Used: {budget_stats['avg']:.2f} ± {budget_stats['std']:.2f}")
    print(f"Min Budget: {budget_stats['min']:.2f}")
    print(f"Max Budget: {budget_stats['max']:.2f}")
    print(f"\nAvg Tool Calls: {tool_stats['avg']:.2f} ± {tool_stats['std']:.2f}")
    print(f"Total Tool Calls: {tool_stats['total']}")
    
    if tool_stats['breakdown']:
        print(f"\nTool Call Breakdown:")
        for tool, count in sorted(tool_stats['breakdown'].items(), key=lambda x: x[1], reverse=True):
            pct = tool_stats['percentages'].get(tool, 0)
            print(f"  {tool:15s}: {count:4d} ({pct:.1f}%)")
    
    print(f"\nAvg Runtime: {avg_runtime:.2f}s per sample")
    print(f"Total Runtime: {sum(runtimes):.2f}s")
    
    print("\n" + "-"*70)
    print("REFINEMENT STATISTICS")
    print("-"*70)
    print(f"Avg Nodes Pruned: {refine_stats['avg_pruned']:.2f} ± {refine_stats['std_pruned']:.2f}")
    print(f"Avg Edges Removed (refine): {refine_stats['avg_edges_removed']:.2f} ± {refine_stats['std_edges_removed']:.2f}")
    print(
        f"Avg Edges Removed (prune): {refine_stats['avg_edges_removed_pruned']:.2f} ± "
        f"{refine_stats['std_edges_removed_pruned']:.2f}"
    )
    print(f"Avg SGS Size: {refine_stats['avg_sgs_size']:.2f} ± {refine_stats['std_sgs_size']:.2f}")
    
    # Error analysis
    errors = [r for r in results if not r["correct"]]
    if errors:
        print("\n" + "-"*70)
        print(f"ERROR ANALYSIS ({len(errors)} errors)")
        print("-"*70)
        
        # Show first few errors
        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. {err['sample_id']}")
            print(f"   Claim: {err['claim'][:80]}...")
            print(f"   Predicted: {err['predicted_label']}, Gold: {err['gold_label']}")
            if 'error' in err:
                print(f"   Error: {err['error']}")
        
        if len(errors) > 5:
            print(f"\n   ... and {len(errors) - 5} more errors")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Summarize evaluation results")
    parser.add_argument("results_file", type=str, help="Path to results jsonl file")
    parser.add_argument("--export_csv", type=str, default=None,
                       help="Export summary to CSV file")
    
    args = parser.parse_args()
    
    # Load results
    try:
        results = load_results(args.results_file)
    except FileNotFoundError:
        print(f"ERROR: Results file not found: {args.results_file}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load results: {e}")
        sys.exit(1)
    
    if not results:
        print("ERROR: No results found in file")
        sys.exit(1)
    
    # Print summary
    print_summary(results, args.results_file)
    
    # Export to CSV if requested
    if args.export_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.export_csv, index=False)
        print(f"\nExported to CSV: {args.export_csv}")


if __name__ == "__main__":
    main()
