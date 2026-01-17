"""
metrics.py
Metrics calculation utilities for evaluation.
"""

from typing import List, Dict, Any
import numpy as np
from collections import Counter


def calculate_accuracy(predicted: List[bool], gold: List[bool]) -> float:
    """Calculate accuracy"""
    if len(predicted) != len(gold):
        raise ValueError("Predicted and gold must have same length")
    
    if len(predicted) == 0:
        return 0.0
    
    correct = sum(1 for p, g in zip(predicted, gold) if p == g)
    return correct / len(predicted)


def calculate_precision_recall_f1(predicted: List[bool], gold: List[bool]) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 for binary classification.
    
    Positive class: True
    Negative class: False
    """
    if len(predicted) != len(gold):
        raise ValueError("Predicted and gold must have same length")
    
    if len(predicted) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    tp = sum(1 for p, g in zip(predicted, gold) if p == True and g == True)
    fp = sum(1 for p, g in zip(predicted, gold) if p == True and g == False)
    fn = sum(1 for p, g in zip(predicted, gold) if p == False and g == True)
    tn = sum(1 for p, g in zip(predicted, gold) if p == False and g == False)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def aggregate_tool_calls(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate tool call statistics from results.
    
    Returns:
        {
            "total": int,
            "avg": float,
            "std": float,
            "breakdown": {"WEB_SEARCH": count, "PYTHON_EXEC": count, ...},
            "percentages": {"WEB_SEARCH": pct, ...}
        }
    """
    if not results:
        return {"total": 0, "avg": 0.0, "std": 0.0, "breakdown": {}, "percentages": {}}
    
    all_tool_counts = []
    all_tools = Counter()
    
    for r in results:
        tool_calls = r.get("tool_calls", {})
        total_calls = sum(tool_calls.values())
        all_tool_counts.append(total_calls)
        
        for tool, count in tool_calls.items():
            all_tools[tool] += count
    
    total_tools = sum(all_tools.values())
    percentages = {tool: (count / total_tools * 100) if total_tools > 0 else 0
                   for tool, count in all_tools.items()}
    
    return {
        "total": sum(all_tool_counts),
        "avg": float(np.mean(all_tool_counts)) if all_tool_counts else 0.0,
        "std": float(np.std(all_tool_counts)) if all_tool_counts else 0.0,
        "breakdown": dict(all_tools),
        "percentages": percentages,
    }


def aggregate_budget_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate budget usage statistics"""
    if not results:
        return {"avg": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    
    budgets = [r.get("budget_used", 0.0) for r in results]
    
    return {
        "avg": float(np.mean(budgets)),
        "std": float(np.std(budgets)),
        "min": float(np.min(budgets)),
        "max": float(np.max(budgets)),
    }


def aggregate_refinement_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate refinement statistics from results.
    
    Returns:
        {
            "avg_pruned": float,
        "avg_edges_removed": float,
        "avg_edges_removed_pruned": float,
        "avg_sgs_size": float,
        ...
    }
    """
    if not results:
        return {
            "avg_pruned": 0.0,
            "avg_edges_removed": 0.0,
            "avg_edges_removed_pruned": 0.0,
            "avg_sgs_size": 0.0,
        }

    
    pruned = []
    edges_removed = []
    edges_removed_pruned = []
    sgs_sizes = []
    
    for r in results:
        stats = r.get("refinement_stats", {})
        pruned.append(stats.get("pruned", 0))
        edges_removed.append(stats.get("edges_removed", 0))
        edges_removed_pruned.append(stats.get("edges_removed_pruned", 0))
        sgs_sizes.append(stats.get("sgs_size", 0))
    
    return {
        "avg_pruned": float(np.mean(pruned)) if pruned else 0.0,
        "std_pruned": float(np.std(pruned)) if pruned else 0.0,
        "avg_edges_removed": float(np.mean(edges_removed)) if edges_removed else 0.0,
        "std_edges_removed": float(np.std(edges_removed)) if edges_removed else 0.0,
        "avg_edges_removed_pruned": float(np.mean(edges_removed_pruned)) if edges_removed_pruned else 0.0,
        "std_edges_removed_pruned": float(np.std(edges_removed_pruned)) if edges_removed_pruned else 0.0,
        "avg_sgs_size": float(np.mean(sgs_sizes)) if sgs_sizes else 0.0,
        "std_sgs_size": float(np.std(sgs_sizes)) if sgs_sizes else 0.0,
    }


if __name__ == "__main__":
    # Test metrics
    predicted = [True, True, False, True, False]
    gold = [True, False, False, True, True]
    
    acc = calculate_accuracy(predicted, gold)
    print(f"Accuracy: {acc:.2f}")
    
    metrics = calculate_precision_recall_f1(predicted, gold)
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall: {metrics['recall']:.2f}")
    print(f"F1: {metrics['f1']:.2f}")
