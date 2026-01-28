"""
dataset_loader.py
Load TruthfulQA or FEVER datasets for evaluation.

PATCH (TruthfulQA):
- TruthfulQA is a QA benchmark, not a binary True/False label over the question.
- We convert each question into a PAIRWISE comparison:
    truthful_statement  vs  false_statement
  where each statement is a (Question + Answer) pair.
- Gold label is always True in this pairwise setup:
    True => choose truthful_statement over false_statement.

This matches the patched run_eval.py that expects:
- truthful_statement
- false_statement
"""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset


def _make_qa_statement(question: str, answer: str) -> str:
    q = (question or "").strip()
    a = (answer or "").strip()
    # Keep it explicit so the debater/parser knows it's QA-form claim.
    return f"Question: {q}\nAnswer: {a}".strip()


def _pick_false_answer(incorrect_answers: List[str], best_answer: str) -> str:
    """
    Pick a false/misleading answer for the pair.
    Prefer the first incorrect answer that differs from best_answer.
    """
    best = (best_answer or "").strip()
    for x in incorrect_answers or []:
        xs = (x or "").strip()
        if xs and xs != best:
            return xs
    # fallback if list empty or weird
    return "I don't know."


def load_truthfulqa(
    max_samples: Optional[int] = None,
    split: str = "validation",
) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA dataset (generation).

    Returns list of samples with:
    - id: unique identifier
    - dataset: "truthfulqa"
    - question: question text
    - claim: kept as question (for compatibility), but NOT used for binary truth anymore
    - label: True (pairwise gold: truthful_statement is the correct choice)
    - truthful_statement: "Question: ... Answer: <best_answer>"
    - false_statement:    "Question: ... Answer: <one incorrect answer>"
    - gold_answer: best_answer
    - incorrect_answer: chosen incorrect answer
    - correct_answers / incorrect_answers: raw lists for debugging
    """
    print(f"Loading TruthfulQA dataset (split={split})...")

    try:
        dataset = load_dataset("truthful_qa", "generation", split=split)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using fallback minimal dataset...")
        return _get_fallback_truthfulqa(max_samples)

    samples: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break

        question = item.get("question", "") or ""
        best_answer = item.get("best_answer", "") or ""
        correct_answers = item.get("correct_answers", []) or []
        incorrect_answers = item.get("incorrect_answers", []) or []

        incorrect_answer = _pick_false_answer(incorrect_answers, best_answer)

        truthful_statement = _make_qa_statement(question, best_answer)
        false_statement = _make_qa_statement(question, incorrect_answer)

        samples.append(
            {
                "id": f"truthfulqa_{idx:04d}",
                "dataset": "truthfulqa",
                "question": question,
                "claim": question,  # legacy field; not used as binary truth target anymore
                "label": True,  # gold for pairwise: truthful_statement is correct
                "truthful_statement": truthful_statement,
                "false_statement": false_statement,
                "gold_answer": best_answer,
                "incorrect_answer": incorrect_answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
            }
        )

    print(f"Loaded {len(samples)} samples from TruthfulQA")
    return samples


def load_fever(
    max_samples: Optional[int] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load FEVER dataset.

    Returns list of samples with:
    - id: unique identifier
    - dataset: "fever"
    - claim: the claim text
    - label: True (SUPPORTS), False (REFUTES); skips NOT_ENOUGH_INFO
    - evidence: evidence sentences (as provided by dataset)
    """
    print(f"Loading FEVER dataset (split={split})...")

    try:
        dataset = load_dataset("fever", "v1.0", split=split)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using fallback minimal dataset...")
        return _get_fallback_fever(max_samples)

    samples: List[Dict[str, Any]] = []
    kept = 0
    for idx, item in enumerate(dataset):
        if max_samples and kept >= max_samples:
            break

        claim = item.get("claim", "") or ""
        label_str = item.get("label", "NOT ENOUGH INFO") or "NOT ENOUGH INFO"

        if label_str == "SUPPORTS":
            label = True
        elif label_str == "REFUTES":
            label = False
        else:
            continue  # skip NEI

        samples.append(
            {
                "id": f"fever_{item.get('id', idx)}",
                "dataset": "fever",
                "claim": claim,
                "label": label,
                "evidence": item.get("evidence", []) or [],
            }
        )
        kept += 1

    print(f"Loaded {len(samples)} samples from FEVER")
    return samples


def _get_fallback_truthfulqa(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback minimal TruthfulQA dataset for testing (pairwise)."""
    fallback = [
        {
            "id": "truthfulqa_0001",
            "dataset": "truthfulqa",
            "question": "What is the capital of France?",
            "claim": "What is the capital of France?",
            "label": True,
            "truthful_statement": _make_qa_statement("What is the capital of France?", "Paris"),
            "false_statement": _make_qa_statement("What is the capital of France?", "London"),
            "gold_answer": "Paris",
            "incorrect_answer": "London",
            "correct_answers": ["Paris"],
            "incorrect_answers": ["London", "Berlin"],
        },
        {
            "id": "truthfulqa_0002",
            "dataset": "truthfulqa",
            "question": "Is the Earth flat?",
            "claim": "Is the Earth flat?",
            "label": True,
            "truthful_statement": _make_qa_statement("Is the Earth flat?", "No, the Earth is round."),
            "false_statement": _make_qa_statement("Is the Earth flat?", "Yes, the Earth is flat."),
            "gold_answer": "No, the Earth is round.",
            "incorrect_answer": "Yes, the Earth is flat.",
            "correct_answers": ["No", "The Earth is round", "No, the Earth is round."],
            "incorrect_answers": ["Yes", "The Earth is flat", "Yes, the Earth is flat."],
        },
    ]

    if max_samples:
        fallback = fallback[:max_samples]
    return fallback


def _get_fallback_fever(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback minimal FEVER dataset for testing."""
    fallback = [
        {
            "id": "fever_0001",
            "dataset": "fever",
            "claim": "Paris is the capital of France.",
            "label": True,
            "evidence": [],
        },
        {
            "id": "fever_0002",
            "dataset": "fever",
            "claim": "The Earth is flat.",
            "label": False,
            "evidence": [],
        },
    ]

    if max_samples:
        fallback = fallback[:max_samples]
    return fallback


def load_dataset_by_name(dataset_name: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load dataset by name.

    Args:
        dataset_name: "truthfulqa" or "fever"
        max_samples: maximum number of samples to load

    Returns:
        List of samples
    """
    dataset_name = (dataset_name or "").lower().strip()

    if dataset_name == "truthfulqa":
        return load_truthfulqa(max_samples=max_samples)
    if dataset_name == "fever":
        # Prefer local processed file if available, else standard loader
        local_path = "data/benchmarks_processed/fever_labeled.json"
        if os.path.exists(local_path):
            return load_local_benchmark("fever", local_path, max_samples)
        return load_fever(max_samples=max_samples)
    if dataset_name == "copheme":
        return load_local_benchmark("copheme", "data/benchmarks_processed/copheme_labeled.json", max_samples)
    if dataset_name == "copheme_nei_balanced":
        return load_local_benchmark(
            "copheme_nei_balanced",
            "data/benchmarks_processed/copheme_nei_balanced.json",
            max_samples,
            label_map={"NOT ENOUGH INFO": True, "REFUTED": False, "SUPPORTS": True},
        )
    if dataset_name == "hover":
        return load_local_benchmark("hover", "data/benchmarks_processed/hover_labeled.json", max_samples)
    if dataset_name == "hover_balanced":
        return load_local_benchmark("hover_balanced", "data/benchmarks_processed/hover_labeled_balanced.json", max_samples)
    if dataset_name == "scifact":
        return load_local_benchmark("scifact", "data/benchmarks_processed/scifact_labeled.json", max_samples)

    raise ValueError(f"Unknown dataset: {dataset_name}. Supported: truthfulqa, fever, copheme, hover, scifact")

def load_local_benchmark(
    dataset_name: str,
    path: str,
    max_samples: Optional[int] = None,
    label_map: Optional[Dict[str, Optional[bool]]] = None,
) -> List[Dict[str, Any]]:
    print(f"Loading {dataset_name} from {path}...")
    samples = []
    if not os.path.exists(path):
        print(f"⚠️ File not found: {path}")
        return []
        
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    default_map = {
        "SUPPORTS": True,
        "REFUTED": False,
        "REFUTES": False,
    }
    use_map = label_map or default_map

    for idx, item in enumerate(data):
        if max_samples and len(samples) >= max_samples:
            break
            
        # Map labels to boolean if possible
        label_str = str(item.get("label", "")).upper()
        if label_str not in use_map:
            continue
        label = use_map[label_str]
        if label is None:
            continue
            
        samples.append({
            "id": f"{dataset_name}_{item.get('id', idx)}",
            "dataset": dataset_name,
            "claim": item.get("claim"),
            "label": label,
            "evidence": item.get("evidence", [])
        })
    
    print(f"Loaded {len(samples)} samples from {dataset_name}")
    return samples


if __name__ == "__main__":
    # Test loading
    print("\n=== Testing TruthfulQA (pairwise) ===")
    samples = load_truthfulqa(max_samples=2)
    for s in samples:
        print(json.dumps(s, indent=2, ensure_ascii=False))

    print("\n=== Testing FEVER ===")
    samples = load_fever(max_samples=2)
    for s in samples:
        print(json.dumps(s, indent=2, ensure_ascii=False))
