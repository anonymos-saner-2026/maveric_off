"""
dataset_loader.py
Load TruthfulQA or FEVER datasets for evaluation.
"""

import json
from typing import List, Dict, Any, Optional
from datasets import load_dataset


def load_truthfulqa(max_samples: Optional[int] = None, split: str = "validation") -> List[Dict[str, Any]]:
    """
    Load TruthfulQA dataset.
    
    Returns list of samples with:
    - id: unique identifier
    - question: the question text
    - claim: converted question to claim format  
    - label: True/False based on best_answer
    - gold_answer: the correct answer text
    """
    print(f"Loading TruthfulQA dataset (split={split})...")
    
    try:
        dataset = load_dataset("truthful_qa", "generation", split=split)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using fallback minimal dataset...")
        return _get_fallback_truthfulqa(max_samples)
    
    samples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        
        question = item.get("question", "")
        best_answer = item.get("best_answer", "")
        correct_answers = item.get("correct_answers", [])
        incorrect_answers = item.get("incorrect_answers", [])
        
        # Convert to claim format: assume question implies a claim
        # We'll use the question as-is for verification
        # Label: True if best_answer is in correct_answers
        label = best_answer in correct_answers if correct_answers else True
        
        samples.append({
            "id": f"truthfulqa_{idx:04d}",
            "question": question,
            "claim": question,  # Use question directly
            "label": label,
            "gold_answer": best_answer,
            "correct_answers": correct_answers,
            "incorrect_answers": incorrect_answers,
        })
    
    print(f"Loaded {len(samples)} samples from TruthfulQA")
    return samples


def load_fever(max_samples: Optional[int] = None, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load FEVER dataset.
    
    Returns list of samples with:
    - id: unique identifier
    - claim: the claim text
    - label: True (SUPPORTS), False (REFUTES), None (NOT_ENOUGH_INFO)
    - evidence: evidence sentences
    """
    print(f"Loading FEVER dataset (split={split})...")
    
    try:
        dataset = load_dataset("fever", "v1.0", split=split)
    except Exception as e:
        print(f"Error loading from HuggingFace: {e}")
        print("Using fallback minimal dataset...")
        return _get_fallback_fever(max_samples)
    
    samples = []
    for idx, item in enumerate(dataset):
        if max_samples and idx >= max_samples:
            break
        
        claim = item.get("claim", "")
        label_str = item.get("label", "NOT ENOUGH INFO")
        
        # Convert label: SUPPORTS -> True, REFUTES -> False, NOT_ENOUGH_INFO -> None
        if label_str == "SUPPORTS":
            label = True
        elif label_str == "REFUTES":
            label = False
        else:
            label = None  # Skip ambiguous cases
        
        # Skip None labels for now (binary classification)
        if label is None:
            continue
        
        samples.append({
            "id": f"fever_{item.get('id', idx)}",
            "claim": claim,
            "label": label,
            "evidence": item.get("evidence", []),
        })
    
    print(f"Loaded {len(samples)} samples from FEVER")
    return samples


def _get_fallback_truthfulqa(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback minimal TruthfulQA dataset for testing"""
    fallback = [
        {
            "id": "truthfulqa_0001",
            "question": "What is the capital of France?",
            "claim": "What is the capital of France?",
            "label": True,
            "gold_answer": "Paris",
            "correct_answers": ["Paris"],
            "incorrect_answers": ["London", "Berlin"],
        },
        {
            "id": "truthfulqa_0002",
            "question": "Is the Earth flat?",
            "claim": "Is the Earth flat?",
            "label": False,
            "gold_answer": "No",
            "correct_answers": ["No", "The Earth is round"],
            "incorrect_answers": ["Yes", "The Earth is flat"],
        },
    ]
    
    if max_samples:
        fallback = fallback[:max_samples]
    
    return fallback


def _get_fallback_fever(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback minimal FEVER dataset for testing"""
    fallback = [
        {
            "id": "fever_0001",
            "claim": "Paris is the capital of France.",
            "label": True,
            "evidence": [],
        },
        {
            "id": "fever_0002",
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
    dataset_name = dataset_name.lower()
    
    if dataset_name == "truthfulqa":
        return load_truthfulqa(max_samples=max_samples)
    elif dataset_name == "fever":
        return load_fever(max_samples=max_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: truthfulqa, fever")


if __name__ == "__main__":
    # Test loading
    print("\n=== Testing TruthfulQA ===")
    samples = load_truthfulqa(max_samples=2)
    for s in samples:
        print(json.dumps(s, indent=2))
    
    print("\n=== Testing FEVER ===")
    samples = load_fever(max_samples=2)
    for s in samples:
        print(json.dumps(s, indent=2))
