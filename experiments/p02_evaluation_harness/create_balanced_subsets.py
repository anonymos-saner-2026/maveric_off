import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _group_by_label(items: List[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    groups: Dict[str, List[Dict[str, object]]] = {}
    for item in items:
        label = str(item.get("label", ""))
        groups.setdefault(label, []).append(item)
    return groups


def _sample_balanced(items: List[Dict[str, object]], labels: List[str], per_label: int, seed: int) -> List[Dict[str, object]]:
    groups = _group_by_label(items)
    rng = random.Random(seed)

    sampled: List[Dict[str, object]] = []
    for label in labels:
        label_items = groups.get(label, [])
        if not label_items:
            continue
        take = min(per_label, len(label_items))
        sampled.extend(rng.sample(label_items, take))

    rng.shuffle(sampled)
    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(description="Create balanced subsets for hover/copheme")
    parser.add_argument("--hover-per-label", type=int, default=50)
    parser.add_argument("--copheme-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/benchmarks_processed",
        help="Directory containing labeled datasets",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    hover_path = processed_dir / "hover_labeled.json"
    copheme_path = processed_dir / "copheme_labeled.json"

    if not hover_path.exists():
        raise FileNotFoundError(f"Missing hover dataset: {hover_path}")
    if not copheme_path.exists():
        raise FileNotFoundError(f"Missing copheme dataset: {copheme_path}")

    with hover_path.open("r", encoding="utf-8") as f:
        hover_items = json.load(f)
    with copheme_path.open("r", encoding="utf-8") as f:
        copheme_items = json.load(f)

    hover_subset = _sample_balanced(
        hover_items,
        labels=["SUPPORTS", "REFUTED"],
        per_label=args.hover_per_label,
        seed=args.seed,
    )
    copheme_subset = _sample_balanced(
        copheme_items,
        labels=["NOT ENOUGH INFO", "REFUTED"],
        per_label=args.copheme_per_label,
        seed=args.seed,
    )

    hover_out = processed_dir / "hover_labeled_balanced.json"
    copheme_out = processed_dir / "copheme_nei_balanced.json"

    with hover_out.open("w", encoding="utf-8") as f:
        json.dump(hover_subset, f, indent=2, ensure_ascii=False)
    with copheme_out.open("w", encoding="utf-8") as f:
        json.dump(copheme_subset, f, indent=2, ensure_ascii=False)

    print("Balanced subsets created:")
    print(f"  hover: {len(hover_subset)} -> {hover_out}")
    print(f"  copheme (NEI/REFUTED): {len(copheme_subset)} -> {copheme_out}")


if __name__ == "__main__":
    main()
