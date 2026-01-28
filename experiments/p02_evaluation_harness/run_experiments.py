#!/usr/bin/env python3
"""
run_experiments.py
Execute MaVERiC benchmark on 5 datasets: TruthfulQA, FEVER, CoPHEME, HoVer, SciFact.
"""

import subprocess
import sys
import os
import time
import argparse

DATASETS = [
    "truthfulqa",
    "fever",
    "copheme",
    "hover",
    "scifact"
]

def run_cmd(cmd):
    print(f"🚀 Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        print(f"❌ Command failed: {' '.join(cmd)}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Run MaVERiC benchmark on multiple datasets")
    parser.add_argument("--budget", type=float, default=8.0, help="Budget per sample (default: 8.0)")
    parser.add_argument("--max_samples", type=int, default=20, help="Max samples per dataset (0 for all, default: 20)")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets to run")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    run_eval_path = os.path.join(base_dir, "run_eval.py")
    results_dir = os.path.join(base_dir, "results_comprehensive")
    os.makedirs(results_dir, exist_ok=True)

    target_datasets = DATASETS
    if args.datasets:
        target_datasets = [d.strip().lower() for d in args.datasets.split(",")]

    print("="*60)
    print("  MaVERiC Comprehensive Benchmark Execution")
    print(f"  Datasets: {target_datasets}")
    print(f"  Budget: {args.budget}")
    print(f"  Samples per dataset: {'ALL' if args.max_samples <= 0 else args.max_samples}")
    if args.max_samples <= 0 or args.max_samples > 100:
        print("\n  ⚠️  WARNING: Running with many samples (especially FEVER/HoVer)")
        print("     will take a long time (hours to days).")
    print("="*60)

    for ds in target_datasets:
        print(f"\n▶️ Processing Dataset: {ds}")
        start = time.time()
        
        cmd = [
            "python3", run_eval_path,
            "--dataset", ds,
            "--method", "maveric",
            "--budget", str(args.budget),
            "--max_samples", str(args.max_samples),
            "--output_dir", results_dir,
            "--use_graph_cache", "--append_cache_on_miss"
        ]
        
        success = run_cmd(cmd)
        elapsed = time.time() - start
        
        if success:
            print(f"✅ Finished {ds} in {elapsed:.1f}s")
        else:
            print(f"⚠️ Failed {ds} in {elapsed:.1f}s")

    print("\n" + "="*60)
    print("🎉 All benchmarks completed.")
    print(f"Results stored in: {results_dir}")
    print("Use summarize.py to aggregate metrics.")

if __name__ == "__main__":
    main()
