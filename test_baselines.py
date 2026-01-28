#!/usr/bin/env python3
"""
Verification Script: Testing Class A No-Tool Baselines.
"""
from src.baselines.no_tool import CoTBaseline, SelfConsistencyBaseline, MADBaseline, JudgeOnlyBaseline
import time

def test_baselines():
    claim = "The capital of France is Paris." # Simple true claim
    
    print("="*60)
    print("🧪 TESTING CLASS A BASELINES (NO-TOOL)")
    print(f"Claim: {claim}")
    print("="*60)
    
    baselines = [
        ("A1. CoT", CoTBaseline()),
        ("A2. Self-Consistency", SelfConsistencyBaseline(k=3)), # Small K for testing
        ("A3. MAD (Multi-Agent)", MADBaseline(num_rounds=1)),   # Small round count for testing
        ("A4. Judge-Only", JudgeOnlyBaseline())
    ]
    
    for name, bl in baselines:
        print(f"\n[RUNNING: {name}]...")
        start = time.time()
        res = bl.verify(claim)
        dur = time.time() - start
        icon = "✅" if res is True else "❌" if res is False else "❓"
        print(f"Result: {icon} {res} (Time: {dur:.1f}s)")

if __name__ == "__main__":
    test_baselines()
