#!/usr/bin/env python3
"""
Quick test of hybrid multi-hop reasoning enhancement
"""
from src.tools.real_toolkit import RealToolkit

# Test case from HoVer error
test_claim = "Skagen Painter Peder Severin Krøyer favored naturalism along with Theodor Esbern Philipsen and the artist Ossian Elgström studied with in 1907."

print("="*70)
print("TESTING HYBRID MULTI-HOP REASONING")
print("="*70)
print(f"Claim: {test_claim}")
print()

# Test Stage 1: Entity Extraction
print("Stage 1: Entity Detection (Regex)")
entity_info = RealToolkit._extract_bridge_entities(test_claim)
print(f"  Has bridge: {entity_info.get('has_bridge')}")
print(f"  Entities: {entity_info.get('entities')}")
print(f"  Pattern: {entity_info.get('pattern_type')}")
print()

# Test Stage 2: Decomposition
if entity_info.get("has_bridge"):
    print("Stage 2: LLM Decomposition")
    sub_queries = RealToolkit._decompose_multihop_claim(test_claim, entity_info)
    if sub_queries:
        print(f"  Decomposed into {len(sub_queries)} steps:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"    {i}. {sq.get('query')} (type: {sq.get('entity_type')})")
    else:
        print("  ⚠️ Decomposition failed")
    print()
    
    # Test Stage 3: Execution (commented out to save time)
    # if sub_queries:
    #     print("Stage 3: Hybrid Execution")
    #     results = RealToolkit._execute_hybrid_multihop(test_claim, sub_queries)
    #     print(f"  Total evidence: {len(results)} snippets")
else:
    print("⚠️ No bridge entity detected, would use standard search")

print()
print("="*70)
print("✅ Test Complete")
