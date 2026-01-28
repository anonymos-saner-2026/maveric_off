
from src.agents.parser import parse_debate
import json

def verify():
    # Transcript designed to test 3 features:
    # 1. Context-aware splitting: Bob uses "It".
    # 2. Deduplication: Charlie repeats Alice exactly.
    # 3. Temporal Edges: David attacks Bob.
    transcript = """
Moderator: We are discussing Ibuprofen safety.
Alice: Ibuprofen is a safe painkiller for most adults.
Bob: It causes stomach bleeding in elderly patients.
Charlie: Ibuprofen is a safe painkiller for most adults.
David: No, that is wrong. It causes serious bleeding issues.
"""

    print(f"--- Input Transcript ---\n{transcript}\n")
    
    # Use LLM mode to trigger context-aware splitting
    # Disable self-refine (Stage B) to verify A3 edges persist
    g = parse_debate(transcript, split_mode="llm", refine_iters=0)
    
    print("\n--- Resulting Graph ---")
    nodes_by_speaker = {}
    for nid, n in g.nodes.items():
        print(f"[{nid}] ({n.speaker}) {n.content}")
        nodes_by_speaker[n.speaker] = n
        
    print("\n--- Edges ---")
    for u, v, d in g.nx_graph.edges(data=True):
        print(f"{u} -> {v} [{d.get('type')}]")

    # 1. Verify Coreference
    # Bob should say "Ibuprofen causes..." not "It causes..."
    bobs_claims = [n.content for n in g.nodes.values() if n.speaker == "bob"]
    if any("ibuprofen" in c.lower() for c in bobs_claims):
        print("\n✅ Coreference Check: PASSED (Bob's claim is standalone)")
    else:
        print(f"\n❌ Coreference Check: FAILED. Bob said: {bobs_claims}")

    # 2. Verify Deduplication
    # Alice and Charlie said the exact same thing.
    # We expect FEWER nodes than claims.
    # Distinct speakers: Mod, Alice, Bob, Charlie, David.
    # If Alice and Charlie merged, we might only see one node for "safe painkiller".
    # Or we might see edges remapped.
    # Let's check if there are 2 identical nodes or 1.
    contents = [n.content for n in g.nodes.values()]
    if contents.count("Ibuprofen is a safe painkiller for most adults.") == 1:
        print("\n✅ Deduplication Check: PASSED (Duplicate merged)")
    elif contents.count("Ibuprofen is a safe painkiller for most adults.") > 1:
        print("\n❌ Deduplication Check: FAILED (Duplicate remains)")
    else:
        print("\n⚠️ Deduplication Check: Maybe phrasing changed? (OK if core meaning preserved)")

    # 3. Verify Turn-Structure (Implicit)
    # David attacks Bob/Alice.
    # Just checking edge existence.
    if g.nx_graph.number_of_edges() > 0:
        print("\n✅ Relation Extraction: PASSED (Edges created)")
    else:
        print("\n❌ Relation Extraction: FAILED (No edges)")

if __name__ == "__main__":
    verify()
