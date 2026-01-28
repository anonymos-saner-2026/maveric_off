
from src.agents.parser import parse_debate
import json

def test_coreference_resolution():
    transcript = """
Moderator: Today, we discuss Ibuprofen.
Alice: Ibuprofen is a common painkiller.
Bob: It causes stomach ulcers in high doses.
    """
    
    print(f"--- Input Transcript ---\n{transcript}\n------------------------")
    
    # Run parser
    g = parse_debate(transcript, split_mode="llm")
    
    # Check nodes
    print("\n--- Parsed Nodes ---")
    resolved = False
    for nid, node in g.nodes.items():
        print(f"[{nid}] {node.speaker}: {node.content}")
        # We want Bob's claim to say "Ibuprofen causes..." not "It causes..."
        if node.speaker == "Bob" and "ibuprofen" in node.content.lower() and "stomach" in node.content.lower():
            resolved = True
            
    if resolved:
        print("\nSUCCESS: Coreference resolved!")
    else:
        print("\nFAILURE: Coreference NOT resolved (likely 'It' used).")

if __name__ == "__main__":
    test_coreference_resolution()
