
import time
import logging
from src.agents import parser
from src.agents import parser_llm
from src.core.graph import ArgumentationGraph

# Suppress logging
logging.basicConfig(level=logging.CRITICAL)

TRANSCRIPTS = {
    "nuclear_short": """
Moderator: Should we use nuclear energy?
Alice: Nuclear energy is essential because it provides baseload power.
Bob: But it is dangerous. Waste remains radioactive for thousands of years.
Alice: Modern reactors are safe and waste can be recycled.
""",
    "complex_logic": """
Moderator: Is coffee good?
Alice: Coffee improves alertness.
Bob: Coffee improves alertness because caffeine blocks adenosine receptors and increases dopamine.
Alice: However, it causes dependency if you drink it every day.
"""
}

def analyze_graph(g: ArgumentationGraph, name: str):
    print(f"--- Analysis: {name} ---")
    print(f"Nodes: {len(g.nodes)}")
    print(f"Edges: {g.nx_graph.number_of_edges()}")
    
    # Atomicity check (simple heuristic)
    long_nodes = 0
    for n in g.nodes.values():
        if len(n.content.split()) > 15: # Arbitrary threshold
            long_nodes += 1
    print(f"Long nodes (>15 words): {long_nodes}")

    # Coref check
    pronouns = 0
    for n in g.nodes.values():
        lower = n.content.lower()
        if " it " in f" {lower} " or " they " in f" {lower} ":
            pronouns += 1
    print(f"Nodes with 'it'/'they': {pronouns}")

    # Content dump
    print("Node Content:")
    for nid, n in sorted(g.nodes.items()):
        print(f"  [{nid}] {n.content}")
    print("Edges:")
    for u, v, d in g.nx_graph.edges(data=True):
        print(f"  {u} -> {v} ({d.get('type')})")
    print("\n")

def main():
    print("=== Parser Comparison Table ===\n")

    for key, text in TRANSCRIPTS.items():
        print(f"Transcript: {key}")
        print("-" * 40)

        # 1. Standard Parser (parser.py)
        # Note: parse_debate might default to a specific mode, assuming default behavior
        start = time.time()
        try:
            g1 = parser.parse_debate(text)
            dur1 = time.time() - start
            print(f"[Standard Parser] Time: {dur1:.2f}s")
            analyze_graph(g1, "Standard Parser")
        except Exception as e:
            print(f"[Standard Parser] Failed: {e}")

        # 2. LLM Parser (parser_llm.py)
        start = time.time()
        try:
            g2 = parser_llm.parse_debate_llm(text)
            dur2 = time.time() - start
            print(f"[LLM Parser] Time: {dur2:.2f}s")
            analyze_graph(g2, "LLM Parser")
        except Exception as e:
            print(f"[LLM Parser] Failed: {e}")
        
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
