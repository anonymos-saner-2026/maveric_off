
import logging
from src.agents.parser_llm import parse_debate_llm
from src.core.graph import ArgumentationGraph

# Minimal logger
logging.basicConfig(level=logging.ERROR)

transcript = """
Moderator: Topic is nuclear safety.
Alice: If a nuclear plant melts down, it destroys the city. It is a long-term disaster.
Bob: Modern reactors are safe. They have passive safety systems that prevent them from overheating.
"""

print("Running parse_debate_llm on pronoun-heavy transcript...")
g = parse_debate_llm(transcript)

print("\n--- Parsed Nodes ---")
for nid, node in g.nodes.items():
    print(f"[{nid}] {node.content}")

print("\n--- Parsed Edges ---")
for u, v, d in g.nx_graph.edges(data=True):
    print(f"{u} -> {v} ({d.get('type')})")
