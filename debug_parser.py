
import logging
from src.agents import parser
# Monkey patch _build_shortlist to print debug info
original_build_shortlist = parser._build_shortlist
def debug_build_shortlist(nid, candidate_ids, limit):
    print(f"[DEBUG] nid={nid}, candidates_pool_size={len(candidate_ids)}")
    res = original_build_shortlist(nid, candidate_ids, limit)
    print(f"[DEBUG] -> Returned {len(res)} candidates: {[c['id'] for c in res]}")
    return res

parser._build_shortlist = debug_build_shortlist

TRANSCRIPT = """
Moderator: Topic: Is remote work better for productivity?
Alice: Remote work reduces commute time, which increases focus.
Bob: Remote work harms collaboration and knowledge sharing.
Charlie: Hybrid models balance focus time with collaboration.
David: Remote work can improve work-life balance but blur boundaries.
"""

def main():
    print("=== Debug: Shortlist Tracing ===\n")
    try:
        g = parser.parse_debate(TRANSCRIPT)
        print(f"Nodes: {len(g.nodes)}")
        print(f"Edges: {g.nx_graph.number_of_edges()}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    main()
