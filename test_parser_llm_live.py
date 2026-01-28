# test_parser_llm_live.py
# Live LLM parser tests (no mocking). Runs a few small transcripts
# through multiple parser modes and validates basic graph invariants.

from __future__ import annotations

import time
from typing import Callable

from src.agents.parser import parse_debate, parse_debate_fast, parse_debate_paper_safe
from src.agents.parser_llm import parse_debate_llm


def _basic_checks(graph, label: str) -> None:
    assert graph is not None, f"{label}: graph is None"
    assert "A1" in graph.nodes, f"{label}: missing root node A1"
    assert graph.root_id_override == "A1", f"{label}: root_id_override not set to A1"
    assert len(graph.nodes) >= 1, f"{label}: empty graph"

    for nid, node in graph.nodes.items():
        assert node.content and node.content.strip(), f"{label}: empty content for node {nid}"

    for u, v, d in graph.nx_graph.edges(data=True):
        assert u != v, f"{label}: self-loop edge {u}->{v}"
        edge_type = (d or {}).get("type")
        assert edge_type in {"attack", "support"}, f"{label}: invalid edge type {edge_type}"


def _run_case(name: str, transcript: str, parser_fn: Callable[[str], object]) -> None:
    start = time.time()
    graph = parser_fn(transcript)
    elapsed = time.time() - start
    _basic_checks(graph, name)

    total_nodes = len(graph.nodes)
    total_edges = graph.nx_graph.number_of_edges()
    orphans = [nid for nid in graph.nodes if nid != "A1" and graph.nx_graph.in_degree(nid) == 0]
    orphan_ratio = (len(orphans) / total_nodes) if total_nodes else 0.0

    print(f"== {name} ==")
    print(f"nodes={total_nodes} edges={total_edges} orphan_ratio={orphan_ratio:.2f} time={elapsed:.2f}s")


def main() -> None:
    transcript_short = (
        "Moderator: Topic is whether nuclear power reduces emissions.\n"
        "Alice: Nuclear power emits less CO2 than coal because reactors do not burn fossil fuels.\n"
        "Bob: Nuclear power has long construction times and high costs, so it is not practical."
    )

    transcript_long = (
        "Moderator: Topic is whether remote work improves productivity.\n"
        "Alice: Remote work reduces commute time and gives employees more focused hours.\n"
        "Bob: Remote work can reduce collaboration and increase miscommunication.\n"
        "Charlie: Hybrid work balances focus and collaboration, so hybrid work is best."
    )

    cases = [
        ("default_parser_short", transcript_short, lambda t: parse_debate(t, split_mode="hybrid")),
        ("paper_safe_short", transcript_short, parse_debate_paper_safe),
        ("fast_short", transcript_short, parse_debate_fast),
        ("llm_only_short", transcript_short, parse_debate_llm),
        ("default_parser_long", transcript_long, lambda t: parse_debate(t, split_mode="hybrid")),
        ("paper_safe_long", transcript_long, parse_debate_paper_safe),
        ("fast_long", transcript_long, parse_debate_fast),
        ("llm_only_long", transcript_long, parse_debate_llm),
    ]

    print("== Running live LLM parser tests ==")
    for name, transcript, parser_fn in cases:
        _run_case(name, transcript, parser_fn)


if __name__ == "__main__":
    main()
