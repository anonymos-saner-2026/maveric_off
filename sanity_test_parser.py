#!/usr/bin/env python3
"""Sanity checks for parser atomic claim splitting."""

import os
from src.agents import parser


def _print_graph(af):
    nodes = sorted(af.nodes.values(), key=lambda n: n.id)
    print("Nodes:")
    for node in nodes:
        print(f"  {node.id}: {node.content}")
    print("Edges:")
    for u, v, d in sorted(af.nx_graph.edges(data=True)):
        et = (d or {}).get("type", "?")
        print(f"  {u} -> {v} ({et})")
    warnings = getattr(af, "atomicity_warnings", [])
    if warnings:
        print(f"Atomicity warnings: {warnings}")


def _extract_candidates(text: str):
    candidates = []
    root = None
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, claim = line.split(":", 1)
            claim = claim.strip()
            if speaker.strip().lower() == "moderator":
                root = claim
            elif claim:
                candidates.append(claim)
        else:
            candidates.append(line)
    if root is None:
        root = candidates[0] if candidates else "Topic claim."
    return root, candidates


def _mock_llm_split(text: str):
    clauses = [
        "Coffee improves alertness.",
        "Caffeine blocks adenosine receptors.",
        "Caffeine increases dopamine.",
        "Coffee causes dehydration.",
        "Coffee makes people jittery.",
    ]
    links = [
        parser._ClauseLink(child_idx=1, parent_idx=0, connector="because", direction="support", edge_type="support"),
        parser._ClauseLink(child_idx=2, parent_idx=0, connector="and", direction="support", edge_type="support"),
        parser._ClauseLink(child_idx=3, parent_idx=0, connector="but", direction="support", edge_type="attack"),
        parser._ClauseLink(child_idx=4, parent_idx=0, connector="and", direction="support", edge_type="attack"),
    ]
    cleaned = [c for c in clauses if c]
    return cleaned, links


def _split_with_optional_mock(text: str, split_mode: str):
    use_mock = os.environ.get("PARSER_MOCK_LLM", "").strip() == "1"
    if split_mode == "llm" and use_mock:
        return _mock_llm_split(text)
    if split_mode == "hybrid" and use_mock:
        llm_result = _mock_llm_split(text)
        deterministic = parser._split_atomic_claims(text)
        return parser._merge_split_results(llm_result, deterministic)
    if parser._needs_atomic_split(text):
        return parser._split_atomic_claims_with_mode(text, split_mode)
    return [text], []


def _atomicity_metrics_from_split(text: str, split_mode: str) -> dict:
    root, candidates = _extract_candidates(text)
    total_clauses = 0
    total_links = 0
    offenders = 0
    all_clauses = []
    all_links = []

    for claim in [root] + candidates:
        clauses, links = _split_with_optional_mock(claim, split_mode)
        total_clauses += len(clauses)
        total_links += len(links)
        offenders += sum(1 for clause in clauses if parser._needs_atomic_split(clause))
        all_clauses.append(clauses)
        all_links.append(links)

    ratio = (offenders / total_clauses) if total_clauses else 0.0
    return {
        "clauses": total_clauses,
        "links": total_links,
        "offenders": offenders,
        "ratio": ratio,
        "by_claim_clauses": all_clauses,
        "by_claim_links": all_links,
    }


def _run_case(title: str, text: str, split_mode: str) -> dict:
    print("\n" + "=" * 72)
    print(f"{title} | mode={split_mode}")
    metrics = _atomicity_metrics_from_split(text, split_mode)
    print(
        f"clauses={metrics['clauses']} links={metrics['links']} offenders={metrics['offenders']} ratio={metrics['ratio']:.2f}"
    )

    root, candidates = _extract_candidates(text)
    claims = [root] + candidates
    for idx, claim in enumerate(claims):
        print(f"  Claim {idx + 1}: {claim}")
        clauses = metrics["by_claim_clauses"][idx]
        links = metrics["by_claim_links"][idx]
        for c_idx, clause in enumerate(clauses):
            print(f"    [{c_idx}] {clause}")
        if links:
            print("    Links:")
            for link in links:
                direction = link.direction
                edge = link.edge_type
                connector = link.connector or ""
                print(
                    f"      {link.child_idx} -> {link.parent_idx} {edge} dir={direction} connector={connector}"
                )
    return metrics


def main() -> None:
    case_1 = (
        "Moderator: The topic is whether coffee improves alertness.\n"
        "Alice: Coffee improves alertness because caffeine blocks adenosine receptors and increases dopamine.\n"
        "Bob: Coffee causes dehydration and makes people jittery."
    )
    case_2 = (
        "Moderator: Solar power is cost-effective.\n"
        "Alice: Solar installations are cheaper each year; therefore solar power is cost-effective."
    )
    case_3 = (
        "Moderator: Electric cars reduce emissions.\n"
        "Alice: Electric cars reduce emissions in cities, but battery manufacturing emits CO2."
    )
    case_4 = (
        "Moderator: Vaccines are safe.\n"
        "Alice: Large trials show vaccines are safe and they prevent severe disease and hospitalization."
    )

    modes = ["deterministic", "llm", "hybrid"]
    cases = [
        ("Case 1: because + and", case_1),
        ("Case 2: therefore", case_2),
        ("Case 3: but", case_3),
        ("Case 4: multiple and", case_4),
    ]

    use_mock = os.environ.get("PARSER_MOCK_LLM", "").strip() == "1"
    if use_mock:
        print("PARSER_MOCK_LLM=1 (LLM split mocked)")

    summary = {mode: {"clauses": 0, "links": 0, "offenders": 0} for mode in modes}
    for title, text in cases:
        for mode in modes:
            metrics = _run_case(title, text, mode)
            summary[mode]["clauses"] += metrics["clauses"]
            summary[mode]["links"] += metrics["links"]
            summary[mode]["offenders"] += metrics["offenders"]

    print("\n" + "=" * 72)
    print("Aggregate atomicity metrics")
    for mode in modes:
        total_clauses = summary[mode]["clauses"]
        total_links = summary[mode]["links"]
        offenders = summary[mode]["offenders"]
        ratio = (offenders / total_clauses) if total_clauses else 0.0
        print(
            f"{mode}: clauses={total_clauses}, links={total_links}, offenders={offenders}, ratio={ratio:.2f}"
        )


if __name__ == "__main__":
    main()
