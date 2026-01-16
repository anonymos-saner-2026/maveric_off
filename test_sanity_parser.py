# test_sanity_parser.py
#
# Sanity + Harder sanity checks for src/agents/parser.py (MaVERiC parser)
#
# What this script tries to catch (the big 3 failure modes you care about):
#   H1) Undercut retargeting:
#       "not reliable evidence" should ATTACK an evidence node (trial/study/report),
#       and should NOT SUPPORT the root factual claim.
#
#   H3) Scope / negation / quantifiers:
#       Scope+negation should ATTACK the general claim.
#       Universal 'always' claim should SUPPORT the general claim (or support an intermediate).
#
#   H6) Relevance gate (placeholder-root case):
#       When A1 is a placeholder topic claim, generic related facts (e.g., "France is in Europe")
#       should NOT SUPPORT A1 directly.
#
# Usage:
#   python test_sanity_parser.py

from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Tuple, Optional

if __name__ == "__main__":
    sys.path.append(os.path.abspath("."))

from src.agents.parser import parse_debate  # noqa: E402


# -------------------------
# Pretty print helpers
# -------------------------
def _print_header(title: str) -> None:
    print("\n" + "=" * 44)
    print(f"RUN: {title}")
    print("=" * 44 + "\n")


def _id_num(nid: str) -> int:
    m = re.findall(r"\d+", str(nid))
    if not m:
        return 10**9
    try:
        return int(m[-1])
    except Exception:
        return 10**9


def _collect_edges(af) -> List[Tuple[str, str, str]]:
    edges = []
    for u, v, d in list(af.nx_graph.edges(data=True)):
        et = (d or {}).get("type", "support")
        edges.append((u, v, et))
    edges.sort(key=lambda x: (_id_num(x[0]), _id_num(x[1]), x[2]))
    return edges


def _show_graph(af) -> None:
    print("\n=========================")
    print("NODES")
    print("=========================")
    for nid in sorted(list(af.nodes.keys()), key=_id_num):
        node = af.nodes[nid]
        print(f"- {nid} | speaker={node.speaker} | tool={node.tool_type} | cost={node.verification_cost} | {node.content}")

    print("\n=========================")
    print("EDGES")
    print("=========================")
    for u, v, et in _collect_edges(af):
        print(f"- {u} -> {v} | type={et}")


def _find_nodes_containing(af, *phrases: str) -> List[str]:
    out = []
    for nid, node in af.nodes.items():
        s = (node.content or "").lower()
        ok = True
        for p in phrases:
            if p.lower() not in s:
                ok = False
                break
        if ok:
            out.append(nid)
    return out


def _has_edge(af, u: str, v: str, etype: Optional[str] = None) -> bool:
    if not af.nx_graph.has_edge(u, v):
        return False
    if etype is None:
        return True
    return (af.nx_graph[u][v].get("type") == etype)


def _edges_from(af, u: str) -> List[Tuple[str, str]]:
    out = []
    for _, v, d in af.nx_graph.out_edges(u, data=True):
        out.append((v, (d or {}).get("type", "support")))
    return out


def _warn(msg: str) -> None:
    print(f"⚠️  {msg}")


def _fail(msg: str) -> None:
    raise AssertionError(msg)


# -------------------------
# Generic invariant checks
# -------------------------
def _check_invariants(af) -> None:
    if "A1" not in af.nodes:
        _fail("Missing root node A1")

    for u, v, d in list(af.nx_graph.edges(data=True)):
        if u == v:
            _fail(f"Self-loop edge found: {u} -> {v}")
        et = (d or {}).get("type", "")
        if et not in {"attack", "support"}:
            _fail(f"Invalid edge type: {u} -> {v} type={et}")

    for u, v, _ in _collect_edges(af):
        if u not in af.nodes or v not in af.nodes:
            _fail(f"Dangling edge: {u}->{v} references missing node")


# -------------------------
# Targeted checks for H1/H3/H6
# -------------------------
def _check_H1_undercut(af) -> None:
    root = "A1"

    # undercut candidates
    n_undercut = (
        _find_nodes_containing(af, "not reliable")
        + _find_nodes_containing(af, "unreliable")
        + _find_nodes_containing(af, "not trustworthy")
        + _find_nodes_containing(af, "cannot be trusted")
        + _find_nodes_containing(af, "biased")
    )
    n_undercut = list(dict.fromkeys(n_undercut))
    if not n_undercut:
        _warn("H1: Could not find undercut conclusion node. Skipping strict H1 checks.")
        return
    und = sorted(n_undercut, key=lambda nid: len(af.nodes[nid].content or ""))[0]

    # evidence candidates (exclude undercut node itself)
    ev = []
    for k in ("trial", "study", "paper", "reported", "report"):
        ev += _find_nodes_containing(af, k)
    ev = [nid for nid in dict.fromkeys(ev) if nid != und]

    if _has_edge(af, und, root, "support"):
        _fail(f"H1: Undercut node {und} SUPPORTS root A1 (should not).")

    attacked = any(_has_edge(af, und, e, "attack") for e in ev)
    if not attacked:
        _warn(f"H1: Undercut node {und} does not ATTACK any obvious evidence node (trial/study/report). Risky for MaVERiC.")

    # Optional: reasons -> undercut support
    reasons = []
    reasons += _find_nodes_containing(af, "funded")
    reasons += _find_nodes_containing(af, "never released")
    reasons = list(dict.fromkeys(reasons))
    if reasons and not any(_has_edge(af, r, und, "support") for r in reasons):
        _warn("H1: Found undercut reasons (funded/never released) but none SUPPORT the undercut node. Possible flattening.")


def _check_H3_scope_quantifiers(af) -> None:
    root = "A1"

    # scope-negation: negation + any scope cue
    neg_nodes = (
        _find_nodes_containing(af, "does not")
        + _find_nodes_containing(af, "do not")
        + _find_nodes_containing(af, "not cause")
    )
    scope_cues = ("moderate", "most", "for most", "typical", "normal", "in practice", "generally")
    n_scope_neg = []
    for nid in neg_nodes:
        s = (af.nodes[nid].content or "").lower()
        if any(c in s for c in scope_cues):
            n_scope_neg.append(nid)
    n_scope_neg = list(dict.fromkeys(n_scope_neg))

    if n_scope_neg:
        sneg = sorted(n_scope_neg, key=lambda nid: len(af.nodes[nid].content or ""))[0]
        if not _has_edge(af, sneg, root, "attack"):
            _fail(f"H3: Scope/negation node {sneg} should ATTACK A1 but does not.")
    else:
        _warn("H3: Could not find scope+negation node robustly (negation + scope cue).")

    n_always = _find_nodes_containing(af, "always")
    if n_always:
        always = sorted(n_always, key=lambda nid: ("dehydr" not in (af.nodes[nid].content or "").lower(), len(af.nodes[nid].content or "")))[0]
        if not (_has_edge(af, always, root, "support") or any(et == "support" and v != root for v, et in _edges_from(af, always))):
            _warn(f"H3: 'always' node {always} does not SUPPORT A1 or any intermediate. Might be mis-typed.")
    else:
        _warn("H3: Could not find an 'always' node.")


def _check_H6_relevance_gate(af) -> None:
    root = "A1"
    n_europe = _find_nodes_containing(af, "france", "europe")
    if not n_europe:
        _warn("H6: Could not find node 'France is in Europe'. Skipping strict relevance check.")
        return

    eu = n_europe[0]
    if _has_edge(af, eu, root, "support"):
        _fail(f"H6: Relevance gate violated: '{af.nodes[eu].content}' SUPPORTS A1 directly.")


# -------------------------
# Tests
# -------------------------
def _run_test(name: str, transcript: str, checks: List) -> None:
    _print_header(name)
    af = parse_debate(transcript)
    _show_graph(af)

    _check_invariants(af)

    for fn in checks:
        fn(af)

    print("\n✅ SANITY PASS (checks ok).")


def main() -> None:
    print("=== Harder Parser Sanity Check (H1/H3/H6 focused) ===")

    t_h1 = """
Moderator: Topic is whether the new drug X reduces blood pressure in adults.
Alice: Drug X reduces blood pressure by about 10 mmHg in adults.
Alice: A 2023 clinical trial reported a statistically significant reduction of 10 mmHg.
Bob: That 2023 trial was funded by the manufacturer.
Bob: The dataset was never released.
Bob: Therefore, the trial is not reliable evidence for Drug X reducing blood pressure.
Carol: Independent meta-analyses usually find smaller effects.
Carol: The smaller effects are around 2 to 3 mmHg.
""".strip()

    t_h3 = """
Moderator: Topic claim is that coffee causes dehydration.
Alice: Coffee is a mild diuretic.
Alice: In moderate amounts, coffee does not cause net dehydration for most people.
Bob: Coffee always causes dehydration.
Bob: Coffee makes you urinate more.
""".strip()

    t_h6 = """
Moderator: Topic claim.
Alice: Paris is the capital of France.
Alice: France is in Europe.
Bob: Paris is not the capital of France.
Bob: France is not in Europe.
""".strip()

    tests = [
        ("H1_undercut_source_trust", t_h1, [_check_H1_undercut]),
        ("H3_hedge_and_scope", t_h3, [_check_H3_scope_quantifiers]),
        ("H6_relevance_gate_multitarget", t_h6, [_check_H6_relevance_gate]),
    ]

    for name, tx, checks in tests:
        try:
            _run_test(name, tx, checks)
        except AssertionError as e:
            print(f"\n❌ SANITY FAIL: {e}\n")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
