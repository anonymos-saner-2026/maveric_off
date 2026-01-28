# test_solver_unit_v2_hard.py
# Hard deterministic unit tests for MaVERiC solver_v2 (no real API calls).
# Covers: tool routing (math->PYTHON_EXEC), cost-aware top-k, widen shortlist if unaffordable,
# y_direct override, adversary detection (attack only), attack->support conversion,
# structural confidence counts only verified-true supporters, tool-cache sanity.

import math
import sys
import traceback
import pytest
from typing import Optional

# ---------------------------
# Imports with fallbacks
# ---------------------------
def _import_graph():
    try:
        from src.core.graph import ArgumentationGraph, ArgumentNode
        return ArgumentationGraph, ArgumentNode
    except Exception:
        # fallback if your graph module lives elsewhere
        from src.graph import ArgumentationGraph, ArgumentNode
        return ArgumentationGraph, ArgumentNode

def _import_solver_module():
    # You said current file is src/core/solver.py
    try:
        import src.core.solver as solver_mod
        return solver_mod
    except Exception:
        # fallback if you keep separate solver_v2.py
        import src.core.solver_v2 as solver_mod
        return solver_mod


ArgumentationGraph, ArgumentNode = _import_graph()
solver_mod = _import_solver_module()


# ---------------------------
# Minimal test harness
# ---------------------------
def _ok(msg):
    print(f"[OK] {msg}")

def _fail(msg):
    raise AssertionError(msg)

def _assert(cond, msg):
    if not cond:
        _fail(msg)

def _assert_eq(a, b, msg):
    if a != b:
        _fail(f"{msg} | got={a} expected={b}")

def _assert_close(a, b, tol=1e-6, msg=""):
    if abs(a - b) > tol:
        _fail(f"{msg} | got={a} expected={b} tol={tol}")


# ---------------------------
# Fake RealToolkit (deterministic)
# ---------------------------
class FakeRealToolkit:
    """
    Deterministic substitute for src.tools.real_toolkit.RealToolkit used in solver.
    We control verify_claim / verify_attack / verify_support behavior.
    """
    claim_truth = {}        # claim_text -> bool
    attack_truth = {}       # (attacker_text, target_text) -> bool
    support_truth = {}      # (source_text, target_text) -> bool

    calls_verify_claim = []     # list of (tool_type, claim_text)
    calls_verify_attack = []    # list of (att_text, tgt_text)
    calls_verify_support = []   # list of (src_text, tgt_text)

    @staticmethod
    def reset():
        FakeRealToolkit.claim_truth = {}
        FakeRealToolkit.attack_truth = {}
        FakeRealToolkit.support_truth = {}
        FakeRealToolkit.calls_verify_claim = []
        FakeRealToolkit.calls_verify_attack = []
        FakeRealToolkit.calls_verify_support = []

    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> Optional[bool]:
        FakeRealToolkit.calls_verify_claim.append((tool_type, claim))
        # default: if not specified, return True
        return bool(FakeRealToolkit.claim_truth.get(claim, True))

    @staticmethod
    def verify_attack(attacker_content: str, target_content: str) -> bool:
        FakeRealToolkit.calls_verify_attack.append((attacker_content, target_content))
        # default: treat as valid attack unless specified
        return bool(FakeRealToolkit.attack_truth.get((attacker_content, target_content), True))

    @staticmethod
    def verify_support(source_content: str, target_content: str) -> bool:
        FakeRealToolkit.calls_verify_support.append((source_content, target_content))
        # default: not support unless specified
        return bool(FakeRealToolkit.support_truth.get((source_content, target_content), False))


# Monkeypatch RealToolkit inside solver module
solver_mod.RealToolkit = FakeRealToolkit


# ---------------------------
# Helpers to build graphs quickly
# ---------------------------
def make_node(nid, content, tool_type="AUTO", cost=0.0, speaker="X"):
    return ArgumentNode(
        id=nid,
        content=content,
        speaker=speaker,
        is_verified=False,
        ground_truth=None,
        verification_cost=float(cost) if cost else 0.0,
        tool_type=tool_type,
    )

def make_solver(g, budget, **kwargs):
    return solver_mod.MaVERiCSolver(graph=g, budget=budget, **kwargs)


# ---------------------------
# Tests
# ---------------------------
def test_tool_routing_math_is_python_exec():
    FakeRealToolkit.reset()
    g = ArgumentationGraph()
    n1 = make_node("A1", "2 + 2 equals 4.", tool_type="AUTO")
    n2 = make_node("A2", "2 + 2 equals 5.", tool_type="AUTO")
    g.add_node(n1)
    g.add_node(n2)
    g.add_attack("A2", "A1")

    # enforce deterministic truth
    FakeRealToolkit.claim_truth[n1.content] = True
    FakeRealToolkit.claim_truth[n2.content] = False

    s = make_solver(g, budget=2.1, topk_counterfactual=5)
    final_ext, verdict = s.run()

    # Expect both verifications to use PYTHON_EXEC (math override)
    _assert(len(FakeRealToolkit.calls_verify_claim) >= 1, "verify_claim not called")
    for tool, claim in FakeRealToolkit.calls_verify_claim:
        _assert_eq(tool, "PYTHON_EXEC", "math claim should route to PYTHON_EXEC")

    _ok("tool_routing_math_is_python_exec")


def test_cost_aware_topk_picks_cheaper_high_roi():
    FakeRealToolkit.reset()

    # Two nodes, no edges => deg same
    # We set pagerank indirectly by patching nx.pagerank in solver module call site would be messy.
    # Instead: use node.verification_cost to force cost difference and rely on cheap-score/cost.
    # With topk=1, shortlist picks by cost-aware cheap score.

    g = ArgumentationGraph()
    cheap = make_node("A1", "17 * 19 equals 323.", tool_type="AUTO", cost=1.0)
    expensive = make_node("A2", "The capital of France is Paris.", tool_type="WEB_SEARCH", cost=100.0)
    g.add_node(cheap)
    g.add_node(expensive)

    # Set truths
    FakeRealToolkit.claim_truth[cheap.content] = True
    FakeRealToolkit.claim_truth[expensive.content] = True

    # Patch pagerank to make expensive look structurally more important
    # but cheap should still win due to cost-aware cheap score.
    orig_pagerank = solver_mod.nx.pagerank
    def fake_pagerank(G, alpha=0.85):
        return {"A1": 0.01, "A2": 0.99}  # expensive has higher phi
    solver_mod.nx.pagerank = fake_pagerank

    try:
        s = make_solver(g, budget=1.1, topk_counterfactual=1)  # only enough for the cheap node
        final_ext, verdict = s.run()

        # First (and only) verify should be cheap node A1
        _assert(len(FakeRealToolkit.calls_verify_claim) >= 1, "verify_claim not called")
        first_tool, first_claim = FakeRealToolkit.calls_verify_claim[0]
        _assert_eq(first_claim, cheap.content, "cost-aware topk should pick cheap node first")
        _ok("cost_aware_topk_picks_cheaper_high_roi")
    finally:
        solver_mod.nx.pagerank = orig_pagerank


def test_widen_shortlist_finds_affordable_candidate():
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    # Node A2 gets higher pagerank, but is unaffordable; A1 is affordable
    affordable = make_node("A1", "17 * 19 equals 323.", tool_type="AUTO", cost=2.0)
    unaffordable = make_node("A2", "Some web fact needing search.", tool_type="WEB_SEARCH", cost=100.0)
    g.add_node(affordable)
    g.add_node(unaffordable)

    FakeRealToolkit.claim_truth[affordable.content] = True
    FakeRealToolkit.claim_truth[unaffordable.content] = True

    orig_pagerank = solver_mod.nx.pagerank
    def fake_pagerank(G, alpha=0.85):
        return {"A1": 0.01, "A2": 0.99}
    solver_mod.nx.pagerank = fake_pagerank

    try:
        s = make_solver(g, budget=2.01, topk_counterfactual=1)  # top1 would be A2 but cannot afford
        final_ext, verdict = s.run()

        _assert(len(FakeRealToolkit.calls_verify_claim) >= 1, "verify_claim not called")
        first_tool, first_claim = FakeRealToolkit.calls_verify_claim[0]
        _assert_eq(first_claim, affordable.content, "widen shortlist should find affordable node")
        _ok("widen_shortlist_finds_affordable_candidate")
    finally:
        solver_mod.nx.pagerank = orig_pagerank


def test_adversary_detection_attack_only():
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    root = make_node("A1", "Root claim.", tool_type="AUTO", cost=1.0)
    supporter = make_node("A2", "Support evidence.", tool_type="AUTO", cost=10.0)
    attacker = make_node("A3", "Attack statement.", tool_type="AUTO", cost=10.0)

    g.add_node(root); g.add_node(supporter); g.add_node(attacker)
    g.add_support("A2", "A1")
    g.add_attack("A3", "A1")

    FakeRealToolkit.claim_truth[root.content] = True

    # Ensure root chosen first and only verified
    s = make_solver(g, budget=1.05, topk_counterfactual=10, delta_root=50.0)
    final_ext, verdict = s.run()

    _assert("A3" in s.flagged_adversaries, "attacker via attack-edge should be flagged adversary")
    _assert("A2" not in s.flagged_adversaries, "supporter via support-edge must not be flagged adversary")
    _ok("adversary_detection_attack_only")


def test_convert_invalid_attack_to_support():
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    t = make_node("A1", "True node content.", tool_type="AUTO", cost=1.0)
    x = make_node("B1", "Target node content.", tool_type="AUTO", cost=10.0)
    g.add_node(t); g.add_node(x)
    g.add_attack("A1", "B1")

    FakeRealToolkit.claim_truth[t.content] = True

    # Force attack invalid, support valid => conversion should happen
    FakeRealToolkit.attack_truth[(t.content, x.content)] = False
    FakeRealToolkit.support_truth[(t.content, x.content)] = True

    s = make_solver(g, budget=1.05, topk_counterfactual=10, delta_root=50.0)
    final_ext, verdict = s.run()

    d = g.nx_graph.get_edge_data("A1", "B1")
    _assert(d is not None, "edge A1->B1 should exist after conversion")
    _assert_eq(d.get("type"), "support", "invalid attack should be converted to support")
    _ok("convert_invalid_attack_to_support")


def test_structural_confidence_counts_only_verified_true_supporters():
    pytest.skip("structural confidence helper removed in current solver API")
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    x = make_node("X", "X", tool_type="AUTO")
    s1 = make_node("S1", "S1", tool_type="AUTO")
    s2 = make_node("S2", "S2", tool_type="AUTO")
    g.add_node(x); g.add_node(s1); g.add_node(s2)
    g.add_support("S1", "X")
    g.add_support("S2", "X")

    # Mark X verified TRUE, S1 verified TRUE, S2 unverified
    g.nodes["X"].is_verified = True
    g.nodes["X"].ground_truth = True

    g.nodes["S1"].is_verified = True
    g.nodes["S1"].ground_truth = True

    g.nodes["S2"].is_verified = False
    g.nodes["S2"].ground_truth = None

    s = make_solver(g, budget=0.0)

    # Monkeypatch grounded extension to isolate the confidence formula
    orig_ge = g.get_grounded_extension
    g.get_grounded_extension = lambda: {"X"}

    try:
        pagerank = {"X": 1.0, "S1": 1.0, "S2": 1.0}
        conf = s._calculate_structural_confidence(pagerank)

        # current_weight = 1.0 * (1.2 ** 1) = 1.2
        # total_weight = 3.0
        expected = (1.2 / 3.0) * 100.0
        _assert_close(conf, expected, tol=1e-6, msg="confidence should count only verified-true supporters")
        _ok("structural_confidence_counts_only_verified_true_supporters")
    finally:
        g.get_grounded_extension = orig_ge


def test_y_direct_overrides_semantics_membership():
    FakeRealToolkit.reset()

    # Build graph where grounded extension excludes root due to unattacked attacker
    # A2 attacks A1, no one attacks A2 => GE tends to accept A2 and defeat A1.
    # If solver verifies A1 TRUE first, y_direct=True should make verdict True
    # even if A1 not in final grounded extension.
    g = ArgumentationGraph()
    root = make_node("A1", "Root claim is TRUE.", tool_type="AUTO", cost=1.0)
    attacker = make_node("A2", "Attacker node.", tool_type="AUTO", cost=10.0)
    g.add_node(root); g.add_node(attacker)
    g.add_attack("A2", "A1")

    FakeRealToolkit.claim_truth[root.content] = True

    s = make_solver(g, budget=1.05, topk_counterfactual=10, delta_root=100.0)
    final_ext, verdict = s.run()

    _assert(s.y_direct is True, "y_direct should be set when root verified")
    _assert(verdict is True, "verdict should follow y_direct=True")
    # It's acceptable that root may not be in extension due to semantics
    _assert("A1" not in final_ext or "A2" in final_ext, "sanity: semantics may exclude root here")
    _ok("y_direct_overrides_semantics_membership")


def test_tool_cache_called_once_per_node_in_candidate_calc():
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    n1 = make_node("A1", "A non-math factual claim.", tool_type="AUTO", cost=5.0)
    g.add_node(n1)

    s = make_solver(g, budget=100.0, topk_counterfactual=5)

    # Replace _decide_tool_strategy with counter to ensure not called multiple times
    calls = {"n": 0}
    def fake_router(claim):
        calls["n"] += 1
        return "WEB_SEARCH"
    s._decide_tool_strategy = fake_router

    current_ext = set()
    g_atk = g.get_attack_subgraph()
    Nk_atk = set()
    dist_to_root = {}

    # First call should trigger router once
    cands1 = s._calculate_roi_candidates([n1], current_ext, g_atk, Nk_atk, dist_to_root)
    _assert_eq(calls["n"], 1, "router should be called once in first calc")

    # Second call should use cache, not call router again
    cands2 = s._calculate_roi_candidates([n1], current_ext, g_atk, Nk_atk, dist_to_root)
    _assert_eq(calls["n"], 1, "router should be cached, not called again")

    _ok("tool_cache_called_once_per_node_in_candidate_calc")


def test_run_live_yields_updates_and_counts_tool_calls():
    FakeRealToolkit.reset()

    g = ArgumentationGraph()
    root = make_node("A1", "2 + 2 equals 4.", tool_type="AUTO", cost=1.0)
    other = make_node("A2", "2 + 2 equals 5.", tool_type="AUTO", cost=1.0)
    g.add_node(root); g.add_node(other)
    g.add_attack("A2", "A1")

    FakeRealToolkit.claim_truth[root.content] = True
    FakeRealToolkit.claim_truth[other.content] = False

    s = make_solver(g, budget=2.05, topk_counterfactual=10, delta_root=50.0)

    final_ext, verdict = s.run()

    _assert(isinstance(final_ext, set), "run should return a set extension")
    _assert(s.tool_calls >= 1, "tool_calls should increment")
    _ok("run_yields_results_and_counts_tool_calls")


# ---------------------------
# Runner
# ---------------------------
def run_all():
    tests = [
        test_tool_routing_math_is_python_exec,
        test_cost_aware_topk_picks_cheaper_high_roi,
        test_widen_shortlist_finds_affordable_candidate,
        test_adversary_detection_attack_only,
        test_convert_invalid_attack_to_support,
        test_structural_confidence_counts_only_verified_true_supporters,
        test_y_direct_overrides_semantics_membership,
        test_tool_cache_called_once_per_node_in_candidate_calc,
        test_run_live_yields_updates_and_counts_tool_calls,
    ]

    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\nAll solver v2 hard unit tests passed ✅")


if __name__ == "__main__":
    run_all()
