# test_logic_graph.py
# Minimal, deterministic tests for MaVERiC solver_v2 ROI v2 selection logic.
#
# What this script validates:
# 1) Stage-1 proxy shortlist ranks the intended node highest (cost-aware).
# 2) Stage-2 bounded root-aware impact g(v) is computed correctly:
#       Delta_root (XOR root membership flip) + Delta_local (localized symmdiff)
# 3) Final ROI(v) matches the ROI v2 formula and selects the intended node.
# 4) ROI scales inversely with cost (sanity check).
#
# How to run:
#   python test_logic_graph.py
#
# Notes:
# - We do NOT rely on any external tools. We stub RealToolkit calls.
# - We bypass run() randomness by directly calling _calculate_roi_candidates
#   with fixed pagerank scores.
# - Grounded extension semantics are mocked deterministically in DummyGraph.

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass
from typing import Dict, Optional, Set

import networkx as nx
from typing import Optional

# Import your merged solver. Adjust this import path if needed.
import src.core.solver_v2 as solver_v2


# -----------------------------
# Dummy node & graph wrappers
# -----------------------------
@dataclass
class DummyNode:
    id: str
    content: str
    tool_type: Optional[str] = None
    verification_cost: Optional[float] = None
    speaker: str = "UNK"
    is_verified: bool = False
    ground_truth: Optional[bool] = None


class DummyGraph:
    """
    A tiny graph wrapper that matches the solver expectations:
    - nx_graph: networkx.DiGraph with edge attribute "type" in {"attack","support"}.
    - nodes: dict[str, DummyNode]
    - remove_node(node_id)
    - find_semantic_root() -> node_id
    - get_grounded_extension() -> set[node_id]
    """

    def __init__(self):
        self.nx_graph = nx.DiGraph()
        self.nodes: Dict[str, DummyNode] = {}

    def add_node(self, node: DummyNode):
        self.nodes[node.id] = node
        self.nx_graph.add_node(node.id)

    def add_edge(self, u: str, v: str, edge_type: str):
        assert edge_type in {"attack", "support"}
        self.nx_graph.add_edge(u, v, type=edge_type)

    def remove_node(self, node_id: str):
        if node_id in self.nx_graph:
            self.nx_graph.remove_node(node_id)
        if node_id in self.nodes:
            del self.nodes[node_id]

    def find_semantic_root(self) -> str:
        return "r"

    def get_shielded_nodes(self):
        return []

    def get_grounded_extension(self) -> Set[str]:
        """
        Deterministic "mock" semantics:

        If attacker 'a' exists, root 'r' is NOT accepted:
          GE = {'a','c'}.
        If 'a' is removed, root becomes accepted:
          GE = {'r','c'}.

        Node 'b' is irrelevant and never appears in GE.

        This ensures:
        - Removing 'a' flips root membership (Delta_root = 1).
        - Removing 'c' does NOT flip root membership but causes local change (Delta_local > 0).
        """
        nodes = set(self.nx_graph.nodes())
        if "a" in nodes:
            ext = {"a", "c"} if "c" in nodes else {"a"}
        else:
            ext = {"r", "c"} if "c" in nodes else {"r"}
        # Ensure extension only includes present nodes
        return set([x for x in ext if x in nodes])


# -----------------------------
# Stub out RealToolkit calls
# -----------------------------
class DummyToolkit:
    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> Optional[bool]:
        # Not used in ROI tests; keep deterministic anyway.
        return True

    @staticmethod
    def verify_attack(src_claim: str, tgt_claim: str) -> bool:
        return True

    @staticmethod
    def verify_support(src_claim: str, tgt_claim: str) -> bool:
        return True


def patch_toolkit():
    # solver_v2 imported RealToolkit into its module namespace.
    solver_v2.RealToolkit = DummyToolkit


# -----------------------------
# Assertions / helpers
# -----------------------------
def assert_close(a: float, b: float, tol: float = 1e-6, msg: str = ""):
    if abs(a - b) > tol:
        raise AssertionError(f"{msg} | expected {b}, got {a} (tol={tol})")


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


# -----------------------------
# Build deterministic test graph
# -----------------------------
def build_graph_for_roi_tests() -> DummyGraph:
    """
    Nodes:
      r: root (expensive)
      a: attacker of root (cheap) -- should be selected by ROI v2
      c: supporter of root (cheap)
      b: irrelevant/disconnected (cheap)

    Edges:
      a -> r (attack)
      c -> r (support)
    """
    g = DummyGraph()

    # Costs:
    # Root is expensive so we don't trivially pick it due to proximity.
    g.add_node(DummyNode("r", "Root claim", verification_cost=5.0))
    g.add_node(DummyNode("a", "Attacker claim", verification_cost=1.0))
    g.add_node(DummyNode("c", "Supporter claim", verification_cost=1.0))
    g.add_node(DummyNode("b", "Irrelevant claim", verification_cost=1.0))

    g.add_edge("a", "r", "attack")
    g.add_edge("c", "r", "support")
    # b is disconnected

    return g


# -----------------------------
# Tests
# -----------------------------
def test_stage1_proxy_ranking_and_shortlist():
    """
    Confirms Stage-1 proxy ROI_tilde ranks 'a' above others under chosen pagerank.
    """
    g = build_graph_for_roi_tests()

    solver = solver_v2.MaVERiCSolver(
        graph=g,
        budget=10.0,
        # neutralize priority boosts to focus on ROI mechanics
        root_boost=1.0,
        adversary_boost=1.0,
        support_to_root_boost=1.0,
        topk_counterfactual=1,  # force shortlist size 1
        k_hop_root=1,
        beta_root_flip=0.7,
        gamma_struct=0.8,
        alpha_proxy=0.6,
        roi_eps=1e-6,
    )
    solver.root_id = g.find_semantic_root()

    # Fixed pagerank to make a > c > r > b structurally
    pagerank = {"a": 0.40, "c": 0.30, "r": 0.20, "b": 0.10}

    active = [n for n in g.nodes.values() if not n.is_verified]
    current_ext = g.get_grounded_extension()

    candidates = solver._calculate_roi_candidates(active, pagerank, current_ext)
    assert_true(len(candidates) == 1, "Expected shortlist size 1 to yield 1 candidate under budget.")
    best_node, best_roi, (dr, dl), tool, cost = candidates[0]
    assert_true(best_node.id == "a", f"Stage-1 proxy shortlist should pick 'a', got '{best_node.id}'.")


def test_roi_v2_components_and_selection():
    """
    Validates g(v) calculation and final ROI ranking selects node 'a'.
    """
    g = build_graph_for_roi_tests()

    solver = solver_v2.MaVERiCSolver(
        graph=g,
        budget=10.0,
        root_boost=1.0,
        adversary_boost=1.0,
        support_to_root_boost=1.0,
        topk_counterfactual=25,
        k_hop_root=1,
        beta_root_flip=0.7,
        gamma_struct=0.8,
        alpha_proxy=0.6,
        roi_eps=1e-6,
    )
    solver.root_id = g.find_semantic_root()

    pagerank = {"a": 0.40, "c": 0.30, "r": 0.20, "b": 0.10}

    active = [n for n in g.nodes.values() if not n.is_verified]
    current_ext = g.get_grounded_extension()  # {'a','c'}

    # Compute candidates and pick best by ROI
    candidates = solver._calculate_roi_candidates(active, pagerank, current_ext)
    assert_true(len(candidates) > 0, "No candidates returned.")

    best_node, best_roi, best_delta, tool, cost = max(candidates, key=lambda x: x[1])
    assert_true(best_node.id == "a", f"ROI v2 should select 'a' as keystone, got '{best_node.id}'.")

    # Now validate g(a) approximately, using the known dummy semantics.
    # current_ext = {'a','c'}
    # removing 'a' => new_ext = {'r','c'}
    current_ext_set = set(current_ext)
    temp = build_graph_for_roi_tests()
    temp.remove_node("a")
    new_ext_set = set(temp.get_grounded_extension())  # {'r','c'}

    # Nk(root) with k=1 includes {r, a, c} (undirected neighbors + self)
    Nk = solver._k_hop_neighborhood("r", 1)
    assert_true("r" in Nk and "a" in Nk and "c" in Nk, f"Unexpected Nk(root)={Nk}")

    g_val, delta_root, delta_local = solver._bounded_root_impact(
        current_ext_set=current_ext_set,
        new_ext_set=new_ext_set,
        Nk=Nk,
    )

    # Expected:
    # delta_root = 1 (root membership flips from False to True)
    assert_true(delta_root == 1, f"Expected delta_root=1 for removing 'a', got {delta_root}")

    # delta_local:
    # symdiff({'a','c'},{'r','c'}) = {'a','r'}; intersect Nk = {'a','r'} -> 2
    # |Nk| = 3 => 2/3
    expected_delta_local = 2.0 / 3.0
    assert_close(delta_local, expected_delta_local, tol=1e-6, msg="delta_local mismatch")

    # g = 0.7*1 + 0.3*(2/3) = 0.7 + 0.2 = 0.9
    expected_g = 0.7 * 1.0 + 0.3 * expected_delta_local
    assert_close(g_val, expected_g, tol=1e-6, msg="g(v) mismatch")


def test_roi_scales_with_cost():
    """
    If everything else equal, doubling cost should roughly halve ROI.
    """
    g = build_graph_for_roi_tests()
    # Make 'a' cost = 2 instead of 1
    g.nodes["a"].verification_cost = 2.0

    solver = solver_v2.MaVERiCSolver(
        graph=g,
        budget=10.0,
        root_boost=1.0,
        adversary_boost=1.0,
        support_to_root_boost=1.0,
        topk_counterfactual=25,
        k_hop_root=1,
        beta_root_flip=0.7,
        gamma_struct=0.8,
        alpha_proxy=0.6,
        roi_eps=1e-6,
    )
    solver.root_id = g.find_semantic_root()

    pagerank = {"a": 0.40, "c": 0.30, "r": 0.20, "b": 0.10}
    active = [n for n in g.nodes.values() if not n.is_verified]
    current_ext = g.get_grounded_extension()

    candidates = solver._calculate_roi_candidates(active, pagerank, current_ext)
    # Extract ROI for 'a'
    roi_a = None
    for node, roi, _, _, cost in candidates:
        if node.id == "a":
            roi_a = roi
            assert_true(abs(cost - 2.0) < 1e-9, "Cost for 'a' not applied correctly.")
    assert_true(roi_a is not None, "Did not find 'a' among candidates.")

    # Now compare to baseline where cost=1
    g2 = build_graph_for_roi_tests()
    solver2 = solver_v2.MaVERiCSolver(
        graph=g2,
        budget=10.0,
        root_boost=1.0,
        adversary_boost=1.0,
        support_to_root_boost=1.0,
        topk_counterfactual=25,
        k_hop_root=1,
        beta_root_flip=0.7,
        gamma_struct=0.8,
        alpha_proxy=0.6,
        roi_eps=1e-6,
    )
    solver2.root_id = g2.find_semantic_root()
    active2 = [n for n in g2.nodes.values() if not n.is_verified]
    candidates2 = solver2._calculate_roi_candidates(active2, pagerank, g2.get_grounded_extension())
    roi_a2 = None
    for node, roi, _, _, cost in candidates2:
        if node.id == "a":
            roi_a2 = roi
            assert_true(abs(cost - 1.0) < 1e-9, "Baseline cost for 'a' not correct.")
    assert_true(roi_a2 is not None, "Did not find baseline 'a' among candidates.")

    # ROI should be ~ half when cost doubles (allow tolerance due to eps factors)
    ratio = roi_a2 / roi_a if roi_a > 0 else float("inf")
    assert_true(1.8 <= ratio <= 2.2, f"Expected ~2x ratio when cost halves; got ratio={ratio:.3f}")


# -----------------------------
# Runner
# -----------------------------
def main():
    patch_toolkit()

    tests = [
        ("test_stage1_proxy_ranking_and_shortlist", test_stage1_proxy_ranking_and_shortlist),
        ("test_roi_v2_components_and_selection", test_roi_v2_components_and_selection),
        ("test_roi_scales_with_cost", test_roi_scales_with_cost),
    ]

    ok = 0
    print("== Running MaVERiC ROI v2 logic tests ==")
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            ok += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()

    print(f"\nSummary: {ok}/{len(tests)} tests passed.")
    if ok != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
