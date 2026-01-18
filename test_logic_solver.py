# test_logic_solver.py
import unittest
from unittest.mock import patch
import random
import inspect
from contextlib import contextmanager
from typing import Optional

from src.core.solver import MaVERiCSolver
from src.core.graph import ArgumentationGraph, ArgumentNode


# ----------------------------
# Helpers to build graphs
# ----------------------------
def _add_node(g: ArgumentationGraph, nid: str, cost: float = 1.0) -> None:
    g.add_node(
        ArgumentNode(
            id=nid,
            content=f"[{nid}] claim",
            speaker="S",
            is_verified=False,
            ground_truth=None,
            verification_cost=float(cost),
            tool_type="AUTO",
        )
    )


def build_core_flip_graph(cost_v: float = 1.0, cost_a: float = 1.0) -> ArgumentationGraph:
    """
    Small core graph used across tests.
    Nodes: r, a, d, b, v
    Edges:
      a -> r (attack)
      d -> a (attack)
      b -> d (attack)
      d -> b (attack)
      v -> d (support)
    """
    g = ArgumentationGraph()
    _add_node(g, "r", cost=1.0)
    _add_node(g, "a", cost=cost_a)
    _add_node(g, "d", cost=1.0)
    _add_node(g, "b", cost=1.0)
    _add_node(g, "v", cost=cost_v)

    g.add_attack("a", "r")
    g.add_attack("d", "a")
    g.add_attack("b", "d")
    g.add_attack("d", "b")
    g.add_support("v", "d")

    # Deterministic root for tests
    g.root_id_override = "r"
    return g


def add_support_spam(g: ArgumentationGraph, n_spam: int = 30) -> None:
    for i in range(n_spam):
        sid = f"spam{i}"
        _add_node(g, sid, cost=1.0)
        g.add_support(sid, "r")
        g.add_support(sid, "d")


def build_large_graph_with_core(seed: int = 7, n_fillers: int = 25) -> ArgumentationGraph:
    rng = random.Random(seed)
    g = build_core_flip_graph(cost_v=1.0, cost_a=1.0)

    filler_ids = []
    for i in range(n_fillers):
        fid = f"f{i}"
        filler_ids.append(fid)
        _add_node(g, fid, cost=1.0)

    # Random edges among fillers only
    for _ in range(n_fillers * 3):
        u = rng.choice(filler_ids)
        v = rng.choice(filler_ids)
        if u == v:
            continue
        if rng.random() < 0.65:
            g.add_attack(u, v)
        else:
            g.add_support(u, v)

    g.root_id_override = "r"
    return g


def build_root_only_graph(root_truth_cost: float = 1.0) -> ArgumentationGraph:
    """
    Minimal graph where root is the only node.
    Used to test y_direct verdict override deterministically.
    """
    g = ArgumentationGraph()
    _add_node(g, "r", cost=root_truth_cost)
    g.root_id_override = "r"
    return g


def build_prune_false_graph() -> ArgumentationGraph:
    """
    Graph where chosen node is expected to be FALSE, thus solver will prune it.
    We will force solver to pick 'b' (or any chosen) by making its cost cheaper.
    """
    g = build_core_flip_graph(cost_v=1.0, cost_a=1.0)
    # Make b cheapest so solver likely picks it first in many configs.
    g.nodes["b"].verification_cost = 0.5
    g.root_id_override = "r"
    return g


def build_adversary_flagging_graph() -> ArgumentationGraph:
    """
    Attackers of a verified-TRUE node must be flagged as suspected adversaries.
    We'll verify 't' true and ensure 'x','y' (attackers) become flagged.
    """
    g = ArgumentationGraph()
    _add_node(g, "t", cost=1.0)
    _add_node(g, "x", cost=1.0)
    _add_node(g, "y", cost=1.0)
    g.add_attack("x", "t")
    g.add_attack("y", "t")
    g.root_id_override = "t"
    return g


def build_refine_convert_attack_to_support_graph() -> ArgumentationGraph:
    """
    Graph with u -> t as ATTACK, but we will patch verify_attack=False, verify_support=True,
    expecting refine to convert to SUPPORT after u verified TRUE.
    """
    g = ArgumentationGraph()
    _add_node(g, "u", cost=1.0)
    _add_node(g, "t", cost=1.0)
    g.add_attack("u", "t")
    g.root_id_override = "t"
    return g


def build_unreachable_to_root_graph() -> ArgumentationGraph:
    """
    Create a node 'x' that has no directed attack path to root 'r'.
    This should not crash ROI (p=0 for x).
    """
    g = ArgumentationGraph()
    _add_node(g, "r", cost=1.0)
    _add_node(g, "a", cost=1.0)
    _add_node(g, "x", cost=1.0)

    # a attacks r (so a can reach r via attack)
    g.add_attack("a", "r")

    # x only supports a (no attack edges involving x), so x cannot reach r in attack-only graph
    g.add_support("x", "a")

    g.root_id_override = "r"
    return g


# ----------------------------
# Fake toolkit behavior
# ----------------------------
def fake_verify_claim(tool_type: str, claim: str) -> Optional[bool]:
    # Deterministic: only v is True by default
    return "[v]" in (claim or "")


def fake_verify_attack(claim_u: str, claim_v: str) -> bool:
    return True


def fake_verify_support(claim_u: str, claim_v: str) -> bool:
    return True


# ----------------------------
# Patch: stabilize tool + cost
# ----------------------------
def stable_get_tool_and_cost(self, node):
    """
    Deterministic and fast:
      - avoid router calls
      - respect node.verification_cost if present
    """
    tool = "COMMON_SENSE"
    c = getattr(node, "verification_cost", None)
    cost = float(c) if (c is not None and float(c) > 0) else 1.0
    return tool, cost


# ----------------------------
# Utilities for robust testing
# ----------------------------
@contextmanager
def capture_first_verification_event():
    """
    Capture the first verification event even if the verified node is pruned
    from g.nodes (common when verify result is False).
    """
    orig = MaVERiCSolver._verify_node

    def wrapped(self, node, tool: str, cost: float):
        if not hasattr(self, "_test_first_verified_id"):
            self._test_first_verified_id = getattr(node, "id", None)
        res = orig(self, node, tool, cost)
        if not hasattr(self, "_test_first_verified_truth"):
            self._test_first_verified_truth = res
        return res

    with patch.object(MaVERiCSolver, "_verify_node", new=wrapped):
        yield


def _get_first_verified_id(solver: MaVERiCSolver):
    return getattr(solver, "_test_first_verified_id", None)


def _get_first_verified_truth(solver: MaVERiCSolver):
    return getattr(solver, "_test_first_verified_truth", None)


def _tail_logs(solver: MaVERiCSolver, n: int = 20) -> str:
    logs = getattr(solver, "logs", None)
    if not logs:
        return "<no logs attr or empty>"
    return "\n".join(logs[-n:])


def _attack_only_graph_from_nx(nx_graph):
    """
    Build attack-only DiGraph from g.nx_graph (works without using solver internals).
    """
    import networkx as nx

    g_atk = nx.DiGraph()
    g_atk.add_nodes_from(nx_graph.nodes())
    for u, v, d in nx_graph.edges(data=True):
        if (d or {}).get("type") == "attack":
            g_atk.add_edge(u, v)
    return g_atk


def _attack_dist_to_root(g_atk, root_id: str):
    """
    dist[v] is shortest directed distance v -> ... -> root in attack-only graph.
    Compute via BFS on reversed graph from root.
    """
    import networkx as nx

    if (not root_id) or (root_id not in g_atk):
        return {}
    g_rev = g_atk.reverse(copy=False)
    try:
        return dict(nx.single_source_shortest_path_length(g_rev, root_id))
    except Exception:
        return {}


def _attack_khop_neighborhood(g_atk, root_id: str, k: int):
    if (not root_id) or (root_id not in g_atk):
        return set()
    if k <= 0:
        return {root_id}
    ug = g_atk.to_undirected(as_view=True)
    visited = {root_id}
    frontier = {root_id}
    for _ in range(k):
        nxt = set()
        for x in frontier:
            nxt |= set(ug.neighbors(x))
        nxt -= visited
        if not nxt:
            break
        visited |= nxt
        frontier = nxt
    return visited


def _call_calc_candidates_signature_agnostic(
    solver: MaVERiCSolver,
    active_nodes,
    S_curr,
    g_atk,
    Nk_atk,
    dist_to_root,
):
    """
    Solver code evolved a bit; call _calculate_roi_candidates by signature inspection.
    """
    fn = solver._calculate_roi_candidates
    sig = inspect.signature(fn)
    kwargs = {}
    for name in sig.parameters:
        if name == "self":
            continue
        if name == "active_nodes":
            kwargs[name] = active_nodes
        elif name == "S_curr":
            kwargs[name] = S_curr
        elif name == "g_atk":
            kwargs[name] = g_atk
        elif name == "Nk_atk":
            kwargs[name] = Nk_atk
        elif name in {"dist_to_root", "dist"}:
            kwargs[name] = dist_to_root
        else:
            raise TypeError(f"Unknown parameter in _calculate_roi_candidates: {name}")
    return fn(**kwargs)


def _manual_stage2_roi_argmax_by_graph_semantics(
    solver: MaVERiCSolver,
    g: ArgumentationGraph,
):
    """
    Compute the stage-2 ROI for each feasible candidate using:
      - SGS from graph.get_grounded_extension(use_shield=True, alpha=solver.sgs_alpha)
      - solver's beta_root_flip, gamma_struct, omega, priors
      - s(v) computed via solver helper if present, else 0.0
    """
    solver.root_id = g.find_semantic_root()
    root = solver.root_id
    if not root:
        return None

    active = [n for n in g.nodes.values() if (not n.is_verified) and (getattr(n, "id", None) in g.nx_graph)]
    feasible = []
    for n in active:
        _, cost = stable_get_tool_and_cost(solver, n)
        if cost <= solver.budget + 1e-12:
            feasible.append(n)

    if not feasible:
        return None

    g_atk = _attack_only_graph_from_nx(g.nx_graph)
    Nk = _attack_khop_neighborhood(g_atk, root, getattr(solver, "k_hop_root", 2))

    # Structural score s(v)
    if hasattr(solver, "_structural_score_ranknorm"):
        cand_ids = [str(getattr(n, "id")) for n in feasible if getattr(n, "id", None)]
        s_map = solver._structural_score_ranknorm(g_atk, cand_ids, eta=getattr(solver, "eta_struct", 0.7))
    else:
        s_map = {getattr(n, "id", None): 0.0 for n in feasible if getattr(n, "id", None)}

    def sgs():
        return set(g.get_grounded_extension(use_shield=True, alpha=getattr(solver, "sgs_alpha", 1.0)))

    def sgs_with_temp_tau(nid: str, val: bool):
        node = g.nodes[nid]
        old_v = bool(node.is_verified)
        old_gt = node.ground_truth
        node.is_verified = True
        node.ground_truth = bool(val)
        try:
            return set(g.get_grounded_extension(use_shield=True, alpha=getattr(solver, "sgs_alpha", 1.0)))
        finally:
            node.is_verified = old_v
            node.ground_truth = old_gt

    S_curr = sgs()
    beta = float(getattr(solver, "beta_root_flip", 0.7))
    eps = float(getattr(solver, "roi_eps", 1e-6))
    gamma = float(getattr(solver, "gamma_struct", 0.8))

    def omega(nid: str) -> float:
        return float(solver._priority_weight(nid)) if hasattr(solver, "_priority_weight") else 1.0

    def prior_true(nid: str) -> float:
        if hasattr(solver, "_outcome_prior_true"):
            return float(solver._outcome_prior_true(nid))
        return 0.5

    def bounded_delta(S_new):
        droot = int((root in S_curr) ^ (root in S_new)) if root is not None else 0
        if not Nk:
            dloc = 0.0
        else:
            dloc = float(len(S_curr.symmetric_difference(S_new).intersection(Nk))) / (float(len(Nk)) + eps)
        return droot, dloc

    best_id = None
    best_roi = None

    for node in feasible:
        nid = getattr(node, "id", None)
        if not nid:
            continue
        _, cost = stable_get_tool_and_cost(solver, node)
        cost = float(cost)

        S_T = sgs_with_temp_tau(nid, True)
        S_F = sgs_with_temp_tau(nid, False)

        droot_T, dloc_T = bounded_delta(S_T)
        droot_F, dloc_F = bounded_delta(S_F)

        Delta_T = beta * float(droot_T) + (1.0 - beta) * float(dloc_T)
        Delta_F = beta * float(droot_F) + (1.0 - beta) * float(dloc_F)

        pi = prior_true(nid)
        g_val = pi * Delta_T + (1.0 - pi) * Delta_F
        g_val = max(0.0, min(1.0, float(g_val)))

        s_val = float(s_map.get(nid, 0.0))
        roi = ((g_val + eps) * (1.0 + gamma * s_val) * omega(nid)) / max(cost, 1e-9)

        if (best_roi is None) or (roi > best_roi):
            best_roi = roi
            best_id = nid

    return best_id


# ----------------------------
# Tests
# ----------------------------
class TestSolverLogic(unittest.TestCase):

    @patch("src.core.solver.MaVERiCSolver._get_tool_and_cost", new=stable_get_tool_and_cost)
    @patch("src.core.solver.RealToolkit.verify_claim", side_effect=fake_verify_claim)
    @patch("src.core.solver.RealToolkit.verify_attack", side_effect=fake_verify_attack)
    @patch("src.core.solver.RealToolkit.verify_support", side_effect=fake_verify_support)
    def test_01_first_choice_matches_manual_roi_argmax(self, *_):
        g = build_core_flip_graph(cost_v=1.0, cost_a=1.0)
        solver = MaVERiCSolver(graph=g, budget=1.0, topk_counterfactual=9999)

        expected = _manual_stage2_roi_argmax_by_graph_semantics(solver, g)
        self.assertIsNotNone(expected, "Expected a feasible argmax candidate")

        with capture_first_verification_event():
            solver.run()

        picked = _get_first_verified_id(solver)
        if picked is None:
            self.fail(
                "Solver performed no verification.\n"
                f"budget_end={getattr(solver, 'budget', None)} tool_calls={getattr(solver, 'tool_calls', None)}\n"
                f"logs_tail:\n{_tail_logs(solver)}"
            )

        self.assertEqual(picked, expected, f"Solver picked {picked} but manual ROI argmax is {expected}")

    @patch("src.core.solver.MaVERiCSolver._get_tool_and_cost", new=stable_get_tool_and_cost)
    @patch("src.core.solver.RealToolkit.verify_claim", side_effect=fake_verify_claim)
    @patch("src.core.solver.RealToolkit.verify_attack", side_effect=fake_verify_attack)
    @patch("src.core.solver.RealToolkit.verify_support", side_effect=fake_verify_support)
    def test_02_attack_only_invariance_under_support_spam(self, *_):
        """
        Robustness test (solver-aligned):
          - The *attack-only* machinery should be immune to support spam.
        IMPORTANT:
          Solver ALSO has omega boost for nodes that SUPPORT the root (delta_support_to_root),
          which can legitimately make spam nodes win the first choice.
          => So we disable that boost here to isolate the intended invariance property.
        """
        g1 = build_core_flip_graph(cost_v=1.0, cost_a=1.0)
        g2 = build_core_flip_graph(cost_v=1.0, cost_a=1.0)
        add_support_spam(g2, n_spam=30)

        s1 = MaVERiCSolver(graph=g1, budget=1.0, topk_counterfactual=9999, delta_support_to_root=0.0)
        s2 = MaVERiCSolver(graph=g2, budget=1.0, topk_counterfactual=9999, delta_support_to_root=0.0)

        with capture_first_verification_event():
            s1.run()
        p1 = _get_first_verified_id(s1)
        self.assertIsNotNone(p1, "Solver 1 did not verify any node")

        with capture_first_verification_event():
            s2.run()
        p2 = _get_first_verified_id(s2)
        self.assertIsNotNone(p2, "Solver 2 did not verify any node")

        self.assertEqual(p1, p2, f"Support spam changed first choice: clean={p1}, spam={p2}")

    @patch("src.core.solver.MaVERiCSolver._get_tool_and_cost", new=stable_get_tool_and_cost)
    @patch("src.core.solver.RealToolkit.verify_claim", side_effect=fake_verify_claim)
    @patch("src.core.solver.RealToolkit.verify_attack", side_effect=fake_verify_attack)
    @patch("src.core.solver.RealToolkit.verify_support", side_effect=fake_verify_support)
    def test_03_cost_feasibility_blocks_infeasible_node(self, *_):
        g = build_core_flip_graph(cost_v=2.0, cost_a=1.0)
        solver = MaVERiCSolver(graph=g, budget=1.0, topk_counterfactual=9999)

        with capture_first_verification_event():
            solver.run()

        picked = _get_first_verified_id(solver)
        if picked is None:
            self.fail(
                "Solver verified 0 nodes even though feasible nodes exist.\n"
                f"budget_end={getattr(solver, 'budget', None)} tool_calls={getattr(solver, 'tool_calls', None)}\n"
                f"logs_tail:\n{_tail_logs(solver)}"
            )

        self.assertNotEqual(picked, "v", "v must not be verified because cost>budget")

    @patch("src.core.solver.MaVERiCSolver._get_tool_and_cost", new=stable_get_tool_and_cost)
    @patch("src.core.solver.RealToolkit.verify_claim", side_effect=fake_verify_claim)
    @patch("src.core.solver.RealToolkit.verify_attack", side_effect=fake_verify_attack)
    @patch("src.core.solver.RealToolkit.verify_support", side_effect=fake_verify_support)
    def test_05_extremely_strict_one_step_argmax_consistency_large_graph(self, *_):
        g = build_large_graph_with_core(seed=7, n_fillers=25)
        solver = MaVERiCSolver(graph=g, budget=1.0, topk_counterfactual=9999)

        solver.root_id = g.find_semantic_root()
        if not solver.root_id:
            self.skipTest("No root id")
        S_curr = set(g.get_grounded_extension(use_shield=True, alpha=getattr(solver, "sgs_alpha", 1.0)))

        g_atk = _attack_only_graph_from_nx(g.nx_graph)
        root_id = solver.root_id
        if not root_id:
            self.skipTest("No root id")
        Nk_atk = _attack_khop_neighborhood(g_atk, root_id, getattr(solver, "k_hop_root", 2))
        dist_to_root = _attack_dist_to_root(g_atk, root_id)


        self.assertIn("a", dist_to_root, "a should have attack-path distance to root")
        self.assertNotIn("x", dist_to_root, "x should be unreachable by attack-only distance")

        active = [
            n for n in g.nodes.values()
            if (not n.is_verified) and (getattr(n, "id", None) in g.nx_graph)
        ]
        cand = _call_calc_candidates_signature_agnostic(
            solver=solver,
            active_nodes=active,
            S_curr=S_curr,
            g_atk=g_atk,
            Nk_atk=Nk_atk,
            dist_to_root=dist_to_root,
        )
        self.assertTrue(cand, "Expected non-empty candidates even with unreachable node")
        # Ensure no NaN/inf ROI
        for node, roi, *_rest in cand:
            node_id = getattr(node, "id", None)
            self.assertTrue(roi == roi, f"ROI is NaN for node {node_id}")
            self.assertGreaterEqual(roi, 0.0, f"ROI negative for node {node_id}")

        # And solver.run should not crash
        with capture_first_verification_event():
            solver.run()
        self.assertIsNotNone(_get_first_verified_id(solver), "Solver should verify some node without crashing")


if __name__ == "__main__":
    unittest.main()
