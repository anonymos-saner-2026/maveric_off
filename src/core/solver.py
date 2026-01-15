# src/core/solver.py
# Drop-in replacement for your current solver.py
#
# MaVERiC Solver v2.2 (Patched with Root-Markov-Blanket Evidence ROI)
#
# Patches implemented:
# A) Evidence set around root is directional "Markov blanket":
#    - direct attackers/supporters: 1-hop incoming edges to root (attack/support)
#    - optional k-hop attackers (attack-directed only) for refutation value
#    => avoids support-chain spam like c7 -> c2 -> c0 (support noise)
#
# B) Evidence gain shaping:
#    - gain=1.0 for direct attacker/supporter (incoming 1-hop)
#    - gain=0.3 for k-hop attackers (attack-directed), excluding direct
#    - gain=0.0 for multi-hop supporters (treated as support noise by default)
#
# C) Post-root prioritization:
#    - boost attacker refutation nodes (direct + k-hop attackers) with bounded additive weight
#
# ROI (two-stage):
#   Stage1 proxy:
#     - pre-root: proxy = rho*p + (1-rho)*s
#     - post-root: proxy = rho*evidence_gain + (1-rho)*s     (no proximity p)
#
#   Stage2:
#     - pre-root: g = expected counterfactual impact (root flip + local delta)
#     - post-root: g_post = lambda_evidence * evidence_gain + (1-lambda)*g
#                + bounded attacker-refutation boost via omega(v)
#
# NOTE:
# - Topology refinement logic kept as you had (verify_attack/support).
# - Verified-false nodes are pruned.

import time
from typing import Dict, List, Optional, Tuple, Set

import networkx as nx

from src.tools.real_toolkit import RealToolkit

# Default tool costs (override if you already store per-node costs)
TOOL_COSTS = {
    "WEB_SEARCH": 5.0,
    "PYTHON_EXEC": 2.0,
    "WIKIPEDIA": 1.0,
    "COMMON_SENSE": 0.5,
}


class MaVERiCSolver:
    """
    MaVERiC Solver v2.2

    SGS:
      Uses graph.get_grounded_extension(use_shield=True, alpha=sgs_alpha).

    ROI (two-stage):
      Stage 1 (cheap proxy within budget):
        - pre-root:
            ROI_tilde(v) = ((rho * p(v) + (1-rho) * s(v)) * omega(v)) / C(v)
        - post-root:
            ROI_tilde(v) = ((rho * eg(v) + (1-rho) * s(v)) * omega_post(v)) / C(v)

      Stage 2 (shortlist only):
        - pre-root:
            ROI(v) = ((g(v)+eps) * (1 + gamma*s(v)) * omega(v)) / C(v)
        - post-root:
            ROI(v) = ((g_post(v)+eps) * (1 + gamma*s(v)) * omega_post(v)) / C(v)

    Where:
      p(v): attack-only proximity to root (directed v->...->root distance)
      s(v): rank-normalized structural influence (attack-only PageRank+deg)
      omega(v): bounded priority boosts (root/adv/support-to-root)
      omega_post(v): omega(v) + bounded attacker-refutation boost if v is attacker-of-root
      g(v): expected impact via verification-aligned counterfactual SGS
      eg(v): evidence_gain around root (Markov blanket + k-hop attackers)
      g_post(v): lambda_evidence*eg(v) + (1-lambda_evidence)*g(v)

    """

    def __init__(
        self,
        graph,
        budget: float,
        tool_costs: Optional[Dict[str, float]] = None,
        topk_counterfactual: int = 25,
        # SGS hyperparam
        sgs_alpha: float = 1.0,
        # ROI hyperparams
        k_hop_root: int = 2,
        beta_root_flip: float = 0.7,
        gamma_struct: float = 0.8,
        rho_proxy: float = 0.6,
        roi_eps: float = 1e-6,
        # bounded priority boosts (additive)
        delta_root: float = 1.5,
        delta_adv: float = 0.5,
        delta_support_to_root: float = 0.5,
        # structural mix
        eta_struct: float = 0.7,
        # outcome prior
        prior_true_default: float = 0.5,
        prior_true_adv: float = 0.3,
        # --- Post-root evidence settings ---
        lambda_evidence: float = 0.8,          # how much post-root objective is evidence_gain
        delta_refute_attackers: float = 0.5,   # bounded additive boost for attacker refutation
        eg_direct: float = 1.0,                # evidence gain for direct in-neighbors of root
        eg_khop_attack: float = 0.3,           # evidence gain for k-hop attackers (attack-directed)
        # IMPORTANT: multi-hop supporters are treated as noise by default (eg=0.0)
    ):
        self.graph = graph
        self.budget = float(budget)

        self.tool_calls = 0
        self.logs: List[str] = []

        self.flagged_adversaries: Set[str] = set()
        self.root_id: Optional[str] = None
        self.y_direct: Optional[bool] = None

        self.TOOL_COSTS = dict(tool_costs) if tool_costs else dict(TOOL_COSTS)

        # perf
        self.topk_counterfactual = int(topk_counterfactual)

        # SGS
        self.sgs_alpha = float(sgs_alpha)

        # ROI knobs
        self.k_hop_root = int(k_hop_root)
        self.beta_root_flip = float(beta_root_flip)
        self.gamma_struct = float(gamma_struct)
        self.rho_proxy = float(rho_proxy)
        self.roi_eps = float(roi_eps)

        # bounded boosts
        self.delta_root = float(delta_root)
        self.delta_adv = float(delta_adv)
        self.delta_support_to_root = float(delta_support_to_root)

        # structural
        self.eta_struct = float(eta_struct)

        # priors
        self.prior_true_default = float(prior_true_default)
        self.prior_true_adv = float(prior_true_adv)

        # post-root evidence params
        self.lambda_evidence = float(lambda_evidence)
        self.delta_refute_attackers = float(delta_refute_attackers)
        self.eg_direct = float(eg_direct)
        self.eg_khop_attack = float(eg_khop_attack)

        # caches
        self._tool_cache: Dict[str, str] = {}  # node_id -> tool

        # runtime caches for post-root evidence sets (recomputed each loop)
        self._direct_attackers: Set[str] = set()
        self._direct_supporters: Set[str] = set()
        self._khop_attackers: Set[str] = set()

    # --------------------------
    # Logging
    # --------------------------
    def _add_log(self, message: str) -> str:
        self.logs.append(message)
        return message

    # --------------------------
    # Budgeting
    # --------------------------
    def _spend(self, amount: float) -> bool:
        amount = float(amount)
        if amount <= 0:
            return True
        if self.budget + 1e-12 < amount:
            return False
        self.budget -= amount
        if self.budget < 0:
            self.budget = 0.0
        return True

    # --------------------------
    # Node pruning
    # --------------------------
    def _prune_node(self, node_id: str) -> None:
        self.graph.remove_node(node_id)
        self.flagged_adversaries.discard(node_id)

    # --------------------------
    # Verification state helpers
    # --------------------------
    def _is_verified_true(self, node_id: str) -> bool:
        n = self.graph.nodes.get(node_id)
        return bool(n and n.is_verified and n.ground_truth is True)

    def _is_verified_false(self, node_id: str) -> bool:
        n = self.graph.nodes.get(node_id)
        return bool(n and n.is_verified and n.ground_truth is False)

    def _root_verified_true(self) -> bool:
        if not self.root_id:
            return False
        # If root verified directly, use y_direct.
        if self.y_direct is True:
            return True
        # Otherwise, check node state.
        return self._is_verified_true(self.root_id)

    # --------------------------
    # Tool routing
    # --------------------------
    def _decide_tool_strategy(self, claim: str) -> str:
        """
        Semantic router. Cached per node to avoid repeated calls.
        """
        import re

        s = (claim or "").lower()

        if re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and ("=" in s or "equal" in s):
            return "PYTHON_EXEC"
        if re.search(r"square\s*root|\bsqrt\b", s):
            return "PYTHON_EXEC"
        if re.search(r"\bleap\s+year\b", s):
            return "PYTHON_EXEC"

        prompt = f"""
Role: Tool Router.
Task: Select the tool to verify: "{claim}"

Selection Logic:
1. PYTHON_EXEC:
   - ONLY for explicit MATH calculations or DATE/TIME logic.
2. WEB_SEARCH:
   - Use for most factual claims.
3. COMMON_SENSE:
   - Only for obvious truths.

Output: PYTHON_EXEC, WEB_SEARCH, or COMMON_SENSE.
"""
        try:
            from src.config import client, JUDGE_MODEL

            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            tool = res.choices[0].message.content.strip().upper()
            if "PYTHON" in tool:
                return "PYTHON_EXEC"
            if "SEARCH" in tool:
                return "WEB_SEARCH"
            return "COMMON_SENSE"
        except Exception:
            return "WEB_SEARCH"

    def _get_tool_and_cost(self, node) -> Tuple[str, float]:
        import re

        s = (node.content or "").lower()
        looks_math = bool(re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and (("equal" in s) or ("=" in s)))
        looks_sqrt = bool(re.search(r"square\s*root|\bsqrt\b", s))
        looks_leap = "leap year" in s

        tool = getattr(node, "tool_type", None)
        if tool:
            tool = str(tool).upper()

        if (tool is None) or (tool in {"AUTO", "UNKNOWN", ""}) or (tool == "COMMON_SENSE"):
            if looks_math or looks_sqrt or looks_leap:
                tool = "PYTHON_EXEC"
            else:
                if node.id in self._tool_cache:
                    tool = self._tool_cache[node.id]
                else:
                    tool = self._decide_tool_strategy(node.content)
                    self._tool_cache[node.id] = tool

        node_cost = getattr(node, "verification_cost", None)
        if node_cost is not None and float(node_cost) > 0:
            cost = float(node_cost)
        else:
            cost = float(self.TOOL_COSTS.get(tool, 5.0))

        return tool, cost

    # --------------------------
    # Graph views for ROI
    # --------------------------
    def _attack_only_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nx_graph.nodes())
        for u, v, d in self.graph.nx_graph.edges(data=True):
            if (d or {}).get("type") == "attack":
                g.add_edge(u, v)
        return g

    def _support_only_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nx_graph.nodes())
        for u, v, d in self.graph.nx_graph.edges(data=True):
            if (d or {}).get("type") == "support":
                g.add_edge(u, v)
        return g

    def _combined_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nx_graph.nodes())
        for u, v, d in self.graph.nx_graph.edges(data=True):
            t = (d or {}).get("type")
            if t in {"attack", "support"}:
                g.add_edge(u, v)
        return g

    def _attack_distance_to_root(self, g_atk: nx.DiGraph, root_id: str) -> Dict[str, int]:
        """
        dist_to_root[v] is shortest directed distance v -> ... -> root in ATTACK-only graph.
        Implemented by reversing edges and doing single-source BFS from root.
        """
        if not root_id or root_id not in g_atk:
            return {}
        g_rev = g_atk.reverse(copy=False)
        try:
            dist = nx.single_source_shortest_path_length(g_rev, root_id)
            return dict(dist)
        except Exception:
            return {}

    def _combined_distance_to_root(self, g_all: nx.DiGraph, root_id: str) -> Dict[str, int]:
        """
        dist_to_root[v] shortest directed distance v -> ... -> root on combined (attack+support).
        Reverse BFS from root.
        """
        if not root_id or root_id not in g_all:
            return {}
        g_rev = g_all.reverse(copy=False)
        try:
            dist = nx.single_source_shortest_path_length(g_rev, root_id)
            return dict(dist)
        except Exception:
            return {}

    # --------------------------
    # Root Markov blanket evidence sets (PATCH A)
    # --------------------------
    def _compute_root_evidence_sets(self) -> None:
        """
        Compute directional evidence sets around root:
          - direct attackers/supporters: 1-hop incoming edges to root
          - k-hop attackers: nodes that can reach root via attack-directed path of length <=k
            (excluding direct attackers to keep gain tiers clean)
        Multi-hop supporters are intentionally NOT included (support-chain treated as noise).
        """
        self._direct_attackers = set()
        self._direct_supporters = set()
        self._khop_attackers = set()

        r = self.root_id
        if not r or r not in self.graph.nx_graph:
            return

        # direct 1-hop incoming
        for u, _, d in self.graph.nx_graph.in_edges(r, data=True):
            t = (d or {}).get("type")
            if t == "attack":
                self._direct_attackers.add(u)
            elif t == "support":
                self._direct_supporters.add(u)

        # k-hop attackers (attack-directed)
        k = max(1, int(self.k_hop_root))
        g_atk = self._attack_only_graph()
        dist = self._attack_distance_to_root(g_atk, r)  # v -> ... -> r
        for v, dv in dist.items():
            if v == r:
                continue
            if 1 <= int(dv) <= k:
                self._khop_attackers.add(v)

        # exclude direct attackers from khop set (tier separation)
        self._khop_attackers -= self._direct_attackers

    def _evidence_gain(self, node_id: str) -> float:
        """
        PATCH B:
          - 1.0 if direct attacker/supporter of root
          - 0.3 if k-hop attacker (attack-directed), excluding direct
          - 0.0 otherwise (including multi-hop supporters => support-noise)
        """
        if not self.root_id:
            return 0.0
        if node_id in self._direct_attackers or node_id in self._direct_supporters:
            return float(self.eg_direct)
        if node_id in self._khop_attackers:
            return float(self.eg_khop_attack)
        return 0.0

    def _is_attacker_refutation_candidate(self, node_id: str) -> bool:
        """
        PATCH C target set: attackers of root (direct + k-hop attack).
        """
        return bool(node_id in self._direct_attackers or node_id in self._khop_attackers)

    # --------------------------
    # Structural influence s(v) using rank normalization
    # --------------------------
    @staticmethod
    def _percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        items = list(values.items())
        sorted_vals = sorted([v for _, v in items])
        n = len(sorted_vals)

        def rank(v: float) -> float:
            import bisect
            idx = bisect.bisect_right(sorted_vals, v)
            return float(idx) / float(n) if n > 0 else 0.0

        return {k: rank(v) for k, v in items}

    def _structural_score_ranknorm(
        self,
        g_atk: nx.DiGraph,
        candidates: List[str],
        eta: float,
    ) -> Dict[str, float]:
        if not candidates:
            return {}

        try:
            pr = nx.pagerank(g_atk, alpha=0.85)
        except Exception:
            pr = {nid: 0.0 for nid in g_atk.nodes()}

        pr_c = {v: float(pr.get(v, 0.0)) for v in candidates}
        deg_c = {v: float(g_atk.degree(v)) for v in candidates}

        rpr = self._percentile_ranks(pr_c)
        rdg = self._percentile_ranks(deg_c)

        out = {}
        for v in candidates:
            out[v] = float(eta) * float(rpr.get(v, 0.0)) + (1.0 - float(eta)) * float(rdg.get(v, 0.0))
        return out

    # --------------------------
    # Priority weights omega(v): bounded additive boosts
    # --------------------------
    def _supports_root(self, node_id: str) -> bool:
        r = self.root_id
        if not r:
            return False
        if not self.graph.nx_graph.has_edge(node_id, r):
            return False
        d = self.graph.nx_graph.get_edge_data(node_id, r) or {}
        return d.get("type") == "support"

    def _priority_weight(self, node_id: str) -> float:
        """
        omega(v) = 1
                  + I[v=root]*delta_root
                  + I[v in suspected adversaries]*delta_adv
                  + I[v supports root directly]*delta_support_to_root
        """
        w = 1.0
        if self.root_id and node_id == self.root_id:
            w += self.delta_root
        if node_id in self.flagged_adversaries:
            w += self.delta_adv
        if self._supports_root(node_id):
            w += self.delta_support_to_root
        return float(w)

    def _priority_weight_post_root(self, node_id: str) -> float:
        """
        PATCH C: post-root bounded additive boost for attacker refutation candidates.
        """
        w = self._priority_weight(node_id)
        if self._is_attacker_refutation_candidate(node_id):
            w += self.delta_refute_attackers
        return float(w)

    # --------------------------
    # Outcome prior pi_v
    # --------------------------
    def _outcome_prior_true(self, node_id: str) -> float:
        p = self.prior_true_default
        if node_id in self.flagged_adversaries:
            p = self.prior_true_adv
        return float(max(0.0, min(1.0, p)))

    # --------------------------
    # SGS calls
    # --------------------------
    def _sgs(self) -> Set[str]:
        return set(self.graph.get_grounded_extension(use_shield=True, alpha=self.sgs_alpha))

    def _sgs_with_temp_tau(self, node_id: str, value: bool) -> Set[str]:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return self._sgs()

        old_is_verified = bool(node.is_verified)
        old_gt = node.ground_truth

        node.is_verified = True
        node.ground_truth = bool(value)
        try:
            out = set(self.graph.get_grounded_extension(use_shield=True, alpha=self.sgs_alpha))
        finally:
            node.is_verified = old_is_verified
            node.ground_truth = old_gt

        return out

    # --------------------------
    # Impact computation g(v)
    # --------------------------
    def _bounded_delta(
        self,
        S_curr: Set[str],
        S_new: Set[str],
        Nk_atk: Set[str],
    ) -> Tuple[int, float]:
        r = self.root_id
        delta_root = 0
        if r is not None:
            delta_root = int((r in S_curr) ^ (r in S_new))

        if not Nk_atk:
            delta_local = 0.0
        else:
            symdiff = S_curr.symmetric_difference(S_new)
            local = symdiff.intersection(Nk_atk)
            delta_local = float(len(local) / (float(len(Nk_atk)) + self.roi_eps))

        return delta_root, delta_local

    def _expected_impact(self, node_id: str, S_curr: Set[str], Nk_atk: Set[str]) -> Tuple[float, Tuple[int, float, int, float]]:
        """
        g(v) = pi * Delta_T + (1-pi) * Delta_F
        Delta_* = beta * Delta_root_* + (1-beta) * Delta_local_*
        """
        pi = self._outcome_prior_true(node_id)

        S_T = self._sgs_with_temp_tau(node_id, True)
        S_F = self._sgs_with_temp_tau(node_id, False)

        droot_T, dloc_T = self._bounded_delta(S_curr, S_T, Nk_atk)
        droot_F, dloc_F = self._bounded_delta(S_curr, S_F, Nk_atk)

        beta = self.beta_root_flip
        Delta_T = beta * float(droot_T) + (1.0 - beta) * float(dloc_T)
        Delta_F = beta * float(droot_F) + (1.0 - beta) * float(dloc_F)

        g = pi * Delta_T + (1.0 - pi) * Delta_F
        g = max(0.0, min(1.0, float(g)))

        return g, (int(droot_T), float(dloc_T), int(droot_F), float(dloc_F))

    # --------------------------
    # ROI computation (two-stage)
    # --------------------------
    def _calculate_roi_candidates(
        self,
        active_nodes: List,
        S_curr: Set[str],
        g_atk: nx.DiGraph,
        Nk_atk: Set[str],
        dist_to_root: Dict[str, int],
    ) -> List[Tuple[object, float, Tuple[int, float, int, float], str, float]]:
        """
        Returns list:
          (node, roi, (droot_T, dloc_T, droot_F, dloc_F), tool, cost)
        """
        # recompute post-root evidence sets each call (cheap for small graphs)
        self._compute_root_evidence_sets()
        post_root = self._root_verified_true()

        # Candidate ids within budget
        cand_nodes: List[Tuple[object, str, str, float]] = []
        for n in active_nodes:
            tool, cost = self._get_tool_and_cost(n)
            if cost <= self.budget + 1e-12:
                cand_nodes.append((n, n.id, tool, float(cost)))

        if not cand_nodes:
            return []

        cand_ids = [nid for _, nid, _, _ in cand_nodes]
        s_map = self._structural_score_ranknorm(g_atk, cand_ids, eta=self.eta_struct)

        # Stage 1: cheap proxy
        stage1 = []
        for node, nid, tool, cost in cand_nodes:
            s = float(s_map.get(nid, 0.0))

            if post_root:
                eg = self._evidence_gain(nid)
                omega = self._priority_weight_post_root(nid)
                proxy = self.rho_proxy * float(eg) + (1.0 - self.rho_proxy) * float(s)
            else:
                # pre-root: use proximity p on attack-only dist
                if nid in dist_to_root:
                    p = 1.0 / (float(dist_to_root[nid]) + 1.0)
                else:
                    p = 0.0
                omega = self._priority_weight(nid)
                proxy = self.rho_proxy * float(p) + (1.0 - self.rho_proxy) * float(s)

            roi_tilde = (float(proxy) * float(omega)) / max(cost, 1e-9)
            stage1.append((roi_tilde, node, nid, tool, cost, s, omega))

        stage1.sort(key=lambda x: x[0], reverse=True)
        K = max(1, int(self.topk_counterfactual))
        shortlist = stage1[:K]

        # Stage 2: verification-aligned expected impact (+ post-root evidence objective)
        candidates = []
        lam = float(max(0.0, min(1.0, self.lambda_evidence)))
        for _, node, nid, tool, cost, s, omega in shortlist:
            g_val, dbg = self._expected_impact(nid, S_curr=S_curr, Nk_atk=Nk_atk)

            if post_root:
                eg = float(self._evidence_gain(nid))
                g_eff = lam * eg + (1.0 - lam) * float(g_val)
                omega_eff = self._priority_weight_post_root(nid)
            else:
                g_eff = float(g_val)
                omega_eff = self._priority_weight(nid)

            roi = ((g_eff + self.roi_eps) * (1.0 + self.gamma_struct * float(s)) * float(omega_eff)) / max(cost, 1e-9)
            candidates.append((node, float(roi), dbg, tool, float(cost)))

        return candidates

    # --------------------------
    # Topology refinement
    # --------------------------
    def _flag_attackers_of_truth(self, node_id: str) -> None:
        for u, _, d in self.graph.nx_graph.in_edges(node_id, data=True):
            if (d or {}).get("type") == "attack":
                if u in self.graph.nodes:
                    self.flagged_adversaries.add(u)

    def _refine_topology_after_true(self, node_id: str) -> None:
        """
        After verifying node_id is TRUE:
        - Flag ATTACK predecessors as suspected adversaries.
        - For outgoing edges:
            * invalid ATTACK may be removed and optionally converted to SUPPORT if verify_support succeeds.
            * invalid SUPPORT may be pruned conservatively.
        - Remove ATTACK between two verified-TRUE nodes.
        """
        if node_id not in self.graph.nx_graph:
            return

        current_node = self.graph.nodes.get(node_id)
        if current_node is None:
            return

        self._flag_attackers_of_truth(node_id)

        out_edges = list(self.graph.nx_graph.out_edges(node_id, data=True))
        for _, tid, d in out_edges:
            if tid not in self.graph.nodes or tid not in self.graph.nx_graph:
                continue

            edge_type = (d or {}).get("type")
            target_node = self.graph.nodes[tid]

            if edge_type == "attack":
                # remove truth-on-truth attack
                if target_node.is_verified and target_node.ground_truth is True:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self._add_log(f"✂️ Removed truth-on-truth ATTACK: {node_id} -> {tid}")
                    continue

                is_valid_attack = RealToolkit.verify_attack(current_node.content, target_node.content)
                if not is_valid_attack:
                    try:
                        is_support = RealToolkit.verify_support(current_node.content, target_node.content)
                    except Exception:
                        is_support = False

                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)

                    # tiny bookkeeping cost (optional)
                    self._spend(0.05)

                    if is_support:
                        self.graph.nx_graph.add_edge(node_id, tid, type="support")
                        self._add_log(f"🔄 Converted invalid ATTACK to SUPPORT: {node_id} -> {tid}")
                    else:
                        self._add_log(f"✂️ Pruned invalid ATTACK: {node_id} -> {tid}")

            elif edge_type == "support":
                try:
                    is_support = RealToolkit.verify_support(current_node.content, target_node.content)
                except Exception:
                    # be conservative on failures
                    is_support = True

                if not is_support:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                    self._spend(0.05)
                    self._add_log(f"✂️ Pruned invalid SUPPORT: {node_id} -> {tid}")

    # --------------------------
    # Verification
    # --------------------------
    def _verify_node(self, node, tool: str, cost: float) -> Optional[bool]:
        if not self._spend(cost):
            return None
        self.tool_calls += 1
        node.is_verified = True
        is_true = RealToolkit.verify_claim(tool_type=tool, claim=node.content)
        node.ground_truth = bool(is_true)
        return bool(is_true)

    # --------------------------
    # Core runners
    # --------------------------
    def run(self) -> Tuple[set, bool]:
        """
        Returns (final_SGS_extension, verdict).
        Verdict:
          - If root verified directly, use y_direct.
          - Else use membership of root in final SGS extension.
        """
        self.root_id = self.graph.find_semantic_root()

        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            # current SGS extension
            S_curr = self._sgs()

            # attack-only objects for ROI
            g_atk = self._attack_only_graph()

            # locality Nk_atk can stay attack-only (paper-aligned)
            # Here we compute Nk_atk as k-hop undirected neighborhood on ATTACK-only graph
            # (keeps support-spam from polluting locality delta)
            Nk_atk = set()
            if self.root_id and self.root_id in g_atk:
                ug = g_atk.to_undirected(as_view=True)
                visited = {self.root_id}
                frontier = {self.root_id}
                for _ in range(max(0, int(self.k_hop_root))):
                    nxt = set()
                    for x in frontier:
                        nxt |= set(ug.neighbors(x))
                    nxt -= visited
                    if not nxt:
                        break
                    visited |= nxt
                    frontier = nxt
                Nk_atk = visited

            dist_to_root = self._attack_distance_to_root(g_atk, self.root_id) if self.root_id else {}

            candidates = self._calculate_roi_candidates(
                active_nodes=active,
                S_curr=S_curr,
                g_atk=g_atk,
                Nk_atk=Nk_atk,
                dist_to_root=dist_to_root,
            )
            if not candidates:
                break

            best_node, best_roi, dbg, tool, cost = max(candidates, key=lambda x: x[1])

            is_true = self._verify_node(best_node, tool, cost)
            if is_true is None:
                break

            # direct override if root verified
            if self.root_id and best_node.id == self.root_id:
                self.y_direct = bool(is_true)

            if not is_true:
                self._prune_node(best_node.id)
            else:
                self._refine_topology_after_true(best_node.id)

        final_ext = self._sgs()

        if self.y_direct is not None:
            verdict = bool(self.y_direct)
        else:
            verdict = bool(self.root_id in final_ext) if self.root_id else False

        return final_ext, verdict

    def run_live(self):
        """
        Live run: generator yielding logs + update dicts for UI.
        """
        self._add_log(f"🚀 MaVERiC Solver started. Budget: ${self.budget:.2f}")

        # reset node states
        for node in self.graph.nodes.values():
            node.is_verified = False
            node.ground_truth = None

        self._add_log("--- ATOMIC CLAIMS EXTRACTED ---")
        for nid, n in self.graph.nodes.items():
            self._add_log(f"🔹 [{nid}] ({getattr(n, 'speaker', 'UNK')}): {n.content}")

        self.root_id = self.graph.find_semantic_root()
        yield self._add_log(f"📍 Auto-detected Semantic Root: {self.root_id}")
        yield "start"

        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                yield self._add_log("ℹ️ Strategic verification complete (no active nodes).")
                break

            S_curr = self._sgs()
            g_atk = self._attack_only_graph()

            Nk_atk = set()
            if self.root_id and self.root_id in g_atk:
                ug = g_atk.to_undirected(as_view=True)
                visited = {self.root_id}
                frontier = {self.root_id}
                for _ in range(max(0, int(self.k_hop_root))):
                    nxt = set()
                    for x in frontier:
                        nxt |= set(ug.neighbors(x))
                    nxt -= visited
                    if not nxt:
                        break
                    visited |= nxt
                    frontier = nxt
                Nk_atk = visited

            dist_to_root = self._attack_distance_to_root(g_atk, self.root_id) if self.root_id else {}

            candidates = self._calculate_roi_candidates(
                active_nodes=active,
                S_curr=S_curr,
                g_atk=g_atk,
                Nk_atk=Nk_atk,
                dist_to_root=dist_to_root,
            )
            if not candidates:
                yield self._add_log("ℹ️ No candidates within budget.")
                break

            best_node, best_roi, dbg, tool, cost = max(candidates, key=lambda x: x[1])

            if self.budget < cost:
                yield self._add_log("ℹ️ Budget insufficient for next verification.")
                break

            droot_T, dloc_T, droot_F, dloc_F = dbg
            yield self._add_log(
                f"🔍 Verifying {best_node.id} (ROI: {best_roi:.3f}, "
                f"T:[Δroot={droot_T},Δloc={dloc_T:.3f}] "
                f"F:[Δroot={droot_F},Δloc={dloc_F:.3f}], "
                f"tool={tool}, cost={cost:.2f})..."
            )

            is_true = self._verify_node(best_node, tool, cost)
            if is_true is None:
                yield self._add_log("ℹ️ Budget insufficient for verification.")
                break

            if self.root_id and best_node.id == self.root_id:
                self.y_direct = bool(is_true)

            if not is_true:
                impacted = list(self.graph.nx_graph.successors(best_node.id)) if best_node.id in self.graph.nx_graph else []
                self._prune_node(best_node.id)
                yield self._add_log(f"💥 FALSE. Pruned {best_node.id} and {len(impacted)} dependent claims.")
            else:
                self._refine_topology_after_true(best_node.id)
                yield self._add_log(f"🛡️ TRUE. Refinement updated via {best_node.id}.")

            yield {
                "type": "update",
                "nx_graph": self.graph.nx_graph.copy(),
                "budget": self.budget,
                "highlight_node": best_node.id,
                "root_id": self.root_id,
                "y_direct": self.y_direct,
                "tool_calls": self.tool_calls,
                "flagged_adversaries": list(self.flagged_adversaries),
            }

            time.sleep(0.2)

        yield self._add_log("🏁 Strategic verification complete.")
