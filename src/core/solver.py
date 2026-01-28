# src/core/solver.py
# Drop-in replacement for your current solver.py
#
# MaVERiC Solver v2.2 (Patched with Root-Markov-Blanket Evidence ROI)
#
# + Patch: track refinement stats
#   - self.pruned_count
#   - self.edges_removed_count
#   - self.edges_removed_false_refine_count
#   - self.edges_removed_prune_count
#
# + Patch: SGS evidence-gated mode supported via graph.get_grounded_extension(..., require_evidence=...)
#   - default require_evidence=True (recommended)
#
# + Patch: reset budget correctly at each run()
#   - keeps self.initial_budget
#
# NOTE:
# - Topology refinement logic kept as in your version.
# - Verified-false nodes are pruned.

import time
from typing import Dict, List, Optional, Tuple, Set, Callable

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
      Uses graph.get_grounded_extension(use_shield=True, alpha=sgs_alpha, require_evidence=sgs_require_evidence).

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
        sgs_require_evidence: bool = True,
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
        lambda_evidence: float = 0.8,
        delta_refute_attackers: float = 0.5,
        eg_direct: float = 1.0,
        eg_khop_attack: float = 0.3,
        forced_root_id: Optional[str] = None,
        root_tie_topk: int = 3,
        root_tie_margin: float = 0.05,
        root_llm_tiebreaker: Optional[Callable[[str, List[str]], Optional[str]]] = None,
    ):
        self.graph = graph

        self.initial_budget = float(budget)
        self.budget = float(budget)

        self.tool_calls = 0
        self.logs: List[str] = []

        self.flagged_adversaries: Set[str] = set()
        self.root_id: Optional[str] = forced_root_id
        self.y_direct: Optional[bool] = None

        # refinement stats
        self.pruned_count: int = 0
        self.edges_removed_count: int = 0  # removed by topology refinement (true)
        self.edges_removed_false_refine_count: int = 0  # removed by topology refinement (false)
        self.edges_removed_prune_count: int = 0  # removed due to node pruning

        self.TOOL_COSTS = dict(tool_costs) if tool_costs else dict(TOOL_COSTS)

        self.topk_counterfactual = int(topk_counterfactual)
        self.sgs_alpha = float(sgs_alpha)
        self.sgs_require_evidence = bool(sgs_require_evidence)

        self.k_hop_root = int(k_hop_root)
        self.beta_root_flip = float(beta_root_flip)
        self.gamma_struct = float(gamma_struct)
        self.rho_proxy = float(rho_proxy)
        self.roi_eps = float(roi_eps)

        self.delta_root = float(delta_root)
        self.delta_adv = float(delta_adv)
        self.delta_support_to_root = float(delta_support_to_root)

        self.eta_struct = float(eta_struct)

        self.prior_true_default = float(prior_true_default)
        self.prior_true_adv = float(prior_true_adv)

        self.lambda_evidence = float(lambda_evidence)
        self.delta_refute_attackers = float(delta_refute_attackers)
        self.eg_direct = float(eg_direct)
        self.eg_khop_attack = float(eg_khop_attack)

        self._tool_cache: Dict[str, str] = {}

        self.root_tie_topk = int(root_tie_topk)
        self.root_tie_margin = float(root_tie_margin)
        self.root_llm_tiebreaker = root_llm_tiebreaker or (
            lambda claim, candidates: MaVERiCSolver._llm_root_tiebreaker(claim, candidates, self.graph)
        )

        self._direct_attackers: Set[str] = set()
        self._direct_supporters: Set[str] = set()
        self._khop_attackers: Set[str] = set()

        self.verified_true_ids: Set[str] = set()
        self.verified_false_ids: Set[str] = set()

        self.verify_error: bool = False

    # --------------------------
    # Convenience: stats for eval harness
    # --------------------------
    @property
    def edges_removed_total(self) -> int:
        return int(self.edges_removed_count + self.edges_removed_false_refine_count + self.edges_removed_prune_count)

    # --------------------------
    # Logging
    # --------------------------
    def _add_log(self, message: str) -> str:
        self.logs.append(message)
        return message

    # --------------------------
    # Root LLM tie-breaker
    # --------------------------
    @staticmethod
    def _llm_root_tiebreaker(claim: str, candidate_ids: List[str], graph) -> Optional[str]:
        if not candidate_ids:
            return None

        lines = []
        for idx, nid in enumerate(candidate_ids, start=1):
            node = graph.nodes.get(nid)
            if not node:
                continue
            lines.append(f"[{idx}] {nid}: {node.content}")

        if not lines:
            return None

        prompt = f"""
Choose the node ID that best matches the core claim.
If none match clearly, output "ABSTAIN".

Claim:
{claim}

Candidate nodes:
""" + "\n".join(lines) + """

Reply with only one of: the node ID (exact) or ABSTAIN.
"""
        try:
            from src.config import client, JUDGE_MODEL

            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            if not raw or "ABSTAIN" in raw.upper():
                return None

            for nid in candidate_ids:
                if nid == raw:
                    return nid
            return None
        except Exception:
            return None

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
        if node_id in self.graph.nx_graph:
            self.pruned_count += 1
            try:
                self.edges_removed_prune_count += int(self.graph.nx_graph.degree(node_id))
            except Exception:
                self.edges_removed_prune_count += 0
        self.graph.remove_node(node_id)
        self.flagged_adversaries.discard(node_id)

    # --------------------------
    # Verification state helpers
    # --------------------------
    def _is_verified_true(self, node_id: str) -> bool:
        n = self.graph.nodes.get(node_id)
        return bool(n and n.is_verified and n.ground_truth is True)

    def _root_verified_true(self) -> bool:
        if not self.root_id:
            return False
        if self.y_direct is True:
            return True
        return self._is_verified_true(self.root_id)

    # --------------------------
    # Tool routing
    # --------------------------
    def _decide_tool_strategy(self, claim: str) -> str:
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
            tool = (res.choices[0].message.content or "").strip().upper()
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

    def _attack_distance_to_root(self, g_atk: nx.DiGraph, root_id: str) -> Dict[str, int]:
        if not root_id or root_id not in g_atk:
            return {}
        g_rev = g_atk.reverse(copy=False)
        try:
            dist = nx.single_source_shortest_path_length(g_rev, root_id)
            return dict(dist)
        except Exception:
            return {}

    # --------------------------
    # Root Markov blanket evidence sets
    # --------------------------
    def _compute_root_evidence_sets(self) -> None:
        self._direct_attackers = set()
        self._direct_supporters = set()
        self._khop_attackers = set()

        r = self.root_id
        if not r or r not in self.graph.nx_graph:
            return

        for u, _, d in self.graph.nx_graph.in_edges(r, data=True):
            t = (d or {}).get("type")
            if t == "attack":
                self._direct_attackers.add(u)
            elif t == "support":
                self._direct_supporters.add(u)

        k = max(1, int(self.k_hop_root))
        g_atk = self._attack_only_graph()
        dist = self._attack_distance_to_root(g_atk, r)
        for v, dv in dist.items():
            if v == r:
                continue
            if 1 <= int(dv) <= k:
                self._khop_attackers.add(v)

        self._khop_attackers -= self._direct_attackers

    def _evidence_gain(self, node_id: str) -> float:
        if not self.root_id:
            return 0.0
        if node_id in self._direct_attackers or node_id in self._direct_supporters:
            return float(self.eg_direct)
        if node_id in self._khop_attackers:
            return float(self.eg_khop_attack)
        return 0.0

    def _is_attacker_refutation_candidate(self, node_id: str) -> bool:
        return bool(node_id in self._direct_attackers or node_id in self._khop_attackers)

    # --------------------------
    # Structural influence s(v)
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
    # Priority weights omega(v)
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
        w = 1.0
        if self.root_id and node_id == self.root_id:
            w += self.delta_root
        if node_id in self.flagged_adversaries:
            w += self.delta_adv
        if self._supports_root(node_id):
            w += self.delta_support_to_root
        return float(w)

    def _priority_weight_post_root(self, node_id: str) -> float:
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
        return set(
            self.graph.get_grounded_extension(
                use_shield=True,
                alpha=self.sgs_alpha,
                require_evidence=self.sgs_require_evidence,
            )
        )

    def _sgs_with_temp_tau(self, node_id: str, value: bool) -> Set[str]:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return self._sgs()

        old_is_verified = bool(node.is_verified)
        old_gt = node.ground_truth

        node.is_verified = True
        node.ground_truth = bool(value)
        try:
            out = set(
                self.graph.get_grounded_extension(
                    use_shield=True,
                    alpha=self.sgs_alpha,
                    require_evidence=self.sgs_require_evidence,
                )
            )
        finally:
            node.is_verified = old_is_verified
            node.ground_truth = old_gt

        return out

    # --------------------------
    # Impact computation g(v)
    # --------------------------
    def _bounded_delta(self, S_curr: Set[str], S_new: Set[str], Nk_atk: Set[str]) -> Tuple[int, float]:
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

    def _expected_impact(
        self,
        node_id: str,
        S_curr: Set[str],
        Nk_atk: Set[str],
    ) -> Tuple[float, Tuple[int, float, int, float]]:
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
        self._compute_root_evidence_sets()
        post_root = self._root_verified_true()

        cand_nodes: List[Tuple[object, str, str, float]] = []
        for n in active_nodes:
            tool, cost = self._get_tool_and_cost(n)
            node_id = getattr(n, "id", None)
            if node_id and cost <= self.budget + 1e-12:
                cand_nodes.append((n, str(node_id), tool, float(cost)))

        if not cand_nodes:
            return []

        cand_ids = [nid for _, nid, _, _ in cand_nodes]
        s_map = self._structural_score_ranknorm(g_atk, cand_ids, eta=self.eta_struct)

        stage1 = []
        for node, nid, tool, cost in cand_nodes:
            s = float(s_map.get(nid, 0.0))

            if post_root:
                eg = self._evidence_gain(nid)
                omega = self._priority_weight_post_root(nid)
                proxy = self.rho_proxy * float(eg) + (1.0 - self.rho_proxy) * float(s)
            else:
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

        candidates = []
        lam = float(max(0.0, min(1.0, self.lambda_evidence)))
        for _, node, nid, tool, cost, s, _omega_stage1 in shortlist:
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
                if target_node.is_verified and target_node.ground_truth is True:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self.edges_removed_count += 1
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
                        self.edges_removed_count += 1

                    # tiny refine charge
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
                    is_support = True

                if not is_support:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self.edges_removed_count += 1
                    self._spend(0.05)
                    self._add_log(f"✂️ Pruned invalid SUPPORT: {node_id} -> {tid}")

    def _refine_topology_after_false(self, node_id: str) -> None:
        if node_id not in self.graph.nx_graph:
            return

        current_node = self.graph.nodes.get(node_id)
        if current_node is None:
            return

        out_edges = list(self.graph.nx_graph.out_edges(node_id, data=True))
        for _, tid, d in out_edges:
            if tid not in self.graph.nodes or tid not in self.graph.nx_graph:
                continue

            edge_type = (d or {}).get("type")
            target_node = self.graph.nodes[tid]

            if edge_type == "support":
                if self.graph.nx_graph.has_edge(node_id, tid):
                    self.graph.nx_graph.remove_edge(node_id, tid)
                    self.edges_removed_false_refine_count += 1
                self._add_log(f"✂️ Removed SUPPORT from false node: {node_id} -> {tid}")
                continue

            if edge_type == "attack":
                is_valid_attack = RealToolkit.verify_attack(current_node.content, target_node.content)
                if is_valid_attack is False:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self.edges_removed_false_refine_count += 1
                    self._spend(0.05)
                    self._add_log(f"✂️ Removed invalid ATTACK from false node: {node_id} -> {tid}")

    # --------------------------
    # Verification
    # --------------------------
    def _verify_node(self, node, tool: str, cost: float) -> Optional[bool]:
        if not self._spend(cost):
            return None
        self.tool_calls += 1

        node.is_verified = True
        try:
            is_true = RealToolkit.verify_claim(tool_type=tool, claim=node.content)
        except Exception as e:
            node.ground_truth = None
            self.verify_error = True
            self._add_log(f"⚠️ verify_claim exception on {node.id}: {e}")
            return None

        if is_true is None:
            node.ground_truth = None
            self.verify_error = True
            return None

        node.ground_truth = bool(is_true)

        if node.ground_truth is True:
            self.verified_true_ids.add(node.id)
        else:
            self.verified_false_ids.add(node.id)

        return bool(is_true)

    # --------------------------
    # Core runner
    # --------------------------
    def run(self) -> Tuple[Set[str], Optional[bool]]:
        # Reset budget per run (IMPORTANT)
        self.budget = float(self.initial_budget)

        # Reset per-node verification state
        for node in self.graph.nodes.values():
            node.is_verified = False
            node.ground_truth = None

        self.tool_calls = 0
        self.logs = []
        self.flagged_adversaries = set()
        self.y_direct = None
        self.verify_error = False
        self._tool_cache = {}

        self.verified_true_ids = set()
        self.verified_false_ids = set()

        self.pruned_count = 0
        self.edges_removed_count = 0
        self.edges_removed_false_refine_count = 0
        self.edges_removed_prune_count = 0

        # Root selection
        if not self.root_id:
            claim = getattr(self.graph, "claim", None)
            self.root_id = self.graph.find_semantic_root(
                claim=claim,
                llm_tiebreaker=self.root_llm_tiebreaker if claim else None,
                tie_topk=self.root_tie_topk,
                tie_margin=self.root_tie_margin,
            )

        # Main loop
        while self.budget > 1e-12:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            S_curr = self._sgs()
            g_atk = self._attack_only_graph()

            Nk_atk: Set[str] = set()
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
            best_node_id = getattr(best_node, "id", None)

            is_true = self._verify_node(best_node, tool, cost)
            if is_true is None:
                break

            if self.root_id and best_node_id and best_node_id == self.root_id:
                self.y_direct = bool(is_true)

            if not is_true:
                if best_node_id:
                    self._refine_topology_after_false(best_node_id)
                    self._prune_node(best_node_id)
            else:
                if best_node_id:
                    self._refine_topology_after_true(best_node_id)

        final_ext = self._sgs()

        evidence_used = (self.tool_calls > 0) or (len(self.verified_true_ids) + len(self.verified_false_ids) > 0)

        if not evidence_used:
            verdict = None
        elif self.y_direct is not None:
            verdict = bool(self.y_direct)
        else:
            verdict = bool(self.root_id in final_ext) if self.root_id else False

        return final_ext, verdict

    # --------------------------
    # Pairwise score (optional debug scalar)
    # --------------------------
    def pairwise_score(self, final_ext: Set[str]) -> float:
        """
        A bounded-ish scalar for debugging comparisons across statements.
        Not used as a single decisive metric.

        + prefers y_direct True strongly
        + prefers root in extension slightly
        + prefers more verified_true than verified_false (tanh-bounded)
        + penalizes tool_calls slightly (tanh-bounded)
        """
        import math

        r = self.root_id
        root_in = bool(r and r in final_ext)

        if self.y_direct is True:
            root_term = 1.8
        elif self.y_direct is False:
            root_term = -2.2
        else:
            root_term = 0.4 if root_in else -0.4

        vt = len(self.verified_true_ids)
        vf = len(self.verified_false_ids)
        net = vt - vf
        cov = vt + vf

        ev_term = 0.9 * math.tanh(net / 3.0) + 0.25 * math.tanh(cov / 6.0)
        cost_term = -0.25 * math.tanh(float(self.tool_calls) / 12.0)

        return float(root_term + ev_term + cost_term)
