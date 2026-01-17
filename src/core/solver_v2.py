# solver_v2.py
# Drop-in replacement for your current solver.py
#
# MaVERiC Solver v2 (Merged, polished)
#
# Fixes and features:
# - Correct root usage (graph.find_semantic_root()).
# - ROI v2 as the main selection rule:
#     Stage 1: cheap proxy shortlist (proximity-to-root + normalized structural score) / cost.
#     Stage 2: bounded, root-aware counterfactual impact g(v) + structural tie-break / cost.
# - Enforces cost <= remaining budget.
# - Flags only ATTACK predecessors as suspected adversaries.
# - Topology refinement respects edge types and conservatively rewires:
#     invalid ATTACK may be removed and optionally converted to SUPPORT if verify_support succeeds.
#     invalid SUPPORT may be pruned conservatively.
# - Removes truth-on-truth ATTACK conflicts between verified-TRUE nodes.
# - y_direct override if root is verified directly.
# - Tool routing cache per node id to reduce repeated router calls.
# - Optional top-K counterfactual to reduce deepcopy cost.

import copy
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
    MaVERiC Solver v2

    Core idea:
      - Build/receive a noisy typed claim graph (attack/support).
      - Under a budget, verify a small set of high-value nodes.
      - Use verified outcomes to conservatively refine topology and flag suspected adversaries.
      - Output verdict based on grounded extension membership of the semantic root
        (or direct verification override if root was verified).

    ROI v2 (main):
      Stage 1 (cheap proxy):
        ROI_tilde(v) = ((alpha * p(v) + (1-alpha) * s(v)) * omega(v)) / C(v)
      Stage 2 (shortlist only):
        ROI(v) = ((g(v) + eps) * (1 + gamma * s(v)) * omega(v)) / C(v)

      where:
        p(v) in [0,1] is proximity to root via shortest path distance
        s(v) in [0,1] is normalized structural influence (PageRank + degree)
        omega(v) is priority weight (root/adversary/support-to-root boosts)
        g(v) in [0,1] is bounded root-aware counterfactual impact:
          g(v) = beta * Delta_root(v) + (1-beta) * Delta_local(v)
          Delta_root is XOR flip of root membership in grounded extension
          Delta_local is localized symmetric difference near Nk(root)
    """

    def __init__(
        self,
        graph,
        budget: float,
        tool_costs: Optional[Dict[str, float]] = None,
        topk_counterfactual: int = 25,
        adversary_boost: float = 2.0,
        root_boost: float = 20.0,
        support_to_root_boost: float = 2.5,
        degree_boost_alpha: float = 0.1,
        # ROI v2 hyperparams
        k_hop_root: int = 2,
        beta_root_flip: float = 0.7,
        gamma_struct: float = 0.8,
        alpha_proxy: float = 0.6,
        roi_eps: float = 1e-6,
    ):
        self.graph = graph
        self.budget = float(budget)

        self.tool_calls = 0
        self.logs: List[str] = []

        self.flagged_adversaries = set()
        self.root_id: Optional[str] = None
        self.y_direct: Optional[bool] = None

        self.TOOL_COSTS = dict(tool_costs) if tool_costs else dict(TOOL_COSTS)

        # perf / strategy knobs
        self.topk_counterfactual = int(topk_counterfactual)
        self.adversary_boost = float(adversary_boost)
        self.root_boost = float(root_boost)
        self.support_to_root_boost = float(support_to_root_boost)
        self.degree_boost_alpha = float(degree_boost_alpha)

        # ROI v2 knobs
        self.k_hop_root = int(k_hop_root)
        self.beta_root_flip = float(beta_root_flip)
        self.gamma_struct = float(gamma_struct)
        self.alpha_proxy = float(alpha_proxy)
        self.roi_eps = float(roi_eps)

        # caches
        self._tool_cache: Dict[str, str] = {}  # node_id -> tool

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
        """
        Spend budget safely.
        Return True if spent, False if insufficient.
        Never allows budget to go negative.
        """
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
        # remove from graph (both nx_graph and node registry via your graph wrapper)
        self.graph.remove_node(node_id)

        # keep adversary set consistent with "active graph"
        if node_id in self.flagged_adversaries:
            self.flagged_adversaries.discard(node_id)

        # if root got pruned, semantics-based verdict will collapse naturally
        # y_direct remains as-is (only set if root verified directly)

    # --------------------------
    # Tool routing
    # --------------------------
    def _decide_tool_strategy(self, claim: str) -> str:
        """
        Semantic router.
        Keep as-is (LLM router) but cache results per node to avoid repeated calls.
        """
        import re

        s = (claim or "").lower()

        # arithmetic pattern (robust)
        if re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and ("=" in s or "equal" in s):
            return "PYTHON_EXEC"

        # sqrt
        if re.search(r"square\s*root|\bsqrt\b", s):
            return "PYTHON_EXEC"

        # leap year
        if re.search(r"\bleap\s+year\b", s):
            return "PYTHON_EXEC"

        prompt = f"""
Role: Tool Router.
Task: Select the tool to verify: "{claim}"

Selection Logic:
1. PYTHON_EXEC:
   - ONLY for explicit MATH calculations (e.g., "sqrt of 144", "10% of 50").
   - ONLY for DATE/TIME logic (e.g., "Was 2020 a leap year?").
   - DO NOT use for historical facts, heights, distances, or statistics.

2. WEB_SEARCH:
   - Use for EVERYTHING ELSE: History, Science facts, Biology, Geography, Current Events.

3. COMMON_SENSE:
   - Only for obvious truths ("Fire is hot").

Output: PYTHON_EXEC, WEB_SEARCH, or COMMON_SENSE.
"""
        try:
            from src.config import client, JUDGE_MODEL  # local import to avoid circular issues

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
        """
        Returns (tool, cost). Respects node.tool_type / node.verification_cost if provided,
        otherwise uses router + default TOOL_COSTS.
        """
        import re

        s = (node.content or "").lower()
        looks_math = bool(
            re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", s) and (("equal" in s) or ("=" in s))
        )
        looks_sqrt = bool(re.search(r"square\s*root|\bsqrt\b", s))
        looks_leap = "leap year" in s

        tool = getattr(node, "tool_type", None)
        if tool:
            tool = str(tool).upper()

        # Treat COMMON_SENSE default as AUTO, unless user explicitly set it
        if (tool is None) or (tool in {"AUTO", "UNKNOWN", ""}) or (tool == "COMMON_SENSE"):
            # override to PYTHON_EXEC if math-like
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
    # Priority weights / boosts
    # --------------------------
    def _priority_weight(self, node_id: str) -> float:
        """
        omega(v) priority weights:
        - root boost
        - adversary boost
        """
        w = 1.0
        if self.root_id and node_id == self.root_id:
            w *= self.root_boost
        if node_id in self.flagged_adversaries:
            w *= self.adversary_boost
        return w

    def _deg_boost(self, node_id: str) -> float:
        # connectivity matters
        return 1.0 + self.degree_boost_alpha * float(self.graph.nx_graph.degree(node_id))

    def _support_to_root_bonus(self, node_id: str) -> float:
        """
        Extra boost for nodes that SUPPORT the root claim (potential "fake shields").
        """
        if not self.root_id:
            return 1.0
        if not self.graph.nx_graph.has_edge(node_id, self.root_id):
            return 1.0
        d = self.graph.nx_graph.get_edge_data(node_id, self.root_id) or {}
        if d.get("type") == "support":
            return self.support_to_root_boost
        return 1.0

    # --------------------------
    # ROI v2 helpers
    # --------------------------
    def _k_hop_neighborhood(self, root_id: str, k: int) -> set:
        """
        Returns N_k(root): nodes within k hops from root (undirected neighborhood for locality).
        """
        if (not root_id) or (root_id not in self.graph.nx_graph):
            return set()
        if k <= 0:
            return {root_id}

        ug = self.graph.nx_graph.to_undirected(as_view=True)
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

    def _proximity_to_root(self, node_id: str) -> float:
        """
        p(v) in [0,1], higher is closer to root.
        p(v) = 1 / (dist(v, r) + 1) with dist computed on undirected graph.
        """
        r = self.root_id
        if (not r) or (r not in self.graph.nx_graph) or (node_id not in self.graph.nx_graph):
            return 0.0
        if node_id == r:
            return 1.0
        ug = self.graph.nx_graph.to_undirected(as_view=True)
        try:
            dist = nx.shortest_path_length(ug, source=node_id, target=r)
            return float(1.0 / (float(dist) + 1.0))
        except Exception:
            return 0.0

    def _normalized_structural_scores(self, pagerank_scores: Dict[str, float]) -> Dict[str, float]:
        """
        s(v) in [0,1], normalized structural influence.
        We combine PageRank and degree, then min-max normalize over current graph nodes.
        """
        nodes = list(self.graph.nx_graph.nodes())
        if not nodes:
            return {}

        pr_vals = {nid: float(pagerank_scores.get(nid, 0.0)) for nid in nodes}
        deg_vals = {nid: float(self.graph.nx_graph.degree(nid)) for nid in nodes}

        pr_min, pr_max = min(pr_vals.values()), max(pr_vals.values())
        dg_min, dg_max = min(deg_vals.values()), max(deg_vals.values())

        def norm(x: float, a: float, b: float) -> float:
            if b - a < 1e-12:
                return 0.0
            return float((x - a) / (b - a))

        w_pr = 0.7
        w_dg = 0.3

        raw = {}
        for nid in nodes:
            pr_n = norm(pr_vals[nid], pr_min, pr_max)
            dg_n = norm(deg_vals[nid], dg_min, dg_max)
            raw[nid] = w_pr * pr_n + w_dg * dg_n

        rmin, rmax = min(raw.values()), max(raw.values())
        out = {nid: norm(raw[nid], rmin, rmax) for nid in nodes}
        return out

    def _bounded_root_impact(
        self,
        current_ext_set: set,
        new_ext_set: set,
        Nk: set,
    ) -> Tuple[float, int, float]:
        """
        Returns (g, delta_root, delta_local) with g in [0,1].
        delta_root in {0,1} is XOR flip of root membership.
        delta_local in [0,1] is localized symmetric difference around Nk(root).
        """
        eps = self.roi_eps
        r = self.root_id

        if r is None:
            delta_root = 0
        else:
            in_curr = (r in current_ext_set)
            in_new = (r in new_ext_set)
            delta_root = int(in_curr ^ in_new)

        if not Nk:
            delta_local = 0.0
        else:
            symdiff = current_ext_set.symmetric_difference(new_ext_set)
            local = symdiff.intersection(Nk)
            delta_local = float(len(local) / (float(len(Nk)) + eps))

        beta = self.beta_root_flip
        g = beta * float(delta_root) + (1.0 - beta) * float(delta_local)
        g = max(0.0, min(1.0, g))
        return g, delta_root, delta_local

    # --------------------------
    # ROI v2 computation
    # --------------------------
    def _calculate_roi_candidates(
        self,
        active_nodes: List,
        pagerank_scores: Dict[str, float],
        current_ext: set,
    ) -> List[Tuple[object, float, Tuple[int, float], str, float]]:
        """
        ROI v2 (two-stage).

        Stage 1 proxy shortlist:
          ROI_tilde(v) = ((alpha * p(v) + (1-alpha) * s(v)) * omega(v)) / C(v)

        Stage 2 exact on shortlist:
          ROI(v) = ((g(v) + eps) * (1 + gamma * s(v)) * omega(v)) / C(v)

        Returns list of tuples:
          (node, roi, (delta_root, delta_local), tool, cost)
        """
        current_ext_set = set(current_ext)

        s_map = self._normalized_structural_scores(pagerank_scores)
        Nk = self._k_hop_neighborhood(self.root_id, self.k_hop_root) if self.root_id else set()

        scored = []
        for n in active_nodes:
            nid = n.id
            tool, cost = self._get_tool_and_cost(n)
            cost = max(cost, 1e-9)

            p = self._proximity_to_root(nid)
            s = float(s_map.get(nid, 0.0))

            omega = self._priority_weight(nid) * self._support_to_root_bonus(nid)

            proxy = (self.alpha_proxy * p + (1.0 - self.alpha_proxy) * s)
            roi_tilde = (proxy * omega) / cost

            scored.append((roi_tilde, n, tool, cost, s, omega))

        scored.sort(key=lambda x: x[0], reverse=True)

        k = max(1, int(self.topk_counterfactual))
        shortlist = scored[:k]

        if shortlist and all(cost > self.budget for _, _, _, cost, *_ in shortlist):
            widen_k = min(len(scored), max(k * 4, 100))
            shortlist = scored[:widen_k]

        candidates = []
        for _, node, tool, cost, s, omega in shortlist:
            if cost > self.budget:
                continue

            temp_g = copy.deepcopy(self.graph)
            if node.id in temp_g.nx_graph:
                temp_g.remove_node(node.id)
                new_ext_set = set(temp_g.get_grounded_extension())
            else:
                new_ext_set = set(current_ext_set)

            g, delta_root, delta_local = self._bounded_root_impact(
                current_ext_set=current_ext_set,
                new_ext_set=new_ext_set,
                Nk=Nk,
            )

            roi = ((g + self.roi_eps) * (1.0 + self.gamma_struct * float(s)) * float(omega)) / max(cost, 1e-9)
            candidates.append((node, float(roi), (int(delta_root), float(delta_local)), tool, float(cost)))

        return candidates

    # --------------------------
    # Topology refinement
    # --------------------------
    def _flag_attackers_of_truth(self, node_id: str) -> None:
        for u, _, d in self.graph.nx_graph.in_edges(node_id, data=True):
            if d.get("type") == "attack":
                if u in self.graph.nodes:
                    self.flagged_adversaries.add(u)

    def _refine_topology_after_true(self, node_id: str) -> None:
        """
        After verifying node_id is TRUE:
        - Flag ATTACK predecessors as suspected adversaries.
        - For outgoing edges:
            * If edge is attack: verify_attack; if invalid remove, if verify_support then convert to support.
            * If edge is support: optionally verify_support; if invalid remove (conservative).
        - Remove truth-on-truth conflicts: remove ATTACK edges between verified-TRUE nodes.
        """
        if node_id not in self.graph.nx_graph:
            return

        current_node = self.graph.nodes.get(node_id)
        if current_node is None:
            return

        # 1) flag suspected attackers
        self._flag_attackers_of_truth(node_id)

        # 2) process outgoing edges (copy list first)
        out_edges = list(self.graph.nx_graph.out_edges(node_id, data=True))
        for _, tid, d in out_edges:
            if tid not in self.graph.nodes or tid not in self.graph.nx_graph:
                continue

            edge_type = d.get("type")
            target_node = self.graph.nodes[tid]

            # prune truth-on-truth ATTACK conflicts
            if edge_type == "attack":
                if target_node.is_verified and target_node.ground_truth is True:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                        self._add_log(f"✂️ Removed truth-on-truth ATTACK: {node_id} -> {tid}")
                    continue

                # validate attack relation
                is_valid_attack = RealToolkit.verify_attack(current_node.content, target_node.content)
                if not is_valid_attack:
                    # check if it's actually support
                    try:
                        is_support = RealToolkit.verify_support(current_node.content, target_node.content)
                    except Exception:
                        is_support = False

                    # remove wrong attack
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)

                    # tiny bookkeeping burn (optional)
                    self._spend(0.05)

                    if is_support:
                        self.graph.nx_graph.add_edge(node_id, tid, type="support")
                        self._add_log(f"🔄 Converted invalid ATTACK to SUPPORT: {node_id} -> {tid}")
                    else:
                        self._add_log(f"✂️ Pruned fallacious ATTACK: {node_id} -x-> {tid}")

            elif edge_type == "support":
                # optional support consistency check; conservative: keep if tool errors
                try:
                    is_support = RealToolkit.verify_support(current_node.content, target_node.content)
                except Exception:
                    is_support = True

                if not is_support:
                    if self.graph.nx_graph.has_edge(node_id, tid):
                        self.graph.nx_graph.remove_edge(node_id, tid)
                    self._spend(0.05)
                    self._add_log(f"✂️ Pruned invalid SUPPORT: {node_id} -/> {tid}")

    # --------------------------
    # Confidence
    # --------------------------
    def _calculate_structural_confidence(self, pagerank_scores: Dict[str, float]) -> float:
        """
        Confidence based on PageRank mass inside grounded extension.
        Bonus for verified-true nodes that have verified-true supporters.
        """
        current_ge = set(self.graph.get_grounded_extension())
        if not current_ge:
            return 0.0

        total_weight = sum(float(pagerank_scores.get(nid, 0.0)) for nid in self.graph.nx_graph.nodes())
        if total_weight <= 0:
            return 0.0

        current_weight = 0.0
        for nid in current_ge:
            w = float(pagerank_scores.get(nid, 0.0))

            verified_true_supporters = 0
            for u, _, d in self.graph.nx_graph.in_edges(nid, data=True):
                if d.get("type") != "support":
                    continue
                nu = self.graph.nodes.get(u)
                if nu and nu.is_verified and nu.ground_truth is True:
                    verified_true_supporters += 1

            node_obj = self.graph.nodes.get(nid)
            if node_obj and node_obj.is_verified and node_obj.ground_truth is True:
                bonus = 1.2 ** verified_true_supporters
                current_weight += w * bonus
            else:
                current_weight += w

        conf = (current_weight / total_weight) * 100.0

        # if root is absent, confidence collapses
        if self.root_id and self.root_id not in current_ge:
            return 0.0

        return float(min(conf, 100.0))

    # --------------------------
    # Verification
    # --------------------------
    def _verify_node(self, node, tool: str, cost: float) -> bool:
        if not self._spend(cost):
            return False

        self.tool_calls += 1
        node.is_verified = True

        is_true = RealToolkit.verify_claim(tool_type=tool, claim=node.content)
        node.ground_truth = bool(is_true)
        return bool(is_true)

    # --------------------------
    # Core runners
    # --------------------------
    def run(self) -> Tuple[Set[str], Optional[bool]]:
        """
        Batch run: returns (final_grounded_extension, verdict).

        Verdict:
          - If root verified directly, use y_direct.
          - Else use membership of root in final grounded extension.
        """
        self.root_id = self.graph.find_semantic_root(claim=getattr(self.graph, "claim", None))

        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            try:
                pagerank_scores = nx.pagerank(self.graph.nx_graph, alpha=0.85)
            except Exception:
                pagerank_scores = {nid: 1.0 for nid in self.graph.nx_graph.nodes}

            current_ext = set(self.graph.get_grounded_extension())

            candidates = self._calculate_roi_candidates(active, pagerank_scores, current_ext)
            if not candidates:
                break

            best_node, best_roi, best_delta, tool, cost = max(candidates, key=lambda x: x[1])
            best_node_id = getattr(best_node, "id", None)

            is_true = self._verify_node(best_node, tool, cost)

            # direct override if root verified
            if self.root_id and best_node_id and best_node_id == self.root_id:
                self.y_direct = is_true

            if not is_true:
                if best_node_id:
                    self._prune_node(best_node_id)
            else:
                if best_node_id:
                    self._refine_topology_after_true(best_node_id)

        final_ext = set(self.graph.get_grounded_extension())

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

        self.root_id = self.graph.find_semantic_root(claim=getattr(self.graph, "claim", None))
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

            try:
                pagerank_scores = nx.pagerank(self.graph.nx_graph, alpha=0.85)
            except Exception:
                pagerank_scores = {nid: 1.0 for nid in self.graph.nx_graph.nodes}

            current_ext = set(self.graph.get_grounded_extension())

            candidates = self._calculate_roi_candidates(active, pagerank_scores, current_ext)
            if not candidates:
                yield self._add_log("ℹ️ No candidates within budget.")
                break

            best_node, best_roi, best_delta, tool, cost = max(candidates, key=lambda x: x[1])

            if self.budget < cost:
                yield self._add_log("ℹ️ Budget insufficient for next verification.")
                break

            delta_root, delta_local = best_delta
            best_node_id = getattr(best_node, "id", None)
            yield self._add_log(
                f"🔍 Verifying Keystone {best_node_id} (ROI: {best_roi:.3f}, "
                f"Δroot={delta_root}, Δlocal={delta_local:.3f}, tool={tool}, cost={cost:.2f})..."
            )

            is_true = self._verify_node(best_node, tool, cost)

            if self.root_id and best_node_id and best_node_id == self.root_id:
                self.y_direct = is_true

            if not is_true:
                impacted = (
                    list(self.graph.nx_graph.successors(best_node_id))
                    if best_node_id and best_node_id in self.graph.nx_graph
                    else []
                )
                if best_node_id:
                    self._prune_node(best_node_id)
                yield self._add_log(f"💥 FALSE. Pruned {best_node_id} and {len(impacted)} dependent claims.")
            else:
                if best_node_id:
                    self._refine_topology_after_true(best_node_id)
                yield self._add_log(f"🛡️ TRUE. Refinement updated via {best_node_id}.")

            conf_score = self._calculate_structural_confidence(pagerank_scores)

            yield {
                "type": "update",
                "nx_graph": self.graph.nx_graph.copy(),
                "budget": self.budget,
                "pagerank": pagerank_scores,
                "confidence": conf_score,
                "highlight_node": best_node_id,
                "shielded": self.graph.get_shielded_nodes() if hasattr(self.graph, "get_shielded_nodes") else [],
                "root_id": self.root_id,
                "y_direct": self.y_direct,
                "tool_calls": self.tool_calls,
                "flagged_adversaries": list(self.flagged_adversaries),
            }

            time.sleep(0.6)

        yield self._add_log("🏁 Strategic verification complete.")
