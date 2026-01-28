"""
Class D baselines - Graph-aware selector ablations (share MaVERiC refinement)

This module implements selector-only ablations that keep the full MaVERiC
verification loop and refinement rules, but replace the node selector:
  - D1: RandomSelector
  - D2: UncertaintySelector (cheap probe, no tools)
  - D3: CentralitySelector (pagerank/degree/betweenness)
  - D4: DistanceToRootSelector (attack-only proximity)
  - D5: ProxyOnlySelector (Stage-1 ROI proxy only)
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Callable, Tuple

import networkx as nx

from src.config import client, JUDGE_MODEL
from src.core.solver import MaVERiCSolver


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _attack_only_graph(graph) -> nx.DiGraph:
    if hasattr(graph, "get_attack_subgraph"):
        return graph.get_attack_subgraph()
    g = nx.DiGraph()
    g.add_nodes_from(graph.nx_graph.nodes())
    for u, v, d in graph.nx_graph.edges(data=True):
        if (d or {}).get("type") == "attack":
            g.add_edge(u, v)
    return g


def _tau_map(graph) -> Dict[str, str]:
    tau: Dict[str, str] = {}
    for nid, node in graph.nodes.items():
        if node.is_verified and node.ground_truth is True:
            tau[nid] = "TRUE"
        elif node.is_verified and node.ground_truth is False:
            tau[nid] = "FALSE"
        else:
            tau[nid] = "UNK"
    return tau


def _reverse_dist_to_root(g_atk: nx.DiGraph, root_id: Optional[str]) -> Dict[str, int]:
    if not root_id or root_id not in g_atk:
        return {}
    g_rev = g_atk.reverse(copy=False)
    try:
        return dict(nx.single_source_shortest_path_length(g_rev, root_id))
    except Exception:
        return {}


def _is_direct_support_to_root(graph, node_id: str, root_id: Optional[str]) -> bool:
    if not root_id:
        return False
    if not graph.nx_graph.has_edge(node_id, root_id):
        return False
    d = graph.nx_graph.get_edge_data(node_id, root_id) or {}
    return d.get("type") == "support"


# -----------------------------------------------------------------------------
# Selectors
# -----------------------------------------------------------------------------
class BaseSelector:
    def __init__(self) -> None:
        self._cost_fn: Optional[Callable[[object], float]] = None

    def set_cost_fn(self, cost_fn: Callable[[object], float]) -> None:
        self._cost_fn = cost_fn

    def _get_cost(self, node: object, default_cost: float = 5.0) -> float:
        if self._cost_fn is not None:
            try:
                return float(self._cost_fn(node))
            except Exception:
                return float(default_cost)
        node_cost = getattr(node, "verification_cost", None)
        if node_cost is not None and float(node_cost) > 0:
            return float(node_cost)
        return float(default_cost)


class RandomSelector(BaseSelector):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = random.Random(seed)

    def select(self, C, G, tau, r, A, B_rem):
        return self._rng.choice(C) if C else None


class UncertaintySelector(BaseSelector):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        self._rng = random.Random(seed)
        self._cache: Dict[str, float] = {}

    def _cheap_llm_prob_true(self, claim_text: str) -> float:
        if not claim_text:
            return 0.5
        if client is None:
            return 0.5

        prompt = (
            "Return a probability between 0 and 1 that the claim is true. "
            "Only output a number.\n\n"
            f"Claim: {claim_text}\n"
        )
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=6,
            )
            raw = (res.choices[0].message.content or "").strip()
            num = None
            for tok in raw.replace("%", " ").split():
                try:
                    num = float(tok)
                    break
                except Exception:
                    continue
            if num is None:
                return 0.5
            if num > 1.0:
                num = num / 100.0
            return float(max(0.0, min(1.0, num)))
        except Exception:
            return 0.5

    def select(self, C, G, tau, r, A, B_rem):
        best_v = None
        best_h = -1.0
        eps = 1e-9
        for v in C:
            vid = getattr(v, "id", None) or ""
            if vid in self._cache:
                p = self._cache[vid]
            else:
                p = self._cheap_llm_prob_true(getattr(v, "content", ""))
                self._cache[vid] = p
            h = -p * math.log(p + eps) - (1.0 - p) * math.log(1.0 - p + eps)
            if (h > best_h) or (h == best_h and (vid or "") < (getattr(best_v, "id", "") or "")):
                best_h = h
                best_v = v
        return best_v


class CentralitySelector(BaseSelector):
    def __init__(self, kind: str = "pagerank") -> None:
        super().__init__()
        self.kind = str(kind or "pagerank").lower()

    def _centrality(self, g_atk: nx.DiGraph) -> Dict[str, float]:
        if self.kind == "degree":
            return {nid: float(g_atk.degree(nid)) for nid in g_atk.nodes()}
        if self.kind == "betweenness":
            try:
                return nx.betweenness_centrality(g_atk, normalized=True)
            except Exception:
                return {nid: 0.0 for nid in g_atk.nodes()}
        try:
            return nx.pagerank(g_atk, alpha=0.85)
        except Exception:
            return {nid: 0.0 for nid in g_atk.nodes()}

    def select(self, C, G, tau, r, A, B_rem):
        if not C:
            return None
        g_atk = _attack_only_graph(G)
        cent = self._centrality(g_atk)
        return max(
            C,
            key=lambda v: (float(cent.get(getattr(v, "id", ""), 0.0)), getattr(v, "id", "")),
        )


class DistanceToRootSelector(BaseSelector):
    def select(self, C, G, tau, r, A, B_rem):
        if not C:
            return None
        g_atk = _attack_only_graph(G)
        dist = _reverse_dist_to_root(g_atk, r)

        def prox(node) -> float:
            nid = getattr(node, "id", None)
            if nid is None:
                return 0.0
            d = dist.get(nid, None)
            if d is None:
                return 0.0
            return 1.0 / (float(d) + 1.0)

        return max(C, key=lambda v: (prox(v), getattr(v, "id", "")))


class ProxyOnlySelector(BaseSelector):
    def __init__(self, rho: float = 0.5) -> None:
        super().__init__()
        self.rho = float(rho)

    def select(self, C, G, tau, r, A, B_rem):
        if not C:
            return None

        g_atk = _attack_only_graph(G)
        dist = _reverse_dist_to_root(g_atk, r)
        try:
            pr = nx.pagerank(g_atk, alpha=0.85)
        except Exception:
            pr = {nid: 0.0 for nid in g_atk.nodes()}

        def p(node) -> float:
            nid = getattr(node, "id", None)
            if nid is None:
                return 0.0
            d = dist.get(nid, None)
            if d is None:
                return 0.0
            return 1.0 / (float(d) + 1.0)

        def omega(node) -> float:
            nid = getattr(node, "id", None)
            w = 1.0
            if nid and r and nid == r:
                w *= 2.0
            if nid and nid in A:
                w *= 1.5
            if nid and _is_direct_support_to_root(G, nid, r):
                w *= 1.2
            return float(w)

        def roi_proxy(node) -> float:
            cost = self._get_cost(node)
            if cost > B_rem:
                return -1e12
            nid = getattr(node, "id", None) or ""
            s = float(pr.get(nid, 0.0))
            score = (self.rho * p(node) + (1.0 - self.rho) * s) * omega(node)
            return float(score) / max(float(cost), 1e-9)

        return max(C, key=lambda v: (roi_proxy(v), getattr(v, "id", "")))


# -----------------------------------------------------------------------------
# Group D solver (selector ablations, shared refinement)
# -----------------------------------------------------------------------------
class SelectorAblationSolver(MaVERiCSolver):
    def __init__(self, *args, selector: Optional[BaseSelector] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.selector: BaseSelector = selector or RandomSelector(seed=0)

    def _feasible_candidates(self, active_nodes: List) -> Tuple[List, Dict[str, float], Dict[str, str]]:
        candidates: List = []
        cost_map: Dict[str, float] = {}
        tool_map: Dict[str, str] = {}
        for node in active_nodes:
            tool, cost = self._get_tool_and_cost(node)
            if float(cost) <= self.budget + 1e-12:
                nid = getattr(node, "id", None)
                if nid:
                    candidates.append(node)
                    cost_map[nid] = float(cost)
                    tool_map[nid] = str(tool)
        return candidates, cost_map, tool_map

    def run(self):
        self.budget = float(self.initial_budget)

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

        if not self.root_id:
            claim = getattr(self.graph, "claim", None)
            self.root_id = self.graph.find_semantic_root(
                claim=claim,
                llm_tiebreaker=self.root_llm_tiebreaker if claim else None,
                tie_topk=self.root_tie_topk,
                tie_margin=self.root_tie_margin,
            )

        while self.budget > 1e-12:
            active = [
                n for n in self.graph.nodes.values()
                if (not n.is_verified) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            candidates, cost_map, tool_map = self._feasible_candidates(active)
            if not candidates:
                break

            if hasattr(self.selector, "set_cost_fn"):
                self.selector.set_cost_fn(lambda node: cost_map.get(getattr(node, "id", ""), 5.0))

            tau = _tau_map(self.graph)
            pick = self.selector.select(candidates, self.graph, tau, self.root_id, self.flagged_adversaries, self.budget)
            if pick is None:
                break

            pick_id = getattr(pick, "id", None)
            tool = tool_map.get(pick_id, None)
            cost = cost_map.get(pick_id, None)
            if tool is None or cost is None:
                tool, cost = self._get_tool_and_cost(pick)

            is_true = self._verify_node(pick, tool, float(cost))
            if is_true is None:
                break

            if self.root_id and pick_id and pick_id == self.root_id:
                self.y_direct = bool(is_true)

            if not is_true:
                if pick_id:
                    self._refine_topology_after_false(pick_id)
                    self._prune_node(pick_id)
            else:
                if pick_id:
                    self._refine_topology_after_true(pick_id)

        final_ext = self._sgs()

        evidence_used = (self.tool_calls > 0) or (len(self.verified_true_ids) + len(self.verified_false_ids) > 0)
        if not evidence_used:
            verdict = None
        elif self.y_direct is not None:
            verdict = bool(self.y_direct)
        else:
            verdict = bool(self.root_id in final_ext) if self.root_id else False

        return final_ext, verdict


class D1RandomRefineSolver(SelectorAblationSolver):
    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, selector=RandomSelector(seed=seed), **kwargs)


class D2UncertaintyRefineSolver(SelectorAblationSolver):
    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, selector=UncertaintySelector(seed=seed), **kwargs)


class D3CentralityRefineSolver(SelectorAblationSolver):
    def __init__(self, *args, kind: str = "pagerank", **kwargs) -> None:
        super().__init__(*args, selector=CentralitySelector(kind=kind), **kwargs)


class D4DistanceRefineSolver(SelectorAblationSolver):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, selector=DistanceToRootSelector(), **kwargs)


class D5ProxyOnlyRefineSolver(SelectorAblationSolver):
    def __init__(self, *args, rho: float = 0.5, **kwargs) -> None:
        super().__init__(*args, selector=ProxyOnlySelector(rho=rho), **kwargs)


__all__ = [
    "SelectorAblationSolver",
    "RandomSelector",
    "UncertaintySelector",
    "CentralitySelector",
    "DistanceToRootSelector",
    "ProxyOnlySelector",
    "D1RandomRefineSolver",
    "D2UncertaintyRefineSolver",
    "D3CentralityRefineSolver",
    "D4DistanceRefineSolver",
    "D5ProxyOnlyRefineSolver",
]
