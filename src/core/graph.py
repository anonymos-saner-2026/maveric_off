# src/core/graph.py

import networkx as nx
from dataclasses import dataclass
from typing import Set, Dict, Optional, Callable, List


@dataclass
class ArgumentNode:
    """
    Container for a single argument in the debate graph.
    """
    id: str
    content: str
    speaker: str

    # State updated by the solver
    is_verified: bool = False
    ground_truth: Optional[bool] = None  # True / False / None (unknown)

    # Cost and tool metadata
    verification_cost: float = 0.0
    tool_type: str = "AUTO"


class ArgumentationGraph:
    """
    Argumentation graph with both attack and support edges.

    Nodes are identified by string ids and stored in both:
      - self.nodes: mapping id -> ArgumentNode
      - self.nx_graph: networkx DiGraph with edges labeled by 'type' in {attack, support}
    """

    def __init__(self) -> None:
        self.nx_graph: nx.DiGraph = nx.DiGraph()
        self.nodes: Dict[str, ArgumentNode] = {}
        self.claim: Optional[str] = None
        self.root_id_override: Optional[str] = None
        self.atomicity_warnings: List[str] = []

    # ------------------------------------------------------------------
    # Basic graph construction
    # ------------------------------------------------------------------
    def add_node(self, node: ArgumentNode) -> None:
        self.nodes[node.id] = node
        self.nx_graph.add_node(node.id)

    def add_attack(self, attacker: str, target: str) -> None:
        if attacker in self.nodes and target in self.nodes:
            self.nx_graph.add_edge(attacker, target, type="attack")

    def add_support(self, supporter: str, target: str) -> None:
        if supporter in self.nodes and target in self.nodes:
            self.nx_graph.add_edge(supporter, target, type="support")

    def remove_node(self, node_id: str) -> None:
        if node_id in self.nx_graph:
            self.nx_graph.remove_node(node_id)
        if node_id in self.nodes:
            del self.nodes[node_id]

    # ------------------------------------------------------------------
    # Helper: verification state
    # ------------------------------------------------------------------
    def _tau(self, node_id: str) -> Optional[bool]:
        """
        Return verification truth value:
          - True if verified true
          - False if verified false
          - None if unknown
        """
        n = self.nodes.get(node_id)
        if not n or not n.is_verified:
            return None
        return bool(n.ground_truth) if n.ground_truth is not None else None

    def _is_verified_false(self, node_id: str) -> bool:
        n = self.nodes.get(node_id)
        return bool(n and n.is_verified and n.ground_truth is False)

    def _is_verified_true(self, node_id: str) -> bool:
        n = self.nodes.get(node_id)
        return bool(n and n.is_verified and n.ground_truth is True)

    # ------------------------------------------------------------------
    # Helper views
    # ------------------------------------------------------------------
    def get_attack_subgraph(self) -> nx.DiGraph:
        """
        Return a DiGraph that only contains attack edges.
        """
        g_attack = nx.DiGraph()
        g_attack.add_nodes_from(self.nx_graph.nodes())
        for u, v, d in self.nx_graph.edges(data=True):
            if (d or {}).get("type") == "attack":
                g_attack.add_edge(u, v)
        return g_attack

    def get_shielded_nodes(self) -> Set[str]:
        """
        Nodes v that have at least one verified-true supporter u with support(u -> v).
        """
        shielded: Set[str] = set()
        for u, v, d in self.nx_graph.edges(data=True):
            if (d or {}).get("type") != "support":
                continue
            if self._is_verified_true(u):
                shielded.add(v)
        return shielded

    # ------------------------------------------------------------------
    # Root detection heuristic (optional)
    # ------------------------------------------------------------------
    def find_semantic_root(
        self,
        prefer_attack_only: bool = True,
        claim: Optional[str] = None,
        llm_tiebreaker: Optional[Callable[[str, List[str]], Optional[str]]] = None,
        tie_topk: int = 3,
        tie_margin: float = 0.05,
    ) -> Optional[str]:
        """
        Heuristic to identify a semantic root node.

        Important: if prefer_attack_only=True, compute centrality on the attack-only subgraph
        to reduce sensitivity to support-spam.

        Hybrid tie-breaker:
        - Use topology score first.
        - If top-k scores are close, optionally rerank with LLM to match claim.
        """
        if self.root_id_override:
            return self.root_id_override

        if not self.nx_graph.nodes:
            return None

        g = self.get_attack_subgraph() if prefer_attack_only else self.nx_graph

        try:
            pagerank = nx.pagerank(g, alpha=0.85)
        except Exception:
            pagerank = {n: 0.0 for n in g.nodes}

        in_degree = dict(g.in_degree())
        max_in_deg = max(in_degree.values()) if in_degree else 1.0

        scores: Dict[str, float] = {}
        for node_id in g.nodes:
            indeg_norm = float(in_degree.get(node_id, 0)) / float(max_in_deg or 1.0)
            scores[node_id] = 0.6 * float(pagerank.get(node_id, 0.0)) + 0.4 * indeg_norm

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if not ranked:
            return None

        top_id, top_score = ranked[0]
        if not claim or llm_tiebreaker is None:
            return top_id

        topk = ranked[: max(1, int(tie_topk))]
        if len(topk) <= 1:
            return top_id

        cutoff = float(top_score) * (1.0 - float(tie_margin))
        close = [nid for nid, score in topk if score >= cutoff]
        if len(close) <= 1:
            return top_id

        pick = llm_tiebreaker(claim, close)
        return pick if pick in close else top_id

    # ------------------------------------------------------------------
    # Shielded grounded semantics (SGS)
    # ------------------------------------------------------------------
    def get_grounded_extension(
        self,
        use_shield: bool = True,
        alpha: float = 1.0,
        require_evidence: bool = True,
    ) -> Set[str]:
        """
        Compute the SGS accepted set as a grounded-style least fixed point with a shield rule.

        Key behaviors:
        - Verified-false nodes are excluded from active_nodes.
        - Conflict-free (practical):
            * never accept a node attacked by an already accepted node (v in defeated)
            * never accept a node that attacks an already accepted node
            * within-round: drop candidates attacked by other candidates

        Evidence-gated variant (recommended):
        - If require_evidence=True:
            * even if v has no alive attackers, we only accept v if:
                - v is verified_true, OR
                - v has at least one verified_true supporter (i.e., is shielded)
        This avoids accepting "uncontested" but unverified / noisy atomic claims.

        Shield rule:
        - If v has alive attackers, it can still be accepted if:
            sup_count >= alpha * atk_count
        where sup_count counts ONLY verified_true supporters (when use_shield=True).
        """
        if alpha <= 0:
            raise ValueError("alpha must be > 0")

        # Active nodes: exclude verified-false nodes
        active_nodes: Set[str] = {nid for nid in self.nx_graph.nodes() if not self._is_verified_false(nid)}
        if not active_nodes:
            return set()

        attackers_of: Dict[str, Set[str]] = {v: set() for v in active_nodes}
        supporters_true_of: Dict[str, Set[str]] = {v: set() for v in active_nodes}
        outgoing_attack: Dict[str, Set[str]] = {u: set() for u in active_nodes}

        # Build local adjacency views restricted to active_nodes
        for u, v, d in self.nx_graph.edges(data=True):
            if u not in active_nodes or v not in active_nodes:
                continue
            et = (d or {}).get("type")
            if et == "attack":
                attackers_of[v].add(u)
                outgoing_attack[u].add(v)
            elif et == "support":
                if use_shield and self._is_verified_true(u):
                    supporters_true_of[v].add(u)

        accepted: Set[str] = set()   # A
        defeated: Set[str] = set()   # Def(A): nodes attacked by accepted nodes

        while True:
            candidates: Set[str] = set()

            for v in active_nodes:
                if v in accepted:
                    continue

                # cannot accept nodes attacked by accepted nodes
                if v in defeated:
                    continue

                # cannot accept nodes that attack already-accepted nodes
                if outgoing_attack.get(v, set()) & accepted:
                    continue

                alive_attackers = attackers_of[v] - defeated

                # evidence_ok: verified_true(v) OR has >=1 verified_true supporter (shielded)
                evidence_ok = self._is_verified_true(v) or (len(supporters_true_of.get(v, set())) > 0)

                # Case 1) No alive attackers
                if not alive_attackers:
                    if (not require_evidence) or evidence_ok:
                        candidates.add(v)
                    continue

                # Case 2) Has alive attackers -> shield rule
                if use_shield:
                    sup_count = len(supporters_true_of.get(v, set()))
                    atk_count = len(alive_attackers)
                    if float(sup_count) >= float(alpha) * float(atk_count):
                        candidates.add(v)

            if not candidates:
                break

            # Conservative within-round filtering: drop candidates attacked by other candidates
            attacked_inside: Set[str] = set()
            for u in candidates:
                attacked_inside |= (outgoing_attack.get(u, set()) & candidates)

            newly_accepted = candidates - attacked_inside
            if not newly_accepted:
                break

            accepted |= newly_accepted

            # update defeated set: anything attacked by newly accepted nodes
            for a in newly_accepted:
                defeated |= outgoing_attack.get(a, set())

        return accepted
