"""
Class F baselines - Collusion-oriented defenses/filters.

F1: MAD with adversary filtering (drop outlier agents by disagreement)
F2: Judge with evidence requirement (hard citation gate)
F3: Graph consistency gating (reject if TRUE-TRUE ATTACK remains)
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from src.config import client, JUDGE_MODEL, TOOLS_CONFIG
from src.core.baselines import _build_pseudo_transcript_from_graph
from src.core.solver import MaVERiCSolver
from src.baselines.linear_tool import retrieve_evidence
from src.baselines.selector_ablation import (
    CentralitySelector,
    DistanceToRootSelector,
    RandomSelector,
)
from src.tools.real_toolkit import RealToolkit, _summarize_hits, _safe_json_loads


def _tool_cost(tool_name: str, default_cost: float = 5.0) -> float:
    return float(TOOLS_CONFIG.get(tool_name, {}).get("cost", default_cost))


def _parse_agent_line(line: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^\s*\*?\*?\[?([A-Za-z][\w\- ]{0,40})\]?\*?\*?\s*:\s*(.+)$", line)
    if not m:
        return None
    name = m.group(1).strip()
    msg = m.group(2).strip()
    if not name or not msg:
        return None
    return name, msg


def _parse_verdict_from_text(text: str) -> Optional[bool]:
    if not text:
        return None
    up = text.upper()
    if "VERDICT" in up:
        if "TRUE" in up and "FALSE" not in up:
            return True
        if "FALSE" in up:
            return False
    if "[TRUE]" in up or re.search(r"\bTRUE\b", up):
        if "NOT TRUE" not in up and "FALSE" not in up:
            return True
    if "[FALSE]" in up or re.search(r"\bFALSE\b", up):
        return False
    if "REFUTE" in up or "INACCURATE" in up:
        return False
    if "SUPPORT" in up or "ACCURATE" in up:
        return True
    if "ABSTAIN" in up or "NEI" in up or "UNCERTAIN" in up:
        return None
    return None


def _majority_vote(labels: List[Optional[bool]], tie_break: str = "skeptical") -> Optional[bool]:
    true_cnt = sum(1 for v in labels if v is True)
    false_cnt = sum(1 for v in labels if v is False)
    if true_cnt == 0 and false_cnt == 0:
        return None
    if true_cnt > false_cnt:
        return True
    if false_cnt > true_cnt:
        return False
    return False if tie_break == "skeptical" else None


class F1MADAdversaryFilterBaseline:
    def __init__(
        self,
        drop_m: int = 2,
        disagreement_mode: str = "pairwise",
        tie_break: str = "skeptical",
        use_llm_extractor: bool = True,
    ) -> None:
        self.drop_m = int(drop_m)
        self.disagreement_mode = str(disagreement_mode or "pairwise").lower()
        self.tie_break = tie_break
        self.use_llm_extractor = bool(use_llm_extractor)
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def _extract_verdict_llm(self, claim: str, message: str) -> Optional[bool]:
        if client is None:
            return None
        prompt = f"""
You are a verdict extractor.

Claim:
{claim}

Agent message:
{message}

Task: infer the agent's final stance on the claim.
Output ONLY one of: TRUE, FALSE, ABSTAIN.
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip().upper()
            if "TRUE" in raw and "FALSE" not in raw:
                return True
            if "FALSE" in raw:
                return False
            return None
        except Exception:
            return None

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        graph: Optional[object] = None,
        budget: float = 0.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}

        if not transcript and graph is not None:
            transcript = _build_pseudo_transcript_from_graph(graph)

        transcript = transcript or ""
        lines = [ln.strip() for ln in transcript.splitlines() if ln.strip()]
        messages_by_agent: Dict[str, List[str]] = {}
        for ln in lines:
            parsed = _parse_agent_line(ln)
            if not parsed:
                continue
            name, msg = parsed
            messages_by_agent.setdefault(name, []).append(msg)

        if not messages_by_agent:
            return None

        verdicts: Dict[str, Optional[bool]] = {}
        for agent, msgs in messages_by_agent.items():
            last_msg = msgs[-1] if msgs else ""
            v = _parse_verdict_from_text(last_msg)
            if v is None and self.use_llm_extractor:
                v = self._extract_verdict_llm(claim, last_msg)
            verdicts[agent] = v

        agents = [a for a in verdicts.keys() if isinstance(a, str)]
        n = len(agents)
        drop_m = max(0, min(self.drop_m, max(0, n - 1)))

        scores: Dict[str, int] = {}
        if self.disagreement_mode == "diff_majority":
            majority = _majority_vote(list(verdicts.values()), tie_break=self.tie_break)
            for a in agents:
                scores[a] = 0 if majority is None else int(verdicts[a] != majority)
        else:
            for a in agents:
                scores[a] = sum(
                    1 for b in agents if b != a and verdicts[a] != verdicts[b]
                )

        sorted_agents = sorted(agents, key=lambda x: scores.get(str(x), 0))
        kept = sorted_agents[:-drop_m] if drop_m > 0 else sorted_agents
        kept_verdicts = [verdicts[a] for a in kept]

        final = _majority_vote(kept_verdicts, tie_break=self.tie_break)
        majority_all = _majority_vote(list(verdicts.values()), tie_break=self.tie_break)
        kept_true = sum(1 for v in kept_verdicts if v is True)
        kept_false = sum(1 for v in kept_verdicts if v is False)
        kept_margin = abs(kept_true - kept_false)

        self.stats = {
            "agents_total": int(n),
            "agents_dropped": int(drop_m),
            "agents_kept": int(len(kept)),
            "disagreement_mode": self.disagreement_mode,
            "majority_all": majority_all,
            "kept_majority_margin": int(kept_margin),
            "verdict_counts": {
                "true": int(sum(1 for v in verdicts.values() if v is True)),
                "false": int(sum(1 for v in verdicts.values() if v is False)),
                "abstain": int(sum(1 for v in verdicts.values() if v is None)),
            },
            "kept_verdict_counts": {
                "true": int(kept_true),
                "false": int(kept_false),
                "abstain": int(sum(1 for v in kept_verdicts if v is None)),
            },
            "scores": scores,
            "agents_dropped_list": [a for a in agents if a not in kept],
        }
        return final


class F2EvidenceRequirementBaseline:
    def __init__(self, per_claim_cap: int = 1) -> None:
        self.per_claim_cap = int(per_claim_cap)
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def _judge_draft(self, claim: str, transcript: str) -> Dict[str, List[str]]:
        if client is None:
            return {"support_claims": [claim], "refute_claims": []}
        prompt = f"""
You are drafting evidence-backed claims for a fact-checking judge.

Main claim:
{claim}

Transcript (if any):
{transcript}

Task:
1) Extract up to 4 atomic claims that SUPPORT the main claim.
2) Extract up to 4 atomic claims that REFUTE the main claim.
3) Only include claims that are verifiable with citations.

Output STRICT JSON:
{{
  "support_claims": ["..."],
  "refute_claims": ["..."]
}}
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            data = m.group(0) if m else "{}"
            obj = _safe_json_loads(data)
            if not isinstance(obj, dict):
                return {"support_claims": [claim], "refute_claims": []}
            sup = obj.get("support_claims") or []
            ref = obj.get("refute_claims") or []
            sup = [str(s).strip() for s in sup if s]
            ref = [str(s).strip() for s in ref if s]
            return {"support_claims": sup, "refute_claims": ref}
        except Exception:
            return {"support_claims": [claim], "refute_claims": []}

    def _choose_tool(self, claim: str) -> str:
        s = (claim or "").lower()
        if re.search(r"\d", s) and re.search(r"[=<>+\-*/]", s):
            return "PYTHON_EXEC"
        if re.search(r"\b(percent|percentage|average|mean|sum|ratio|square|sqrt|root)\b", s):
            return "PYTHON_EXEC"
        return "WEB_SEARCH"

    def _citations_consistent(
        self,
        verdict: Optional[bool],
        support_ids: List[int],
        refute_ids: List[int],
        max_id: int,
    ) -> bool:
        if verdict is True:
            if not support_ids:
                return False
            if len(support_ids) < len(refute_ids):
                return False
        if verdict is False:
            if not refute_ids:
                return False
            if len(refute_ids) < len(support_ids):
                return False
        for i in support_ids + refute_ids:
            if i < 1 or i > max_id:
                return False
        return True

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        graph: Optional[object] = None,
        budget: float = 0.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}

        if not transcript and graph is not None:
            transcript = _build_pseudo_transcript_from_graph(graph)

        transcript = transcript or ""
        draft = self._judge_draft(claim, transcript)
        support_claims = draft.get("support_claims", [])
        refute_claims = draft.get("refute_claims", [])

        validated: List[Tuple[bool, int]] = []
        attempted = 0
        evidence_found = 0
        invalid_citations = 0
        remaining = float(budget)

        def validate_one(c: str) -> None:
            nonlocal remaining
            if remaining <= 0:
                return
            nonlocal attempted, evidence_found, invalid_citations
            attempted += 1
            tool = self._choose_tool(c)
            if tool == "PYTHON_EXEC":
                cost = _tool_cost("PYTHON_EXEC", default_cost=8.0)
                if cost > remaining:
                    return
                res = RealToolkit.verify_claim("PYTHON_EXEC", c)
                self.tool_calls += 1
                self.budget_spent += cost
                remaining -= cost
                if res is True:
                    validated.append((True, 1))
                elif res is False:
                    validated.append((False, 1))
                return

            snippets, spent = retrieve_evidence(c, transcript=None, budget=remaining, max_rounds=self.per_claim_cap)
            if spent <= 0 or not snippets:
                return
            remaining -= spent
            self.tool_calls += 1
            self.budget_spent += spent
            evidence_found += 1

            evidence_lines = _summarize_hits(snippets, max_n=8)
            jr = RealToolkit._rag_judge_with_calibrated_conf(c, evidence_lines, snippets)

            max_id = len(snippets)
            if self._citations_consistent(jr.verdict, jr.support_ids, jr.refute_ids, max_id):
                if jr.verdict is True:
                    validated.append((True, len(jr.support_ids)))
                elif jr.verdict is False:
                    validated.append((False, len(jr.refute_ids)))
            else:
                invalid_citations += 1

        for c in support_claims:
            validate_one(c)

        for c in refute_claims:
            validate_one(c)

        true_votes = sum(1 for v, _ in validated if v is True)
        false_votes = sum(1 for v, _ in validated if v is False)

        if true_votes == 0 and false_votes == 0:
            verdict = False
        elif false_votes > 0 and true_votes > 0:
            verdict = False
        elif true_votes > false_votes:
            verdict = True
        else:
            verdict = False

        self.stats = {
            "support_claims": int(len(support_claims)),
            "refute_claims": int(len(refute_claims)),
            "claims_attempted": int(attempted),
            "evidence_found": int(evidence_found),
            "invalid_citations": int(invalid_citations),
            "validated_true": int(true_votes),
            "validated_false": int(false_votes),
            "budget_utilization": round(float(self.budget_spent) / max(1e-9, float(budget)), 3),
        }

        return verdict


class F3GraphConsistencyGatingSolver(MaVERiCSolver):
    def __init__(self, *args, selector: str = "centrality", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        sel = str(selector or "centrality").lower()
        if sel == "proximity":
            self.selector = DistanceToRootSelector()
        elif sel == "random":
            self.selector = RandomSelector(seed=0)
        else:
            self.selector = CentralitySelector(kind="pagerank")
        self.gate_triggered = False
        self.true_true_attack_edges: List[Tuple[str, str]] = []

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

        self.gate_triggered = False
        self.true_true_attack_edges = []

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

            tau = {}
            for nid, node in self.graph.nodes.items():
                if node.is_verified and node.ground_truth is True:
                    tau[nid] = "TRUE"
                elif node.is_verified and node.ground_truth is False:
                    tau[nid] = "FALSE"
                else:
                    tau[nid] = "UNK"
            pick = self.selector.select(candidates, self.graph, tau, self.root_id, self.flagged_adversaries, self.budget)
            if pick is None:
                break

            pick_id = getattr(pick, "id", None)
            if not pick_id:
                break
            key: str = str(pick_id)
            tool = tool_map.get(key, None)
            cost = cost_map.get(key, None)
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

        for u, v, d in self.graph.nx_graph.edges(data=True):
            if (d or {}).get("type") != "attack":
                continue
            if self._is_verified_true(u) and self._is_verified_true(v):
                self.true_true_attack_edges.append((u, v))

        if self.true_true_attack_edges:
            self.gate_triggered = True

        evidence_used = (self.tool_calls > 0) or (len(self.verified_true_ids) + len(self.verified_false_ids) > 0)
        if not evidence_used:
            verdict = None
        elif self.y_direct is not None:
            verdict = bool(self.y_direct)
        else:
            verdict = bool(self.root_id in final_ext) if self.root_id else False

        if self.gate_triggered:
            verdict = False

        return final_ext, verdict


__all__ = [
    "F1MADAdversaryFilterBaseline",
    "F2EvidenceRequirementBaseline",
    "F3GraphConsistencyGatingSolver",
]
