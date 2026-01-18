# src/core/baselines.py

import random
from typing import Optional, Tuple, Set

from src.config import TOOLS_CONFIG, client, JUDGE_MODEL
from src.tools.real_toolkit import RealToolkit


# ----------------------------
# Helpers
# ----------------------------
def _as_set(x) -> Set[str]:
    if x is None:
        return set()
    if isinstance(x, set):
        return x
    if isinstance(x, (list, tuple)):
        return set(x)
    # if x is a single string id
    if isinstance(x, str):
        return {x}
    return set()


def _fallback_root_id(graph) -> Optional[str]:
    """
    Robust root resolver:
    - Prefer graph.find_semantic_root() if exists.
    - Else pick earliest node by (numeric part, then lexicographic).
    """
    if graph is None:
        return None

    if hasattr(graph, "find_semantic_root"):
        try:
            rid = graph.find_semantic_root(claim=getattr(graph, "claim", None))
            if rid:
                return rid
        except Exception:
            pass

    # fallback: pick smallest node id
    try:
        ids = [nid for nid in graph.nodes.keys() if nid in graph.nx_graph]
        if not ids:
            return None

        def key_fn(s: str):
            digits = "".join(ch for ch in s if ch.isdigit())
            return (int(digits) if digits else 10**9, s)

        ids.sort(key=key_fn)
        return ids[0]
    except Exception:
        return None


def _tool_cost(tool_type: str, default_cost: float = 5.0) -> float:
    tt = (tool_type or "WEB_SEARCH").upper()
    return float(TOOLS_CONFIG.get(tt, {}).get("cost", default_cost))


def _build_pseudo_transcript_from_graph(graph, max_chars: int = 20000) -> str:
    """
    MADSolver needs a transcript-like text.
    If we do not have the raw debate transcript, we reconstruct a pseudo transcript
    from atomic claims in the graph.
    """
    if graph is None or not getattr(graph, "nodes", None):
        return ""

    # Sort nodes by numeric id part for a stable order (A1, A2, ...)
    ids = list(graph.nodes.keys())

    def key_fn(s: str):
        digits = "".join(ch for ch in s if ch.isdigit())
        return (int(digits) if digits else 10**9, s)

    ids.sort(key=key_fn)

    lines = []
    for nid in ids:
        n = graph.nodes.get(nid)
        if n is None:
            continue
        speaker = getattr(n, "speaker", "UNK")
        content = getattr(n, "content", "")
        if not content:
            continue
        lines.append(f"{speaker}: {content}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED]"
    return text


# =========================================================
# Baselines
# =========================================================
class RandomSolver:
    """
    Random verification baseline:
    - Randomly pick unverified nodes, verify until budget exhausted.
    - Verdict is whether the semantic root survives in the final extension.
    """

    def __init__(self, graph, budget: float):
        self.graph = graph
        self.budget = float(budget)
        self.tool_calls = 0
        self.root_id = _fallback_root_id(graph)

    def run(self) -> Tuple[Set[str], Optional[bool]]:
        while self.budget > 0:
            active = [
                n for n in self.graph.nodes.values()
                if (not getattr(n, "is_verified", False)) and (n.id in self.graph.nx_graph)
            ]
            if not active:
                break

            node = random.choice(active)

            tool_type = getattr(node, "tool_type", "WEB_SEARCH")
            cost = _tool_cost(tool_type, default_cost=5.0)
            if cost > self.budget:
                break

            self.budget -= cost
            self.tool_calls += 1
            node.is_verified = True

            is_true = RealToolkit.verify_claim(tool_type, node.content)
            if is_true is None:
                node.ground_truth = None
                break
            node.ground_truth = bool(is_true)

            if not is_true:
                self.graph.remove_node(node.id)

        final_ext = _as_set(self.graph.get_grounded_extension())
        verdict = bool(self.root_id in final_ext) if self.root_id else None
        return final_ext, verdict


class CRITICSolver:
    """
    Sequential verification baseline:
    - Verify nodes in id order until budget exhausted.
    - Verdict based on semantic root membership in final extension.
    """

    def __init__(self, graph, budget: float):
        self.graph = graph
        self.budget = float(budget)
        self.tool_calls = 0
        self.root_id = _fallback_root_id(graph)

    def run(self) -> Tuple[Set[str], Optional[bool]]:
        nodes = sorted(
            [n for n in self.graph.nodes.values() if n.id in self.graph.nx_graph],
            key=lambda x: x.id,
        )

        for node in nodes:
            tool_type = getattr(node, "tool_type", "WEB_SEARCH")
            cost = _tool_cost(tool_type, default_cost=5.0)
            if self.budget < cost:
                break

            self.budget -= cost
            self.tool_calls += 1
            node.is_verified = True

            is_true = RealToolkit.verify_claim(tool_type, node.content)
            if is_true is None:
                node.ground_truth = None
                break
            node.ground_truth = bool(is_true)

            if not is_true:
                self.graph.remove_node(node.id)

        final_ext = _as_set(self.graph.get_grounded_extension())
        verdict = bool(self.root_id in final_ext) if self.root_id else None
        return final_ext, verdict


class MADSolver:
    """
    Majority-Aggregation baseline (social consensus):
    - Compatible with main_experiments: init(graph, budget)
    - run() returns (extension_set, verdict_bool)

    Note:
    - MAD does not verify nodes and does not use budget.
    - We reconstruct a pseudo transcript from the graph's atomic claims.
    """

    def __init__(self, graph, budget: float, topic: Optional[str] = None):
        self.graph = graph
        self.budget = float(budget)
        self.topic = topic or getattr(graph, "topic", None) or ""
        self.tool_calls = 0  # MAD does not call tools by design
        self.reason = ""
        self.root_id = _fallback_root_id(graph)

    def run(self) -> Tuple[Set[str], Optional[bool]]:
        transcript = _build_pseudo_transcript_from_graph(self.graph)

        if not self.topic:
            # fallback: if topic not provided, use root claim content (if exists)
            if self.root_id and self.root_id in self.graph.nodes:
                self.topic = getattr(self.graph.nodes[self.root_id], "content", "") or ""

        prompt = f"""
Role: Social Consensus Analyzer.
Topic: "{self.topic}"
Transcript:
{transcript}

Task:
1) Ignore your own scientific knowledge.
2) Estimate whether the majority of participants SUPPORT the topic or OPPOSE it.
3) Output a verdict that reflects the majority belief in the transcript.

Output format (exactly):
VERDICT: [ACCURATE/INACCURATE]
REASON: [one sentence]
"""

        verdict_bool: Optional[bool] = None
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            output = (res.choices[0].message.content or "").strip()

            # parse verdict
            out_upper = output.upper()
            is_accurate = ("VERDICT:" in out_upper) and ("ACCURATE" in out_upper) and ("INACCURATE" not in out_upper)
            is_inaccurate = ("VERDICT:" in out_upper) and ("INACCURATE" in out_upper)
            if is_accurate:
                verdict_bool = True
            elif is_inaccurate:
                verdict_bool = False
            else:
                # fallback heuristic
                verdict_bool = True if "ACCURATE" in out_upper and "INACCURATE" not in out_upper else False

            if "REASON:" in output:
                self.reason = output.split("REASON:", 1)[-1].strip()
            else:
                self.reason = "Majority-based judgment."

        except Exception:
            verdict_bool = None
            self.reason = "Error in MAD judgment."

        # MAD cannot produce a full accepted set, so we return a minimal proxy:
        # accept root if verdict true, else accept empty set.
        if self.root_id and verdict_bool is True:
            return {self.root_id}, True
        if self.root_id and verdict_bool is False:
            return set(), False

        # if root is missing or verdict is None
        return set(), verdict_bool


# Optional: keep your ReActAgent elsewhere, but do NOT mix its schema with graph baselines
