# src/agents/parser.py
# LLM-first atomic-claim parser for MaVERiC with self-refine loops.
#
# Pipeline:
#   Stage A (LLM Parse): transcript -> atomic arguments + typed relations (attack/support)
#   Stage B (LLM Self-Refine x N): graph -> patch operations (edge-level only) -> apply -> repeat
#   Stage C (Guardrails): minimal deterministic validation + root relevance gate
#
# Self-refine design:
# - Runs up to REFINE_ITERS (default 4) iterations.
# - Each iteration: ask LLM for minimal edge patch ops to fix issues (H1/H3/H6 + general).
# - Apply ops safely (no new nodes, no content edits, only edges).
# - Deterministic: temperature=0, json schema, explicit "minimal edits" rubric.
# - Early stop when no ops or when the ops do not change the graph.
#
# Keeps heuristics minimal:
# - id normalization / validation / no self-loops / allowed types
# - light relevance gate only for SUPPORT into A1

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

_ALLOWED_TOOLS = {"WEB_SEARCH", "PYTHON_EXEC", "WIKIPEDIA", "COMMON_SENSE", "AUTO"}
_ALLOWED_REL_TYPES = {"attack", "support"}

# Self-refine knobs
REFINE_ITERS = 4
REFINE_MAX_OPS = 24  # safeguard: cap number of ops we apply per iteration


# -----------------------------
# Tool + cost normalization
# -----------------------------
def _norm_tool(x: Any) -> str:
    s = str(x or "").strip().upper().replace(" ", "_")
    if s in _ALLOWED_TOOLS:
        return s
    if "SEARCH" in s or "BROWSE" in s:
        return "WEB_SEARCH"
    if "PYTHON" in s or "CALC" in s or "MATH" in s:
        return "PYTHON_EXEC"
    if "WIKI" in s:
        return "WIKIPEDIA"
    if "COMMON" in s or "SENSE" in s:
        return "COMMON_SENSE"
    if not s:
        return "AUTO"
    return "WEB_SEARCH"


def _safe_cost(tool: str) -> float:
    cfg = TOOLS_CONFIG.get(tool, None)
    if isinstance(cfg, dict):
        try:
            c = float(cfg.get("cost", 5.0))
            return c if c > 0 else 5.0
        except Exception:
            return 5.0
    return 5.0


# -----------------------------
# Minimal text helpers
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "because", "so",
    "thus", "therefore", "however", "also", "in", "on", "at", "to", "of", "for",
    "with", "without", "is", "are", "was", "were", "be", "been", "being", "as",
    "it", "this", "that", "these", "those", "you", "we", "they", "i", "he", "she",
    "them", "him", "her", "your", "our", "their", "its", "into", "from", "by",
    "about", "around", "over", "under",
}

# Used ONLY for root support relevance gate
_ROOT_PREDICATE_TOKENS = {
    "capital", "cause", "causes", "prevent", "prevents",
    "reduce", "reduces", "increase", "increases",
    "equal", "equals", "reliable", "unreliable", "evidence",
    "trial", "study", "report", "reported", "dehydration",
    "saves", "save",
}


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _tokenize(s: str) -> List[str]:
    s2 = _norm_text(s)
    toks = re.findall(r"[a-z0-9%]+", s2)
    return [t for t in toks if t and t not in _STOPWORDS]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _root_support_gate(source_txt: str, root_txt: str) -> bool:
    """
    Light, conservative gate to prevent obvious irrelevant SUPPORT into A1.
    (We intentionally do not try to be perfect. Self-refine handles semantics.)
    """
    st = set(_tokenize(source_txt))
    rt = set(_tokenize(root_txt))

    if _jaccard(list(st), list(rt)) >= 0.18:
        return True

    shared = st & rt
    if len(shared) >= 1 and (st & _ROOT_PREDICATE_TOKENS):
        return True

    return False


# -----------------------------
# ID utilities
# -----------------------------
def _dedupe_ids(args: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    seen: Set[str] = set()
    id_map: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []

    for item in args:
        old = str(item.get("id", "")).strip()
        if not old:
            continue

        new_id = old
        if new_id in seen:
            k = 2
            while f"{old}_{k}" in seen:
                k += 1
            new_id = f"{old}_{k}"

        seen.add(new_id)
        id_map[old] = new_id

        item2 = dict(item)
        item2["id"] = new_id
        out.append(item2)

    return out, id_map


def _id_num(nid: str) -> int:
    m = re.findall(r"\d+", str(nid))
    if not m:
        return 10**9
    try:
        return int(m[-1])
    except Exception:
        return 10**9


# -----------------------------
# Prompting: Stage A (Parse)
# -----------------------------
def _build_stageA_prompt(transcript: str) -> str:
    few_shot = r"""
FEW-SHOT 1 (Atomicity + correct attachment)
INPUT:
Moderator: Topic is whether eating watermelon seeds makes them grow in your stomach.
Alice: Seeds do not germinate in the stomach because stomach acid prevents sprouting and digestion prevents sprouting.
Bob: Seeds grow inside you and you will have a watermelon plant in your belly.

OUTPUT:
{
  "arguments":[
    {"id":"A1","speaker":"moderator","content":"Eating watermelon seeds makes them grow in your stomach.","tool":"WEB_SEARCH"},
    {"id":"A2","speaker":"alice","content":"Seeds do not germinate in the stomach.","tool":"WEB_SEARCH"},
    {"id":"A3","speaker":"alice","content":"Stomach acid prevents sprouting.","tool":"COMMON_SENSE"},
    {"id":"A4","speaker":"alice","content":"Digestion prevents sprouting.","tool":"COMMON_SENSE"},
    {"id":"A5","speaker":"bob","content":"Seeds grow inside you.","tool":"WEB_SEARCH"},
    {"id":"A6","speaker":"bob","content":"You will have a watermelon plant in your belly.","tool":"WEB_SEARCH"}
  ],
  "relations":[
    {"from":"A2","to":"A1","type":"attack"},
    {"from":"A3","to":"A2","type":"support"},
    {"from":"A4","to":"A2","type":"support"},
    {"from":"A6","to":"A5","type":"support"},
    {"from":"A5","to":"A1","type":"support"}
  ]
}

FEW-SHOT 2 (Undercut evidence structure)
INPUT:
Moderator: Drug X reduces blood pressure in adults.
Alice: A 2023 clinical trial reported a 10 mmHg reduction.
Bob: The trial was funded by the manufacturer and the dataset was never released, so the trial is not reliable evidence.

OUTPUT:
{
  "arguments":[
    {"id":"A1","speaker":"moderator","content":"Drug X reduces blood pressure in adults.","tool":"WEB_SEARCH"},
    {"id":"A2","speaker":"alice","content":"A 2023 clinical trial reported a 10 mmHg reduction.","tool":"WEB_SEARCH"},
    {"id":"A3","speaker":"bob","content":"The trial was funded by the manufacturer.","tool":"WEB_SEARCH"},
    {"id":"A4","speaker":"bob","content":"The dataset from the trial was never released.","tool":"WEB_SEARCH"},
    {"id":"A5","speaker":"bob","content":"The trial is not reliable evidence.","tool":"WEB_SEARCH"}
  ],
  "relations":[
    {"from":"A2","to":"A1","type":"support"},
    {"from":"A3","to":"A5","type":"support"},
    {"from":"A4","to":"A5","type":"support"},
    {"from":"A5","to":"A2","type":"attack"}
  ]
}
"""

    instructions = r"""
ROLE: Argument Graph Parser.

OUTPUT CONTRACT (MUST FOLLOW):
- Output strictly valid JSON with exactly keys: "arguments", "relations".
- arguments: list of {id, speaker, content, tool}
- relations: list of {from, to, type} where type in {"attack","support"} lowercase.
- Every id unique. Every relation references existing ids.
- MUST include root node id="A1".
- A1 must be a declarative affirmative main proposition (not a question).

ATOMICITY:
- Split turns into single-fact claims.
- Split conjunction reasons into separate nodes.

EDGE SEMANTICS:
- support: reason/evidence/justification that increases belief in target.
- attack: contradiction/negation/refutation OR undercutting the evidential basis.

TOOL:
- PYTHON_EXEC: deterministic math/equations/unit conversion.
- COMMON_SENSE: everyday physical truths.
- WEB_SEARCH: open-domain factual claims.
- If unsure, WEB_SEARCH.

IMPORTANT: Prefer structured attachment:
- Reasons attach to the claim they directly justify, not always to A1.
- Avoid cross-linking unrelated subthreads.
"""

    return f"{instructions}\n\n{few_shot}\n\nTRANSCRIPT:\n{transcript}\n"


# -----------------------------
# Prompting: Stage B (Self-refine patch ops)
# -----------------------------
def _build_stageB_prompt(root_id: str, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], iteration: int) -> str:
    rubric = r"""
ROLE: Self-Refine Repair Agent for Argumentation Graphs.

You receive:
- root_id (usually "A1")
- nodes: list of nodes with {id, speaker, content, tool}
- edges: list of edges with {from,to,type}

TASK:
Return a JSON object {"ops":[...]} where each op is one of:
- {"op":"add_edge","from":ID,"to":ID,"type":"attack|support","reason":STRING}
- {"op":"remove_edge","from":ID,"to":ID,"type":"attack|support","reason":STRING}
- {"op":"set_edge_type","from":ID,"to":ID,"type":"attack|support","reason":STRING}

ABSOLUTE CONSTRAINTS:
- Do NOT change node contents.
- Do NOT create new nodes.
- Only modify edges.
- Only use ids that exist.
- Only use type in {"attack","support"}.
- Output ONLY valid JSON.

SELF-REFINE STYLE (VERY IMPORTANT):
- Make MINIMAL edits that strictly improve correctness.
- Prefer removing wrong edges over adding many speculative edges.
- Avoid "graph drift": do not rewire everything. Fix only clear issues.
- If graph already seems reasonable, output {"ops":[]}.

HIGH-PRIORITY FIX RULES:
(H1) UNDERCUT STRUCTURE:
- "not reliable evidence / unreliable evidence / cannot be trusted / biased evidence" node is UNDERCUT CONCLUSION.
- Reasons like "funded by", "dataset not released", "conflict of interest", "p-hacking" should SUPPORT the undercut conclusion.
- Undercut conclusion should ATTACK the specific evidence claim (trial/study/report claim) it undermines.
- Undercut conclusion should NOT ATTACK its own reasons.

(H3) SCOPE / NEGATION / QUANTIFIERS:
- Scope-limited negation (e.g., "In moderate amounts, X does not cause Y for most people") should ATTACK the general claim "X causes Y".
- Universal intensifier ("X always causes Y") should SUPPORT "X causes Y".
- Optional: universal can ATTACK the scope-negation if directly contradictory.

(H6) RELEVANCE / ATTACHMENT:
- Generic related fact sharing only an entity ("France is in Europe") must NOT SUPPORT an unrelated predicate claim ("Paris is capital of France").
- Such facts can be disconnected or only attach where explicitly used.

GENERAL CLEANUP:
- Remove edges that are backward (conclusion supporting its premise).
- Remove edges that connect unrelated claims without logical justification.
- Avoid attaching too many nodes directly to root unless clearly about the root proposition.
"""

    payload = {
        "iteration": iteration,
        "root_id": root_id,
        "nodes": nodes,
        "edges": edges,
    }
    return rubric + "\n\nINPUT_GRAPH_JSON:\n" + json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Deterministic LLM call
# -----------------------------
def _chat_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    res = client.chat.completions.create(
        model=PARSER_MODEL,
        messages=messages,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(res.choices[0].message.content)


# -----------------------------
# Graph extraction helpers
# -----------------------------
def _collect_graph_edges(af: ArgumentationGraph) -> List[Dict[str, Any]]:
    out = []
    for u, v, d in list(af.nx_graph.edges(data=True)):
        et = (d or {}).get("type", "support")
        out.append({"from": u, "to": v, "type": et})
    # stable ordering for deterministic prompt diffs
    out.sort(key=lambda e: (e["from"], e["to"], e["type"]))
    return out


def _graph_signature(af: ArgumentationGraph) -> str:
    """
    Cheap stable signature of edges to detect if iteration changed anything.
    """
    edges = _collect_graph_edges(af)
    return json.dumps(edges, sort_keys=True, ensure_ascii=False)


# -----------------------------
# Apply ops safely
# -----------------------------
def _apply_ops(af: ArgumentationGraph, ops: List[Dict[str, Any]]) -> int:
    """
    Apply LLM-proposed ops safely. Returns number of applied changes.
    """
    if not ops:
        return 0

    applied = 0
    # cap to avoid runaway edits
    ops = ops[:REFINE_MAX_OPS]

    for op in ops:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("op", "")).strip()
        u = str(op.get("from", "")).strip()
        v = str(op.get("to", "")).strip()
        t = str(op.get("type", "")).strip().lower()

        if kind not in {"add_edge", "remove_edge", "set_edge_type"}:
            continue
        if t not in _ALLOWED_REL_TYPES:
            continue
        if not u or not v or u == v:
            continue
        if u not in af.nodes or v not in af.nodes:
            continue

        if kind == "remove_edge":
            if af.nx_graph.has_edge(u, v) and af.nx_graph[u][v].get("type") == t:
                af.nx_graph.remove_edge(u, v)
                applied += 1

        elif kind == "add_edge":
            if af.nx_graph.has_edge(u, v):
                continue
            if t == "attack":
                af.add_attack(u, v)
            else:
                af.add_support(u, v)
            applied += 1

        elif kind == "set_edge_type":
            if af.nx_graph.has_edge(u, v):
                cur = af.nx_graph[u][v].get("type")
                if cur != t:
                    af.nx_graph[u][v]["type"] = t
                    applied += 1
            else:
                if t == "attack":
                    af.add_attack(u, v)
                else:
                    af.add_support(u, v)
                applied += 1

    return applied


# -----------------------------
# Guardrails (minimal)
# -----------------------------
def _ensure_root(af: ArgumentationGraph, fallback_text: str) -> None:
    if "A1" in af.nodes:
        return
    af.add_node(
        ArgumentNode(
            id="A1",
            speaker="moderator",
            content=(fallback_text or "Topic claim.").strip()[:500],
            tool_type="WEB_SEARCH",
            verification_cost=_safe_cost("WEB_SEARCH"),
        )
    )


def _sanitize_graph(af: ArgumentationGraph) -> None:
    """
    Minimal deterministic cleanup:
    - drop invalid edge types / dangling edges / self-loops
    - enforce root support relevance gate for SUPPORT -> A1
    """
    _ensure_root(af, "Topic claim.")
    root_txt = af.nodes["A1"].content

    for u, v, d in list(af.nx_graph.edges(data=True)):
        if u == v:
            af.nx_graph.remove_edge(u, v)
            continue
        if u not in af.nodes or v not in af.nodes:
            af.nx_graph.remove_edge(u, v)
            continue
        et = (d or {}).get("type", "")
        if et not in _ALLOWED_REL_TYPES:
            af.nx_graph.remove_edge(u, v)
            continue
        if v == "A1" and et == "support":
            if not _root_support_gate(af.nodes[u].content, root_txt):
                af.nx_graph.remove_edge(u, v)


# -----------------------------
# Public API
# -----------------------------
def parse_debate(text: str) -> ArgumentationGraph:
    transcript = (text or "").strip()

    # -------------------
    # Stage A: Parse
    # -------------------
    try:
        promptA = _build_stageA_prompt(transcript)
        dataA = _chat_json(
            [
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": promptA},
            ]
        )
    except Exception:
        af = ArgumentationGraph()
        _ensure_root(af, transcript or "Topic claim.")
        af.find_semantic_root = lambda prefer_attack_only=True: "A1"
        return af

    args = dataA.get("arguments", [])
    rels = dataA.get("relations", [])
    if not isinstance(args, list):
        args = []
    if not isinstance(rels, list):
        rels = []

    # dedupe ids
    args, id_map = _dedupe_ids([a for a in args if isinstance(a, dict)])

    # ensure A1 exists
    if not any(str(a.get("id", "")).strip() == "A1" for a in args):
        proto = None
        for a in args:
            c = str(a.get("content", "")).strip()
            if c:
                proto = c
                break
        args = [{"id": "A1", "speaker": "moderator", "content": proto or "Topic claim.", "tool": "WEB_SEARCH"}] + args
        args, id_map = _dedupe_ids(args)

    # build graph
    af = ArgumentationGraph()

    # nodes
    for item in args:
        nid = str(item.get("id", "")).strip()
        content = str(item.get("content", "")).strip()
        if not nid or not content:
            continue

        speaker = str(item.get("speaker", "unk") or "unk").strip().lower()
        tool = _norm_tool(item.get("tool", "WEB_SEARCH"))
        cost = _safe_cost(tool)

        node = ArgumentNode(
            id=nid,
            speaker=speaker,
            content=content,
            tool_type=tool,
            verification_cost=cost,
        )
        node.is_verified = False
        af.add_node(node)

    _ensure_root(af, transcript or "Topic claim.")

    # edges
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        src0 = str(rel.get("from", "")).strip()
        dst0 = str(rel.get("to", "")).strip()
        rtype = str(rel.get("type", "")).strip().lower()

        src = id_map.get(src0, src0)
        dst = id_map.get(dst0, dst0)

        if rtype not in _ALLOWED_REL_TYPES:
            continue
        if src not in af.nodes or dst not in af.nodes:
            continue
        if src == dst:
            continue

        if rtype == "attack":
            af.add_attack(src, dst)
        else:
            af.add_support(src, dst)

    # sanitize before refine (keeps the prompt clean)
    _sanitize_graph(af)

    # -------------------
    # Stage B: Self-refine loop
    # -------------------
    for it in range(1, REFINE_ITERS + 1):
        before_sig = _graph_signature(af)

        nodes_payload = [
            {
                "id": nid,
                "speaker": af.nodes[nid].speaker,
                "content": af.nodes[nid].content,
                "tool": af.nodes[nid].tool_type,
            }
            for nid in sorted(af.nodes.keys(), key=_id_num)
        ]
        edges_payload = _collect_graph_edges(af)

        try:
            promptB = _build_stageB_prompt("A1", nodes_payload, edges_payload, it)
            dataB = _chat_json(
                [
                    {"role": "system", "content": "You output strictly valid JSON only."},
                    {"role": "user", "content": promptB},
                ]
            )
        except Exception:
            break

        ops = dataB.get("ops", [])
        if not isinstance(ops, list):
            ops = []

        # Early stop if no ops
        if not ops:
            break

        _apply_ops(af, ops)
        _sanitize_graph(af)

        after_sig = _graph_signature(af)
        # Early stop if no net change
        if after_sig == before_sig:
            break

    # -------------------
    # Finalize
    # -------------------
    _sanitize_graph(af)
    af.find_semantic_root = lambda prefer_attack_only=True: "A1"
    return af
