# Full LLM pipeline parser for MaVERiC.
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode
from src.agents.parser import _ALLOWED_REL_TYPES, _ALLOWED_TOOLS, _safe_cost, _norm_tool, _sanitize_graph, _dedupe_ids

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def _chat_json(
    messages: List[Dict[str, str]],
    max_retries: int = 2,
    retry_delay: float = 0.8,
) -> Dict[str, Any]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            res = client.chat.completions.create(
                model=PARSER_MODEL,
                messages=messages,  # type: ignore[arg-type]
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = res.choices[0].message.content
            if not content:
                raise ValueError("empty response")
            return json.loads(content)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_delay * (attempt + 1))
    if last_error:
        return {}
    return {}


def _build_atomic_prompt(transcript: str) -> str:
    rubric = r"""
ROLE: Atomic Claim Parser.

TASK:
Extract atomic, standalone claims from the transcript.
Return JSON with keys:
- root: {id:"A1", content: "..."}
- claims: list of {id, speaker, content, tool}

RULES:
- Root A1 must be the main debate proposition (affirmative, declarative).
- Each claim MUST be a single atomic statement (no conjunctions).
- Rewrite pronouns to standalone nouns.
- Use tools: WEB_SEARCH, COMMON_SENSE, PYTHON_EXEC, WIKIPEDIA, AUTO.
- Ensure unique ids: A1, A2, A3...

FEW-SHOT:
Transcript:
Moderator: Topic: Does social media harm teen mental health?
Alice: Social media increases anxiety.
Bob: Correlation is not causation.

Output:
{"root":{"id":"A1","content":"Social media harms teen mental health."},
 "claims":[
  {"id":"A2","speaker":"alice","content":"Social media increases anxiety in teens.","tool":"WEB_SEARCH"},
  {"id":"A3","speaker":"bob","content":"Correlation does not imply causation.","tool":"COMMON_SENSE"}
 ]}
"""
    return rubric + "\n\nTRANSCRIPT:\n" + transcript


def _build_rel_prompt(nodes: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Relation Builder.

TASK:
Given atomic claims, return JSON {"relations":[...]} where each relation is:
- {"from": ID, "to": ID, "type": "attack|support"}

RULES:
- Only use provided IDs.
- No self-loops.
- Prefer linking to the most specific claim (not always A1).
- Support: evidence, elaboration, agreement.
- Attack: contradiction, refutation, undercutting.
"""
    return rubric + "\n\nNODES:\n" + json.dumps({"nodes": nodes}, ensure_ascii=False)


def _build_orphan_prompt(orphan_claims: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Orphan Attachment.

TASK:
Attach orphan claims to the best-matching prior claims.
Return JSON {"relations":[...]} where each relation is:
- {"from": ID, "to": ID, "type": "attack|support"}

RULES:
- Only add edges FROM orphan claims.
- At most 1 edge per orphan.
- Use attack for contradiction, support for evidence.
- If no reasonable link exists, omit that orphan.
"""
    payload = {"orphans": orphan_claims, "nodes": nodes}
    return rubric + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)


def _build_edge_audit_prompt(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Edge Semantic Auditor.

TASK:
Check each edge and return JSON {"results": [...]}
- {"from":ID, "to":ID, "label":"correct|incorrect", "fix":"keep|flip|remove"}

RULES:
- Flip if type is wrong but relation exists.
- Remove if unrelated.
- Keep if correct.
"""
    payload = {"nodes": nodes, "edges": edges}
    return rubric + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False)


def parse_debate_llm(transcript: str) -> ArgumentationGraph:
    transcript = (transcript or "").strip()
    af = ArgumentationGraph()
    af.claim = transcript

    try:
        prompt = _build_atomic_prompt(transcript)
        data = _chat_json(
            [
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception:
        data = {}

    root = data.get("root", {}) if isinstance(data, dict) else {}
    claims = data.get("claims", []) if isinstance(data, dict) else []
    if not isinstance(claims, list):
        claims = []

    root_text = str(root.get("content", "")).strip() if isinstance(root, dict) else ""
    if not root_text:
        root_text = transcript or "Topic claim."

    raw_items: List[Dict[str, Any]] = [
        {"id": "A1", "speaker": "moderator", "content": root_text, "tool": "WEB_SEARCH"}
    ]
    for item in claims:
        if isinstance(item, dict):
            raw_items.append(dict(item))
    deduped_items, _ = _dedupe_ids(raw_items)

    for item in deduped_items:
        if not isinstance(item, dict):
            continue
        nid = str(item.get("id", "")).strip() or ""
        content = str(item.get("content", "")).strip()
        if not content or not nid:
            continue
        if nid == "A1":
            root_node = ArgumentNode(
                id="A1",
                speaker="moderator",
                content=root_text,
                tool_type="WEB_SEARCH",
                verification_cost=_safe_cost("WEB_SEARCH"),
            )
            root_node.is_verified = False
            af.add_node(root_node)
            continue
        speaker = str(item.get("speaker", "unk") or "unk").strip().lower()
        tool = _norm_tool(item.get("tool", "WEB_SEARCH"))
        if tool not in _ALLOWED_TOOLS:
            tool = "WEB_SEARCH"
        node = ArgumentNode(
            id=nid,
            speaker=speaker,
            content=content,
            tool_type=tool,
            verification_cost=_safe_cost(tool),
        )
        node.is_verified = False
        af.add_node(node)

    nodes_payload = [
        {"id": nid, "speaker": af.nodes[nid].speaker, "content": af.nodes[nid].content, "tool": af.nodes[nid].tool_type}
        for nid in sorted(af.nodes.keys())
    ]

    # Relations
    try:
        rel_prompt = _build_rel_prompt(nodes_payload)
        rel_data = _chat_json(
            [
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": rel_prompt},
            ]
        )
    except Exception:
        rel_data = {}

    relations = rel_data.get("relations", []) if isinstance(rel_data, dict) else []
    if not isinstance(relations, list):
        relations = []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        src = str(rel.get("from", "")).strip()
        dst = str(rel.get("to", "")).strip()
        rtype = str(rel.get("type", "")).strip().lower()
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

    # Orphan attach
    orphan_ids = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
    if orphan_ids:
        orphan_payload = [{"id": nid, "content": af.nodes[nid].content} for nid in orphan_ids]
        try:
            orphan_prompt = _build_orphan_prompt(orphan_payload, nodes_payload)
            orphan_data = _chat_json(
                [
                    {"role": "system", "content": "You output strictly valid JSON only."},
                    {"role": "user", "content": orphan_prompt},
                ]
            )
        except Exception:
            orphan_data = {}
        orphan_rels = orphan_data.get("relations", []) if isinstance(orphan_data, dict) else []
        if not isinstance(orphan_rels, list):
            orphan_rels = []
        for rel in orphan_rels:
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("from", "")).strip()
            dst = str(rel.get("to", "")).strip()
            rtype = str(rel.get("type", "")).strip().lower()
            if rtype not in _ALLOWED_REL_TYPES:
                continue
            if src not in af.nodes or dst not in af.nodes:
                continue
            if src == dst:
                continue
            if af.nx_graph.has_edge(src, dst):
                continue
            if rtype == "attack":
                af.add_attack(src, dst)
            else:
                af.add_support(src, dst)

    # Edge audit (all edges)
    edges_payload = [
        {"from": u, "to": v, "type": d.get("type")}
        for u, v, d in af.nx_graph.edges(data=True)
    ]
    if edges_payload:
        try:
            audit_prompt = _build_edge_audit_prompt(nodes_payload, edges_payload)
            audit_data = _chat_json(
                [
                    {"role": "system", "content": "You output strictly valid JSON only."},
                    {"role": "user", "content": audit_prompt},
                ]
            )
        except Exception:
            audit_data = {}
        results = audit_data.get("results", []) if isinstance(audit_data, dict) else []
        if isinstance(results, list):
            for res in results:
                if not isinstance(res, dict):
                    continue
                src = str(res.get("from", "")).strip()
                dst = str(res.get("to", "")).strip()
                fix = str(res.get("fix", "keep")).strip().lower()
                if not src or not dst:
                    continue
                if not af.nx_graph.has_edge(src, dst):
                    continue
                if fix == "flip":
                    cur = af.nx_graph[src][dst].get("type")
                    af.nx_graph[src][dst]["type"] = "attack" if cur == "support" else "support"
                elif fix == "remove":
                    af.nx_graph.remove_edge(src, dst)

    _sanitize_graph(af)
    af.root_id_override = "A1"
    return af
