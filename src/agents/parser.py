import json
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

_ALLOWED_TOOLS = {"WEB_SEARCH", "PYTHON_EXEC", "WIKIPEDIA", "COMMON_SENSE", "AUTO"}
_ALLOWED_REL_TYPES = {"attack", "support"}


def _norm_tool(x: Any) -> str:
    s = str(x or "").strip().upper().replace(" ", "_")
    if s in _ALLOWED_TOOLS:
        return s
    # common aliases
    if "SEARCH" in s:
        return "WEB_SEARCH"
    if "PYTHON" in s:
        return "PYTHON_EXEC"
    if "WIKI" in s:
        return "WIKIPEDIA"
    if s == "":
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


def _dedupe_ids(args: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Ensure ids are unique.
    Returns (new_args, id_map_old_to_new).
    """
    seen = set()
    id_map: Dict[str, str] = {}
    new_args: List[Dict[str, Any]] = []

    for item in args:
        old = str(item.get("id", "")).strip()
        if not old:
            continue

        new_id = old
        if new_id in seen:
            # make unique by suffixing _k
            k = 2
            while f"{old}_{k}" in seen:
                k += 1
            new_id = f"{old}_{k}"

        seen.add(new_id)
        id_map[old] = new_id
        item2 = dict(item)
        item2["id"] = new_id
        new_args.append(item2)

    return new_args, id_map


def parse_debate(text: str) -> ArgumentationGraph:
    prompt = f"""
ROLE: Expert Logic & Argumentation Analyst.
TASK: Convert the provided debate into a High-Granularity Atomic Typed Argumentation Graph.

REQUIREMENTS:
- Output MUST be strictly valid JSON with exactly two keys: "arguments" and "relations".
- MUST include a root claim with id "A1" that represents the debate topic claim.
- All argument ids MUST be unique strings.
- Each relation must satisfy:
  - "from" and "to" reference existing argument ids
  - "type" is exactly "attack" or "support" (lowercase)

DECOMPOSITION RULES:
1) ATOMICITY: Break each speaker's turn into single verifiable factual claims.
2) KEYSTONE IDENTIFICATION: Extract key assumptions (numbers, laws, definitions) as separate nodes.
3) TARGET MAPPING: Claims should ultimately connect to A1 by support or attack when appropriate.

TOOL ASSIGNMENT (field name: "tool"):
- "PYTHON_EXEC": explicit arithmetic, unit conversions, simple deterministic logic.
- "WEB_SEARCH": open-domain factual claims needing retrieval.
- "WIKIPEDIA": canonical definitions / entities (optional).
- "COMMON_SENSE": obvious everyday truths.
- If unsure, output "WEB_SEARCH".

JSON OUTPUT FORMAT EXAMPLE:
{{
  "arguments": [
    {{"id": "A1", "speaker": "Moderator", "content": "TOPIC CLAIM HERE", "tool": "WEB_SEARCH"}},
    {{"id": "A2", "speaker": "Alice", "content": "Some supporting fact", "tool": "WEB_SEARCH"}}
  ],
  "relations": [
    {{"from": "A2", "to": "A1", "type": "support"}}
  ]
}}

DEBATE TRANSCRIPT:
{text}
"""

    try:
        res = client.chat.completions.create(
            model=PARSER_MODEL,
            messages=[
                {"role": "system", "content": "You output strictly valid JSON for atomic claim extraction."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(res.choices[0].message.content)
    except Exception:
        # fail-safe: return minimal graph with A1
        af = ArgumentationGraph()
        af.add_node(
            ArgumentNode(
                id="A1",
                speaker="Moderator",
                content=str(text).strip()[:500] if text else "Topic claim.",
                tool_type="WEB_SEARCH",
                verification_cost=_safe_cost("WEB_SEARCH"),
            )
        )
        return af

    args = data.get("arguments", [])
    rels = data.get("relations", [])

    if not isinstance(args, list):
        args = []
    if not isinstance(rels, list):
        rels = []

    # Ensure unique ids
    args, id_map = _dedupe_ids([a for a in args if isinstance(a, dict)])

    # Build graph nodes
    af = ArgumentationGraph()

    # Make sure A1 exists; if missing, synthesize it from transcript header
    has_a1 = any(str(a.get("id", "")).strip() == "A1" for a in args)
    if not has_a1:
        args = [{
            "id": "A1",
            "speaker": "Moderator",
            "content": "Debate topic claim (root).",
            "tool": "WEB_SEARCH",
        }] + args
        # refresh id_map after insertion (A1 unique anyway)
        args, id_map = _dedupe_ids(args)

    for item in args:
        nid = str(item.get("id", "")).strip()
        if not nid:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue

        speaker = str(item.get("speaker", "UNK") or "UNK").strip()
        tool = _norm_tool(item.get("tool", "WEB_SEARCH"))
        cost = _safe_cost(tool)

        node = ArgumentNode(
            id=nid,
            content=content,
            speaker=speaker,
            tool_type=tool,
            verification_cost=cost,
        )
        node.is_verified = False
        af.add_node(node)

    # Add relations with validation + id remap
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

    return af
