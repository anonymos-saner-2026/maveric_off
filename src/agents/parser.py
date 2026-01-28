# src/agents/parser.py
# LLM-first atomic-claim parser for MaVERiC with self-refine loops.
#
# Pipeline:
#   Stage A1 (LLM Parse): transcript -> candidate arguments (no relations)
#   Stage A2 (Atomic Split): enforce one-claim-per-node (deterministic/LLM/hybrid)
#   Stage A3 (LLM Relations): atomic arguments -> typed relations (attack/support)
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
from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, PARSER_MODEL, TOOLS_CONFIG
from src.core.graph import ArgumentationGraph, ArgumentNode
from src.agents.deduplication import merge_redundant_nodes

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

_ALLOWED_TOOLS = {"WEB_SEARCH", "PYTHON_EXEC", "WIKIPEDIA", "COMMON_SENSE", "AUTO"}
_ALLOWED_REL_TYPES = {"attack", "support"}

# Self-refine knobs
REFINE_ITERS = 4
REFINE_MAX_OPS = 24  # safeguard: cap number of ops we apply per iteration


@dataclass
class ParserConfig:
    mode: str = "default"
    enable_topic_linker: bool = True
    enable_orphan_cluster_llm: bool = True
    enable_orphan_global_llm: bool = True
    enable_orphan_deterministic: bool = True
    enable_orphan_root_fallback: bool = True
    enable_component_link: bool = True
    enable_edge_sanity: bool = True
    enable_attack_balance: bool = True
    enable_edge_audit: bool = True
    enable_dedup: bool = True
    enable_self_refine: bool = True
    refine_iters_override: Optional[int] = None
    split_mode_override: Optional[str] = None


def get_parser_config(mode: str) -> ParserConfig:
    mode = (mode or "default").strip().lower()
    if mode == "paper_safe":
        return ParserConfig(
            mode=mode,
            enable_topic_linker=True,
            enable_orphan_cluster_llm=False,
            enable_orphan_global_llm=False,
            enable_orphan_deterministic=False,
            enable_orphan_root_fallback=False,
            enable_component_link=False,
            enable_edge_sanity=False,
            enable_attack_balance=False,
            enable_edge_audit=False,
            enable_dedup=True,
            enable_self_refine=True,
            refine_iters_override=1,
        )
    if mode == "fast":
        return ParserConfig(
            mode=mode,
            enable_topic_linker=True,
            enable_orphan_cluster_llm=False,
            enable_orphan_global_llm=False,
            enable_orphan_deterministic=True,
            enable_orphan_root_fallback=True,
            enable_component_link=False,
            enable_edge_sanity=False,
            enable_attack_balance=False,
            enable_edge_audit=False,
            enable_dedup=True,
            enable_self_refine=False,
            refine_iters_override=0,
            split_mode_override="deterministic",
        )
    return ParserConfig(mode="default")


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


def _normalize_token(token: str) -> str:
    token = token.lower()
    synonyms = {
        "costs": "cost",
        "costing": "cost",
        "subsidies": "subsidy",
        "subsidized": "subsidy",
        "subsidise": "subsidy",
        "subsidize": "subsidy",
        "batteries": "battery",
        "jobs": "job",
        "emissions": "emission",
        "renewables": "renewable",
        "reactors": "reactor",
        "studies": "study",
        "teens": "teen",
        "tax": "subsidy",
        "credits": "credit",
        "credit": "credit",
        "pollution": "emission",
        "greenhouse": "emission",
        "safety": "risk",
        "danger": "risk",
        "hazard": "risk",
        "toxic": "risk",
        "costly": "cost",
        "expensive": "cost",
        "cheap": "cost",
        "price": "cost",
        "prices": "cost",
        "subsidized": "subsidy",
        "subsidized": "subsidy",
        "reskilling": "training",
        "training": "training",
        "jobs": "job",
        "workers": "worker",
        "teens": "teen",
        "adolescents": "teen",
    }
    token = synonyms.get(token, token)
    for suffix in ("ing", "ed", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)]
            break
    return token


def _tokenize(s: str) -> List[str]:
    s2 = _norm_text(s)
    toks = re.findall(r"[a-z0-9%]+", s2)
    normalized = [_normalize_token(t) for t in toks if t and t not in _STOPWORDS]
    return [t for t in normalized if t]


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
# Atomic claim enforcement
# -----------------------------
@dataclass(frozen=True)
class _ClauseSplit:
    text: str
    connector: Optional[str]


@dataclass(frozen=True)
class _ClauseLink:
    child_idx: int
    parent_idx: int
    connector: Optional[str]
    direction: str
    edge_type: str


_SPLIT_CONNECTORS = {
    "because",
    "since",
    "as",
    "so",
    "therefore",
    "thus",
    "hence",
    "but",
    "however",
    "although",
    "though",
    "while",
    "whereas",
    "and",
    "or",
}

_CAUSE_CONNECTORS = {"because", "since", "as"}
_CONSEQUENCE_CONNECTORS = {"therefore", "thus", "hence", "so"}
_CONTRAST_CONNECTORS = {"but", "however", "although", "though", "while", "whereas"}
_DISJUNCT_CONNECTORS = {"or"}

_LLM_SPLIT_MAX_CLAUSES = 8

_NEGATION_WORDS = {"no", "not", "never", "cannot", "can't", "without", "none", "neither", "nor"}
_STRONG_TOPIC_TERMS = {
    "nuclear", "solar", "wind", "waste", "reactor", "safety", "risk", "cost", "subsidy",
    "emission", "hydrogen", "battery", "ai", "job", "automation",
    "social", "media", "mental", "health", "teen", "anxiety", "depression",
    "diet", "protein", "fiber", "remote", "work", "productivity",
    "commute", "collaboration", "hybrid", "burnout", "isolation", "morale",
}

_RELATIVE_CONNECTORS = {"which", "that", "who", "whom", "whose", "where", "when"}
_RELATIVE_CLAUSE_RE = re.compile(r"^(.*?),\s*(which|that|who|whom|whose|where|when)\s+(.*)$", re.IGNORECASE)
_THAT_CLAUSE_RE = re.compile(r"^(.*?)\s+that\s+(.+)$", re.IGNORECASE)
_NOT_ONLY_RE = re.compile(r"^not only (.+?) but also (.+)$", re.IGNORECASE)

_PUNCT_SPLIT = re.compile(r"[;:\n]+")


def _normalize_clause_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if cleaned.endswith(","):
        cleaned = cleaned[:-1].strip()
    return cleaned


def _is_sentence_like(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"[A-Za-z0-9%]+", text)
    if len(tokens) < 3:
        return False
    return True


def _verb_count(text: str) -> int:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if not tokens:
        return 0
    verb_markers = {
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    }
    count = 0
    for tok in tokens:
        if tok in verb_markers or tok.endswith("ed") or tok.endswith("ing"):
            count += 1
    return count


def _split_relative_clause(text: str) -> Optional[Tuple[str, str, str]]:
    match = _RELATIVE_CLAUSE_RE.match(text)
    if match:
        head = _normalize_clause_text(match.group(1))
        rel = _normalize_clause_text(match.group(3))
        connector = match.group(2).lower()
        if head and rel:
            return head, rel, connector

    that_match = _THAT_CLAUSE_RE.match(text)
    if that_match:
        head = _normalize_clause_text(that_match.group(1))
        rel = _normalize_clause_text(that_match.group(2))
        if head and rel:
            return head, rel, "that"

    return None


def _split_conjunction_chain(text: str) -> List[str]:
    if " and " not in text.lower() and " or " not in text.lower():
        return [text]
    not_only_match = _NOT_ONLY_RE.match(text.strip())
    if not_only_match:
        left = _normalize_clause_text(not_only_match.group(1))
        right = _normalize_clause_text(not_only_match.group(2))
        if left and right:
            return [left, right]
    tokens = re.split(r"\b(and|or)\b", text, flags=re.IGNORECASE)
    if len(tokens) < 3:
        return [text]
    parts = []
    buffer = ""
    for tok in tokens:
        if tok.strip().lower() in {"and", "or"}:
            if buffer.strip():
                parts.append(_normalize_clause_text(buffer))
                buffer = ""
            continue
        buffer = f"{buffer} {tok}" if buffer else tok
    if buffer.strip():
        parts.append(_normalize_clause_text(buffer))
    parts = [p for p in parts if p]
    if len(parts) <= 1:
        return [text]
    verb_count = sum(_verb_count(p) for p in parts)
    if verb_count <= 1:
        long_parts = [p for p in parts if len(p.split()) >= 6]
        if len(long_parts) <= 1:
            return [text]
    return parts


def _split_by_connectors(text: str) -> List[_ClauseSplit]:
    rel_split = _split_relative_clause(text)
    if rel_split:
        head, rel, connector = rel_split
        return [_ClauseSplit(text=head, connector=None), _ClauseSplit(text=rel, connector=connector)]

    tokens = re.split(r"\s+", text)
    if len(tokens) < 7:
        return [_ClauseSplit(text=text, connector=None)]

    parts: List[_ClauseSplit] = []
    buffer: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        low = tok.lower().strip(",;:()")
        if low in _SPLIT_CONNECTORS and buffer:
            clause = _normalize_clause_text(" ".join(buffer))
            if clause:
                parts.append(_ClauseSplit(text=clause, connector=low))
            buffer = []
            i += 1
            continue
        buffer.append(tok)
        i += 1

    last_clause = _normalize_clause_text(" ".join(buffer))
    if last_clause:
        parts.append(_ClauseSplit(text=last_clause, connector=None))

    if len(parts) <= 1:
        conj_parts = _split_conjunction_chain(text)
        if len(conj_parts) > 1:
            return [_ClauseSplit(text=p, connector="and") for p in conj_parts]
        return [_ClauseSplit(text=text, connector=None)]

    # If the split results are too small, keep original text
    tiny = [p for p in parts if len(re.findall(r"\w+", p.text)) < 3]
    if tiny:
        return [_ClauseSplit(text=text, connector=None)]

    return parts


def _split_atomic_claims(text: str) -> Tuple[List[str], List[_ClauseLink]]:
    if not text:
        return [], []

    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return [], []

    rough_segments = [seg for seg in _PUNCT_SPLIT.split(cleaned) if seg.strip()]
    clauses: List[str] = []
    links: List[_ClauseLink] = []

    for seg in rough_segments:
        seg = seg.strip()
        if not seg:
            continue
        splits = _split_by_connectors(seg)
        if len(splits) == 1:
            clause = _normalize_clause_text(splits[0].text)
            if _is_sentence_like(clause):
                clauses.append(clause)
            continue

        start_idx = len(clauses)
        valid_parts: List[_ClauseSplit] = []
        for part in splits:
            clause = _normalize_clause_text(part.text)
            if not _is_sentence_like(clause):
                continue
            valid_parts.append(_ClauseSplit(text=clause, connector=part.connector))
            clauses.append(clause)
        end_idx = len(clauses)

        for rel_idx in range(start_idx + 1, end_idx):
            connector = valid_parts[rel_idx - start_idx].connector
            direction = "support"
            if connector in _CONSEQUENCE_CONNECTORS:
                direction = "reverse"
            elif connector in _CAUSE_CONNECTORS:
                direction = "support"
            edge_type = "support"
            if connector in _CONTRAST_CONNECTORS or connector in _DISJUNCT_CONNECTORS:
                edge_type = "attack"
            links.append(
                _ClauseLink(
                    child_idx=rel_idx,
                    parent_idx=start_idx,
                    connector=connector,
                    direction=direction,
                    edge_type=edge_type,
                )
            )

    if not clauses:
        return [cleaned], []

    # If too many very short clauses, keep the original text as a single claim.
    short_clauses = [c for c in clauses if len(re.findall(r"\w+", c)) < 4]
    if len(short_clauses) >= max(1, len(clauses) // 2):
        return [cleaned], []

    return clauses, links


def _needs_atomic_split(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"\w+", text)
    # Check for pronouns that need resolution regardless of length
    if re.search(r"\b(it|they|he|she|this|that|these|those)\b", text, re.IGNORECASE):
        return True
    if len(tokens) < 10:
        return False
    if _PUNCT_SPLIT.search(text):
        return True
    if re.search(r"\b(and|or|because|since|although|though|while|whereas|but|however|therefore|thus|hence|so|if|which|that|who|whom|whose|where|when)\b", text, re.IGNORECASE):
        return True
    if _verb_count(text) >= 2 and ("," in text or " and " in text.lower() or " or " in text.lower()):
        return True
    return False


def _atomicity_violations(af: ArgumentationGraph) -> List[str]:
    offenders: List[str] = []
    for nid, node in af.nodes.items():
        if nid == "A1":
            continue
        if _needs_atomic_split(node.content):
            offenders.append(nid)
    return offenders


def _derive_tool_for_clause(parent_tool: str, clause_text: str) -> str:
    tool = _norm_tool(parent_tool)
    if tool == "COMMON_SENSE":
        return tool
    clause_lower = clause_text.lower()
    if re.search(r"(-?\d+)\s*[\+\-\*/]\s*(-?\d+)", clause_lower) and ("=" in clause_lower or "equal" in clause_lower):
        return "PYTHON_EXEC"
    if "leap year" in clause_lower or "sqrt" in clause_lower or "square root" in clause_lower:
        return "PYTHON_EXEC"
    return tool or "WEB_SEARCH"


def _merge_split_results(
    base: Tuple[List[str], List[_ClauseLink]],
    fallback: Tuple[List[str], List[_ClauseLink]],
) -> Tuple[List[str], List[_ClauseLink]]:
    base_clauses, base_links = base
    fb_clauses, fb_links = fallback
    if not base_clauses:
        return fallback
    if len(base_clauses) > _LLM_SPLIT_MAX_CLAUSES:
        return fallback
    if any(len(re.findall(r"\w+", c)) < 3 for c in base_clauses):
        return fallback
    return base_clauses, base_links


def _llm_split_atomic_claims(text: str, context: str = "") -> Tuple[List[str], List[_ClauseLink]]:
    if not text:
        return [], []


    context_block = ""
    if context:
        context_block = f"\nCONTEXT (Preceding discussion):\n{context}\n"

    prompt = f"""
Role: Atomic Claim Splitter.
Task: Split the input into minimal atomic factual claims. Do not drop meaning.
CRITICAL: Resolve pronouns (it, they, he, she, that) using the CONTEXT provided. Node content MUST be standalone verifiable.

Rules:
- Use 0-based indices into clauses.
- direction is "support" (child -> parent) or "reverse" (parent -> child).
- edge_type is "support" or "attack".
- Use connector for causal words (because, therefore, but, however, or, etc.).
- Keep clauses short and standalone.
- REWRITE pronouns (it, they) with specific nouns, EVEN IF referring to the same sentence.
- FORBIDDEN: "It destroys...", "They are..." (Must be "The meltdown destroys...", "Batteries are...")

Examples:
Input: "Solar is good, but it is expensive."
Output: [
  {{"content": "Solar is good."}},
  {{"content": "Solar is expensive.", "link": {{"target": 0, "type": "support", "connector": "but"}}}}
]

Input: "If a reactor melts, it irradiates the city."
Output: [
  {{"content": "If a reactor melts, the reactor irradiates the city."}}
]

{context_block}

CRITICAL FINAL INSTRUCTION:
ABSOLUTELY NO PRONOUNS ("it", "they"). REPEAT THE NOUN.
Input: "If a meltdown happens, it destroys..." -> Output: "If a meltdown happens, the meltdown destroys..."

Advanced Splitting for Conjunctions:
Input: "Coffee improves alertness because caffeine blocks receptors and increases dopamine."
Output: [
  {{"content": "Coffee improves alertness."}},
  {{"content": "Caffeine blocks receptors.", "link": {{"target": 0, "type": "support", "connector": "because"}}}},
  {{"content": "Caffeine increases dopamine.", "link": {{"target": 0, "type": "support", "connector": "and"}}}}
]

Before finalizing, review your list of claims. If 'it' or 'they' remains, REWRITE the sentence to be fully standalone.
"""
    try:
        data = _chat_json(
            [
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": prompt + "\nINPUT:\n" + text},
            ]
        )
    except Exception:
        return _split_atomic_claims(text)

    clauses = data.get("clauses", [])
    links = data.get("links", [])
    if not isinstance(clauses, list) or not clauses:
        return _split_atomic_claims(text)

    cleaned_clauses: List[str] = []
    for clause in clauses:
        clause_text = _normalize_clause_text(str(clause))
        if _is_sentence_like(clause_text):
            cleaned_clauses.append(clause_text)

    if not cleaned_clauses:
        return _split_atomic_claims(text)

    built_links: List[_ClauseLink] = []
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            child_raw = link.get("child_idx", None)
            parent_raw = link.get("parent_idx", None)
            if child_raw is None or parent_raw is None:
                continue
            try:
                child_idx = int(child_raw)
                parent_idx = int(parent_raw)
            except Exception:
                continue
            connector = link.get("connector", None)
            direction = str(link.get("direction", "support")).strip().lower()
            edge_type = str(link.get("edge_type", "support")).strip().lower()
            if direction not in {"support", "reverse"}:
                direction = "support"
            if edge_type not in _ALLOWED_REL_TYPES:
                edge_type = "support"
            if child_idx < 0 or parent_idx < 0:
                continue
            if child_idx >= len(cleaned_clauses) or parent_idx >= len(cleaned_clauses):
                continue
            built_links.append(
                _ClauseLink(
                    child_idx=child_idx,
                    parent_idx=parent_idx,
                    connector=str(connector) if connector else None,
                    direction=direction,
                    edge_type=edge_type,
                )
            )

    return cleaned_clauses, built_links


def _repair_pronouns(clauses: List[str], context: str) -> List[str]:
    """
    Identifies clauses with leftover pronouns and asks LLM to fix them.
    Used as a guardrail for the atomic splitter.
    """
    targets = []
    for i, c in enumerate(clauses):
        if re.search(r"\b(it|they|them|their|his|her|these|those)\b", c, re.IGNORECASE):
            targets.append(i)

    if not targets:
        return clauses

    prompt = f"""
ROLE: Coreference Resolution Specialist.
TASK: Rewrite the following specific claims to be 100% standalone by replacing ALL pronouns with the correct nouns from CONTEXT.

CONTEXT:
{context}

CLAIMS TO FIX:
"""
    for idx in targets:
        prompt += f"[{idx}] {clauses[idx]}\n"

    prompt += "\nOutput JSON: {\"fixed_claims\": {\"0\": \"Rewritten claim 0\", ...}}"

    try:
        data = _chat_json([{"role": "user", "content": prompt}])
        fixed = data.get("fixed_claims", {})
        new_clauses = list(clauses)
        for k, v in fixed.items():
            try:
                ki = int(k)
                if 0 <= ki < len(new_clauses):
                    new_clauses[ki] = str(v).strip()
            except ValueError:
                continue
        return new_clauses
    except Exception:
        return clauses


def _repair_root_pronouns(root_text: str, context: str) -> str:
    if not root_text:
        return root_text
    if not re.search(r"\b(it|they|them|their|his|her|these|those)\b", root_text, re.IGNORECASE):
        return root_text

    prompt = f"""
ROLE: Coreference Resolution Specialist.
TASK: Rewrite the root claim to be standalone by replacing ALL pronouns with the correct nouns from CONTEXT.

RULES:
- Do NOT change the meaning.
- Replace pronouns with explicit noun phrases.

CONTEXT:
{context}

ROOT CLAIM:
{root_text}

Output JSON: {{"fixed_root": "Rewritten root claim"}}
"""

    try:
        data = _chat_json([{"role": "user", "content": prompt}])
        fixed = data.get("fixed_root", "")
        return str(fixed).strip() or root_text
    except Exception:
        return root_text


def _split_atomic_claims_with_mode(text: str, mode: str, context: str = "") -> Tuple[List[str], List[_ClauseLink]]:
    if mode == "llm":
        clauses, links = _llm_split_atomic_claims(text, context)
        # Apply Pronoun Guardrail
        clauses = _repair_pronouns(clauses, context or text)
        return clauses, links
    if mode == "hybrid":
        llm_result = _llm_split_atomic_claims(text, context)
        # Apply Pronoun Guardrail to LLM part
        llm_clauses, llm_links = llm_result
        llm_clauses = _repair_pronouns(llm_clauses, context or text)
        llm_result = (llm_clauses, llm_links)
        
        deterministic = _split_atomic_claims(text)
        return _merge_split_results(llm_result, deterministic)
    return _split_atomic_claims(text)


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
# Prompting: Stage A1 (Parse candidates)
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
  ]
}
"""

    instructions = r"""
ROLE: Argument Graph Parser.

OUTPUT CONTRACT (MUST FOLLOW):
- Output strictly valid JSON with exactly key: "arguments".
- arguments: list of {id, speaker, content, tool}
- Every id unique.
- MUST include root node id="A1".
- A1 must be a declarative affirmative main proposition (not a question).

ATOMICITY:
- Prefer one clear claim per node; if two distinct claims exist, split them.
- Avoid tiny fragments that are not standalone claims.
- Rewrite pronouns to be standalone when possible.

TOOL:
- PYTHON_EXEC: deterministic math/equations/unit conversion.
- COMMON_SENSE: everyday physical truths.
- WEB_SEARCH: open-domain factual claims.
- If unsure, WEB_SEARCH.

IMPORTANT:
- Provide only candidate claims; relations will be built later.
"""

    return f"{instructions}\n\n{few_shot}\n\nTRANSCRIPT:\n{transcript}\n"


# -----------------------------
# Prompting: Stage A3 (Relations)
# -----------------------------
def _build_stageA_rel_prompt(root_id: str, turns_data: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Relation Builder for Argumentation Graphs.

You receive a CHRONOLOGICAL CONVERSATION (Turns):
- Each turn has a speaker and a list of claims.
- Claims are atomic facts extracted from the turn.

TASK:
Return JSON {"relations":[...]} linking claims.
- {"from": ID, "to": ID, "type": "attack|support"}

TEMPORAL LOGIC (CRITICAL):
- **Response Edges**: Claims in Turn N usually target claims in Turn N-1 or N-2.
- **Extended Context**: If a claim clearly references a concept from earlier turns, link to that earlier claim.
- **Flow**: Attack/Support usually points BACKWARDS in time (from Responder to Original Claim).

RELAXED COVERAGE & IMPLIED SUPPORT:
- **IMPLIED SUPPORT**: If a claim elaborates, continues a thought, or provides background/context for a prior claim, LINK IT as "support".
- **SEMANTIC RELEVANCE**: Exact keyword overlap is NOT required. If the topics are related, link them.
- **THREAD LOGIC**: If A2 follows A1 in the same turn or response and doesn't explicitly disagree, it likely supports A1.
- **BE GENEROUS**: Prefer creating a "support" edge over leaving a node disconnected (orphan).
- For contradictions, choose "attack"; for evidence/elaboration/context, choose "support".

FEW-SHOT (Attachment):
Input claims:
- A1: "Solar is cost-effective."
- A2: "Solar prices have dropped 80% in 10 years."
- A3: "Batteries are still expensive."
Output relations:
- {"from": "A2", "to": "A1", "type": "support"}
- {"from": "A3", "to": "A1", "type": "attack"}

CONSTRAINTS:
- Only use IDs provided.
- No self-loops.
- type in ["attack", "support"].

RELATION TYPES:
- support: Justification, evidence, agreement, or elaboration.
- attack: Disagreement, correction, counter-evidence, or undercutting.
"""

    payload = {
        "root_id": root_id,
        "conversation_turns": turns_data,
    }
    return rubric + "\n\nINPUT_CONVERSATION_JSON:\n" + json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Prompting: Orphan attachment
# -----------------------------
def _build_orphan_attach_prompt(root_id: str, orphan_claims: List[Dict[str, Any]], candidates: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Orphan Edge Attachment Assistant.

You receive:
- root_id
- orphan_claims: claims with no incoming edges
- candidate_claims: prior claims that can be linked to

TASK:
Return JSON {"relations":[...]} where each relation is:
- {"from": ID, "to": ID, "type": "attack|support"}

RULES:
- Use only provided IDs.
- No self-loops.
- Only add edges FROM orphan_claims.
- Prefer attaching to the most semantically relevant prior claim (avoid root unless it is the best match).
- Use "attack" if the orphan contradicts or negates the candidate.
- Use "support" if it reinforces or provides evidence.
- If no reasonable link exists, omit that orphan.
- Allow 1-2 edges per orphan when two distinct matches exist.

FEW-SHOT:
Orphan: "Batteries are still expensive."
Candidates: ["Solar is cost-effective.", "Solar prices have dropped 80%"]
Output:
- {"from": "A3", "to": "A1", "type": "attack"}
"""

    payload = {
        "root_id": root_id,
        "orphan_claims": orphan_claims,
        "candidate_claims": candidates,
    }
    return rubric + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


def _build_orphan_force_prompt(root_id: str, orphan_tasks: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Semantic Connector for Argumentation Graphs (Orphan Rescue).

You receive:
- root_id
- orphan_tasks: each includes the orphan claim and a shortlist of candidates

TASK:
For each orphan, select the BEST SINGLE parent from its candidates.
Return JSON {"relations":[...]} linking orphan -> parent.

CRITICAL INSTRUCTIONS:
- **ORPHANS ARE BAD**: Your goal is to eliminate orphans.
- **BEST POSSIBLE MATCH**: Even if the connection is weak (broad topical relevance), LINK IT to the best candidate.
- **DEFAULT TO ROOT**: If no specific candidate is relevant, link to the root (Topic Claim) with "support".
- **USE "SUPPORT" FOR RELATED TOPICS**: If the orphan discusses a related concept but doesn't strictly support/attack, use "support" as a "related to" link.
- NO ORPHAN LEFT BEHIND.
"""

    payload = {
        "root_id": root_id,
        "orphan_tasks": orphan_tasks,
    }
    return rubric + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


# -----------------------------
# Prompting: Edge sanity check
# -----------------------------
def _build_edge_sanity_prompt(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Edge Sanity Checker.

You receive:
- nodes: list of nodes with {id, content}
- edges: list of edges with {from, to, type}

TASK:
Return JSON {"ops": [...]} where each op is one of:
- {"op": "flip", "from": ID, "to": ID}
- {"op": "remove", "from": ID, "to": ID}
- {"op": "keep", "from": ID, "to": ID}

GUIDELINES:
- Flip if the current edge type is clearly wrong (negation mismatch, opposite stance, contradiction).
- Remove if the edge is unrelated or unjustified.
- Keep if the relation makes sense.
- If unsure, keep the edge (do not over-delete).

FEW-SHOT:
Nodes:
- A1: "High-protein diets are safe for most adults."
- A2: "High-protein diets strain kidneys in people with kidney disease."
Edge: {"from":"A2","to":"A1","type":"support"}
Output op: {"op":"flip","from":"A2","to":"A1"}

Nodes:
- B1: "Remote work reduces commute time."
- B2: "Reducing commute time improves focus."
Edge: {"from":"B2","to":"B1","type":"support"}
Output op: {"op":"keep","from":"B2","to":"B1"}

Nodes:
- C1: "AI will eliminate more jobs than it creates."
- C2: "History shows net job growth over time."
Edge: {"from":"C2","to":"C1","type":"support"}
Output op: {"op":"flip","from":"C2","to":"C1"}

Nodes:
- D1: "Solar is cost-effective."
- D2: "Solar prices dropped 80%."
Edge: {"from":"D2","to":"D1","type":"support"}
Output op: {"op":"keep","from":"D2","to":"D1"}

CONSTRAINTS:
- Only reference IDs provided.
- Do not invent nodes or edges.
"""

    payload = {"nodes": nodes, "edges": edges}
    return rubric + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


def _build_edge_audit_prompt(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    rubric = r"""
ROLE: Edge Semantic Auditor.

You receive:
- nodes: list of nodes with {id, content}
- edges: list of edges with {from, to, type}

TASK:
Return JSON {"results": [...]} where each result is:
- {"from": ID, "to": ID, "label": "correct|incorrect", "fix": "keep|flip|remove"}

GUIDELINES:
- Use "flip" if the edge type is wrong but a relationship exists.
- Use "remove" if the edge is unrelated or misleading.
- Use "keep" if the edge is correct.

FEW-SHOT:
Nodes:
- N1: "High-protein diets are safe for most adults."
- N2: "High-protein diets strain kidneys in people with kidney disease."
Edge: {"from":"N2","to":"N1","type":"support"}
Result: {"from":"N2","to":"N1","label":"incorrect","fix":"flip"}

Nodes:
- R1: "Remote work improves work-life balance."
- R2: "Remote work can blur boundaries and increase burnout."
Edge: {"from":"R2","to":"R1","type":"support"}
Result: {"from":"R2","to":"R1","label":"incorrect","fix":"flip"}

Nodes:
- S1: "Social media harms teen mental health."
- S2: "Some studies find no clear causal link between social media and mental health."
Edge: {"from":"S2","to":"S1","type":"support"}
Result: {"from":"S2","to":"S1","label":"incorrect","fix":"flip"}

Nodes:
- J1: "AI will eliminate more jobs than it creates."
- J2: "History shows net job growth over time."
Edge: {"from":"J2","to":"J1","type":"support"}
Result: {"from":"J2","to":"J1","label":"incorrect","fix":"flip"}
"""

    payload = {"nodes": nodes, "edges": edges}
    return rubric + "\n\nINPUT_JSON:\n" + json.dumps(payload, ensure_ascii=False)


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
- **PRESERVE CONNECTIVITY**: Do NOT remove "weak" support edges if they are the only link for a node.
- **RESPECT 'RELATED' LINKS**: A 'support' edge can mean "topically related". Do not remove it just because it's not a strong logical proof.
- **Prefer modifying edge type over removing edges**.
- Only remove edges that are CLEARLY WRONG (e.g. backward, self-contradictory).
- Make MINIMAL edits that strictly improve correctness.

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
            raw = res.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_delay * (attempt + 1))
    if last_error:
        return {}
    return {}


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
# Quality gate
# -----------------------------
def quality_gate(
    af: ArgumentationGraph,
    orphan_ratio_max: float = 0.2,
    audit_incorrect_ratio_max: float = 0.2,
    max_wcc: int = 3,
    min_edge_density: float = 0.02,
    attack_ratio_min: float = 0.2,
    attack_ratio_max: float = 0.8,
    audit_incorrect_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    num_nodes = len(af.nodes)
    num_edges = af.nx_graph.number_of_edges()
    orphan_nodes = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
    orphan_ratio = (len(orphan_nodes) / num_nodes) if num_nodes else 0.0
    edge_density = (num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else 0.0
    attack_edges = sum(1 for _, _, d in af.nx_graph.edges(data=True) if d.get("type") == "attack")
    attack_ratio = (attack_edges / num_edges) if num_edges else 0.0

    try:
        import networkx as nx
        wcc = len(list(nx.weakly_connected_components(af.nx_graph)))
    except Exception:
        wcc = 0

    reasons = []
    if orphan_ratio > orphan_ratio_max:
        reasons.append(f"orphan_ratio={orphan_ratio:.2f} > {orphan_ratio_max:.2f}")
    if edge_density < min_edge_density:
        reasons.append(f"edge_density={edge_density:.3f} < {min_edge_density:.3f}")
    if wcc > max_wcc:
        reasons.append(f"wcc={wcc} > {max_wcc}")
    if attack_ratio < attack_ratio_min or attack_ratio > attack_ratio_max:
        reasons.append(f"attack_ratio={attack_ratio:.2f} outside [{attack_ratio_min:.2f},{attack_ratio_max:.2f}]")
    if audit_incorrect_ratio is not None and audit_incorrect_ratio > audit_incorrect_ratio_max:
        reasons.append(
            f"audit_incorrect_ratio={audit_incorrect_ratio:.2f} > {audit_incorrect_ratio_max:.2f}"
        )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "nodes": num_nodes,
            "edges": num_edges,
            "orphan_ratio": orphan_ratio,
            "edge_density": edge_density,
            "attack_ratio": attack_ratio,
            "wcc": wcc,
            "audit_incorrect_ratio": audit_incorrect_ratio,
        },
    }


# -----------------------------
# Public API
# -----------------------------
def parse_debate(
    text: str,
    split_mode: str = "hybrid",
    refine_iters: int = REFINE_ITERS,
    config: Optional[ParserConfig] = None,
    mode: str = "default",
) -> ArgumentationGraph:
    transcript = (text or "").strip()
    if config is None:
        config = get_parser_config(mode)
    split_mode_effective = config.split_mode_override or split_mode
    if config.enable_self_refine:
        refine_iters_effective = config.refine_iters_override if config.refine_iters_override is not None else refine_iters
    else:
        refine_iters_effective = 0

    # -------------------
    # Stage A1: Parse candidates
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
        af.claim = transcript or ""
        _ensure_root(af, transcript or "Topic claim.")
        af.root_id_override = "A1"
        return af

    args = dataA.get("arguments", [])
    if not isinstance(args, list):
        args = []

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
    af.claim = transcript or ""

    # nodes (enforce atomicity)
    split_id_map: Dict[str, List[str]] = {}
    split_link_map: Dict[str, List[Tuple[str, str, Optional[str], str, str]]] = {}
    used_ids: Set[str] = set()
    
    for i, item in enumerate(args):
        # build context from previous items
        ctx_list = []
        for back_i in range(max(0, i-2), i):
             prev = args[back_i]
             p_spk = prev.get("speaker", "unk")
             p_txt = prev.get("content", "")
             ctx_list.append(f"{p_spk}: {p_txt}")
        context_str = "\n".join(ctx_list)

        nid = str(item.get("id", "")).strip()
        content = str(item.get("content", "")).strip()
        if not nid or not content:
            continue

        speaker = str(item.get("speaker", "unk") or "unk").strip().lower()
        tool = _norm_tool(item.get("tool", "WEB_SEARCH"))

        if _needs_atomic_split(content):
            clauses, links = _split_atomic_claims_with_mode(content, split_mode_effective, context_str)
        else:
            clauses, links = [content], []

        # If we have exactly 1 clause, we use the original nid. 
        # But we MUST use the Repair results if available.
        if len(clauses) == 1 and nid != "A1":
            node = ArgumentNode(
                id=nid,
                speaker=speaker,
                content=clauses[0], # Using repaired clause
                tool_type=tool,
                verification_cost=_safe_cost(tool),
            )
            node.is_verified = False
            af.add_node(node)
            split_id_map[nid] = [nid]
            used_ids.add(nid)
            continue

        if nid == "A1":
            clauses = [content]
            links = []

        new_ids: List[str] = []
        for idx, clause in enumerate(clauses, start=1):
            if idx == 1:
                base_id = nid
            else:
                base_id = f"{nid}_{idx}"
            new_id = base_id
            suffix = 2
            while new_id in used_ids:
                new_id = f"{base_id}_{suffix}"
                suffix += 1
            used_ids.add(new_id)
            new_ids.append(new_id)
            clause_tool = _derive_tool_for_clause(tool, clause)
            node = ArgumentNode(
                id=new_id,
                speaker=speaker,
                content=clause,
                tool_type=clause_tool,
                verification_cost=_safe_cost(clause_tool),
            )
            node.is_verified = False
            af.add_node(node)

        split_id_map[nid] = new_ids
        clause_links: List[Tuple[str, str, Optional[str], str, str]] = []
        for link in links:
            if link.child_idx < len(new_ids) and link.parent_idx < len(new_ids):
                clause_links.append(
                    (
                        new_ids[link.child_idx],
                        new_ids[link.parent_idx],
                        link.connector,
                        link.direction,
                        link.edge_type,
                    )
                )
        if clause_links:
            split_link_map[nid] = clause_links

    _ensure_root(af, transcript or "Topic claim.")
    # Repair pronouns in root if needed
    if "A1" in af.nodes:
        af.nodes["A1"].content = _repair_root_pronouns(af.nodes["A1"].content, transcript)

    # Stage A3: relations among atomic nodes (Temporal-Aware)
    turns_data = []
    
    # Reconstruct turns from original args + split_id_map
    # This preserves the chronological flow for the LLM
    for item in args:
        orig_id = str(item.get("id", "")).strip()
        speaker = str(item.get("speaker", "unk")).strip()
        
        # Get atomic IDs derived from this turn
        derived_ids = split_id_map.get(orig_id, [])
        
        # Filter for nodes that still exist (post-dedup) and sort
        valid_ids = sorted(
            [nid for nid in derived_ids if nid in af.nodes],
            key=_id_num
        )
        
        if not valid_ids:
            continue
            
        turn_claims = []
        for nid in valid_ids:
            n = af.nodes[nid]
            turn_claims.append({
                "id": nid,
                "content": n.content,
                "tool": n.tool_type
            })
            
        turns_data.append({
            "speaker": speaker,
            "claims": turn_claims
        })

    try:
        prompt_rel = _build_stageA_rel_prompt("A1", turns_data)
        rel_data = _chat_json(
            [
                {"role": "system", "content": "You output strictly valid JSON only."},
                {"role": "user", "content": prompt_rel},
            ]
        )
    except Exception as e:
        rel_data = {"relations": []}

    rels = rel_data.get("relations", []) if isinstance(rel_data, dict) else []
    if not isinstance(rels, list):
        rels = []

    # edges
    for rel in rels:
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

    # Topic linker (heuristic) for unconnected nodes
    def _topic_tokens(text: str) -> Set[str]:
        return {t for t in _tokenize(text) if t not in _STOPWORDS}

    def _edge_type_guess(src_tokens: Set[str], dst_tokens: Set[str]) -> str:
        if not src_tokens or not dst_tokens:
            return "support"
        neg_src = bool(src_tokens & _NEGATION_WORDS)
        neg_dst = bool(dst_tokens & _NEGATION_WORDS)
        if neg_src != neg_dst:
            return "attack"
        return "support"

    node_ids = list(af.nodes.keys())
    node_tokens = {nid: _topic_tokens(af.nodes[nid].content) for nid in node_ids}
    root_tokens = node_tokens.get("A1", set())

    if config.enable_topic_linker:
        for nid in node_ids:
            if nid == "A1":
                continue
            if af.nx_graph.in_degree(nid) > 0:
                continue
            tokens = node_tokens.get(nid, set())
            if not tokens:
                continue
            candidates = []
            for other_id in node_ids:
                if other_id == nid:
                    continue
                if af.nx_graph.out_degree(other_id) == 0 and af.nx_graph.in_degree(other_id) == 0:
                    continue
                other_tokens = node_tokens.get(other_id, set())
                if not other_tokens:
                    continue
                overlap = len(tokens & other_tokens)
                if overlap >= 2:
                    candidates.append((overlap, other_id))
            if candidates:
                candidates.sort(reverse=True)
                best_overlap = candidates[0][0]
                best_targets = [cid for ov, cid in candidates if ov == best_overlap][:2]
                for target_id in best_targets:
                    edge_type = _edge_type_guess(tokens, node_tokens.get(target_id, set()))
                    if edge_type == "attack":
                        af.add_attack(nid, target_id)
                    else:
                        af.add_support(nid, target_id)
                continue

            # fallback: attach to root if topic overlap exists
            if root_tokens and len(tokens & root_tokens) >= 1:
                edge_type = _edge_type_guess(tokens, root_tokens)
                if edge_type == "attack":
                    af.add_attack(nid, "A1")
                else:
                    af.add_support(nid, "A1")

    # internal links for split clauses (only if not already linked)
    existing_edges = {(u, v, d.get("type")) for u, v, d in af.nx_graph.edges(data=True)}
    for origin_id, links in split_link_map.items():
        for child_id, parent_id, connector, direction, edge_type in links:
            if child_id not in af.nodes or parent_id not in af.nodes:
                continue

            if direction == "reverse":
                source_id = parent_id
                target_id = child_id
            else:
                source_id = child_id
                target_id = parent_id

            if (source_id, target_id, edge_type) in existing_edges:
                continue

            if edge_type == "attack":
                af.add_attack(source_id, target_id)
            else:
                af.add_support(source_id, target_id)

    def _id_distance(a: str, b: str) -> Optional[int]:
        if not (a.startswith("A") and b.startswith("A")):
            return None
        try:
            a_id = int(a.strip("A").split("_")[0])
            b_id = int(b.strip("A").split("_")[0])
        except ValueError:
            return None
        return abs(a_id - b_id)

    def _candidate_score(tokens: Set[str], other_tokens: Set[str], other_id: str) -> float:
        overlap = len(tokens & other_tokens)
        score = overlap * 3.0
        if tokens & root_tokens:
            score += 1.0
        if other_id == "A1":
            score += 1.0
        dist = _id_distance(nid, other_id) if "nid" in locals() else None
        if dist is not None and dist <= 2:
            score += 1.0
        return score

    def _build_shortlist(nid: str, candidate_ids: List[str], limit: int) -> List[Dict[str, str]]:
        tokens = node_tokens.get(nid, set())
        candidates = []
        for cid in candidate_ids:
            if cid == nid:
                continue
            other_tokens = node_tokens.get(cid, set())
            if not other_tokens:
                continue
            # overlap = len(tokens & other_tokens)
            # if overlap == 0 and cid != "A1":
            #     continue
            overlap = len(tokens & other_tokens)
            score = _candidate_score(tokens, other_tokens, cid)
            candidates.append((score, cid))
        candidates.sort(reverse=True)
        return [{"id": cid, "content": af.nodes[cid].content} for _, cid in candidates[:limit]]

    # Cluster-based LLM attach (topic clusters)
    cluster_map: Dict[str, List[str]] = {"misc": []}
    for nid in node_ids:
        keys = node_tokens.get(nid, set()) & _STRONG_TOPIC_TERMS
        if not keys:
            cluster_map["misc"].append(nid)
        else:
            for key in keys:
                cluster_map.setdefault(key, []).append(nid)

    remaining_orphans = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
    if remaining_orphans and config.enable_orphan_cluster_llm:
        orphan_tasks = []
        for nid in remaining_orphans:
            keys = node_tokens.get(nid, set()) & _STRONG_TOPIC_TERMS
            candidate_pool = set()
            if keys:
                for key in keys:
                    candidate_pool.update(cluster_map.get(key, []))
            else:
                candidate_pool.update(cluster_map.get("misc", []))
            candidate_pool.update(["A1"])
            shortlist = _build_shortlist(nid, list(candidate_pool), 8)
            if not shortlist:
                continue
            orphan_tasks.append({"orphan": {"id": nid, "content": af.nodes[nid].content}, "candidates": shortlist})

        if orphan_tasks:
            batch_size = 10
            for i in range(0, len(orphan_tasks), batch_size):
                batch = orphan_tasks[i : i + batch_size]
                try:
                    prompt_force = _build_orphan_force_prompt("A1", batch)
                    force_data = _chat_json(
                        [
                            {"role": "system", "content": "You output strictly valid JSON only."},
                            {"role": "user", "content": prompt_force},
                        ]
                    )
                except Exception:
                    force_data = {"relations": []}

                force_rels = force_data.get("relations", []) if isinstance(force_data, dict) else []
                if not isinstance(force_rels, list):
                    force_rels = []
                for rel in force_rels:
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

    # Global LLM attach for remaining orphans
    remaining_orphans = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
    if remaining_orphans and config.enable_orphan_global_llm:
        orphan_tasks = []
        candidate_ids = list(af.nodes.keys())
        for nid in remaining_orphans:
            shortlist = _build_shortlist(nid, candidate_ids, 10)
            if not shortlist:
                continue
            orphan_tasks.append({"orphan": {"id": nid, "content": af.nodes[nid].content}, "candidates": shortlist})

        if orphan_tasks:
            batch_size = 12
            for i in range(0, len(orphan_tasks), batch_size):
                batch = orphan_tasks[i : i + batch_size]
                try:
                    prompt_force = _build_orphan_force_prompt("A1", batch)
                    force_data = _chat_json(
                        [
                            {"role": "system", "content": "You output strictly valid JSON only."},
                            {"role": "user", "content": prompt_force},
                        ]
                    )
                except Exception:
                    force_data = {"relations": []}

                force_rels = force_data.get("relations", []) if isinstance(force_data, dict) else []
                if not isinstance(force_rels, list):
                    force_rels = []
                for rel in force_rels:
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

    # Deterministic attach for remaining orphans with overlap >= 1
    remaining_orphans = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
    if config.enable_orphan_deterministic:
        for nid in remaining_orphans:
            tokens = node_tokens.get(nid, set())
            if not tokens:
                continue
            best_id = None
            best_overlap = 0
            for other_id in af.nodes:
                if other_id == nid:
                    continue
                other_tokens = node_tokens.get(other_id, set())
                overlap = len(tokens & other_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = other_id
            if best_id and best_overlap >= 1:
                edge_type = _edge_type_guess(tokens, node_tokens.get(best_id, set()))
                if edge_type == "attack":
                    af.add_attack(nid, best_id)
                else:
                    af.add_support(nid, best_id)

        # Force one extra edge for orphans with overlap >= 1
        remaining_orphans = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
        for nid in remaining_orphans:
            tokens = node_tokens.get(nid, set())
            best_id = None
            best_overlap = 0
            for other_id in af.nodes:
                if other_id == nid:
                    continue
                other_tokens = node_tokens.get(other_id, set())
                overlap = len(tokens & other_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = other_id
            if best_id and best_overlap >= 1:
                edge_type = _edge_type_guess(tokens, node_tokens.get(best_id, set()))
                if edge_type == "attack":
                    af.add_attack(nid, best_id)
                else:
                    af.add_support(nid, best_id)

    # Root attach fallback for any remaining orphans
    if config.enable_orphan_root_fallback:
        remaining_orphans = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.in_degree(nid) == 0]
        for nid in remaining_orphans:
            tokens = node_tokens.get(nid, set())
            edge_type = _edge_type_guess(tokens, root_tokens)
            if edge_type == "attack":
                af.add_attack(nid, "A1")
            else:
                af.add_support(nid, "A1")

        # Reattach removed edges to root if overlapping
        removed_candidates = [nid for nid in af.nodes if nid != "A1" and af.nx_graph.out_degree(nid) == 0]
        for nid in removed_candidates:
            tokens = node_tokens.get(nid, set())
            if tokens & root_tokens or tokens & _STRONG_TOPIC_TERMS:
                edge_type = _edge_type_guess(tokens, root_tokens)
                if edge_type == "attack":
                    af.add_attack(nid, "A1")
                else:
                    af.add_support(nid, "A1")

    # Connect small components to root
    if config.enable_component_link:
        try:
            import networkx as nx
            components = list(nx.weakly_connected_components(af.nx_graph))
            for comp in components:
                if "A1" in comp:
                    continue
                candidate = None
                best_overlap = 0
                for nid in comp:
                    tokens = node_tokens.get(nid, set())
                    overlap = len(tokens & root_tokens)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        candidate = nid
                if candidate:
                    edge_type = _edge_type_guess(node_tokens.get(candidate, set()), root_tokens)
                    if edge_type == "attack":
                        af.add_attack(candidate, "A1")
                    else:
                        af.add_support(candidate, "A1")
        except Exception:
            pass

    # LLM edge sanity check for suspicious edges
    suspicious_edges = []
    for u, v, d in list(af.nx_graph.edges(data=True)):
        if d.get("type") not in _ALLOWED_REL_TYPES:
            continue
        src_tokens = node_tokens.get(u, set())
        dst_tokens = node_tokens.get(v, set())
        if not src_tokens or not dst_tokens:
            continue
        neg_src = bool(src_tokens & _NEGATION_WORDS)
        neg_dst = bool(dst_tokens & _NEGATION_WORDS)
        overlap = len(src_tokens & dst_tokens)
        if neg_src != neg_dst or overlap >= 4:
            suspicious_edges.append({"from": u, "to": v, "type": d.get("type")})

    attack_edges = [d for _, _, d in af.nx_graph.edges(data=True) if d.get("type") == "attack"]
    total_edges = af.nx_graph.number_of_edges()
    attack_ratio = (len(attack_edges) / total_edges) if total_edges else 0.0
    if attack_ratio > 0.55:
        for u, v, d in list(af.nx_graph.edges(data=True)):
            if d.get("type") == "attack":
                suspicious_edges.append({"from": u, "to": v, "type": "attack"})

    if suspicious_edges and config.enable_edge_sanity:
        nodes_payload = [{"id": nid, "content": af.nodes[nid].content} for nid in af.nodes]
        try:
            sanity_prompt = _build_edge_sanity_prompt(nodes_payload, suspicious_edges)
            sanity_data = _chat_json(
                [
                    {"role": "system", "content": "You output strictly valid JSON only."},
                    {"role": "user", "content": sanity_prompt},
                ]
            )
        except Exception:
            sanity_data = {"ops": []}

        ops = sanity_data.get("ops", []) if isinstance(sanity_data, dict) else []
        if not isinstance(ops, list):
            ops = []
        for op in ops[:10]:
            if not isinstance(op, dict):
                continue
            action = str(op.get("op", "")).strip().lower()
            src = str(op.get("from", "")).strip()
            dst = str(op.get("to", "")).strip()
            if not src or not dst:
                continue
            if not af.nx_graph.has_edge(src, dst):
                continue
            if action == "flip":
                cur = af.nx_graph[src][dst].get("type")
                af.nx_graph[src][dst]["type"] = "attack" if cur == "support" else "support"
            elif action == "remove":
                af.nx_graph.remove_edge(src, dst)

    # Reduce attack bias when ratio too high
    attack_edges = [
        (u, v, len(node_tokens.get(u, set()) & node_tokens.get(v, set())))
        for u, v, d in af.nx_graph.edges(data=True)
        if d.get("type") == "attack"
    ]
    total_edges = af.nx_graph.number_of_edges()
    attack_ratio = (len(attack_edges) / total_edges) if total_edges else 0.0
    if config.enable_attack_balance and attack_ratio > 0.6 and attack_edges:
        attack_edges.sort(key=lambda x: x[2], reverse=True)
        flip_budget = max(1, int(len(attack_edges) * 0.2))
        for u, v, overlap in attack_edges[:flip_budget]:
            src_tokens = node_tokens.get(u, set())
            dst_tokens = node_tokens.get(v, set())
            neg_src = bool(src_tokens & _NEGATION_WORDS)
            neg_dst = bool(dst_tokens & _NEGATION_WORDS)
            if neg_src == neg_dst and overlap >= 2:
                af.nx_graph[u][v]["type"] = "support"

    # Flip support edges with strong negation mismatch
    for u, v, d in list(af.nx_graph.edges(data=True)):
        if d.get("type") != "support":
            continue
        src_tokens = node_tokens.get(u, set())
        dst_tokens = node_tokens.get(v, set())
        if not src_tokens or not dst_tokens:
            continue
        neg_src = bool(src_tokens & _NEGATION_WORDS)
        neg_dst = bool(dst_tokens & _NEGATION_WORDS)
        if neg_src != neg_dst:
            af.nx_graph[u][v]["type"] = "attack"

    # Edge audit correction pass (sample-based)
    if config.enable_edge_audit:
        try:
            edges_for_audit = []
            for u, v, d in list(af.nx_graph.edges(data=True)):
                edges_for_audit.append({"from": u, "to": v, "type": d.get("type")})
            if edges_for_audit:
                nodes_payload = [{"id": nid, "content": af.nodes[nid].content} for nid in af.nodes]
                batch_size = 10
                max_edges = 60
                max_fixes = 20
                total_incorrect = 0
                total_checked = 0
                applied_fixes = 0

                def _run_audit(pass_edges: List[Dict[str, Any]]) -> None:
                    nonlocal total_incorrect, total_checked, applied_fixes
                    for i in range(0, len(pass_edges), batch_size):
                        if applied_fixes >= max_fixes:
                            break
                        batch = pass_edges[i : i + batch_size]
                        audit_prompt = _build_edge_audit_prompt(nodes_payload, batch)
                        audit_data = _chat_json(
                            [
                                {"role": "system", "content": "You output strictly valid JSON only."},
                                {"role": "user", "content": audit_prompt},
                            ]
                        )
                        results = audit_data.get("results", []) if isinstance(audit_data, dict) else []
                        if not isinstance(results, list):
                            continue
                        for res in results:
                            if not isinstance(res, dict):
                                continue
                            total_checked += 1
                            label = str(res.get("label", "")).strip().lower()
                            if label == "incorrect":
                                total_incorrect += 1
                            src = str(res.get("from", "")).strip()
                            dst = str(res.get("to", "")).strip()
                            fix = str(res.get("fix", "keep")).strip().lower()
                            if not src or not dst:
                                continue
                            if not af.nx_graph.has_edge(src, dst):
                                continue
                            if fix == "flip" and applied_fixes < max_fixes:
                                cur = af.nx_graph[src][dst].get("type")
                                af.nx_graph[src][dst]["type"] = "attack" if cur == "support" else "support"
                                applied_fixes += 1
                            elif fix == "remove" and applied_fixes < max_fixes:
                                af.nx_graph.remove_edge(src, dst)
                                applied_fixes += 1

                _run_audit(edges_for_audit[:max_edges])
                incorrect_ratio = (total_incorrect / total_checked) if total_checked else 0.0
                if incorrect_ratio > 0.3 and applied_fixes < max_fixes:
                    _run_audit(edges_for_audit[max_edges : max_edges * 2])

        except Exception:
            pass

    # Enforce minimum attack ratio if too low
    total_edges = af.nx_graph.number_of_edges()
    attack_edges_count = sum(1 for _, _, d in af.nx_graph.edges(data=True) if d.get("type") == "attack")
    attack_ratio = (attack_edges_count / total_edges) if total_edges else 0.0
    if config.enable_attack_balance and attack_ratio < 0.2:
        flip_budget = 5
        for u, v, d in list(af.nx_graph.edges(data=True)):
            if flip_budget <= 0:
                break
            if d.get("type") != "support":
                continue
            src_tokens = node_tokens.get(u, set())
            dst_tokens = node_tokens.get(v, set())
            if not src_tokens or not dst_tokens:
                continue
            neg_src = bool(src_tokens & _NEGATION_WORDS)
            neg_dst = bool(dst_tokens & _NEGATION_WORDS)
            overlap = len(src_tokens & dst_tokens)
            if neg_src != neg_dst or overlap >= 3:
                af.nx_graph[u][v]["type"] = "attack"
                flip_budget -= 1

    # Deduplication (Consolidate semantically identical nodes)
    if config.enable_dedup:
        _ = merge_redundant_nodes(af)

    # sanitize before refine (keeps the prompt clean)
    _sanitize_graph(af)

    # -------------------
    # Stage B: Self-refine loop
    # -------------------
    for it in range(1, refine_iters_effective + 1):
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
    atomicity_flags = _atomicity_violations(af)
    if atomicity_flags:
        af.atomicity_warnings = atomicity_flags  # type: ignore[attr-defined]
    af.root_id_override = "A1"
    return af


def parse_debate_paper_safe(
    text: str,
    split_mode: str = "hybrid",
    refine_iters: int = REFINE_ITERS,
) -> ArgumentationGraph:
    return parse_debate(text, split_mode=split_mode, refine_iters=refine_iters, mode="paper_safe")


def parse_debate_fast(
    text: str,
    split_mode: str = "hybrid",
    refine_iters: int = REFINE_ITERS,
) -> ArgumentationGraph:
    return parse_debate(text, split_mode=split_mode, refine_iters=refine_iters, mode="fast")
