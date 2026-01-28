# tools/real_toolkit.py
# Patched version (RAG + Judge, stance-aware, evidence-calibrated confidence).
#
# KEY PATCHES (this version):
# 1) Meta-claim rewrite:
#    - Detect "discussion/conversation/topic is about ..." style claims and rewrite into a factual claim.
# 2) Better WEB queries:
#    - Build keyword-focused query rounds (avoid literal full-sentence queries).
#    - Coverage gate: if snippets don't match claim anchors, ask LLM for better queries (1 extra round).
# 3) ABSTAIN / low-confidence policy fix:
#    - If RAG judge ABSTAINS (or is below threshold), fallback properly.
#    - If v_serper and v_ddg are None but COMMON_SENSE returns a vote, use it (do NOT default to FALSE).
# 4) Keep your calibrated confidence clamp:
#    - final_conf = min(llm_conf, evidence_conf)
#    - Use thresholds to accept TRUE/FALSE; otherwise treat as ABSTAIN and fallback.
#
# Notes:
# - Keeps stance-aware heuristic behavior.
# - Heuristic can still short-circuit only for STRONG refutation (FALSE) or extremely strong TRUE (rare).
# - Requires: src.config provides client, SERPER_API_KEY, JUDGE_MODEL.

from __future__ import annotations

import os
import re
import math
import json
import datetime
import random
import io
import contextlib
import time
import requests
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Literal

from src.config import client, SERPER_API_KEY, JUDGE_MODEL, FAST_MODE

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=UserWarning, module="duckduckgo_search")


# -----------------------------
# Utilities
# -----------------------------
Label3 = Optional[bool]  # True / False / None(ABSTAIN)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _domain(url: str) -> str:
    u = (url or "").strip().lower()
    m = re.search(r"https?://([^/]+)", u)
    return m.group(1) if m else ""


def _is_gov_domain(d: str) -> bool:
    return d.endswith(".gov") or ".gov." in d


def _is_edu_domain(d: str) -> bool:
    return d.endswith(".edu") or ".edu." in d


def _is_trusted_domain(url: str) -> bool:
    """
    Trusted domains used for *evidence quality*, not automatic truth.
    Even trusted sources can *mention* a claim to refute it.
    """
    d = _domain(url)
    if not d:
        return False

    trusted_exact_or_contains = [
        # primary science/news references
        "nature.com",
        "science.org",
        "sciencemag.org",
        "cell.com",
        "nejm.org",
        "thelancet.com",
        "reuters.com",
        "apnews.com",
        "bbc.co.uk",
        "bbc.com",
        "nytimes.com",
        "theguardian.com",
        "economist.com",
        # reference works
        "britannica.com",
        "wikipedia.org",
        # space & missions
        "nasa.gov",
        "esa.int",
        "jaxa.jp",
        "isro.gov.in",
        # institutions
        "un.org",
        "who.int",
        "ioc.org",
        "olympics.com",
    ]

    if _is_gov_domain(d) or _is_edu_domain(d):
        return True

    return any(t in d for t in trusted_exact_or_contains)


def _summarize_hits(hits: List[Dict[str, str]], max_n: int = 6) -> str:
    """
    Produce numbered evidence lines so the judge can cite specific snippets.
    """
    out = []
    for i, r in enumerate((hits or [])[:max_n], start=1):
        title = (r.get("title") or "")[:180]
        url = (r.get("url") or "")[:240]
        snip = (r.get("snippet") or "")[:260]
        out.append(f"[{i}] {title} | {url} | {snip}")
    return "\n".join(out)


def _split_sentences_rough(s: str, max_len: int = 260) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:max_len]


def _distinct_domains(hits: List[Dict[str, str]]) -> int:
    doms = set()
    for h in hits or []:
        d = _domain(h.get("url", ""))
        if d:
            doms.add(d)
    return len(doms)


@dataclass
class JudgeResult:
    verdict: Optional[bool]          # True/False/None
    llm_confidence: float            # self-reported by LLM
    evidence_confidence: float       # computed proxy
    final_confidence: float          # min(llm_conf, evidence_conf)
    support_ids: List[int]
    refute_ids: List[int]
    rationale: str


# -----------------------------
# PythonSandbox (unchanged core, minor hardening)
# -----------------------------
class PythonSandbox:
    @staticmethod
    def run(code: str) -> str:
        pattern = r"```python(.*?)```"
        match = re.search(pattern, code, re.DOTALL)
        if match:
            clean_code = match.group(1).strip()
        else:
            clean_code = code.replace("```python", "").replace("```", "").strip()

        forbidden_substrings = [
            "input(",
            "while True",
            "time.sleep",
            "open(",
            "import ",
            "__import__",
            "exec(",
            "eval(",
            "globals(",
            "locals(",
        ]
        for bad in forbidden_substrings:
            if bad in clean_code:
                return f"[Security Block]: Code contains forbidden term '{bad}'."

        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "round": round,
        }

        safe_globals = {
            "__builtins__": safe_builtins,
            "math": math,
            "datetime": datetime,
            "random": random,
        }
        safe_locals: Dict[str, Any] = {}

        output_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(output_buffer):
                compiled = compile(clean_code, "<string>", "exec")
                exec(compiled, safe_globals, safe_locals)

            if "FINAL_RESULT" in safe_locals:
                return str(safe_locals["FINAL_RESULT"])
            if output_buffer.getvalue().strip():
                return output_buffer.getvalue().strip()
            return "[Error]: Code executed but returned no result."
        except Exception as e:
            print(f"      ⚠️ Python execution error: {e}")
            return "[Error]: Python execution failed."


# -----------------------------
# RealToolkit
# -----------------------------
class RealToolkit:
    _cache: Dict[str, Any] = {}
    _llm_error_flag: bool = False

    # ---------- Tuning knobs ----------
    # Gate TRUE more strictly than FALSE, because false-positives are dangerous for pruning/acceptance.
    JUDGE_TRUE_MIN_FINAL_CONF = 0.72
    JUDGE_FALSE_MIN_FINAL_CONF = 0.62

    # Semantic fallback (LLM) threshold
    JUDGE_FALLBACK_MIN_CONF = 0.75

    # For edge prune: only prune when "FALSE" with very high confidence
    EDGE_PRUNE_FALSE_CONF = 0.80
    ATTACK_TRUE_MIN_CONF = 0.5

    @staticmethod
    def _detect_sanity_family(text: str) -> Optional[str]:
        if not text:
            return None
        s = text.lower()

        # Propositional logic detection
        if re.search(r"\b(if|then|implies|therefore|thus)\b", s):
            if re.search(r"\b(true|false|is true|is false)\b", s):
                if re.search(r"\b(and|or|not)\b", s):
                    return "propositional"

        if "leap year" in s:
            return "leap"
        if "square root of" in s:
            return "sqrt"
        if " percent of " in s or "% of" in s:
            return "percent"
        if re.search(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)\s*(equals|=)", s):
            return "arith"
        if re.search(r"(>=|<=|>|<)", s) or "greater than" in s or "less than" in s or "at least" in s or "at most" in s:
            return "compare"
        return None

    @staticmethod
    def _sanity_harness(family: str) -> bool:
        gold: List[Tuple[str, bool]] = []
        if family == "leap":
            gold = [
                ("2000 was a leap year.", True),
                ("1900 was a leap year.", False),
                ("2020 was a leap year.", True),
                ("2100 was a leap year.", False),
            ]
        elif family == "arith":
            gold = [
                ("2 + 2 equals 4", True),
                ("2 + 2 equals 5", False),
                ("17 * 19 equals 323", True),
                ("17 * 19 equals 322", False),
            ]
        elif family == "sqrt":
            gold = [
                ("The square root of 16 is 4.", True),
                ("The square root of 16 is not 4.", False),
                ("The square root of 144 is 12.", True),
                ("The square root of 144 is 13.", False),
            ]
        elif family == "percent":
            gold = [
                ("10 percent of 50 equals 5", True),
                ("10 percent of 50 equals 6", False),
                ("25% of 200 equals 50", True),
                ("25% of 200 equals 40", False),
            ]
        elif family == "compare":
            gold = [
                ("3 > 2", True),
                ("3 < 2", False),
                ("5 is at least 5", True),
                ("5 is at most 4", False),
            ]
        else:
            return True

        for text, expected in gold:
            got = RealToolkit._deterministic_tier0(text)
            if got is None or got != expected:
                return False
        return True

    # ---------- Deterministic tier ----------
    @staticmethod
    def _deterministic_tier0(text: str) -> Optional[bool]:
        if not text:
            return None
        s = text.strip().lower()
        s = re.sub(r"(?<=\d),(?=\d)", "", s)
        tol = 1e-6
        approx = bool(re.search(r"\b(about|around|approximately|approx)\b", s))
        if approx:
            tol = 1e-2

        # Leap year check
        m = re.search(r"(\d{4}).*leap year", s)
        if m:
            year = int(m.group(1))
            is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
            if "not" in s or "isn't" in s or "is not" in s:
                return not is_leap
            return is_leap

        # Square root
        m = re.search(r"square root of\s+(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)", s)
        if m:
            a = float(m.group(1))
            b = float(m.group(2))
            ok = abs((a ** 0.5) - b) <= tol
            if "not" in s or "isn't" in s or "is not" in s:
                return not ok
            return ok

        # Percent variants
        m = re.search(
            r"(\d+(?:\.\d+)?)\s*%?\s*percent of\s*(\d+(?:\.\d+)?)\s*(?:equals|is|=)\s*(\d+(?:\.\d+)?)",
            s,
        )
        if m:
            p = float(m.group(1))
            total = float(m.group(2))
            target = float(m.group(3))
            ok = abs((p / 100.0) * total - target) <= tol
            if "not" in s or "isn't" in s or "is not" in s:
                return not ok
            return ok

        m = re.search(
            r"(\d+(?:\.\d+)?)\s*%?\s*of\s*(\d+(?:\.\d+)?)\s*(?:is|equals|=)\s*(\d+(?:\.\d+)?)",
            s,
        )
        if m:
            p = float(m.group(1))
            total = float(m.group(2))
            target = float(m.group(3))
            ok = abs((p / 100.0) * total - target) <= tol
            if "not" in s or "isn't" in s or "is not" in s:
                return not ok
            return ok

        m = re.search(
            r"(\d+(?:\.\d+)?)\s+is\s+(\d+(?:\.\d+)?)\s*%?\s*of\s+(\d+(?:\.\d+)?)",
            s,
        )
        if m:
            target = float(m.group(1))
            p = float(m.group(2))
            total = float(m.group(3))
            ok = abs((p / 100.0) * total - target) <= tol
            if "not" in s or "isn't" in s or "is not" in s:
                return not ok
            return ok

        # Arithmetic (a op b = c)
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(-?\d+(?:\.\d+)?)\s*(?:equals|=)\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            a = float(m.group(1))
            op = m.group(2)
            b = float(m.group(3))
            c = float(m.group(4))
            if op == "+":
                res = a + b
            elif op == "-":
                res = a - b
            elif op == "*":
                res = a * b
            else:
                if b == 0:
                    return None
                res = a / b
            ok = abs(res - c) <= tol
            if "not" in s or "isn't" in s or "is not" in s or "does not equal" in s:
                return not ok
            return ok

        # Comparisons
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", s)
        if m:
            a = float(m.group(1))
            op = m.group(2)
            b = float(m.group(3))
            if op == ">":
                return a > b
            if op == "<":
                return a < b
            if op == ">=":
                return a >= b
            return a <= b

        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+at\s+least\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) >= float(m.group(2))
        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+at\s+most\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) <= float(m.group(2))
        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+greater\s+than\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) > float(m.group(2))
        m = re.search(r"(-?\d+(?:\.\d+)?)\s+is\s+less\s+than\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            return float(m.group(1)) < float(m.group(2))

        # Range checks
        m = re.search(r"between\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", s)
        if m:
            lo = float(m.group(1))
            hi = float(m.group(2))
            if "is" in s:
                m2 = re.search(r"is\s+(-?\d+(?:\.\d+)?)", s)
                if m2:
                    x = float(m2.group(1))
                    return lo <= x <= hi

        return None

    @staticmethod
    def _eval_propositional(text: str) -> Optional[bool]:
        s = (text or "").strip().lower()
        if "if" not in s and "implies" not in s and "therefore" not in s:
            return None

        if "then" in s:
            parts = s.split("then", 1)
            lhs = parts[0].replace("if", "").strip()
            rhs = parts[1].strip() if len(parts) == 2 else ""
        elif "implies" in s:
            parts = s.split("implies", 1)
            lhs = parts[0].strip()
            rhs = parts[1].strip() if len(parts) == 2 else ""
        else:
            return None

        def _parse_bool_expr(t: str) -> Optional[bool]:
            tokens = t.replace("(", " ").replace(")", " ").split()
            if not tokens:
                return None

            def atom(tok: str) -> Optional[bool]:
                if tok in ("true", "t"):
                    return True
                if tok in ("false", "f"):
                    return False
                return None

            stack: List[Optional[bool]] = []
            op_stack: List[str] = []

            i = 0
            while i < len(tokens):
                tok = tokens[i]
                if tok == "not":
                    i += 1
                    if i < len(tokens):
                        val = atom(tokens[i])
                        if val is None:
                            return None
                        stack.append(not val)
                    else:
                        return None
                elif tok in ("and", "or"):
                    op_stack.append(tok)
                else:
                    val = atom(tok)
                    if val is None:
                        return None
                    stack.append(val)
                i += 1

            if not stack:
                return None
            result = stack[0]
            for idx, op in enumerate(op_stack, start=1):
                if idx >= len(stack):
                    break
                if op == "and":
                    result = bool(result and stack[idx])
                else:
                    result = bool(result or stack[idx])
            return result

        a = _parse_bool_expr(lhs)
        b = _parse_bool_expr(rhs)
        if a is None or b is None:
            return None
        return (not a) or b

    # ---------- Cache helpers ----------
    @staticmethod
    def _cache_key(prefix: str, *parts: str) -> str:
        clean = [prefix] + [str(p).strip().lower() for p in parts if p is not None]
        return "::".join(clean)

    # ---------- LLM codegen for PYTHON_EXEC ----------
    @staticmethod
    def _llm_generate_python_code(clean_fact: str) -> str:
        code_prompt = f"""
Write a Python script to verify the following statement:
"{clean_fact}"

Rules:
- No imports.
- No input(), no file I/O, no network.
- Must terminate quickly.
- Must set FINAL_RESULT = "VERIFIED_TRUE" or FINAL_RESULT = "VERIFIED_FALSE".
- Prefer direct computation, do not guess.

Python code:
"""
        res = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": code_prompt}],
            temperature=0.0,
        )
        return res.choices[0].message.content or ""

    # ---------- Distillation ----------
    @staticmethod
    def _strip_claim_prefix(text: str) -> str:
        s = (text or "").strip()
        s = re.sub(r"^\s*claim\s*:\s*", "", s, flags=re.IGNORECASE)
        return s.strip()

    @staticmethod
    def _distill_claim(text: str) -> str:
        key = RealToolkit._cache_key("distill", text)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        raw = RealToolkit._strip_claim_prefix(text)

        # Extract fingerprints from original claim
        original_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", raw))
        original_numbers = set(re.findall(r"\b\d+\b", raw))

        prompt = f"""
Extract the core factual claim from the text below.
Remove conversational filler and hedging like "I think", "maybe", "in my opinion".
Preserve the main subject/entities and the meaning.
CRITICAL: Do NOT correct numbers, dates, or named entities even if they seem wrong.
If there is no factual claim, rewrite it into a simple checkable statement.

Text: "{raw}"
Core factual claim:
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            clean = (res.choices[0].message.content or "").strip().replace('"', "")
        except Exception:
            clean = raw.strip()

        # Validation: reject distillation if it corrupts critical numbers/years
        distilled_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", clean))
        distilled_numbers = set(re.findall(r"\b\d+\b", clean))

        if original_years and (original_years != distilled_years):
            clean = raw.strip()

        if original_numbers and len(original_numbers.intersection(distilled_numbers)) < len(original_numbers) * 0.7:
            clean = raw.strip()

        RealToolkit._cache[key] = clean
        return clean

    # ---------- Meta-claim rewrite (NEW) ----------
    @staticmethod
    def _is_meta_discourse_claim(s: str) -> bool:
        t = _norm_text(s)
        pats = [
            "the discussion is about",
            "this discussion is about",
            "the conversation is about",
            "this conversation is about",
            "the debate is about",
            "this debate is about",
            "the topic is",
            "topic is",
            "we are discussing",
            "we discuss",
            "the text is about",
            "this passage is about",
        ]
        return any(p in t for p in pats)

    @staticmethod
    def _rewrite_meta_to_factual(clean_fact: str) -> str:
        """
        Deterministic rewrite: remove meta-frame into a checkable factual claim.
        If it cannot be rewritten safely, returns the original.
        """
        key = RealToolkit._cache_key("meta2fact", clean_fact)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Rewrite the statement into a single, checkable factual claim about the underlying subject.
Remove meta framing like "the discussion is about", "the topic is", "we are discussing".
Do NOT introduce new entities, numbers, or dates.

Statement: "{clean_fact}"

Output only the rewritten factual claim (no quotes).
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            out = (res.choices[0].message.content or "").strip().strip('"')
        except Exception:
            out = clean_fact

        if len(out.strip()) < 8:
            out = clean_fact

        RealToolkit._cache[key] = out
        return out

    # ---------- Query builder (IMPROVED) ----------
    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract capitalized multi-word entities and proper nouns."""
        # Match capitalized words/phrases (potential entities)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text or "")
        # Also match all-caps acronyms
        acronyms = re.findall(r"\b[A-Z]{2,6}\b", text or "")
        return list(dict.fromkeys(entities + acronyms))[:5]

    @staticmethod
    def _extract_years(text: str) -> List[str]:
        """Extract 4-digit years from text."""
        return list(dict.fromkeys(re.findall(r"\b(19\d{2}|20\d{2})\b", text or "")))[:3]

    @staticmethod
    def _make_queries(clean_fact: str) -> List[List[str]]:
        """
        Build query rounds with hybrid strategy.
        Round1: full claim + keyword query + optional myth/fact-check
        Round2: encyclopedia sources
        Round3: authoritative sources
        """
        s = (clean_fact or "").strip()
        if not s:
            return [[clean_fact]]

        entities = RealToolkit._extract_entities(clean_fact)
        years = RealToolkit._extract_years(clean_fact)

        stopwords = {
            "discussion", "conversation", "topic", "claim", "about", "therefore",
            "usually", "often", "because", "most", "people", "that", "this",
            "with", "from", "were", "have", "been", "being", "would", "could",
            "should", "which", "there", "their", "they", "what", "when", "where",
        }
        kws = re.findall(r"[A-Za-z]{4,}", clean_fact)
        kws = [w.lower() for w in kws if w.lower() not in stopwords]
        kws = sorted(set(kws), key=lambda x: (-len(x), x))[:6]

        entity_str = " ".join(entities[:3]) if entities else ""
        kw_str = " ".join(kws[:5])
        base_q = f"{entity_str} {kw_str}".strip() if entity_str else kw_str
        base_q = base_q if len(base_q) >= 8 else s[:120]

        year_suffix = f" {years[0]}" if years else ""
        myth_cues = ["myth", "hoax", "rumor", "does", "what happens", "is it true", "debunk", "fact check"]
        is_myth = any(cue in s.lower() for cue in myth_cues)

        r1 = [
            s[:200],
            f"{base_q}{year_suffix}",
        ]
        if is_myth:
            r1.append(f"{base_q} myth")
            r1.append(f"{base_q} fact check")
        else:
            r1.append(f"{base_q} facts")

        r2 = [
            f"site:wikipedia.org {base_q}",
            f"site:britannica.com {entities[0] if entities else base_q}",
        ]

        r3 = [
            f"site:.edu {base_q}",
            f"site:.gov {base_q}",
        ]
        if years:
            r3.append(f"{base_q} {years[0]} official")

        return [r1, r2, r3]

    # ---------- Serper API Key Rotation ----------
    _SERPER_KEYS: List[str] = []
    _CURRENT_KEY_IDX: int = 0

    @staticmethod
    def _load_serper_keys():
        """Load multiple Serper API keys from local file for failover/rotation."""
        if RealToolkit._SERPER_KEYS:
            return
        
        path = "/home/tungnvt5re/research-projects/reinforcement_learning_intro_books/maveric_ijcai/src/api_key/serper.txt"
        keys = []
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    keys = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"        ⚠️ Error reading serper.txt: {e}")
        
        # Fallback to config key if file empty or missing
        if not keys and SERPER_API_KEY:
            keys = [SERPER_API_KEY]
        
        RealToolkit._SERPER_KEYS = keys
        if keys:
            print(f"        📡 Loaded {len(keys)} Serper API keys for rotation.")

    @staticmethod
    def _get_current_serper_key() -> Optional[str]:
        RealToolkit._load_serper_keys()
        if not RealToolkit._SERPER_KEYS:
            return None
        return RealToolkit._SERPER_KEYS[RealToolkit._CURRENT_KEY_IDX % len(RealToolkit._SERPER_KEYS)]

    @staticmethod
    def _rotate_serper_key() -> str:
        RealToolkit._CURRENT_KEY_IDX += 1
        new_idx = RealToolkit._CURRENT_KEY_IDX % len(RealToolkit._SERPER_KEYS)
        print(f"        🔄 Serper quota reached. Rotating to key #{new_idx + 1}...")
        return RealToolkit._SERPER_KEYS[new_idx]

    # ---------- Web search providers ----------
    @staticmethod
    def _serper_search(query: str, num: int = 6, timeout: float = 6.5) -> List[Dict[str, str]]:
        RealToolkit._load_serper_keys()
        if not RealToolkit._SERPER_KEYS:
            return []
            
        # Try all available keys if quota is hit
        max_attempts = len(RealToolkit._SERPER_KEYS)
        
        for attempt in range(max_attempts):
            current_key = RealToolkit._get_current_serper_key()
            if not current_key:
                break
                
            try:
                url = "https://google.serper.dev/search"
                payload = json.dumps({"q": query[:200], "num": int(num)})
                headers = {"X-API-KEY": current_key, "Content-Type": "application/json"}
                resp = requests.post(url, headers=headers, data=payload, timeout=timeout)
                
                # Check for quota limit or unauthorized
                is_quota_error = resp.status_code in (403, 429)
                if not is_quota_error and resp.status_code != 200:
                    # Also check for quota messages in JSON even if status is weird
                    resp_json = {}
                    try:
                        resp_json = resp.json()
                    except:
                        pass
                    msg = str(resp_json.get("message", "")).lower()
                    if "quota" in msg or "unauthorized" in msg or "api key" in msg:
                        is_quota_error = True

                if is_quota_error:
                    if max_attempts > 1 and attempt < max_attempts - 1:
                        RealToolkit._rotate_serper_key()
                        continue
                    else:
                        return []
                        
                if resp.status_code != 200:
                    return []
                    
                organic = resp.json().get("organic", []) or []
                out: List[Dict[str, str]] = []
                for r in organic[: int(num)]:
                    out.append(
                        {
                            "title": (r.get("title") or "")[:200],
                            "url": (r.get("link") or "")[:700],
                            "snippet": (r.get("snippet") or "")[:900],
                            "provider": "serper",
                        }
                    )
                return out
            except Exception:
                # Connection errors or JSON parse errors
                return []
        return []

    @staticmethod
    def _ddg_search(query: str, max_results: int = 6) -> List[Dict[str, str]]:
        try:
            try:
                from ddgs import DDGS  # type: ignore
                ddgs = DDGS()
                res = list(ddgs.text(query, max_results=max_results))
            except Exception:
                from duckduckgo_search import DDGS  # type: ignore
                with DDGS() as ddgs:
                    res = list(ddgs.text(query, max_results=max_results))

            out: List[Dict[str, str]] = []
            for r in (res or [])[:max_results]:
                out.append(
                    {
                        "title": (r.get("title") or "")[:200],
                        "url": (r.get("href") or r.get("url") or "")[:700],
                        "snippet": (r.get("body") or "")[:900],
                        "provider": "ddg",
                    }
                )
            return out
        except Exception:
            return []

    @staticmethod
    def google_search(query: str) -> str:
        """
        Returns JSON:
        {
          "query": "...",
          "serper": [ {title,url,snippet,provider}, ... ],
          "ddg":    [ ... ],
          "ts_ms":  ...
        }
        """
        query = (query or "").strip()
        if not query:
            return json.dumps({"query": query, "serper": [], "ddg": [], "ts_ms": _now_ms()}, ensure_ascii=False)

        serper_hits = RealToolkit._serper_search(query)
        ddg_hits = RealToolkit._ddg_search(query)

        payload = {
            "query": query,
            "serper": serper_hits,
            "ddg": ddg_hits,
            "ts_ms": _now_ms(),
        }
        return json.dumps(payload, ensure_ascii=False)

    # ---------- Evidence heuristics (stance-aware, conservative) ----------
    @staticmethod
    def _is_hoaxy_claim(clean_fact: str) -> bool:
        """
        Enhanced hoax claim detection with semantic patterns and context awareness.
        Returns True if the claim appears to be asserting a conspiracy theory or hoax.
        """
        s = _norm_text(clean_fact)
        
        # Direct hoax markers (original)
        hoax_markers = [
            "hoax",
            "staged",
            "fake",
            "faked",
            "fabricated",
            "soundstage",
            "filmed in a studio",
            "filmed on a set",
            "conspiracy",
            "cover-up",
            "cover up",
            "proof it was",
            "proving it was",
            "would have killed",
            "could not have",
            "impossible that",
            "never happened",
        ]
        
        # Extended semantic patterns for conspiracy theories
        semantic_patterns = [
            r"\b(government|media|elite|they)\s+(is|are|was|were)?\s*(hiding|concealing|covering up|hid|concealed)",
            r"\bmainstream media (lies|lying|lied)",
            r"\bwake up\b.*\btruth",
            r"\bsheeple\b",
            r"\bfalse flag\b",
            r"\bpsyop\b",
            r"\bdeep state\b",
            r"\bnew world order\b",
            r"\billuminati\b",
            r"\bchemtrails\b",
            r"\bcrisis actors\b",
            r"\bpaid actors\b",
            r"\bcontrolled (by|opposition)",
            r"\b(actually|really|truly) (never|didn't) (happen|occur|exist)",
            r"\bproof (that|it) (was|is) (fake|staged|hoax)",
            r"\bevidence (of|that) (cover-?up|conspiracy)",
            r"\b(can't|cannot) be real",
            r"\bphysically impossible\b",
            r"\bdefies (physics|logic|science)",
            r"\bno way (this|that) (happened|is real)",
        ]
        
        # Check direct markers
        if any(m in s for m in hoax_markers):
            return True
        
        # Check semantic patterns
        for pattern in semantic_patterns:
            if re.search(pattern, s):
                return True
        
        return False

    @staticmethod
    def _detect_negation_context(text: str, cue_position: int, window: int = 30) -> bool:
        """
        Check if a cue at given position appears in a negation context.
        Handles double negatives (e.g., "not never" = affirm).
        Returns True if negation is detected within the window.
        """
        start = max(0, cue_position - window)
        context = text[start:cue_position].lower()
        
        negation_words = [
            "not", "no", "never", "neither", "nor", "none", "nobody", "nothing",
            "nowhere", "hardly", "scarcely", "barely", "without", "lack", "lacking",
            "n't", "nt", "cannot", "can't", "won't", "wouldn't", "shouldn't",
            "couldn't", "doesn't", "didn't", "don't", "isn't", "aren't", "wasn't", "weren't"
        ]
        
        # Check for negation words in context
        words = context.split()
        
        # Count negations in last 5 words
        neg_count = sum(1 for w in words[-5:] if w in negation_words)
        
        # Odd number of negations = negated
        # Even number of negations = not negated (double negative)
        # 0 negations = not negated
        return neg_count % 2 == 1

    @staticmethod
    def _has_refute_cues(text_blob: str) -> bool:
        """
        Enhanced refute cue detection with expanded vocabulary and negation handling.
        Returns True if text contains cues that contradict/refute a claim.
        """
        b = _norm_text(text_blob)
        
        # Expanded refute cues (50+ patterns)
        refute_cues = [
            # Original cues
            "conspiracy theory",
            "conspiracy theories",
            "claims that",
            "claim that",
            "alleged",
            "myth",
            "misconception",
            "debunk",
            "debunked",
            "refute",
            "refuted",
            "false",
            "no evidence",
            "without evidence",
            "lack of evidence",
            "disproved",
            "hoax claim",
            "hoax claims",
            "pseudoscience",
            
            # New additions - direct refutation
            "incorrect",
            "inaccurate",
            "untrue",
            "not true",
            "not accurate",
            "not correct",
            "factually wrong",
            "factually incorrect",
            "proven false",
            "shown to be false",
            "demonstrated to be false",
            
            # Evidence-based refutation
            "contradicts",
            "contradicted by",
            "contrary to",
            "inconsistent with",
            "conflicts with",
            "disputed by",
            "challenged by",
            "questioned by",
            
            # Epistemic markers of doubt
            "unsubstantiated",
            "unverified",
            "unfounded",
            "baseless",
            "groundless",
            "unsupported",
            "lacks support",
            "lacks credibility",
            
            # Fact-checking language
            "fact check",
            "fact-check",
            "rated false",
            "pants on fire",
            "mostly false",
            "misleading",
            "misrepresents",
            "misrepresented",
            
            # Scientific refutation
            "no scientific evidence",
            "scientifically unsound",
            "not supported by science",
            "rejected by experts",
            "consensus disagrees",
        ]
        
        # Check for refute cues with negation awareness
        for cue in refute_cues:
            if cue in b:
                # Find position of cue
                pos = b.find(cue)
                # Check if it's negated (e.g., "not debunked" should NOT count as refute)
                if not RealToolkit._detect_negation_context(b, pos):
                    return True
        
        return False

    @staticmethod
    def _compute_refute_intensity(text_blob: str) -> float:
        """
        Compute intensity score [0,1] for refutation strength with density normalization.
        Higher scores indicate stronger, more definitive refutation.
        Accounts for text length to avoid bias from long texts.
        """
        b = _norm_text(text_blob)
        word_count = len(b.split())
        
        # Strong refutation markers (0.8-1.0)
        strong_markers = [
            "proven false", "demonstrated to be false", "definitively false",
            "completely false", "entirely false", "thoroughly debunked",
            "scientifically impossible", "physically impossible",
            "rated false", "pants on fire"
        ]
        
        # Moderate refutation markers (0.5-0.7)
        moderate_markers = [
            "debunked", "refuted", "disproved", "false", "incorrect",
            "inaccurate", "myth", "misconception", "conspiracy theory",
            "no evidence", "lacks evidence", "unsubstantiated"
        ]
        
        # Weak refutation markers (0.3-0.5)
        weak_markers = [
            "disputed", "questioned", "challenged", "alleged",
            "claims that", "claim that", "unverified", "misleading"
        ]
        
        strong_count = 0
        moderate_count = 0
        weak_count = 0
        
        for marker in strong_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    strong_count += 1
        
        for marker in moderate_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    moderate_count += 1
        
        for marker in weak_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    weak_count += 1
        
        # Compute density-adjusted intensity (markers per 100 words)
        if word_count == 0:
            return 0.0
        
        strong_density = strong_count / max(1, word_count / 100)
        moderate_density = moderate_count / max(1, word_count / 100)
        weak_density = weak_count / max(1, word_count / 100)
        
        # Weighted combination with density adjustment
        max_intensity = 0.0
        if strong_density > 0:
            max_intensity = max(max_intensity, min(0.9, 0.7 + strong_density * 0.2))
        if moderate_density > 0:
            max_intensity = max(max_intensity, min(0.7, 0.5 + moderate_density * 0.2))
        if weak_density > 0:
            max_intensity = max(max_intensity, min(0.5, 0.3 + weak_density * 0.2))
        
        return max_intensity

    @staticmethod
    def _has_affirm_cues(text_blob: str) -> bool:
        """
        Enhanced affirm cue detection with expanded vocabulary and negation handling.
        Returns True if text contains cues that support/confirm a claim.
        """
        b = _norm_text(text_blob)
        
        # Expanded affirm cues (40+ patterns)
        affirm_cues = [
            # Original cues
            "confirmed",
            "evidence shows",
            "evidence that",
            "was held in",
            "hosted in",
            "took place in",
            "returned samples",
            "brought back",
            "successfully landed",
            "landed on the moon",
            "astronauts walked",
            "mission returned",
            "official records",
            
            # New additions - verification language
            "verified",
            "verified by",
            "authenticated",
            "validated",
            "substantiated",
            "corroborated",
            "proven",
            "proven true",
            "proven to be true",
            "demonstrated",
            "established",
            
            # Evidence-based affirmation
            "supported by evidence",
            "backed by evidence",
            "evidence confirms",
            "evidence supports",
            "data shows",
            "data confirms",
            "studies show",
            "research shows",
            "research confirms",
            "findings show",
            "findings confirm",
            
            # Expert consensus
            "experts agree",
            "scientific consensus",
            "widely accepted",
            "generally accepted",
            "consensus is",
            "agreed upon",
            
            # Factual statements
            "it is true that",
            "indeed",
            "in fact",
            "actually occurred",
            "actually happened",
            "did occur",
            "did happen",
            "took place",
            "occurred in",
            "happened in",
            
            # Documentary evidence
            "documented",
            "recorded",
            "on record",
            "historical record",
            "archives show",
            "records indicate",
        ]
        
        # Check for affirm cues with negation awareness
        # Important: check each occurrence separately to handle compound sentences
        found_affirm = False
        for cue in affirm_cues:
            pos = 0
            while True:
                pos = b.find(cue, pos)
                if pos == -1:
                    break
                # Check if this specific occurrence is negated
                if not RealToolkit._detect_negation_context(b, pos):
                    found_affirm = True
                    break
                pos += len(cue)
            if found_affirm:
                break
        
        return found_affirm

    @staticmethod
    def _compute_affirm_intensity(text_blob: str) -> float:
        """
        Compute intensity score [0,1] for affirmation strength with density normalization.
        Higher scores indicate stronger, more definitive support.
        Accounts for text length to avoid bias from long texts.
        """
        b = _norm_text(text_blob)
        word_count = len(b.split())
        
        # Strong affirmation markers (0.8-1.0)
        strong_markers = [
            "proven", "proven true", "definitively true", "conclusively proven",
            "scientifically proven", "verified", "authenticated", "confirmed",
            "scientific consensus", "experts agree", "established fact"
        ]
        
        # Moderate affirmation markers (0.5-0.7)
        moderate_markers = [
            "evidence shows", "evidence supports", "data confirms",
            "studies show", "research shows", "documented", "recorded",
            "widely accepted", "generally accepted", "substantiated"
        ]
        
        # Weak affirmation markers (0.3-0.5)
        weak_markers = [
            "suggests", "indicates", "appears to", "seems to",
            "likely", "probably", "reportedly", "allegedly true"
        ]
        
        strong_count = 0
        moderate_count = 0
        weak_count = 0
        
        for marker in strong_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    strong_count += 1
        
        for marker in moderate_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    moderate_count += 1
        
        for marker in weak_markers:
            if marker in b:
                pos = b.find(marker)
                if not RealToolkit._detect_negation_context(b, pos):
                    weak_count += 1
        
        # Compute density-adjusted intensity (markers per 100 words)
        if word_count == 0:
            return 0.0
        
        strong_density = strong_count / max(1, word_count / 100)
        moderate_density = moderate_count / max(1, word_count / 100)
        weak_density = weak_count / max(1, word_count / 100)
        
        # Weighted combination with density adjustment
        max_intensity = 0.0
        if strong_density > 0:
            max_intensity = max(max_intensity, min(0.9, 0.7 + strong_density * 0.2))
        if moderate_density > 0:
            max_intensity = max(max_intensity, min(0.7, 0.5 + moderate_density * 0.2))
        if weak_density > 0:
            max_intensity = max(max_intensity, min(0.5, 0.3 + weak_density * 0.2))
        
        return max_intensity

    @staticmethod
    def _semantic_anchor_match(anchors: set, blob: str) -> Tuple[int, float]:
        """
        Enhanced anchor matching with fuzzy matching and semantic weighting.
        Returns: (exact_matches, weighted_score)
        """
        from difflib import SequenceMatcher
        
        blob_n = _norm_text(blob)
        blob_words = set(blob_n.split())
        
        exact_matches = 0
        weighted_score = 0.0
        
        for anchor in anchors:
            if not anchor:
                continue
            
            anchor_n = _norm_text(str(anchor))
            
            # Exact match
            if anchor_n in blob_n:
                # Weight by anchor type
                if re.match(r'^\d{4}$', str(anchor)):  # Year
                    weighted_score += 2.0  # Increased from 1.5
                elif len(str(anchor)) > 8:  # Likely entity/proper noun
                    weighted_score += 2.5  # Increased from 2.0
                else:  # Regular keyword
                    weighted_score += 1.5  # Increased from 1.0
                exact_matches += 1
            else:
                # Fuzzy match for typos/variations
                best_ratio = 0.0
                for word in blob_words:
                    if len(word) >= 4 and len(anchor_n) >= 4:
                        ratio = SequenceMatcher(None, anchor_n, word).ratio()
                        best_ratio = max(best_ratio, ratio)
                
                # Accept fuzzy matches above threshold
                if best_ratio >= 0.85:
                    weighted_score += 0.5 * best_ratio
                    exact_matches += 0.5
        
        return int(exact_matches), weighted_score

    # ---------- Numerical Reasoning ----------
    @staticmethod
    def _extract_numbers_with_units(text: str) -> List[Tuple[str, float, str]]:
        """
        Extract numbers with their context and units from text.
        Returns: [(context, value, unit), ...]
        
        Examples:
        - "population 2.2 million" -> [("population", 2200000.0, "")]
        - "8,849 meters tall" -> [("tall", 8849.0, "meters")]
        - "GDP of $20 trillion" -> [("gdp", 20000000000000.0, "")]
        """
        text_lower = text.lower()
        results = []
        
        # Unit multipliers
        multipliers = {
            "trillion": 1e12,
            "billion": 1e9,
            "million": 1e6,
            "thousand": 1e3,
            "hundred": 1e2,
        }
        
        # Length conversions (to meters)
        length_units = {
            "meters": 1.0,
            "meter": 1.0,
            "m": 1.0,
            "feet": 0.3048,
            "foot": 0.3048,
            "ft": 0.3048,
            "kilometers": 1000.0,
            "kilometer": 1000.0,
            "km": 1000.0,
            "miles": 1609.34,
            "mile": 1609.34,
            "mi": 1609.34,
        }
        
        # Pattern for numbers with optional commas and decimals
        number_pattern = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
        
        # Find all numbers in text
        for match in re.finditer(number_pattern, text_lower):
            num_str = match.group(1).replace(",", "")
            try:
                base_value = float(num_str)
            except:
                continue
            
            start_pos = match.start()
            end_pos = match.end()
            
            # Extract context (3 words before)
            context_start = max(0, start_pos - 50)
            context = text_lower[context_start:start_pos].strip()
            context_words = context.split()[-5:] if context else []
            context_str = " ".join(context_words)
            
            # Look for unit/multiplier after number (within 20 chars)
            after_text = text_lower[end_pos:end_pos+20].strip()
            
            # Check for multipliers
            final_value = base_value
            unit = ""
            
            # Use regex with word boundary for correct matching
            # e.g. match "m" but not "million"
            
            found_mult = False
            for mult_word, mult_val in multipliers.items():
                if re.match(r"^" + re.escape(mult_word) + r"\b", after_text):
                    final_value *= mult_val
                    # Consume the multiplier text logic would be better, but for now 
                    # we just need to ensure we don't double match if we don't consume.
                    # But to handle "2 million meters", we should ideally look for unit AFTER multiplier.
                    
                    # Let's try to match unit in the REMAINING text if multiplier found
                    match = re.match(r"^" + re.escape(mult_word) + r"\b", after_text)
                    match_end = match.end() if match else 0  # type: ignore[union-attr]
                    remaining = after_text[match_end:].strip()
                    
                    for unit_word, unit_val in length_units.items():
                         if re.match(r"^" + re.escape(unit_word) + r"\b", remaining):
                             unit = "meters"
                             final_value *= unit_val
                             break
                    
                    found_mult = True
                    break
            
            if not found_mult:
                # Check for length units directly (no multiplier)
                for unit_word, unit_val in length_units.items():
                    if re.match(r"^" + re.escape(unit_word) + r"\b", after_text):
                        unit = "meters"
                        final_value *= unit_val
                        break
            
            results.append((context_str, final_value, unit))
        
        return results

    @staticmethod
    def _compare_numerical_values(
        claim_val: float,
        evidence_val: float,
        tolerance: float = 0.1
    ) -> bool:
        """
        Compare numerical values with tolerance.
        tolerance=0.1 means 10% difference is acceptable.
        
        Examples:
        - compare(2200000, 2161000, 0.1) -> True (within 10%)
        - compare(8849, 8850, 0.1) -> True (very close)
        - compare(20, 15, 0.1) -> False (25% difference)
        """
        if claim_val == 0 and evidence_val == 0:
            return True
        
        if claim_val == 0 or evidence_val == 0:
            # One is zero, other is not
            return False
        
        # Calculate relative difference
        max_val = max(abs(claim_val), abs(evidence_val))
        diff = abs(claim_val - evidence_val)
        relative_diff = diff / max_val
        
        return relative_diff <= tolerance

    @staticmethod
    def _verify_numerical_claim(
        claim_numbers: List[Tuple[str, float, str]],
        evidence_numbers: List[Tuple[str, float, str]]
    ) -> Optional[bool]:
        """
        Verify numerical claim against evidence numbers.
        Returns:
        - True if numbers match (within tolerance)
        - False if numbers clearly mismatch
        - None if cannot determine
        """
        if not claim_numbers or not evidence_numbers:
            return None
        
        # Try to match numbers by context similarity
        for claim_ctx, claim_val, claim_unit in claim_numbers:
            for evidence_ctx, evidence_val, evidence_unit in evidence_numbers:
                # If units are specified and different, skip
                if claim_unit and evidence_unit and claim_unit != evidence_unit:
                    continue
                
                # Check if contexts are related (simple word overlap)
                claim_words = set(claim_ctx.split())
                evidence_words = set(evidence_ctx.split())
                
                # If contexts overlap OR both are empty (just numbers)
                if claim_words & evidence_words or (not claim_ctx and not evidence_ctx):
                    # Compare values
                    if RealToolkit._compare_numerical_values(claim_val, evidence_val):
                        return True  # Match found!
        
        # Check if any claim number has a clear mismatch
        # (same context but different value)
        for claim_ctx, claim_val, claim_unit in claim_numbers:
            for evidence_ctx, evidence_val, evidence_unit in evidence_numbers:
                if claim_unit and evidence_unit and claim_unit != evidence_unit:
                    continue
                
                claim_words = set(claim_ctx.split())
                evidence_words = set(evidence_ctx.split())
                
                if claim_words & evidence_words:
                    # Same context but values don't match
                    if not RealToolkit._compare_numerical_values(claim_val, evidence_val, tolerance=0.2):
                        return False  # Clear mismatch!
        
        return None  # Cannot determine

    # ---------- Comparative/Superlative Handling ----------
    @staticmethod
    def _detect_comparative_claim(text: str) -> Optional[Dict[str, Any]]:
        """
        Detect and parse comparative/superlative claims.
        Returns: {
            "type": "comparative" | "superlative",
            "entity": str,
            "attribute": str,
            "comparator": str,
            "reference": str    # comparison target (for comparative)
        }
        """
        text_lower = text.lower()
        
        # Superlative patterns
        # "X is the tallest/most/best..."
        sup_patterns = [
            r"the (most \w+|least \w+|best|worst|tallest|shortest|biggest|smallest|highest|lowest|fastest|slowest|oldest|youngest|first|last) (.+)",
        ]
        
        for p in sup_patterns:
            m = re.search(p, text_lower)
            if m:
                return {
                    "type": "superlative",
                    "comparator": m.group(1),
                    "context": m.group(2)
                }
        
        # Comparative patterns
        # "X is taller/more than Y"
        comp_patterns = [
            r"(.+) is (more \w+|less \w+|better|worse|\w+er) than (.+)",
        ]
        
        for p in comp_patterns:
            m = re.search(p, text_lower)
            if m:
                return {
                    "type": "comparative",
                    "entity_a": m.group(1).strip(" .,!?"),
                    "comparator": m.group(2).strip(),
                    "entity_b": m.group(3).strip(" .,!?")
                }
        
        return None

    @staticmethod
    def _verify_comparative_claim(
        claim_info: Dict[str, Any],
        evidence_text: str
    ) -> Optional[bool]:
        """
        Verify comparative/superlative claims against evidence.
        """
        evidence_lower = evidence_text.lower()
        
        if claim_info["type"] == "superlative":
            comparator = claim_info["comparator"]
            
            # Direct confirmation of superlative
            # e.g. "tallest" -> confirm if evidence says "tallest", "highest peak", "no mountain taller"
            
            # Map comparators to synonyms
            synonyms = {
                "tallest": ["highest", "highest peak", "maximum height"],
                "highest": ["tallest", "highest peak", "maximum height"],
                "biggest": ["largest", "massive", "huge", "gigantic"],
                "largest": ["biggest", "massive", "huge", "gigantic"],
                "fastest": ["quickest", "speed record"],
                "oldest": ["earliest", "ancient", "first"],
                "first": ["initial", "pioneer", "earliest"],
            }
            
            search_terms = [comparator] + synonyms.get(comparator.split()[-1], [])
            
            for term in search_terms:
                if term in evidence_lower:
                    # Check for negation
                    pos = evidence_lower.find(term)
                    if not RealToolkit._detect_negation_context(evidence_lower, pos):
                        return True
            
            # Check for refutation
            if "not the " + comparator in evidence_lower:
                return False
                
            return None

        elif claim_info["type"] == "comparative":
            entity_a = claim_info["entity_a"]
            entity_b = claim_info["entity_b"]
            comparator = claim_info["comparator"]
            
            # 1. Check for direct statement
            # "A is taller than B" or "B is shorter than A"
            if f"{entity_a} is {comparator} than {entity_b}" in evidence_lower:
                return True
            
            # 2. Check for numerical comparison if available
            # This would require extracting numbers for both entities
            # For now, rely on text cues
            
            return None # Difficult to verify without numbers or direct statement

        return None
    
    # ---------- Hybrid Multi-hop Reasoning Enhancement ----------
    @staticmethod
    def _extract_bridge_entities(text: str) -> Dict[str, Any]:
        """
        Stage 1: Enhanced regex-based entity detection.
        Now supports lowercase descriptions and more flexible patterns.
        """
        entities = []
        pattern_type = None
        has_bridge = False
        
        # Helper to clean and extract
        def add_ent(e, p):
            nonlocal has_bridge, pattern_type
            if e and e.strip():
                entities.append(e.strip())
                pattern_type = p
                has_bridge = True

        # Pattern 1: "the [ROLE] that [ENTITY/DESC] [RELATION]"
        # Example: "the artist that the illustrator of Exlex studied with"
        m1 = re.search(
            r"the (artist|painter|athlete|fighter|person|meteorologist|actor|singer|writer) (?:that )?(.+?) (studied with|trained (?:by|with)|worked with|replaced|succeeded|composed (?:for|by))",
            text, re.IGNORECASE
        )
        if m1:
            add_ent(m1.group(2), f"{m1.group(1)}_{m1.group(3)}")

        # Pattern 2: "[ENTITY1] and this/the [DESCRIPTION/ENTITY2] both [ACTION]"
        if not has_bridge:
            m2 = re.search(
                r"(.+?) and (?:this|the) (.+?) both (\w+)",
                text, re.IGNORECASE
            )
            if m2:
                add_ent(m2.group(1), f"dual_entity_{m2.group(3)}")

        # Pattern 3: Temporal/Location evolution
        if not has_bridge:
            m3 = re.search(
                r"at (?:an|a|the) (.+?) that (?:currently|formerly|now|previously) (.+)",
                text, re.IGNORECASE
            )
            if m3:
                add_ent(m3.group(1), "place_evolution")
        
        return {
            "has_bridge": has_bridge,
            "entities": entities,
            "pattern_type": pattern_type
        }
    
    @staticmethod
    def _decompose_multihop_claim(clean_fact: str, entity_info: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
        """
        Stage 2: LLM-powered query decomposition with entity hints.
        Returns list of sub-queries with metadata.
        """
        if not entity_info.get("has_bridge"):
            return None
        
        entities_str = ", ".join(entity_info.get("entities", []))
        pattern = entity_info.get("pattern_type", "unknown")
        
        prompt = f"""Decompose this multi-hop claim into 2-3 sequential sub-questions for fact-checking.

Claim: "{clean_fact}"
Known entities: [{entities_str}]
Pattern type: {pattern}

Rules:
1. First question should resolve the "bridge entity" (the unknown person/thing/place)
2. Use [ENTITY_1], [ENTITY_2] as placeholders in subsequent questions
3. Each question must be searchable on Google (keep it simple)
4. Limit to 3 questions maximum

Output strict JSON:
{{
  "sub_queries": [
    {{"query": "Who/What question to find bridge entity", "entity_type": "person|place|organization"}},
    {{"query": "Verification question using [ENTITY_1]", "entity_type": "attribute"}}
  ]
}}"""
        
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            data = _safe_json_loads(raw)
            if not isinstance(data, dict):
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None
            
            if isinstance(data, dict) and "sub_queries" in data:
                sub_queries = data["sub_queries"]
                if isinstance(sub_queries, list) and len(sub_queries) >= 2:
                    return sub_queries[:3]  # Limit to 3
        except Exception as e:
            print(f"        ⚠️ Multi-hop decomposition failed: {e}")
            return None
        
        return None
    
    @staticmethod
    def _extract_entity_from_evidence(evidence_text: str, entity_type: str, query_context: str) -> str:
        """
        Extract specific entity from search results using LLM.
        Returns the extracted entity name.
        """
        prompt = f"""From the search results below, extract the answer to this question.
Return ONLY the entity name (person/place/thing), nothing else.

Question: {query_context}
Expected type: {entity_type}

Search results:
{evidence_text[:600]}

Answer (entity name only, max 5 words):"""
        
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            entity_name = (res.choices[0].message.content or "").strip().strip('"').strip("'")
            # Clean up common artifacts
            entity_name = entity_name.split("\n")[0]  # Take first line only
            entity_name = entity_name[:50]  # Max 50 chars
            return entity_name if len(entity_name) < 50 else ""
        except Exception:
            return ""
    
    @staticmethod
    def _execute_hybrid_multihop(clean_fact: str, sub_queries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Stage 3: Sequential execution of sub-queries with entity resolution.
        Returns combined evidence from all search steps.
        """
        all_hits = []
        entity_cache = {}
        
        print(f"        🧩 Multi-hop mode: {len(sub_queries)} steps detected")
        
        for i, sub_q_info in enumerate(sub_queries):
            query = sub_q_info.get("query", "")
            entity_type = sub_q_info.get("entity_type", "unknown")
            
            # Substitute cached entities into query
            for placeholder, value in entity_cache.items():
                query = query.replace(placeholder, value)
            
            if not query.strip():
                continue
            
            # Execute search for this step
            print(f"        🔗 Step {i+1}/{len(sub_queries)}: {query[:80]}...")
            
            try:
                serper_hits = RealToolkit._serper_search(query, num=6)
                ddg_hits = RealToolkit._ddg_search(query, max_results=6)
                step_hits = serper_hits + ddg_hits
                all_hits.extend(step_hits)
                
                # Extract entity if needed for next steps
                remaining_steps = sub_queries[i+1:]
                if remaining_steps and any(f"[ENTITY_{i+1}]" in str(s) for s in remaining_steps):
                    evidence_text = _summarize_hits(step_hits, max_n=4)
                    if evidence_text.strip():
                        entity_name = RealToolkit._extract_entity_from_evidence(
                            evidence_text, entity_type, query
                        )
                        if entity_name:
                            entity_cache[f"[ENTITY_{i+1}]"] = entity_name
                            print(f"        ├─ Resolved [ENTITY_{i+1}] = '{entity_name}'")
                        else:
                            print(f"        ├─ ⚠️ Failed to extract [ENTITY_{i+1}]")
                
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                print(f"        ├─ ⚠️ Step {i+1} search error: {e}")
                continue
        
        print(f"        └─ Multi-hop complete: {len(all_hits)} total evidence snippets")
        return all_hits
    
    @staticmethod
    def _detect_multi_hop_claim(text: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced multi-hop detection with LLM fallback.
        """
        # 1. High-speed Regex Detection
        entity_info = RealToolkit._extract_bridge_entities(text)
        if entity_info.get("has_bridge"):
            return {
                "type": "bridge_entity",
                "hint": f"Regex-detected {entity_info.get('pattern_type')}",
                "entity_info": entity_info
            }
        
        # 2. LLM Fallback Detection (for complex phrasing regex missed)
        if len(text.split()) > 10:  # Only for long, complex claims
            prompt = f"""Does the statement below require finding a 'bridge entity' to verify?
(e.g., finding who someone studied with, or what an arena used to be called)

Statement: "{text}"

Answer YES or NO (strictly 1 word):"""
            try:
                res = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=2
                )
                ans = (res.choices[0].message.content or "").strip().upper()
                if "YES" in ans:
                    # Fabricate entity_info for LLM decomposition stage to handle
                    return {
                        "type": "bridge_entity",
                        "hint": "LLM-detected complex relation",
                        "entity_info": {"has_bridge": True, "entities": [], "pattern_type": "complex"}
                    }
            except:
                pass

        # 3. Standard hop patterns (possessives/etc)
        text_lower = text.lower()
        
        # Possessive patterns: "X's Y", "Y of X"
        if "'s " in text_lower:
            return {"type": "possessive", "hint": "Check relationships (possessives) between entities"}
            
        # Relation patterns
        relations = [
            "wife of", "husband of", "son of", "daughter of", "mother of", "father of",
            "author of", "creator of", "inventor of", "founder of", "ceo of",
            "capital of", "president of", "leader of", "member of"
        ]
        
        for rel in relations:
            if rel in text_lower:
                return {"type": "relation", "hint": f"Trace the '{rel}' relationship"}
        
        # Implicit chains (harder to detect using simple regex)
        # But we can look for specific entity bridging phrases
        bridging = [
            "same as", "different from", "related to", "based on"
        ]
        for bridge in bridging:
            if bridge in text_lower:
                return {"type": "chain", "hint": "Follow the connection between entities"}
                
        return None

    @staticmethod
    def _extract_temporal_context(text: str) -> List[str]:
        """
        Extract temporal expressions from text.
        Returns list of year strings found.
        """
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        return years

    @staticmethod
    def _check_temporal_consistency(claim_years: List[str], evidence_years: List[str]) -> bool:
        """
        Check if temporal context in claim matches evidence.
        More lenient to avoid false negatives from historical context.
        Returns True if consistent or no temporal conflict detected.
        """
        if not claim_years or not evidence_years:
            return True  # No temporal info to check
        
        # Check for year mismatches
        claim_year_set = set(claim_years)
        evidence_year_set = set(evidence_years)
        
        # If there's overlap, consider consistent
        if claim_year_set & evidence_year_set:
            return True
        
        # If evidence has MANY years (>= 5), likely historical/background context
        # Don't penalize for this
        if len(evidence_year_set) >= 5:
            return True
        
        # Check if years are close (within 5 years for more leniency)
        min_diff = float('inf')
        for cy in claim_years:
            for ey in evidence_years:
                try:
                    diff = abs(int(cy) - int(ey))
                    min_diff = min(min_diff, diff)
                    if diff <= 5:  # More lenient threshold
                        return True
                except:
                    pass
        
        # Only flag as inconsistent if years are VERY different (>5 years)
        # and evidence doesn't have many years (historical context)
        return min_diff <= 5

    @staticmethod
    def _evidence_heuristic_verdict(clean_fact: str, hits: List[Dict[str, str]]) -> Tuple[Optional[bool], str]:
        """
        ENHANCED Stance-aware heuristic with:
        - Semantic pattern-based hoax detection
        - Expanded cue vocabulary with negation handling
        - Intensity-based scoring
        - Semantic anchor matching
        - Temporal consistency checking
        - Generalized pattern matching
        
        Returns: (verdict, strength)
          - verdict: True / False / None(ABSTAIN)
          - strength: "strong" or "weak" (only meaningful if verdict is not None)
        
        Policy:
          - For hoax-like claims:
              if trusted snippets have strong refute cues => FALSE (strong)
              never return TRUE from mere keyword overlap
          - For non-hoax:
              return TRUE only with strong affirm cues AND good anchor coverage (strong)
          - Use intensity scoring to determine strength
          - Check temporal consistency
          - Otherwise return None
        """
        hoaxy = RealToolkit._is_hoaxy_claim(clean_fact)

        trusted_hits = [h for h in (hits or []) if _is_trusted_domain(h.get("url", ""))]
        if not trusted_hits:
            return None, "weak"
        if len(trusted_hits) < 2 and not RealToolkit._is_hoaxy_claim(clean_fact):
            return None, "weak"

        # Weighted blob by source quality
        weighted_parts = []
        for h in trusted_hits:
            url = h.get("url", "")
            weight = 1.0 if _is_trusted_domain(url) else 0.5
            snippet = (h.get("title", "") + " " + h.get("snippet", ""))
            weighted_parts.append((snippet + " ") * int(max(1, round(weight))))
        blob = " ".join(weighted_parts)
        blob_n = _norm_text(blob)

        # Compute stance intensities
        refute_intensity = RealToolkit._compute_refute_intensity(blob_n)
        affirm_intensity = RealToolkit._compute_affirm_intensity(blob_n)

        # For hoax claims: strong refutation => FALSE
        if hoaxy:
            if RealToolkit._has_refute_cues(blob_n):
                strength = "strong" if refute_intensity >= 0.6 else "weak"
                return False, strength
            # Never affirm hoax claims from heuristic alone
            return None, "weak"

        fact = _norm_text(clean_fact)

        # Temporal consistency check
        claim_years = RealToolkit._extract_temporal_context(clean_fact)
        evidence_years = RealToolkit._extract_temporal_context(blob)
        temporal_consistent = RealToolkit._check_temporal_consistency(claim_years, evidence_years)
        
        if not temporal_consistent:
            # Temporal mismatch suggests claim might be false for the stated time
            return False, "strong"

        # Check for numerical claims (GAP 1)
        claim_numbers = RealToolkit._extract_numbers_with_units(clean_fact)
        evidence_numbers = RealToolkit._extract_numbers_with_units(blob)

        if claim_numbers and evidence_numbers:
            numerical_match = RealToolkit._verify_numerical_claim(
                claim_numbers, evidence_numbers
            )
            if numerical_match is True:
                # Strong signal: numbers match exactly
                return True, "strong (numerical)"
            elif numerical_match is False:
                # Strong signal: numbers mismatch clearly
                return False, "strong (numerical)"

        # Check for comparative/superlative claims (GAP 2)
        comp_info = RealToolkit._detect_comparative_claim(clean_fact)
        if comp_info:
            comp_result = RealToolkit._verify_comparative_claim(comp_info, blob)
            if comp_result is not None:
                # Comparative verification is usually strong
                return comp_result, "strong (comparative)"

        # Extract anchors with ENTITY PRIORITY
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_fact)
        anchors = set(years)

        # Use existing entity extraction for better anchor quality
        entities = RealToolkit._extract_entities(clean_fact)
        for ent in entities:
            anchors.add(ent.lower())

        # Only add keywords if we don't have enough anchors
        if len(anchors) < 5:
            words = re.findall(r"[a-zA-Z]{4,}", clean_fact.lower())
            stop = {
                "that", "this", "with", "from", "were", "held", "host", "hosted",
                "equals", "year", "percent", "square", "root", "claim", "proof",
                "about", "would", "could", "should", "because", "therefore"
            }
            for w in words:
                if w not in stop:
                    anchors.add(w)
                if len(anchors) >= 10:
                    break

        # Semantic anchor matching
        exact_matches, weighted_score = RealToolkit._semantic_anchor_match(anchors, blob_n)

        # Decision logic with CONFLICT RESOLUTION
        has_affirm = RealToolkit._has_affirm_cues(blob_n)
        has_refute = RealToolkit._has_refute_cues(blob_n)

        # CONFLICT: both affirm and refute present
        if has_affirm and has_refute:
            # Use intensity difference to decide
            intensity_diff = affirm_intensity - refute_intensity
            
            if abs(intensity_diff) < 0.2:
                # Too close to call - signals are balanced
                return None, "weak"
            elif intensity_diff > 0.2:
                # Affirm dominates
                if affirm_intensity >= 0.6 and weighted_score >= 4.0:
                    return True, "weak"  # Weak because of conflict
                elif affirm_intensity >= 0.5 and weighted_score >= 5.0:
                    return True, "weak"
            else:
                # Refute dominates (intensity_diff < -0.2)
                if refute_intensity >= 0.6:
                    return False, "weak"  # Weak because of conflict
                elif refute_intensity >= 0.5:
                    return False, "weak"
            
            # Conflict but no clear winner
            return None, "weak"

        # NO CONFLICT: only refute
        if has_refute:
            if refute_intensity >= 0.6:
                return False, "strong"
            elif refute_intensity >= 0.4:
                return False, "weak"
        
        # NO CONFLICT: only affirm
        if has_affirm:
            # Strong affirmation with good coverage
            if affirm_intensity >= 0.6 and weighted_score >= 4.0:
                return True, "strong"
            # Moderate affirmation with excellent coverage
            elif weighted_score >= 6.0 and affirm_intensity >= 0.5:
                return True, "strong"
            # Weak signals
            elif affirm_intensity >= 0.4 and weighted_score >= 3.0:
                return True, "weak"

        return None, "weak"

    # ---------- LLM Judges ----------
    @staticmethod
    def _llm_common_sense_vote(clean_fact: str) -> Optional[bool]:
        prompt = f"""
 You are a careful fact checker.
 Decide whether the statement is TRUE or FALSE using ONLY widely-known knowledge.
 If you are not highly confident, output 'ABSTAIN'.

 Statement: "{clean_fact}"

 Reply with ONLY one of: TRUE, FALSE, ABSTAIN.
 """
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            ans = (res.choices[0].message.content or "").strip().upper()
            if "ABSTAIN" in ans:
                return None
            if "TRUE" in ans:
                return True
            if "FALSE" in ans:
                return False
            return None
        except Exception:
            RealToolkit._llm_error_flag = True
            return None


    @staticmethod
    def _llm_rag_judge_raw(clean_fact: str, evidence_lines: str) -> Tuple[Optional[bool], float, List[int], List[int], str]:
        """
        RAG + judge: stance classification with citations to snippet ids.
        Returns: (verdict, llm_confidence, support_ids, refute_ids, rationale_short)
        """
        multihop_info = RealToolkit._detect_multi_hop_claim(clean_fact)
        hint = ""
        if multihop_info:
            hint = f"\n  IMPORTANT: {multihop_info.get('hint', '')}\n  This claim requires MULTI-HOP reasoning. Combine information from multiple snippets."

        prompt = f"""
 You are an evidence-based fact checker.
 
 Fact:
 {clean_fact}{hint}
 
 Evidence snippets (numbered):
 {evidence_lines}
 
 Task:
 1) Decide whether the evidence SUPPORTS the fact (TRUE), CONTRADICTS it (FALSE), or is INSUFFICIENT (ABSTAIN).
 2) Cite which snippet numbers support (support_ids) and which contradict (refute_ids).
 3) Provide confidence in [0,1] (only high if evidence is direct and unambiguous).
 4) Keep rationale under 2 sentences.
 
 Output STRICT JSON with keys:
 - "verdict": "TRUE" | "FALSE" | "ABSTAIN"
 - "confidence": number
 - "support_ids": array of integers
 - "refute_ids": array of integers
 - "rationale": string
 """
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            data = _safe_json_loads(raw)
            if not isinstance(data, dict):
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None


            if not isinstance(data, dict):
                return None, 0.0, [], [], "Judge parse failed"

            v = str(data.get("verdict", "")).strip().upper()
            conf = float(data.get("confidence", 0.0) or 0.0)
            sup = data.get("support_ids") or []
            ref = data.get("refute_ids") or []
            rat = str(data.get("rationale", "") or "")[:260]

            def _as_int_list(x) -> List[int]:
                out = []
                if isinstance(x, list):
                    for t in x:
                        try:
                            out.append(int(t))
                        except Exception:
                            pass
                return out

            sup_i = _as_int_list(sup)
            ref_i = _as_int_list(ref)
            conf = max(0.0, min(1.0, conf))

            if v == "TRUE":
                return True, conf, sup_i, ref_i, rat
            if v == "FALSE":
                return False, conf, sup_i, ref_i, rat
            return None, conf, sup_i, ref_i, rat
        except Exception:
            RealToolkit._llm_error_flag = True
            return None, 0.0, [], [], "Judge error"


    @staticmethod
    def _llm_semantic_fallback(clean_fact: str, evidence_lines: str) -> Tuple[Optional[bool], float, str]:
        """
        Semantic reasoning fallback - focuses on INDIRECT evidence and INFERENCE.
        Different from RAG judge which requires DIRECT evidence.
        This is used when RAG judge finds evidence insufficient for direct verification.
        Returns: (verdict, confidence, rationale)
        """
        prompt = f"""
You are a reasoning expert. The previous judge found the evidence insufficient for DIRECT verification.
Your task: use INDIRECT evidence, INFERENCE, and REASONING to judge the claim.

Claim:
{clean_fact}

Evidence snippets (may be indirect):
{evidence_lines}

Instructions:
1. Look for INDIRECT support (related facts, context, implications)
2. Use LOGICAL INFERENCE from the evidence
3. Consider what the evidence IMPLIES even if not stated directly
4. Consider domain knowledge and common patterns
5. Only ABSTAIN if evidence is completely irrelevant or contradictory

Confidence guidelines:
- Use 0.5-0.7 for reasonable inferences from indirect evidence
- Use 0.7-0.9 for strong logical implications
- Only use >0.9 if you find overlooked direct evidence

Output STRICT JSON with keys:
- "verdict": "TRUE" | "FALSE" | "ABSTAIN"
- "confidence": number (be more lenient than direct evidence judge)
- "rationale": explain your REASONING PROCESS and what you inferred
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            data = _safe_json_loads(raw)
            if not isinstance(data, dict):
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None

            if not isinstance(data, dict):
                return None, 0.0, "Semantic judge parse failed"

            v = str(data.get("verdict", "")).strip().upper()
            conf = float(data.get("confidence", 0.0) or 0.0)
            rat = str(data.get("rationale", "") or "")[:300]  # Longer rationale for reasoning
            conf = max(0.0, min(1.0, conf))

            if v == "TRUE":
                return True, conf, rat
            if v == "FALSE":
                return False, conf, rat
            return None, conf, rat
        except Exception:
            RealToolkit._llm_error_flag = True
            return None, 0.0, "Semantic judge error"


    # ---------- Confidence calibration ----------
    @staticmethod
    def _evidence_confidence_proxy(
        clean_fact: str,
        evidence_hits: List[Dict[str, str]],
        verdict: Optional[bool],
        support_ids: List[int],
        refute_ids: List[int],
    ) -> float:
        """
        Evidence-based confidence proxy in [0,1], reviewer-friendly:
        - domain diversity among trusted sources
        - citation consistency: if verdict TRUE, needs more support citations than refute (and vice versa)
        - directness proxy: overlap of key anchors (NOT used to auto-TRUE)
        Conservative combination; meant to CLAMP LLM confidence.
        """
        hits = evidence_hits or []
        trusted = [h for h in hits if _is_trusted_domain(h.get("url", ""))]

        dom_div = _distinct_domains(trusted)
        dom_score = min(1.0, dom_div / 3.0)  # 3 distinct trusted domains -> 1.0

        sup = len(support_ids or [])
        ref = len(refute_ids or [])

        cite_score = 0.0
        if sup + ref > 0:
            if verdict is True:
                cite_score = sup / max(1, sup + ref)
            elif verdict is False:
                cite_score = ref / max(1, sup + ref)
            else:
                cite_score = 0.0

        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_fact or "")
        key_terms = re.findall(r"[a-zA-Z]{5,}", (clean_fact or "").lower())[:6]
        anchors = set(years + key_terms)

        blob = " ".join(
            [(h.get("title", "") + " " + h.get("snippet", "")) for h in (trusted[:8] if trusted else hits[:8])]
        )
        blob_n = _norm_text(blob)
        overlap = sum(1 for a in anchors if a and _norm_text(a) in blob_n)
        direct_score = min(1.0, overlap / 4.0)

        econf = 0.45 * dom_score + 0.35 * cite_score + 0.20 * direct_score
        if not trusted:
            econf *= 0.35

        return max(0.0, min(1.0, econf))

    @staticmethod
    def _rag_judge_with_calibrated_conf(
        clean_fact: str,
        evidence_lines: str,
        evidence_hits: List[Dict[str, str]],
    ) -> JudgeResult:
        v, llm_conf, sup_ids, ref_ids, rat = RealToolkit._llm_rag_judge_raw(clean_fact, evidence_lines)
        econf = RealToolkit._evidence_confidence_proxy(clean_fact, evidence_hits, v, sup_ids, ref_ids)
        final_conf = min(max(0.0, min(1.0, llm_conf)), econf)
        return JudgeResult(
            verdict=v,
            llm_confidence=max(0.0, min(1.0, llm_conf)),
            evidence_confidence=econf,
            final_confidence=final_conf,
            support_ids=sup_ids,
            refute_ids=ref_ids,
            rationale=rat,
        )

    # ---------- Voting (IMPROVED with weights) ----------
    @staticmethod
    def _vote_weighted(
        v_serper: Optional[bool],
        v_ddg: Optional[bool],
        v_llm: Optional[bool],
        serper_trusted: int = 0,
        ddg_trusted: int = 0,
    ) -> Optional[bool]:
        """
        Weighted voting that considers trusted source counts.
        Serper/DDG votes weighted by trusted hit ratio.
        LLM vote gets base weight of 1.0.
        """
        true_weight = 0.0
        false_weight = 0.0
        total_weight = 0.0

        # Serper weight based on trusted hits (0.5 to 1.5)
        serper_w = 0.5 + min(1.0, serper_trusted / 4.0)
        if v_serper is True:
            true_weight += serper_w
            total_weight += serper_w
        elif v_serper is False:
            false_weight += serper_w
            total_weight += serper_w

        # DDG weight
        ddg_w = 0.5 + min(1.0, ddg_trusted / 4.0)
        if v_ddg is True:
            true_weight += ddg_w
            total_weight += ddg_w
        elif v_ddg is False:
            false_weight += ddg_w
            total_weight += ddg_w

        # LLM base weight
        llm_w = 1.0
        if v_llm is True:
            true_weight += llm_w
            total_weight += llm_w
        elif v_llm is False:
            false_weight += llm_w
            total_weight += llm_w

        if total_weight < 0.5:
            return None

        # Require >60% weight for decision
        if true_weight / total_weight >= 0.6:
            return True
        if false_weight / total_weight >= 0.6:
            return False
        return None

    @staticmethod
    def _vote_2_of_3(v_serper: Optional[bool], v_ddg: Optional[bool], v_llm: Optional[bool]) -> Optional[bool]:
        """Legacy simple voting for backward compatibility."""
        votes = [v_serper, v_ddg, v_llm]
        t = sum(1 for v in votes if v is True)
        f = sum(1 for v in votes if v is False)
        if t >= 2:
            return True
        if f >= 2:
            return False
        return None

    # ---------- Attack/Support relation checks (LLM) ----------
    @staticmethod
    def _llm_relation_judge(statement_a: str, statement_b: str, mode: Literal["ATTACK", "SUPPORT"]) -> Tuple[str, float]:
        """
        Returns (label, confidence) where label in {"TRUE","FALSE","ABSTAIN"}.
        Note: This confidence is self-reported; we use it only for pruning when confident FALSE.
        """
        a_txt = _norm_text(statement_a)
        b_txt = _norm_text(statement_b)
        a_tokens = set(re.findall(r"[a-z]{3,}", a_txt))
        b_tokens = set(re.findall(r"[a-z]{3,}", b_txt))
        overlap = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))

        # Pre-filter for unrelated statements
        if overlap < 0.2:
            return "FALSE", 0.6 if mode == "ATTACK" else 0.5

        if mode == "ATTACK":
            question = "Does A logically invalidate, contradict, or provide a counter-argument to B?"
            rules = """
- If A says something is TRUE and B says it is FALSE (or vice versa) about the same proposition, output TRUE.
- If A corrects B with a conflicting fact, output TRUE.
- If A is irrelevant or supports B, output FALSE.
- If unclear due to abstraction mismatch, output ABSTAIN.
"""
        else:
            question = "Does A logically support, reinforce, or provide evidence for B?"
            rules = """
- Output TRUE only if A clearly supports B.
- Output FALSE only if A clearly does NOT support B (unrelated or opposite).
- If unclear due to abstraction mismatch, output ABSTAIN.
"""

        prompt = f"""
Task: Relation Check ({mode}).

Statement A: "{_split_sentences_rough(statement_a, 350)}"
Statement B: "{_split_sentences_rough(statement_b, 350)}"

Question: {question}

Rules:
{rules}

Output STRICT JSON:
{{
  "label": "TRUE" | "FALSE" | "ABSTAIN",
  "confidence": number
}}
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = (res.choices[0].message.content or "").strip()
            data = _safe_json_loads(raw)
            if not isinstance(data, dict):
                m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                data = _safe_json_loads(m.group(0)) if m else None
            if not isinstance(data, dict):
                return "ABSTAIN", 0.0
            label = str(data.get("label", "ABSTAIN")).strip().upper()
            conf = float(data.get("confidence", 0.0) or 0.0)
            if label not in ("TRUE", "FALSE", "ABSTAIN"):
                label = "ABSTAIN"
            conf = max(0.0, min(1.0, conf))
            return label, conf
        except Exception:
            return "ABSTAIN", 0.0

    @staticmethod
    def verify_attack(attacker: str, target: str) -> bool:
        """
        Compatibility bool:
          - Return False ONLY when we are confident it is NOT an attack (to prune).
          - Return True only when we are confident it IS an attack.
          - Otherwise return False to avoid false-positive attacks.
        """
        key = RealToolkit._cache_key("attack3", attacker, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        label, conf = RealToolkit._llm_relation_judge(attacker, target, mode="ATTACK")
        out = False
        if label == "TRUE" and conf >= RealToolkit.ATTACK_TRUE_MIN_CONF:
            out = True
        elif label == "FALSE" and conf >= RealToolkit.EDGE_PRUNE_FALSE_CONF:
            out = False

        RealToolkit._cache[key] = out
        return out

    @staticmethod
    def verify_support(source: str, target: str) -> bool:
        """
        Compatibility bool:
          - Return False ONLY when we are confident it is NOT a support edge (to prune).
          - Return True otherwise (keep edge).
        """
        key = RealToolkit._cache_key("support3", source, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        label, conf = RealToolkit._llm_relation_judge(source, target, mode="SUPPORT")
        out = True
        if label == "FALSE" and conf >= RealToolkit.EDGE_PRUNE_FALSE_CONF:
            out = False

        RealToolkit._cache[key] = out
        return out

    # ---------- Coverage gate (IMPROVED) ----------
    @staticmethod
    def _coverage_ok(fact: str, hits: List[Dict[str, str]]) -> bool:
        """Check if search results adequately cover the claim's key terms."""
        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in (hits or [])[:10]])
        blob_n = _norm_text(blob)

        # Extract entities first (higher priority)
        entities = RealToolkit._extract_entities(fact)
        entity_hits = sum(1 for e in entities if _norm_text(e) in blob_n)

        aliases = []
        for ent in entities:
            parts = ent.split()
            if len(parts) >= 2:
                aliases.extend([parts[0], parts[-1]])
        alias_hits = sum(1 for a in aliases if _norm_text(a) in blob_n)

        # Keywords
        stopwords = {"because", "therefore", "usually", "often", "about", "would", "could", "should"}
        terms = re.findall(r"[a-zA-Z]{5,}", (fact or "").lower())
        terms = [t for t in terms if t not in stopwords][:6]

        if not terms and not entities:
            return True

        term_hits = sum(1 for t in terms if t in blob_n)

        # Entities are more important: if at least 1 entity found, lower threshold
        if entity_hits + alias_hits >= 1:
            return term_hits >= max(1, int(0.3 * len(terms))) or entity_hits >= 1

        # Standard threshold
        return term_hits >= max(2, int(0.4 * len(terms)))

    @staticmethod
    def _filter_quality_snippets(hits: List[Dict[str, str]], min_len: int = 40) -> List[Dict[str, str]]:
        """Filter out low-quality snippets (too short, empty, or duplicated content)."""
        seen_content = set()
        out = []
        for h in (hits or []):
            snippet = (h.get("snippet") or "").strip()
            title = (h.get("title") or "").strip()

            # Skip too short
            if len(snippet) < min_len:
                continue

            # Skip duplicated content (first 60 chars as fingerprint)
            fp = _norm_text(snippet[:60])
            if fp in seen_content:
                continue
            seen_content.add(fp)

            out.append(h)
        return out

    @staticmethod
    def _llm_suggest_queries(clean_fact: str) -> List[str]:
        key = f"qs::{_norm_text(clean_fact)}"
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        q_prompt = f"""
Given a statement, suggest 3-5 concise web search queries to verify it.
Return JSON: {{"queries": [..]}}.
Statement: {clean_fact}
"""
        out: List[str] = []
        try:
            rr = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": q_prompt}],
                temperature=0.0,
            )
            qj = _safe_json_loads((rr.choices[0].message.content or "").strip())
            if isinstance(qj, dict):
                qs = qj.get("queries") or []
                if isinstance(qs, list):
                    out = [str(x).strip() for x in qs if isinstance(x, str) and str(x).strip()]
        except Exception:
            RealToolkit._llm_error_flag = True
            out = []

        out = out[:4]
        RealToolkit._cache[key] = out
        return out


    # ---------- Main claim verification ----------
    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> Optional[bool]:
        display_claim = (claim or "")[:80].replace("\n", " ")
        print(f"      🕵️ [Processing]: '{display_claim}...' via {tool_type}")

        RealToolkit._llm_error_flag = False
        cache_key = f"verify||{tool_type}||{(claim or '').strip()}"
        if cache_key in RealToolkit._cache:
            return RealToolkit._cache[cache_key]

        try:
            # ------------------------
            # PYTHON_EXEC robust path
            # ------------------------
            if tool_type == "PYTHON_EXEC":
                det = RealToolkit._deterministic_tier0(claim)
                if det is not None:
                    RealToolkit._cache[cache_key] = det
                    return det

                clean_fact = RealToolkit._distill_claim(claim)

                det2 = RealToolkit._deterministic_tier0(clean_fact)
                if det2 is not None:
                    RealToolkit._cache[cache_key] = det2
                    return det2

                family = RealToolkit._detect_sanity_family(claim) or RealToolkit._detect_sanity_family(clean_fact)

                if family == "propositional":
                    prop_result = RealToolkit._eval_propositional(claim)
                    if prop_result is None:
                        prop_result = RealToolkit._eval_propositional(clean_fact)
                    if prop_result is not None:
                        RealToolkit._cache[cache_key] = bool(prop_result)
                        status_icon = "✅" if prop_result else "❌"
                        print(f"        └─ {status_icon} Result: {'TRUE' if prop_result else 'FALSE'} (propositional)")
                        return bool(prop_result)

                def run_codegen_once() -> Optional[bool]:
                    code = RealToolkit._llm_generate_python_code(clean_fact)
                    result = PythonSandbox.run(code)
                    if "VERIFIED_TRUE" in result:
                        return True
                    if "VERIFIED_FALSE" in result:
                        return False
                    return None

                verdict = run_codegen_once()
                if verdict is None:
                    verdict = run_codegen_once()

                if verdict is None:
                    if family in ("leap", "arith", "sqrt", "percent", "compare", "propositional"):
                        RealToolkit._cache[cache_key] = False
                        return False
                    tool_type = "WEB_SEARCH"

                if family in ("leap", "arith", "sqrt", "percent", "compare"):
                    if not RealToolkit._sanity_harness(family):
                        verdict2 = run_codegen_once()
                        if verdict2 is not None and RealToolkit._sanity_harness(family):
                            verdict = verdict2
                        else:
                            det3 = RealToolkit._deterministic_tier0(clean_fact)
                            verdict = det3 if det3 is not None else False

                RealToolkit._cache[cache_key] = bool(verdict)
                status_icon = "✅" if verdict else "❌"
                print(f"        └─ {status_icon} Result: {'TRUE' if verdict else 'FALSE'}")
                return bool(verdict)

            # ------------------------
            # WEB_SEARCH path (stance-aware RAG+judge + calibrated confidence)
            # ------------------------
            if tool_type == "WEB_SEARCH":
                # Optimization: Skip distillation for short, clean claims
                claim_stripped = RealToolkit._strip_claim_prefix(claim)
                if len(claim_stripped) < 100 and not re.search(r'\b(I think|maybe|probably|in my opinion|perhaps|might be)\b', claim, re.IGNORECASE):
                    clean_fact = claim_stripped
                else:
                    clean_fact = RealToolkit._distill_claim(claim)

                # NEW: meta-claim rewrite
                if RealToolkit._is_meta_discourse_claim(clean_fact):
                    clean_fact = RealToolkit._rewrite_meta_to_factual(clean_fact)

                # HYBRID MULTI-HOP ENHANCEMENT: Try decomposition first
                entity_info = RealToolkit._extract_bridge_entities(clean_fact)
                sub_queries = None
                if entity_info.get("has_bridge"):
                    sub_queries = RealToolkit._decompose_multihop_claim(clean_fact, entity_info)
                
                all_serper: List[Dict[str, str]] = []
                all_ddg: List[Dict[str, str]] = []
                
                def run_round(qs: List[str], round_idx: int) -> None:
                    nonlocal all_serper, all_ddg
                    fast_qs = qs[:2] if FAST_MODE else qs
                    print(f"        🔁 WEB round {round_idx}: {len(fast_qs)} queries")
                    for qi, q in enumerate(fast_qs, start=1):
                        q = (q or "").strip()
                        if not q:
                            continue
                        print(f"        🔎 Q{round_idx}.{qi}: {q}")
                        payload_s = RealToolkit.google_search(q)
                        payload = _safe_json_loads(payload_s) or {}
                        serper_hits = payload.get("serper", []) or []
                        ddg_hits = payload.get("ddg", []) or []
                        all_serper.extend(serper_hits)
                        all_ddg.extend(ddg_hits)
                        time.sleep(0.05)

                # If multi-hop detected and decomposed successfully, use hybrid path
                if sub_queries and len(sub_queries) >= 2:
                    multihop_hits = RealToolkit._execute_hybrid_multihop(clean_fact, sub_queries)
                    # Distribute hits between serper and ddg for compatibility
                    for i, hit in enumerate(multihop_hits):
                        if i % 2 == 0:
                            all_serper.append(hit)
                        else:
                            all_ddg.append(hit)
                else:
                    # Standard search flow (existing code)
                    queries_rounds = RealToolkit._make_queries(clean_fact)
                    if FAST_MODE:
                        queries_rounds = queries_rounds[:1]

                    for ridx, qs in enumerate(queries_rounds, start=1):
                        run_round(qs, ridx)

                        if (len(all_serper) == 0) and (len(all_ddg) == 0):
                            time.sleep(0.25)
                            print(f"        ⚠️ No hits, retrying round {ridx} once...")
                            run_round(qs, ridx)

                        # Optimization: Lower threshold to exit search rounds earlier
                        trusted_cnt = sum(1 for h in (all_serper + all_ddg) if _is_trusted_domain(h.get("url", "")))
                        if trusted_cnt >= (2 if FAST_MODE else 4):
                            break

                def dedup(hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
                    seen = set()
                    out = []
                    for h in hits:
                        k = (h.get("url", ""), h.get("title", ""))
                        if k in seen:
                            continue
                        seen.add(k)
                        out.append(h)
                    return out

                all_serper = dedup(all_serper)
                all_ddg = dedup(all_ddg)
                combined_hits = all_serper + all_ddg

                # NEW: coverage gate + 1 extra LLM query round
                if combined_hits and (not RealToolkit._coverage_ok(clean_fact, combined_hits)):
                    extra_qs = RealToolkit._llm_suggest_queries(clean_fact)
                    if extra_qs and not FAST_MODE:
                        run_round(extra_qs[:4], round_idx=99)
                        all_serper = dedup(all_serper)
                        all_ddg = dedup(all_ddg)
                        combined_hits = all_serper + all_ddg

                trusted_hits = [h for h in combined_hits if _is_trusted_domain(h.get("url", ""))]
                coverage_ok = RealToolkit._coverage_ok(clean_fact, combined_hits) if combined_hits else False
                trusted_cnt = len(trusted_hits)
                allow_heuristic = coverage_ok and trusted_cnt >= 2

                # Only compute heuristic verdicts if quality is sufficient
                if allow_heuristic:
                    v_serper, s_serper = RealToolkit._evidence_heuristic_verdict(clean_fact, all_serper)
                    v_ddg, s_ddg = RealToolkit._evidence_heuristic_verdict(clean_fact, all_ddg)

                    print(
                        f"        🧾 SERPER: hits={len(all_serper)} trusted={sum(1 for h in all_serper if _is_trusted_domain(h.get('url','')))} "
                        f"heuristic={v_serper}({s_serper})"
                    )
                    print(
                        f"        🧾 DDG: hits={len(all_ddg)} trusted={sum(1 for h in all_ddg if _is_trusted_domain(h.get('url','')))} "
                        f"heuristic={v_ddg}({s_ddg})"
                    )

                    v_all, s_all = RealToolkit._evidence_heuristic_verdict(clean_fact, combined_hits)
                    if v_all is not None and s_all.startswith("strong"):
                        verdict = v_all
                        RealToolkit._cache[cache_key] = verdict
                        status_icon = "✅" if verdict else "❌"
                        print(f"        └─ {status_icon} Heuristic(strong) Result: {'TRUE' if verdict else 'FALSE'}")
                        return verdict
                else:
                    # Don't use heuristic verdicts in voting when quality is insufficient
                    v_serper, s_serper = None, "weak"
                    v_ddg, s_ddg = None, "weak"
                    print(
                        f"        🧾 SERPER: hits={len(all_serper)} trusted={sum(1 for h in all_serper if _is_trusted_domain(h.get('url','')))} "
                        f"heuristic=SKIPPED"
                    )
                    print(
                        f"        🧾 DDG: hits={len(all_ddg)} trusted={sum(1 for h in all_ddg if _is_trusted_domain(h.get('url','')))} "
                        f"heuristic=SKIPPED"
                    )
                    print(f"        ⚠️ Heuristic skipped (coverage={coverage_ok}, trusted={trusted_cnt})")

                evidence_base = trusted_hits if trusted_hits else combined_hits
                # Optimization: Reduce evidence pool size for faster LLM processing
                max_evidence = 4 if FAST_MODE else 6
                evidence_lines = _summarize_hits(evidence_base, max_n=max_evidence) if evidence_base else ""

                if not evidence_lines.strip():
                    v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                    print(f"        🧠 COMMON_SENSE vote={v_cs}")
                    final = bool(v_cs) if v_cs is not None else False
                    RealToolkit._cache[cache_key] = final
                    status_icon = "✅" if final else "❌"
                    print(f"        └─ {status_icon} Final (no-evidence) Result: {'TRUE' if final else 'FALSE'}")
                    return final

                jr = RealToolkit._rag_judge_with_calibrated_conf(clean_fact, evidence_lines, evidence_base)
                print(
                    f"        🧠 RAG_JUDGE vote={jr.verdict} "
                    f"llm_conf={jr.llm_confidence:.2f} econf={jr.evidence_confidence:.2f} final_conf={jr.final_confidence:.2f} "
                    f"support={jr.support_ids} refute={jr.refute_ids} rationale={jr.rationale}"
                )

                # Optimization: Early-exit for high-confidence verdicts (≥0.80)
                # This skips semantic fallback and common sense LLM calls when already confident
                if jr.verdict is not None and jr.final_confidence >= 0.80:
                    RealToolkit._cache[cache_key] = bool(jr.verdict)
                    status_icon = "✅" if jr.verdict else "❌"
                    print(f"        └─ {status_icon} Final Result (high-conf): {'TRUE' if jr.verdict else 'FALSE'}")
                    return bool(jr.verdict)

                # NEW: accept only if above threshold; otherwise use semantic fallback then vote
                if jr.verdict is True and jr.final_confidence >= RealToolkit.JUDGE_TRUE_MIN_FINAL_CONF:
                    final = True
                elif jr.verdict is False and jr.final_confidence >= RealToolkit.JUDGE_FALSE_MIN_FINAL_CONF:
                    final = False
                else:
                    if FAST_MODE:
                        v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                        print(f"        🧠 COMMON_SENSE vote={v_cs}")
                        if v_cs is not None and (v_serper is None) and (v_ddg is None):
                            final = bool(v_cs)
                        else:
                            final_vote = RealToolkit._vote_2_of_3(v_serper, v_ddg, v_cs)
                            final = bool(final_vote) if final_vote is not None else False
                    else:
                        sem_v, sem_conf, sem_rat = RealToolkit._llm_semantic_fallback(clean_fact, evidence_lines)
                        print(
                            f"        🧠 SEMANTIC vote={sem_v} conf={sem_conf:.2f} rationale={sem_rat}"
                        )

                        if sem_v is not None and sem_conf >= RealToolkit.JUDGE_FALLBACK_MIN_CONF:
                            final = bool(sem_v)
                        else:
                            v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                            print(f"        🧠 COMMON_SENSE vote={v_cs}")

                            # CRITICAL FIX: don't default to FALSE when only common-sense votes
                            if v_cs is not None and (v_serper is None) and (v_ddg is None):
                                final = bool(v_cs)
                            else:
                                final_vote = RealToolkit._vote_2_of_3(v_serper, v_ddg, v_cs)
                                final = bool(final_vote) if final_vote is not None else False

                if RealToolkit._llm_error_flag:
                    RealToolkit._cache[cache_key] = None
                    print("        └─ ⚠️ Final Result: ABSTAIN (LLM error)")
                    return None

                RealToolkit._cache[cache_key] = bool(final)
                status_icon = "✅" if final else "❌"
                print(f"        └─ {status_icon} Final Result: {'TRUE' if final else 'FALSE'}")
                return bool(final)

            # ------------------------
            # COMMON_SENSE or other
            # ------------------------
            clean_fact = RealToolkit._distill_claim(claim)
            final_prompt = f"""
Return ONLY TRUE or FALSE for the statement below using common sense.

Statement: "{clean_fact}"
Verdict:
"""
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.0,
            )
            verdict_txt = (res.choices[0].message.content or "").strip().upper()
            verdict = "TRUE" in verdict_txt
            RealToolkit._cache[cache_key] = verdict
            return verdict

        except Exception as e:
            RealToolkit._llm_error_flag = True
            print(f"      ⚠️ Verification Error: {e}")
            RealToolkit._cache[cache_key] = None
            return None
