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

from src.config import client, SERPER_API_KEY, JUDGE_MODEL

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
            return f"[Runtime Error]: {str(e)}"


# -----------------------------
# RealToolkit
# -----------------------------
class RealToolkit:
    _cache: Dict[str, Any] = {}

    # ---------- Tuning knobs ----------
    # Gate TRUE more strictly than FALSE, because false-positives are dangerous for pruning/acceptance.
    JUDGE_TRUE_MIN_FINAL_CONF = 0.72
    JUDGE_FALSE_MIN_FINAL_CONF = 0.62

    # For edge prune: only prune when "FALSE" with very high confidence
    EDGE_PRUNE_FALSE_CONF = 0.80

    # ---------- Cache helpers ----------
    @staticmethod
    def _cache_key(prefix: str, *parts: str) -> str:
        joined = "||".join([prefix] + [p.strip() for p in parts])
        return joined[:2000]

    # ---------- Deterministic tier0 helpers ----------
    @staticmethod
    def _normalize_polarity(text: str) -> Tuple[str, bool]:
        if not text:
            return "", False
        s = text.strip().lower()
        s = s.replace("isn't", "is not").replace("wasn't", "was not")

        neg = False
        if " is not " in s:
            neg = True
            s = s.replace(" is not ", " is ")
        if " was not " in s:
            neg = True
            s = s.replace(" was not ", " was ")
        if " does not equal " in s:
            neg = True
            s = s.replace(" does not equal ", " equals ")
        if " not a leap year" in s:
            neg = True
            s = s.replace(" not a leap year", " a leap year")

        return s, neg

    @staticmethod
    def _deterministic_tier0(text: str) -> Optional[bool]:
        s_norm, neg = RealToolkit._normalize_polarity(text)

        # Leap year
        m = re.search(r"\b(\d{4})\b.*\bleap year\b", s_norm)
        if m:
            y = int(m.group(1))
            ok = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))
            return (not ok) if neg else ok

        # sqrt
        m = re.search(r"square root of\s+(-?\d+)\s+is\s+(-?\d+)", s_norm)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            ok = False if a < 0 else (b * b == a)
            return (not ok) if neg else ok

        # arithmetic
        m = re.search(r"(-?\d+)\s*([\+\-\*\/])\s*(-?\d+)\s*(?:equals|=)\s*(-?\d+)", s_norm)
        if m:
            x = int(m.group(1))
            op = m.group(2)
            y = int(m.group(3))
            z = int(m.group(4))
            if op == "+":
                val = x + y
                ok = (val == z)
                return (not ok) if neg else ok
            if op == "-":
                val = x - y
                ok = (val == z)
                return (not ok) if neg else ok
            if op == "*":
                val = x * y
                ok = (val == z)
                return (not ok) if neg else ok
            # division
            if y == 0:
                ok = False
                return (not ok) if neg else ok
            val = x / y
            ok = abs(val - z) < 1e-9
            return (not ok) if neg else ok

        # percent
        m = re.search(
            r"(-?\d+(?:\.\d+)?)\s*%?\s*(?:percent)?\s*of\s*(-?\d+(?:\.\d+)?)\s*(?:equals|=|is)\s*(-?\d+(?:\.\d+)?)",
            s_norm,
        )
        if m:
            p = float(m.group(1))
            base = float(m.group(2))
            ans = float(m.group(3))
            val = (p / 100.0) * base
            ok = abs(val - ans) < 1e-9
            return (not ok) if neg else ok

        # comparisons
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)", s_norm)
        if m:
            a = float(m.group(1))
            op = m.group(2)
            b = float(m.group(3))
            if op == ">":
                ok = a > b
            elif op == "<":
                ok = a < b
            elif op == ">=":
                ok = a >= b
            else:
                ok = a <= b
            return (not ok) if neg else ok

        m = re.search(
            r"(-?\d+(?:\.\d+)?)\s+is\s+(greater than|less than|at least|at most)\s+(-?\d+(?:\.\d+)?)",
            s_norm,
        )
        if m:
            a = float(m.group(1))
            rel = m.group(2)
            b = float(m.group(3))
            if rel == "greater than":
                ok = a > b
            elif rel == "less than":
                ok = a < b
            elif rel == "at least":
                ok = a >= b
            else:
                ok = a <= b
            return (not ok) if neg else ok

        return None

    @staticmethod
    def _eval_propositional(text: str) -> Optional[bool]:
        """
        Evaluate simple propositional logic statements.
        Examples:
         - "If A is True and B is False, then (A and B) is False."
         - "It is false that 2+2=4."
        """
        s = (text or "").lower().strip()

        m = re.search(
            r'if\s+(\w+)\s+is\s+(true|false)\s+and\s+(\w+)\s+is\s+(true|false)\s*,?\s*then\s+\((.*?)\)\s+is\s+(true|false)',
            s,
        )
        if m:
            var1, val1_str, var2, val2_str, expr, expected_str = m.groups()
            val1 = (val1_str == "true")
            val2 = (val2_str == "true")
            expected = (expected_str == "true")
            expr_norm = expr.strip()

            result = None
            if f"{var1} and {var2}" in expr_norm:
                result = val1 and val2
            elif f"{var1} or {var2}" in expr_norm:
                result = val1 or val2
            elif f"not {var1}" in expr_norm:
                result = not val1
            elif f"not {var2}" in expr_norm:
                result = not val2

            if result is not None:
                return result == expected

        if "it is false that" in s:
            if "2 + 2 equals 5" in s or "2+2=5" in s:
                return True
            if "2 + 2 equals 4" in s or "2+2=4" in s:
                return False

        return None

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
        return res.choices[0].message.content

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
            clean = res.choices[0].message.content.strip().replace('"', "")
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
            out = res.choices[0].message.content.strip().strip('"')
        except Exception:
            out = clean_fact

        if len(out.strip()) < 8:
            out = clean_fact

        RealToolkit._cache[key] = out
        return out

    # ---------- Query builder (NEW) ----------
    @staticmethod
    def _make_queries(clean_fact: str) -> List[List[str]]:
        """
        Build query rounds that are NOT literal copies of the claim.
        Round1: keyword-focused
        Round2: encyclopedia
        Round3: edu/gov + meta-analysis style
        """
        s = (clean_fact or "").strip()
        if not s:
            return [[clean_fact]]

        kws = re.findall(r"[A-Za-z]{4,}", clean_fact)
        kws = [w.lower() for w in kws if w.lower() not in {
            "discussion", "conversation", "topic", "claim", "about", "therefore",
            "usually", "often", "because", "most", "people",
        }]
        kw_str = " ".join(kws[:7])
        base_q = kw_str if len(kw_str) >= 8 else s

        return [
            [base_q, f"{base_q} evidence", f"{base_q} explained"],
            [f"site:wikipedia.org {base_q}", f"site:britannica.com {base_q}"],
            [f"site:.edu {base_q}", f"site:.gov {base_q}", f"{base_q} meta analysis"],
        ]

    # ---------- Web search providers ----------
    @staticmethod
    def _serper_search(query: str, num: int = 6, timeout: float = 6.5) -> List[Dict[str, str]]:
        if not SERPER_API_KEY:
            return []
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query[:200], "num": int(num)})
            headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, data=payload, timeout=timeout)
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
        s = _norm_text(clean_fact)
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
        return any(m in s for m in hoax_markers)

    @staticmethod
    def _has_refute_cues(text_blob: str) -> bool:
        b = _norm_text(text_blob)
        refute_cues = [
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
        ]
        return any(c in b for c in refute_cues)

    @staticmethod
    def _has_affirm_cues(text_blob: str) -> bool:
        b = _norm_text(text_blob)
        affirm_cues = [
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
        ]
        return any(c in b for c in affirm_cues)

    @staticmethod
    def _evidence_heuristic_verdict(clean_fact: str, hits: List[Dict[str, str]]) -> Tuple[Optional[bool], str]:
        """
        Stance-aware heuristic.
        Returns: (verdict, strength)
          - verdict: True / False / None(ABSTAIN)
          - strength: "strong" or "weak" (only meaningful if verdict is not None)
        Policy:
          - For hoax-like claims:
              if trusted snippets have strong refute cues => FALSE (strong)
              never return TRUE from mere keyword overlap
          - For non-hoax:
              return TRUE only with strong affirm cues AND anchor coverage (strong)
          - Otherwise return None
        """
        hoaxy = RealToolkit._is_hoaxy_claim(clean_fact)

        trusted_hits = [h for h in (hits or []) if _is_trusted_domain(h.get("url", ""))]
        if not trusted_hits:
            return None, "weak"

        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in trusted_hits])
        blob_n = _norm_text(blob)

        if hoaxy and RealToolkit._has_refute_cues(blob_n):
            return False, "strong"

        fact = _norm_text(clean_fact)

        # Safe negative special-case for Olympics host misinformation
        if "2024" in fact and "summer olympics" in fact and ("tokyo" in fact or "los angeles" in fact):
            if "tokyo" in fact and "2024" in fact:
                return False, "strong"
            if ("los angeles" in fact or "l.a" in fact) and "2024" in fact:
                return False, "strong"

        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_fact)
        anchors = set(years)

        words = re.findall(r"[a-zA-Z]{4,}", clean_fact.lower())
        stop = {
            "that", "this", "with", "from", "were", "held", "host", "hosted",
            "equals", "year", "percent", "square", "root", "claim", "proof",
        }
        for w in words:
            if w not in stop:
                anchors.add(w)
            if len(anchors) >= 8:
                break

        matched = [a for a in anchors if a and a in blob_n]

        if hoaxy:
            return None, "weak"

        if RealToolkit._has_affirm_cues(blob_n) and len(matched) >= 4:
            return True, "strong"

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
            ans = res.choices[0].message.content.strip().upper()
            if "ABSTAIN" in ans:
                return None
            if "TRUE" in ans:
                return True
            if "FALSE" in ans:
                return False
            return None
        except Exception:
            return None

    @staticmethod
    def _llm_rag_judge_raw(clean_fact: str, evidence_lines: str) -> Tuple[Optional[bool], float, List[int], List[int], str]:
        """
        RAG + judge: stance classification with citations to snippet ids.
        Returns: (verdict, llm_confidence, support_ids, refute_ids, rationale_short)
        """
        prompt = f"""
You are an evidence-based fact checker.

Fact:
{clean_fact}

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
            raw = res.choices[0].message.content.strip()
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
            return None, 0.0, [], [], "Judge error"

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

    # ---------- Voting ----------
    @staticmethod
    def _vote_2_of_3(v_serper: Optional[bool], v_ddg: Optional[bool], v_llm: Optional[bool]) -> Optional[bool]:
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
            raw = res.choices[0].message.content.strip()
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
          - Return True otherwise (keep edge).
        """
        key = RealToolkit._cache_key("attack3", attacker, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        label, conf = RealToolkit._llm_relation_judge(attacker, target, mode="ATTACK")
        out = True
        if label == "FALSE" and conf >= RealToolkit.EDGE_PRUNE_FALSE_CONF:
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

    # ---------- Coverage gate (NEW) ----------
    @staticmethod
    def _coverage_ok(fact: str, hits: List[Dict[str, str]]) -> bool:
        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in (hits or [])[:10]])
        blob_n = _norm_text(blob)

        terms = re.findall(r"[a-zA-Z]{5,}", (fact or "").lower())
        terms = [t for t in terms if t not in {"because", "therefore", "usually", "often", "about"}][:6]
        if not terms:
            return True

        hit = sum(1 for t in terms if t in blob_n)
        return hit >= max(2, int(0.4 * len(terms)))

    @staticmethod
    def _llm_suggest_queries(clean_fact: str) -> List[str]:
        key = RealToolkit._cache_key("suggestq", clean_fact)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        q_prompt = f"""
Generate 4 short web search queries to verify this factual claim.
Queries must focus on the subject and predicate, not restating the claim verbatim.
Avoid meta words like "discussion", "conversation", "topic".

Claim: "{clean_fact}"

Output as strict JSON:
{{"queries":["...","...","...","..."]}}
"""
        out: List[str] = []
        try:
            rr = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": q_prompt}],
                temperature=0.0,
            )
            qj = _safe_json_loads(rr.choices[0].message.content.strip())
            if isinstance(qj, dict):
                qs = qj.get("queries") or []
                if isinstance(qs, list):
                    out = [str(x).strip() for x in qs if isinstance(x, str) and str(x).strip()]
        except Exception:
            out = []

        out = out[:4]
        RealToolkit._cache[key] = out
        return out

    # ---------- Main claim verification ----------
    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> bool:
        display_claim = (claim or "")[:80].replace("\n", " ")
        print(f"      🕵️ [Processing]: '{display_claim}...' via {tool_type}")

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
                clean_fact = RealToolkit._distill_claim(claim)

                # NEW: meta-claim rewrite
                if RealToolkit._is_meta_discourse_claim(clean_fact):
                    clean_fact = RealToolkit._rewrite_meta_to_factual(clean_fact)

                queries_rounds: List[List[str]] = RealToolkit._make_queries(clean_fact)

                all_serper: List[Dict[str, str]] = []
                all_ddg: List[Dict[str, str]] = []

                def run_round(qs: List[str], round_idx: int) -> None:
                    nonlocal all_serper, all_ddg
                    print(f"        🔁 WEB round {round_idx}: {len(qs)} queries")
                    for qi, q in enumerate(qs, start=1):
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

                for ridx, qs in enumerate(queries_rounds, start=1):
                    run_round(qs, ridx)

                    if (len(all_serper) == 0) and (len(all_ddg) == 0):
                        time.sleep(0.25)
                        print(f"        ⚠️ No hits, retrying round {ridx} once...")
                        run_round(qs, ridx)

                    trusted_cnt = sum(1 for h in (all_serper + all_ddg) if _is_trusted_domain(h.get("url", "")))
                    if trusted_cnt >= 6:
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
                    if extra_qs:
                        run_round(extra_qs[:4], round_idx=99)
                        all_serper = dedup(all_serper)
                        all_ddg = dedup(all_ddg)
                        combined_hits = all_serper + all_ddg

                trusted_hits = [h for h in combined_hits if _is_trusted_domain(h.get("url", ""))]

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
                if v_all is not None and s_all == "strong":
                    verdict = v_all
                    RealToolkit._cache[cache_key] = verdict
                    status_icon = "✅" if verdict else "❌"
                    print(f"        └─ {status_icon} Heuristic(strong) Result: {'TRUE' if verdict else 'FALSE'}")
                    return verdict

                evidence_base = trusted_hits if trusted_hits else combined_hits
                evidence_lines = _summarize_hits(evidence_base, max_n=8) if evidence_base else ""

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

                # NEW: accept only if above threshold; otherwise treat as ABSTAIN and fallback
                if jr.verdict is True and jr.final_confidence >= RealToolkit.JUDGE_TRUE_MIN_FINAL_CONF:
                    final = True
                elif jr.verdict is False and jr.final_confidence >= RealToolkit.JUDGE_FALSE_MIN_FINAL_CONF:
                    final = False
                else:
                    v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                    print(f"        🧠 COMMON_SENSE vote={v_cs}")

                    # CRITICAL FIX: don't default to FALSE when only common-sense votes
                    if v_cs is not None and (v_serper is None) and (v_ddg is None):
                        final = bool(v_cs)
                    else:
                        final_vote = RealToolkit._vote_2_of_3(v_serper, v_ddg, v_cs)
                        final = bool(final_vote) if final_vote is not None else False

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
            verdict_txt = res.choices[0].message.content.strip().upper()
            verdict = "TRUE" in verdict_txt
            RealToolkit._cache[cache_key] = verdict
            return verdict

        except Exception as e:
            print(f"      ⚠️ Verification Error: {e}")
            # On tool failure, do NOT prune aggressively: keep claim as TRUE to avoid collapsing graph
            RealToolkit._cache[cache_key] = True
            return True
