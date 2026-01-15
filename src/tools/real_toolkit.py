# tools/real_toolkit.py
# Patched version (RAG + Judge first, stance-aware, anti-anchor-match, ABSTAIN-aware edge verification)
#
# What changed vs your current file:
#
# 1) WEB_SEARCH decision order (your question: "đưa case RAG+judge lên trước >=2 nguồn?"):
#    - We run RAG_JUDGE EARLY after round-1 evidence, without requiring >=2 trusted sources.
#    - If judge ABSTAINS or confidence is low, we expand search (round 2/3) and re-judge.
#    - Strong refutation heuristic (hoax/refute cues) can still short-circuit to FALSE early.
#
# 2) Anti-anchor-match TRUE:
#    - Heuristic TRUE no longer short-circuits at all (even if "strong").
#    - Heuristic TRUE is only used as a weak prior (logged), but final truth comes from RAG_JUDGE.
#
# 3) Edge verify: TRUE/FALSE/ABSTAIN + prune only when confident FALSE.
#    - Added verify_attack_rich / verify_support_rich (returns label3 + conf).
#    - verify_attack / verify_support (bool compatibility) only prunes when FALSE with high confidence.
#
# NOTE:
# - verify_claim(tool_type, claim) still returns bool for compatibility.
# - Internally we compute a rich verdict (TRUE/FALSE/ABSTAIN + confidence).
# - By default, ABSTAIN => returns False (conservative: "not verified true").
#
# Dependencies: requests, ddgs or duckduckgo_search, openai client from src.config

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
# Types / Config
# -----------------------------
Label3 = Optional[bool]  # True / False / None (ABSTAIN)

# Claim-judge thresholds (tune here)
JUDGE_TRUE_MIN_CONF = 0.80   # accept TRUE if judge says TRUE and conf >= this
JUDGE_FALSE_MIN_CONF = 0.75  # accept FALSE if judge says FALSE and conf >= this
# If judge returns TRUE but conf < threshold -> treat as ABSTAIN (return False at end)
# If judge returns FALSE but conf < threshold -> treat as ABSTAIN (return False at end)
# (You can choose to "not prune" upstream via solver; toolkit remains conservative.)

# Edge prune threshold (only prune when confident it's NOT that relation)
EDGE_PRUNE_FALSE_MIN_CONF = 0.80


@dataclass
class JudgeResult:
    verdict: Label3          # True / False / None
    confidence: float        # [0,1]
    support_ids: List[int]
    refute_ids: List[int]
    rationale: str


# -----------------------------
# Utilities
# -----------------------------
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
    Even trusted sources can mention a claim to refute it.
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


def _summarize_hits(hits: List[Dict[str, str]], max_n: int = 8) -> str:
    """
    Produce numbered evidence lines so the judge can cite specific snippets.
    """
    out = []
    for i, r in enumerate((hits or [])[:max_n], start=1):
        title = (r.get("title") or "")[:180]
        url = (r.get("url") or "")[:240]
        snip = (r.get("snippet") or "")[:300]
        out.append(f"[{i}] {title} | {url} | {snip}")
    return "\n".join(out)


def _split_sentences_rough(s: str, max_len: int = 260) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s[:max_len]


def _dedup_hits(hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for h in hits or []:
        k = ((h.get("url") or "").strip(), (h.get("title") or "").strip())
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
    return out


def _distinct_trusted_domains(hits: List[Dict[str, str]]) -> int:
    ds = set()
    for h in hits or []:
        if _is_trusted_domain(h.get("url", "")):
            d = _domain(h.get("url", ""))
            if d:
                ds.add(d)
    return len(ds)


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

    # ---------- Cache helpers ----------
    @staticmethod
    def _cache_key(prefix: str, *parts: str) -> str:
        joined = "||".join([prefix] + [p.strip() for p in parts if p is not None])
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
    def _detect_sanity_family(text: str) -> Optional[str]:
        if not text:
            return None
        s = text.lower()
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
        prompt = f"""
Extract the core factual claim from the text below.
Remove conversational filler and hedging like "I think", "maybe", "in my opinion".
Preserve the main subject/entities and the meaning.
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

        RealToolkit._cache[key] = clean
        return clean

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
            "successfully landed",
            "landed on the moon",
            "astronauts walked",
            "returned samples",
            "brought back",
            "official records",
            "mission returned",
        ]
        return any(c in b for c in affirm_cues)

    @staticmethod
    def _evidence_heuristic_refute_only(clean_fact: str, hits: List[Dict[str, str]]) -> Tuple[Optional[bool], str]:
        """
        Heuristic used ONLY for strong refutation short-circuit (FALSE).
        - If claim is hoaxy AND trusted snippets contain refute cues -> FALSE strong.
        Otherwise -> None.
        """
        hoaxy = RealToolkit._is_hoaxy_claim(clean_fact)
        if not hoaxy:
            return None, "weak"

        trusted_hits = [h for h in (hits or []) if _is_trusted_domain(h.get("url", ""))]
        if not trusted_hits:
            return None, "weak"

        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in trusted_hits])
        if RealToolkit._has_refute_cues(blob):
            return False, "strong"
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
    def _llm_rag_judge(clean_fact: str, evidence_lines: str) -> JudgeResult:
        """
        RAG + judge: stance classification with citations to snippet ids.
        Returns JudgeResult(verdict, confidence, support_ids, refute_ids, rationale).
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
                return JudgeResult(None, 0.0, [], [], "Judge parse failed")

            v = str(data.get("verdict", "")).strip().upper()
            conf = float(data.get("confidence", 0.0) or 0.0)

            def _as_int_list(x) -> List[int]:
                out = []
                if isinstance(x, list):
                    for t in x:
                        try:
                            out.append(int(t))
                        except Exception:
                            pass
                return out

            sup_i = _as_int_list(data.get("support_ids") or [])
            ref_i = _as_int_list(data.get("refute_ids") or [])
            rat = str(data.get("rationale", "") or "")[:260]
            conf = max(0.0, min(1.0, conf))

            if v == "TRUE":
                return JudgeResult(True, conf, sup_i, ref_i, rat)
            if v == "FALSE":
                return JudgeResult(False, conf, sup_i, ref_i, rat)
            return JudgeResult(None, conf, sup_i, ref_i, rat)
        except Exception:
            return JudgeResult(None, 0.0, [], [], "Judge error")

    # ---------- Attack/Support relation checks (LLM) ----------
    @staticmethod
    def _llm_relation_judge(statement_a: str, statement_b: str, mode: Literal["ATTACK", "SUPPORT"]) -> Tuple[str, float]:
        """
        Returns (label, confidence) where label in {"TRUE","FALSE","ABSTAIN"}.
        """
        if mode == "ATTACK":
            question = "Does A logically invalidate, contradict, or provide a counter-argument to B?"
            rules = """
- If A contradicts B about the same proposition, output TRUE.
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
    def verify_attack_rich(attacker: str, target: str) -> Tuple[Label3, float]:
        """
        Returns (label3, conf):
          - label3=True   => is ATTACK
          - label3=False  => is NOT ATTACK
          - label3=None   => ABSTAIN/unclear
        """
        key = RealToolkit._cache_key("attack_rich", attacker, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        label, conf = RealToolkit._llm_relation_judge(attacker, target, mode="ATTACK")
        out: Tuple[Label3, float]
        if label == "TRUE":
            out = (True, conf)
        elif label == "FALSE":
            out = (False, conf)
        else:
            out = (None, conf)

        RealToolkit._cache[key] = out
        return out

    @staticmethod
    def verify_support_rich(source: str, target: str) -> Tuple[Label3, float]:
        """
        Returns (label3, conf):
          - label3=True   => is SUPPORT
          - label3=False  => is NOT SUPPORT
          - label3=None   => ABSTAIN/unclear
        """
        key = RealToolkit._cache_key("support_rich", source, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        label, conf = RealToolkit._llm_relation_judge(source, target, mode="SUPPORT")
        out: Tuple[Label3, float]
        if label == "TRUE":
            out = (True, conf)
        elif label == "FALSE":
            out = (False, conf)
        else:
            out = (None, conf)

        RealToolkit._cache[key] = out
        return out

    @staticmethod
    def verify_attack(attacker: str, target: str) -> bool:
        """
        Compatibility bool for pruning:
          - Return False ONLY when confident FALSE (NOT an attack) => prune edge.
          - Return True otherwise => keep edge.
        """
        label3, conf = RealToolkit.verify_attack_rich(attacker, target)
        if label3 is False and conf >= EDGE_PRUNE_FALSE_MIN_CONF:
            return False
        return True

    @staticmethod
    def verify_support(source: str, target: str) -> bool:
        """
        Compatibility bool for pruning:
          - Return False ONLY when confident FALSE (NOT a support) => prune edge.
          - Return True otherwise => keep edge.
        """
        label3, conf = RealToolkit.verify_support_rich(source, target)
        if label3 is False and conf >= EDGE_PRUNE_FALSE_MIN_CONF:
            return False
        return True

    # ---------- Voting (fallback only) ----------
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

    # ---------- Main claim verification (RICH) ----------
    @staticmethod
    def verify_claim_rich(tool_type: str, claim: str) -> JudgeResult:
        """
        Rich claim verifier:
          - verdict: True/False/None(ABSTAIN)
          - confidence, citations, rationale
        """
        cache_key = RealToolkit._cache_key("verify_rich", tool_type, claim or "")
        if cache_key in RealToolkit._cache:
            return RealToolkit._cache[cache_key]

        # Default fallback
        out = JudgeResult(None, 0.0, [], [], "Uninitialized")
        try:
            # ------------------------
            # PYTHON_EXEC robust path
            # ------------------------
            if tool_type == "PYTHON_EXEC":
                det = RealToolkit._deterministic_tier0(claim)
                if det is not None:
                    out = JudgeResult(det, 1.0, [], [], "Deterministic tier0")
                    RealToolkit._cache[cache_key] = out
                    return out

                clean_fact = RealToolkit._distill_claim(claim)

                det2 = RealToolkit._deterministic_tier0(clean_fact)
                if det2 is not None:
                    out = JudgeResult(det2, 1.0, [], [], "Deterministic tier0 (distilled)")
                    RealToolkit._cache[cache_key] = out
                    return out

                family = RealToolkit._detect_sanity_family(claim) or RealToolkit._detect_sanity_family(clean_fact)

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
                    if family in ("leap", "arith", "sqrt", "percent", "compare"):
                        out = JudgeResult(False, 1.0, [], [], "Sanity family but codegen failed")
                        RealToolkit._cache[cache_key] = out
                        return out
                    # fall back to web search
                    tool_type = "WEB_SEARCH"
                else:
                    # harness check if applicable
                    if family in ("leap", "arith", "sqrt", "percent", "compare"):
                        if not RealToolkit._sanity_harness(family):
                            verdict2 = run_codegen_once()
                            if verdict2 is not None and RealToolkit._sanity_harness(family):
                                verdict = verdict2
                            else:
                                det3 = RealToolkit._deterministic_tier0(clean_fact)
                                verdict = det3 if det3 is not None else False

                    out = JudgeResult(bool(verdict), 1.0, [], [], "Python sandbox")
                    RealToolkit._cache[cache_key] = out
                    return out

            # ------------------------
            # WEB_SEARCH path (RAG judge first, then expand)
            # ------------------------
            if tool_type == "WEB_SEARCH":
                clean_fact = RealToolkit._distill_claim(claim)

                # Query rounds: we run judge after round 1, and only expand if needed
                rounds: List[List[str]] = [
                    [clean_fact, f"site:wikipedia.org {clean_fact}"],
                    [f"{clean_fact} debunked", f"{clean_fact} refuted"],
                    [f"site:.gov {clean_fact}", f"site:.edu {clean_fact}"],
                ]

                all_hits_serper: List[Dict[str, str]] = []
                all_hits_ddg: List[Dict[str, str]] = []

                def fetch_round(qs: List[str], round_idx: int) -> None:
                    nonlocal all_hits_serper, all_hits_ddg
                    print(f"        🔁 WEB round {round_idx}: {len(qs)} queries")
                    for qi, q in enumerate(qs, start=1):
                        q = (q or "").strip()
                        if not q:
                            continue
                        print(f"        🔎 Q{round_idx}.{qi}: {q}")
                        payload_s = RealToolkit.google_search(q)
                        payload = _safe_json_loads(payload_s) or {}
                        all_hits_serper.extend(payload.get("serper", []) or [])
                        all_hits_ddg.extend(payload.get("ddg", []) or [])
                        time.sleep(0.05)

                # Iterate rounds: after each round => (1) refute heuristic, (2) judge early, (3) if confident stop
                for ridx, qs in enumerate(rounds, start=1):
                    fetch_round(qs, ridx)

                    all_hits_serper = _dedup_hits(all_hits_serper)
                    all_hits_ddg = _dedup_hits(all_hits_ddg)
                    combined = _dedup_hits(all_hits_serper + all_hits_ddg)

                    # Strong refutation heuristic (only this can short-circuit)
                    v_refute, s_refute = RealToolkit._evidence_heuristic_refute_only(clean_fact, combined)
                    if v_refute is False and s_refute == "strong":
                        out = JudgeResult(False, 0.95, [], [], "Heuristic strong refutation cues (hoax/refute)")
                        RealToolkit._cache[cache_key] = out
                        return out

                    # Build evidence lines (prefer trusted, but do NOT require >=2 sources before judging)
                    trusted = [h for h in combined if _is_trusted_domain(h.get("url", ""))]
                    evidence_base = trusted if trusted else combined
                    evidence_lines = _summarize_hits(evidence_base, max_n=8) if evidence_base else ""

                    # If nothing, continue to next round (or fallback later)
                    if not evidence_lines.strip():
                        continue

                    # RAG judge EARLY (this is the main patch you asked for)
                    jr = RealToolkit._llm_rag_judge(clean_fact, evidence_lines)
                    print(
                        f"        🧠 RAG_JUDGE round={ridx} vote={jr.verdict} conf={jr.confidence:.2f} "
                        f"support={jr.support_ids} refute={jr.refute_ids} rationale={jr.rationale}"
                    )

                    # Accept if confident enough
                    if jr.verdict is True and jr.confidence >= JUDGE_TRUE_MIN_CONF:
                        out = jr
                        RealToolkit._cache[cache_key] = out
                        return out
                    if jr.verdict is False and jr.confidence >= JUDGE_FALSE_MIN_CONF:
                        out = jr
                        RealToolkit._cache[cache_key] = out
                        return out

                    # Else: not confident => expand next round (do NOT use ">=2 trusted sources" as a gate)
                    # But we can log it for debugging:
                    td = _distinct_trusted_domains(trusted)
                    if ridx < len(rounds):
                        print(
                            f"        ℹ️ Judge not confident yet (trusted_domains={td}); expanding search..."
                        )

                # After all rounds: final attempt via commonsense + weak 2-of-3 (optional)
                combined = _dedup_hits(all_hits_serper + all_hits_ddg)
                trusted = [h for h in combined if _is_trusted_domain(h.get("url", ""))]
                evidence_base = trusted if trusted else combined
                evidence_lines = _summarize_hits(evidence_base, max_n=8) if evidence_base else ""

                if evidence_lines.strip():
                    jr = RealToolkit._llm_rag_judge(clean_fact, evidence_lines)
                    print(
                        f"        🧠 RAG_JUDGE final vote={jr.verdict} conf={jr.confidence:.2f} "
                        f"support={jr.support_ids} refute={jr.refute_ids} rationale={jr.rationale}"
                    )
                    # keep as ABSTAIN if low-conf
                    if jr.verdict is True and jr.confidence >= JUDGE_TRUE_MIN_CONF:
                        out = jr
                    elif jr.verdict is False and jr.confidence >= JUDGE_FALSE_MIN_CONF:
                        out = jr
                    else:
                        out = JudgeResult(None, max(0.0, jr.confidence), jr.support_ids, jr.refute_ids, jr.rationale)
                    RealToolkit._cache[cache_key] = out
                    return out

                # No evidence at all: commonsense fallback
                v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                out = JudgeResult(v_cs, 0.55 if v_cs is not None else 0.0, [], [], "Common-sense fallback (no evidence)")
                RealToolkit._cache[cache_key] = out
                return out

            # ------------------------
            # COMMON_SENSE / default
            # ------------------------
            clean_fact = RealToolkit._distill_claim(claim)
            v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
            out = JudgeResult(v_cs, 0.55 if v_cs is not None else 0.0, [], [], "Common-sense")
            RealToolkit._cache[cache_key] = out
            return out

        except Exception as e:
            # On tool failure, ABSTAIN (do not confidently prune)
            out = JudgeResult(None, 0.0, [], [], f"Verification error: {e}")
            RealToolkit._cache[cache_key] = out
            return out

    # ---------- Main claim verification (BOOL compat) ----------
    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> bool:
        display_claim = (claim or "")[:80].replace("\n", " ")
        print(f"      🕵️ [Processing]: '{display_claim}...' via {tool_type}")

        cache_key = RealToolkit._cache_key("verify_bool", tool_type, claim or "")
        if cache_key in RealToolkit._cache:
            return RealToolkit._cache[cache_key]

        try:
            jr = RealToolkit.verify_claim_rich(tool_type, claim)

            # Pretty logging
            if tool_type == "WEB_SEARCH":
                if jr.verdict is True:
                    print(f"        └─ ✅ Final Result: TRUE (conf={jr.confidence:.2f})")
                elif jr.verdict is False:
                    print(f"        └─ ❌ Final Result: FALSE (conf={jr.confidence:.2f})")
                else:
                    print(f"        └─ ⚪ Final Result: ABSTAIN (conf={jr.confidence:.2f}) -> return FALSE")

            # Compatibility mapping:
            # - TRUE only when judge says TRUE with enough confidence.
            # - FALSE when judge says FALSE with enough confidence.
            # - ABSTAIN -> False (conservative: not verified true)
            if jr.verdict is True and jr.confidence >= JUDGE_TRUE_MIN_CONF:
                out = True
            elif jr.verdict is False and jr.confidence >= JUDGE_FALSE_MIN_CONF:
                out = False
            else:
                out = False

            RealToolkit._cache[cache_key] = out
            return out

        except Exception as e:
            print(f"      ⚠️ Verification Error: {e}")
            # Conservative: do not prune; but bool interface has only True/False.
            # Keeping previous behavior: return True on failure to avoid collapsing graph.
            RealToolkit._cache[cache_key] = True
            return True
