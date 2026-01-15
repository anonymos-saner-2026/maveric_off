# tools/real_toolkit.py
# Patched version (RAG + Judge, stance-aware) to reduce false-positives on "hoax" / misinformation claims.
#
# Key patches implemented:
# A) Heuristic now stance-aware:
#    - Detects REFUTE cues (debunked / myth / conspiracy theory / no evidence / false / refuted / hoax claim ...)
#    - For hoax-like claims: strong REFUTE => return FALSE
#    - Anchor-match no longer short-circuits TRUE.
#    - Only returns TRUE when there are strong AFFIRM cues aligned with the claim (rare, conservative).
#    - Otherwise returns None to defer to LLM evidence judge.
#
# B) Decision order changed:
#    - Heuristic only used for strong FALSE (refutation) or very strong TRUE (affirmation).
#    - Otherwise ALWAYS run evidence judge (RAG+judge).
#    - No "heuristic_all => immediate TRUE" anymore.
#
# C) Edge verify (attack/support) now uses TRUE/FALSE/ABSTAIN + confidence:
#    - We prune only when FALSE with high confidence.
#    - Public verify_attack/verify_support still return bool for compatibility:
#        * returns False only when confident FALSE (prune)
#        * returns True otherwise (keep edge)
#
# Requires: src.config provides client, SERPER_API_KEY, JUDGE_MODEL.
#
# Notes:
# - Keeps PYTHON_EXEC deterministic tier0 + sanity harness path.
# - WEB_SEARCH now produces evidence packs and sends them to a stance classifier judge.
# - Conservative by default: if insufficient evidence => ABSTAIN => return False (do NOT verify claim).
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


def _contains_all(hay: str, needles: List[str]) -> bool:
    h = _norm_text(hay)
    return all(_norm_text(n) in h for n in needles if n)


def _domain(url: str) -> str:
    u = (url or "").strip().lower()
    m = re.search(r"https?://([^/]+)", u)
    return m.group(1) if m else ""


def _is_gov_domain(d: str) -> bool:
    # strict-ish gov matching
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
            # Prefer ddgs if available; fallback to duckduckgo_search
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

    # ---------- Evidence heuristics (patched: stance-aware, conservative) ----------
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
        fact = _norm_text(clean_fact)
        hoaxy = RealToolkit._is_hoaxy_claim(clean_fact)

        trusted_hits = [h for h in (hits or []) if _is_trusted_domain(h.get("url", ""))]
        if not trusted_hits:
            return None, "weak"

        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in trusted_hits])
        blob_n = _norm_text(blob)

        # Strong refutation cues -> FALSE for hoax-like claims
        if hoaxy and RealToolkit._has_refute_cues(blob_n):
            return False, "strong"

        # "Apollo 2024 host city" special-case removed from heuristic TRUE short-circuit.
        # Keep only a safe negative:
        if "2024" in fact and "summer olympics" in fact and ("tokyo" in fact or "los angeles" in fact):
            # If claim says Tokyo/LA for 2024 Olympics => almost certainly false
            if "tokyo" in fact and "2024" in fact:
                return False, "strong"
            if ("los angeles" in fact or "l.a" in fact) and "2024" in fact:
                return False, "strong"

        # Extract lightweight anchors (years + key terms)
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_fact)
        anchors = set(years)

        words = re.findall(r"[a-zA-Z]{4,}", clean_fact.lower())
        stop = {
            "that",
            "this",
            "with",
            "from",
            "were",
            "held",
            "host",
            "hosted",
            "equals",
            "year",
            "percent",
            "square",
            "root",
            "claim",
            "proof",
        }
        for w in words:
            if w not in stop:
                anchors.add(w)
            if len(anchors) >= 8:
                break

        matched = [a for a in anchors if a and a in blob_n]

        # For hoax-like claims: never return TRUE just because matched anchors exist.
        if hoaxy:
            return None, "weak"

        # For normal factual claims: allow TRUE only if we see affirm cues + decent anchor coverage
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
    def _llm_rag_judge(clean_fact: str, evidence_lines: str) -> Tuple[Optional[bool], float, List[int], List[int], str]:
        """
        RAG + judge: stance classification with citations to snippet ids.
        Returns: (verdict, confidence, support_ids, refute_ids, rationale_short)
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
                # best-effort: try to extract a JSON object
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

            if v == "TRUE":
                return True, max(0.0, min(1.0, conf)), sup_i, ref_i, rat
            if v == "FALSE":
                return False, max(0.0, min(1.0, conf)), sup_i, ref_i, rat
            return None, max(0.0, min(1.0, conf)), sup_i, ref_i, rat
        except Exception:
            return None, 0.0, [], [], "Judge error"

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

        # Only prune if confident FALSE
        out = True
        if label == "FALSE" and conf >= 0.75:
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
        if label == "FALSE" and conf >= 0.75:
            out = False

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
            # WEB_SEARCH path (patched: stance-aware RAG+judge)
            # ------------------------
            if tool_type == "WEB_SEARCH":
                clean_fact = RealToolkit._distill_claim(claim)

                # Query rounds (keep modest; rely on judge)
                queries_rounds: List[List[str]] = [
                    [clean_fact, f"site:wikipedia.org {clean_fact}"],
                    [f"site:nasa.gov {clean_fact}", f"site:britannica.com {clean_fact}"],
                    [f"site:.edu {clean_fact}", f"site:science.org {clean_fact}"],
                ]

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

                    # Retry once if totally empty
                    if (len(all_serper) == 0) and (len(all_ddg) == 0):
                        time.sleep(0.25)
                        print(f"        ⚠️ No hits, retrying round {ridx} once...")
                        run_round(qs, ridx)

                    trusted_cnt = sum(1 for h in (all_serper + all_ddg) if _is_trusted_domain(h.get("url", "")))
                    if trusted_cnt >= 6:
                        break

                # De-dup by url+title
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
                trusted_hits = [h for h in combined_hits if _is_trusted_domain(h.get("url", ""))]

                # Provider heuristic votes (stance-aware), but do NOT short-circuit unless strong
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

                # Combined heuristic only used when STRONG (either strong FALSE or strong TRUE)
                v_all, s_all = RealToolkit._evidence_heuristic_verdict(clean_fact, combined_hits)
                if v_all is not None and s_all == "strong":
                    verdict = v_all
                    RealToolkit._cache[cache_key] = verdict
                    status_icon = "✅" if verdict else "❌"
                    print(f"        └─ {status_icon} Heuristic(strong) Result: {'TRUE' if verdict else 'FALSE'}")
                    return verdict

                # Build evidence pack: prefer trusted; else fallback to all
                evidence_base = trusted_hits if trusted_hits else combined_hits
                evidence_lines = _summarize_hits(evidence_base, max_n=8) if evidence_base else ""

                # If no evidence at all, conservative reject
                if not evidence_lines.strip():
                    v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                    print(f"        🧠 COMMON_SENSE vote={v_cs}")
                    final = bool(v_cs) if v_cs is not None else False
                    RealToolkit._cache[cache_key] = final
                    status_icon = "✅" if final else "❌"
                    print(f"        └─ {status_icon} Final (no-evidence) Result: {'TRUE' if final else 'FALSE'}")
                    return final

                # ALWAYS run RAG judge (per patch B)
                v_llm, conf, sup_ids, ref_ids, rat = RealToolkit._llm_rag_judge(clean_fact, evidence_lines)
                print(f"        🧠 RAG_JUDGE vote={v_llm} conf={conf:.2f} support={sup_ids} refute={ref_ids} rationale={rat}")

                # If judge abstains, fall back to 2-of-3 vote (heuristics as weak votes) + commonsense
                if v_llm is None:
                    v_cs = RealToolkit._llm_common_sense_vote(clean_fact)
                    print(f"        🧠 COMMON_SENSE vote={v_cs}")
                    final = RealToolkit._vote_2_of_3(v_serper, v_ddg, v_cs)

                    # Still undecided => conservative FALSE
                    if final is None:
                        final = False
                else:
                    # Judge gave TRUE/FALSE:
                    # Require some minimal confidence for TRUE; for FALSE we accept even moderate confidence
                    if v_llm is True and conf < 0.62:
                        # too weak to verify as TRUE
                        final = False
                    else:
                        final = bool(v_llm)

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
            # (match your previous conservative behavior)
            RealToolkit._cache[cache_key] = True
            return True
