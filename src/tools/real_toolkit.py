# tools/real_toolkit.py
# Patched version (your code + RAG + stance-judge):
#   - WEB_SEARCH now uses "RAG + Judge (stance classification)" to avoid stance confusion
#     (e.g., debunk pages mentioning a claim but refuting it).
#   - Retrieve k sources (prioritize whitelist/trusted domains), feed snippets to LLM judge:
#       stance ∈ {SUPPORT, REFUTE, UNKNOWN} + cite snippet ids used.
#   - Aggregate with your existing 2-of-3 voting (Serper heuristic + DDG heuristic + Judge).
#   - Keeps PYTHON_EXEC deterministic tier0 + sanity harness path intact.
#
# Requires: src.config provides client, SERPER_API_KEY, JUDGE_MODEL.

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
from typing import Optional, Dict, Any, Tuple, List

from src.config import client, SERPER_API_KEY, JUDGE_MODEL

warnings.filterwarnings("ignore", category=RuntimeWarning, module="duckduckgo_search")
warnings.filterwarnings("ignore", category=UserWarning, module="duckduckgo_search")


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


def _contains_all(hay: str, needles: List[str]) -> bool:
    h = _norm_text(hay)
    return all(_norm_text(n) in h for n in needles if n)


def _domain(url: str) -> str:
    u = (url or "").strip().lower()
    m = re.search(r"https?://([^/]+)", u)
    return m.group(1) if m else ""


def _summarize_hits(hits: List[Dict[str, str]], max_n: int = 5) -> str:
    out = []
    for r in (hits or [])[:max_n]:
        title = (r.get("title") or "")[:160]
        url = (r.get("url") or "")[:220]
        snip = (r.get("snippet") or "")[:220]
        out.append(f"- {title} | {url} | {snip}")
    return "\n".join(out)


# -----------------------------
# Trust / Whitelist policy
# -----------------------------
# "Domain whitelist" (your requested style for papers): prioritize these first for RAG.
# You can freely extend this list.
DOMAIN_WHITELIST = [
    # science / reference
    "nasa.gov",
    "esa.int",
    "noaa.gov",
    "who.int",
    "un.org",
    "science.org",
    "nature.com",
    "britannica.com",
    "wikipedia.org",
    # major news wires (still useful for timely facts)
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    # olympics-specific
    "olympics.com",
    "ioc.org",
    # common .edu / .gov handled by suffix checks below
]

# "Trusted domains" for heuristic layer (kept from your code, slightly expanded & normalized)
TRUSTED_DOMAIN_SUBSTRINGS = [
    "olympics.com",
    "ioc.org",
    "wikipedia.org",
    "britannica.com",
    "reuters.com",
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",
    "who.int",
    "un.org",
]

# You can flip this to True if you want to ONLY feed whitelisted sources to the judge.
# Default: False (still prefers whitelist, but allows high-quality non-whitelist if needed).
STRICT_RAG_WHITELIST = False


def _is_trusted_domain(url: str) -> bool:
    d = _domain(url)
    if not d:
        return False
    if d.endswith(".gov") or ".gov." in d:
        return True
    if d.endswith(".edu") or ".edu." in d:
        return True
    return any(t in d for t in TRUSTED_DOMAIN_SUBSTRINGS)


def _is_whitelisted_domain(url: str) -> bool:
    d = _domain(url)
    if not d:
        return False
    if d.endswith(".gov") or ".gov." in d:
        return True
    if d.endswith(".edu") or ".edu." in d:
        return True
    return any(w in d for w in DOMAIN_WHITELIST)


def _domain_weight(url: str, provider: str) -> float:
    """
    Ranking weight for RAG selection.
    - Whitelisted gets higher weight.
    - Trusted (news wires etc) gets decent.
    - Serper slightly preferred over DDG for stability.
    """
    w = 0.5
    if _is_whitelisted_domain(url):
        w += 1.5
    elif _is_trusted_domain(url):
        w += 1.0

    if provider == "serper":
        w += 0.10
    elif provider == "ddg":
        w += 0.00

    # small nudge for Wikipedia/Britannica/major science sites
    d = _domain(url)
    if "nasa.gov" in d or "science.org" in d or "nature.com" in d:
        w += 0.35
    if "britannica.com" in d:
        w += 0.30
    if "wikipedia.org" in d:
        w += 0.15

    return w


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
        # Avoid confusing judges with "Claim:" prefix.
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
    def _serper_search(query: str, num: int = 6, timeout: float = 6.0) -> List[Dict[str, str]]:
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
                        "url": (r.get("link") or "")[:600],
                        "snippet": (r.get("snippet") or "")[:700],
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
                        "url": (r.get("href") or r.get("url") or "")[:600],
                        "snippet": (r.get("body") or "")[:700],
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

    # ---------- Attack/Support relation checks (LLM) ----------
    @staticmethod
    def verify_attack(attacker: str, target: str) -> bool:
        key = RealToolkit._cache_key("attack", attacker, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Task: Logic Consistency Check.
Statement A (Attacker): "{attacker}"
Statement B (Target): "{target}"

Question: Does A logically invalidate, contradict, or provide a counter-argument to B?

Rules:
- If A says something is TRUE and B says it is FALSE (or vice versa) about the same proposition, return TRUE.
- If A corrects B with a conflicting fact, return TRUE.
- If A is irrelevant or supports B, return FALSE.

Reply strictly with ONLY 'TRUE' or 'FALSE'.
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            ans = res.choices[0].message.content.strip().upper()
            out = "TRUE" in ans
        except Exception:
            out = False

        RealToolkit._cache[key] = out
        return out

    @staticmethod
    def verify_support(source: str, target: str) -> bool:
        key = RealToolkit._cache_key("support", source, target)
        if key in RealToolkit._cache:
            return RealToolkit._cache[key]

        prompt = f"""
Task: Support Relation Check.
Statement A (Source): "{source}"
Statement B (Target): "{target}"

Question: Does A logically support, reinforce, or provide evidence for B?

Reply strictly with ONLY 'TRUE' or 'FALSE'.
"""
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            ans = res.choices[0].message.content.strip().upper()
            out = "TRUE" in ans
        except Exception:
            out = False

        RealToolkit._cache[key] = out
        return out

    # ---------- Evidence heuristics (kept, minor robustness) ----------
    @staticmethod
    def _evidence_heuristic_verdict(clean_fact: str, hits: List[Dict[str, str]]) -> Optional[bool]:
        """
        Returns True/False if we can decide from trusted snippets deterministically-ish.
        Returns None to abstain.
        """
        fact = _norm_text(clean_fact)

        trusted_hits = [h for h in (hits or []) if _is_trusted_domain(h.get("url", ""))]
        blob = " ".join([(h.get("title", "") + " " + h.get("snippet", "")) for h in trusted_hits])
        blob_n = _norm_text(blob)

        if len(trusted_hits) == 0:
            return None

        # Special-case: Olympics host city statements (your Paris 2024 case)
        if "2024" in fact and "summer olympics" in fact and ("held in" in fact or "hosted" in fact or "host city" in fact):
            paris = "paris" in fact
            tokyo = "tokyo" in fact
            los_angeles = ("los angeles" in fact) or ("la " in fact) or ("l.a" in fact)
            if paris:
                if ("paris" in blob_n) and ("2024" in blob_n) and ("summer olympics" in blob_n or "olympics" in blob_n):
                    return True
                if ("paris" in blob_n) and ("olympics" in blob_n) and ("2024" in blob_n):
                    return True
                if ("los angeles" in blob_n and "2028" in blob_n) or ("tokyo" in blob_n and "2020" in blob_n):
                    return False
                return None
            if tokyo:
                if ("tokyo" in blob_n) and ("2020" in blob_n) and ("olympics" in blob_n):
                    return False
                if ("paris" in blob_n) and ("2024" in blob_n) and ("olympics" in blob_n):
                    return False
                return None
            if los_angeles:
                if ("los angeles" in blob_n) and ("2028" in blob_n) and ("olympics" in blob_n):
                    return False
                if ("paris" in blob_n) and ("2024" in blob_n) and ("olympics" in blob_n):
                    return False
                return None

        # Generic lexical consistency: require at least 3 anchors appear
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", clean_fact)
        anchors = set([y for y in years])

        words = re.findall(r"[a-zA-Z]{4,}", clean_fact.lower())
        stop = {"that", "this", "with", "from", "were", "held", "host", "hosted", "equals", "year", "percent", "square", "root"}
        for w in words:
            if w not in stop:
                anchors.add(w)
            if len(anchors) >= 6:
                break

        matched = [a for a in anchors if a and a in blob_n]
        if len(matched) >= 3:
            return True

        if "not" in fact and (" not " not in blob_n) and len(matched) >= 2:
            return False

        return None

    # ---------- RAG selection + stance judge (NEW PATCH) ----------
    @staticmethod
    def _dedup_hits(hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        out = []
        for h in hits or []:
            url = (h.get("url") or "").strip()
            title = (h.get("title") or "").strip()
            if not url:
                continue
            k = (url, title)
            if k in seen:
                continue
            seen.add(k)
            out.append(h)
        return out

    @staticmethod
    def _rank_hits_for_rag(hits: List[Dict[str, str]], k: int = 6) -> List[Dict[str, str]]:
        """
        Pick top-k hits with:
          - whitelist/trust preference
          - provider slight preference
          - domain diversity (1 per domain)
        """
        hits = RealToolkit._dedup_hits(hits)

        # optionally strict whitelist filtering
        if STRICT_RAG_WHITELIST:
            hits = [h for h in hits if _is_whitelisted_domain(h.get("url", ""))]

        scored = []
        for h in hits:
            url = h.get("url", "") or ""
            provider = (h.get("provider", "") or "").lower()
            w = _domain_weight(url, provider)
            scored.append((w, h))

        scored.sort(key=lambda x: x[0], reverse=True)

        picked = []
        used_domains = set()
        for w, h in scored:
            if len(picked) >= k:
                break
            d = _domain(h.get("url", ""))
            if d in used_domains:
                continue
            used_domains.add(d)
            h2 = dict(h)
            h2["domain"] = d
            h2["rag_weight"] = float(w)
            picked.append(h2)

        return picked

    @staticmethod
    def _rag_stance_judge(clean_fact: str, hits: List[Dict[str, str]], k: int = 6) -> Dict[str, Any]:
        """
        Returns dict:
        {
          "stance": "SUPPORT"|"REFUTE"|"UNKNOWN",
          "confidence": float 0..1,
          "support_ids": [int],
          "refute_ids": [int],
          "used_ids": [int],
          "rationale": str
        }
        """
        top_hits = RealToolkit._rank_hits_for_rag(hits, k=k)

        # If nothing, abstain
        if not top_hits:
            return {"stance": "UNKNOWN", "confidence": 0.0, "support_ids": [], "refute_ids": [], "used_ids": [], "rationale": "no evidence"}

        evidence = []
        for i, h in enumerate(top_hits):
            evidence.append(
                {
                    "id": i,
                    "title": (h.get("title") or "")[:200],
                    "url": (h.get("url") or "")[:600],
                    "domain": (h.get("domain") or _domain(h.get("url", "")))[:120],
                    "snippet": (h.get("snippet") or "")[:900],
                    "provider": (h.get("provider") or "")[:20],
                }
            )

        system = """
You are a strict evidence stance judge for fact-checking.

Given:
- A claim
- A list of evidence snippets (each may mention the claim to either SUPPORT or REFUTE it)

Your task:
1) Determine the stance of the evidence toward the claim:
   - SUPPORT: evidence affirms the claim as true
   - REFUTE: evidence contradicts the claim
   - UNKNOWN: evidence is insufficient/unclear/mixed with no strong conclusion
2) Important: Debunk articles often mention a false claim while refuting it. Do NOT treat "mentions claim" as support.
3) Cite which evidence ids support/refute the claim.

Output STRICT JSON ONLY with keys:
{
  "stance": "SUPPORT"|"REFUTE"|"UNKNOWN",
  "confidence": number in [0,1],
  "support_ids": [int...],
  "refute_ids": [int...],
  "used_ids": [int...],
  "rationale": "short 1-3 sentence justification"
}

Rules:
- If evidence is insufficient, stance must be UNKNOWN and confidence <= 0.6.
- support_ids and refute_ids must be subsets of used_ids.
- used_ids should include the ids you actually relied on.
""".strip()

        user = json.dumps(
            {
                "claim": clean_fact,
                "evidence": evidence,
            },
            ensure_ascii=False,
        )

        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.0,
            )
            txt = (res.choices[0].message.content or "").strip()

            # Extract JSON object if wrapped
            m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
            js = m.group(0) if m else txt
            data = _safe_json_loads(js) or {}
        except Exception:
            data = {}

        stance = str(data.get("stance", "UNKNOWN")).strip().upper()
        if stance not in ("SUPPORT", "REFUTE", "UNKNOWN"):
            stance = "UNKNOWN"

        try:
            conf = float(data.get("confidence", 0.5))
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))

        def _clean_ids(x) -> List[int]:
            if not isinstance(x, list):
                return []
            out = []
            for v in x:
                try:
                    iv = int(v)
                    if 0 <= iv < len(evidence):
                        out.append(iv)
                except Exception:
                    pass
            # unique keep order
            seen = set()
            out2 = []
            for iv in out:
                if iv in seen:
                    continue
                seen.add(iv)
                out2.append(iv)
            return out2

        support_ids = _clean_ids(data.get("support_ids", []))
        refute_ids = _clean_ids(data.get("refute_ids", []))
        used_ids = _clean_ids(data.get("used_ids", []))

        # enforce subsets
        support_ids = [i for i in support_ids if i in used_ids] if used_ids else support_ids
        refute_ids = [i for i in refute_ids if i in used_ids] if used_ids else refute_ids

        rationale = str(data.get("rationale", "") or "").strip()
        if len(rationale) > 600:
            rationale = rationale[:600]

        return {
            "stance": stance,
            "confidence": conf,
            "support_ids": support_ids,
            "refute_ids": refute_ids,
            "used_ids": used_ids,
            "rationale": rationale,
            "top_hits": top_hits,  # keep for logging/citation use
        }

    @staticmethod
    def _stance_to_vote(stance_pack: Dict[str, Any]) -> Optional[bool]:
        stance = (stance_pack.get("stance") or "UNKNOWN").upper()
        conf = float(stance_pack.get("confidence") or 0.0)

        # conservative threshold: avoid over-committing on weak evidence
        if stance == "SUPPORT" and conf >= 0.65:
            return True
        if stance == "REFUTE" and conf >= 0.65:
            return False
        return None

    # ---------- Optional common sense (kept as last resort) ----------
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
            # WEB_SEARCH path (patched with RAG + stance judge)
            # ------------------------
            if tool_type == "WEB_SEARCH":
                clean_fact = RealToolkit._distill_claim(claim)

                # Build a small set of queries (rounds)
                # Round 1: raw + wikipedia
                # Round 2: optionally add a domain-specific query (olympics as in your code)
                queries_rounds: List[List[str]] = [
                    [clean_fact, f"site:wikipedia.org {clean_fact}"],
                    [f"site:olympics.com {clean_fact}"],
                ]

                all_serper: List[Dict[str, str]] = []
                all_ddg: List[Dict[str, str]] = []

                def run_round(qs: List[str], round_idx: int) -> None:
                    nonlocal all_serper, all_ddg
                    print(f"        🔁 WEB round {round_idx}: {len(qs)} queries")
                    for qi, q in enumerate(qs, start=1):
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

                    trusted = [h for h in (all_serper + all_ddg) if _is_trusted_domain(h.get("url", ""))]
                    if len(trusted) >= 4:
                        break

                # De-dup
                all_serper = RealToolkit._dedup_hits(all_serper)
                all_ddg = RealToolkit._dedup_hits(all_ddg)
                combined_hits = RealToolkit._dedup_hits(all_serper + all_ddg)

                trusted_hits = [h for h in combined_hits if _is_trusted_domain(h.get("url", ""))]
                whitelisted_hits = [h for h in combined_hits if _is_whitelisted_domain(h.get("url", ""))]

                # Provider votes: heuristic per-provider (kept)
                v_serper = RealToolkit._evidence_heuristic_verdict(clean_fact, all_serper)
                v_ddg = RealToolkit._evidence_heuristic_verdict(clean_fact, all_ddg)

                print(
                    f"        🧾 SERPER: hits={len(all_serper)} trusted={sum(1 for h in all_serper if _is_trusted_domain(h.get('url','')))} vote={v_serper}"
                )
                print(
                    f"        🧾 DDG: hits={len(all_ddg)} trusted={sum(1 for h in all_ddg if _is_trusted_domain(h.get('url','')))} vote={v_ddg}"
                )

                # Strong heuristic on combined trusted hits (kept early-accept)
                heuristic_all = RealToolkit._evidence_heuristic_verdict(clean_fact, combined_hits)
                if heuristic_all is not None:
                    verdict = heuristic_all
                    RealToolkit._cache[cache_key] = verdict
                    status_icon = "✅" if verdict else "❌"
                    print(f"        └─ {status_icon} Heuristic(trusted) Result: {'TRUE' if verdict else 'FALSE'}")
                    return verdict

                # ---- NEW: RAG + stance judge ----
                # Prefer whitelisted hits first for judge input; fallback to combined if empty.
                rag_pool = whitelisted_hits if len(whitelisted_hits) > 0 else combined_hits
                stance_pack = RealToolkit._rag_stance_judge(clean_fact, rag_pool, k=6)
                v_judge = RealToolkit._stance_to_vote(stance_pack)

                stance = stance_pack.get("stance", "UNKNOWN")
                conf = float(stance_pack.get("confidence", 0.0))
                sup_ids = stance_pack.get("support_ids", [])
                ref_ids = stance_pack.get("refute_ids", [])
                used_ids = stance_pack.get("used_ids", [])
                rationale = stance_pack.get("rationale", "")

                print(f"        🧑‍⚖️ RAG_JUDGE stance={stance} conf={conf:.2f} vote={v_judge}")
                if used_ids:
                    print(f"        🔗 Judge used evidence ids={used_ids} support={sup_ids} refute={ref_ids}")
                if rationale:
                    print(f"        📝 Judge rationale: {rationale[:220]}")

                # If judge abstains (UNKNOWN/lowconf), optionally call common-sense as tie-breaker
                v_llm = v_judge
                if v_llm is None:
                    v_llm = RealToolkit._llm_common_sense_vote(clean_fact)
                    print(f"        🧠 COMMON_SENSE vote={v_llm}")

                # 2-of-3 voting
                final = RealToolkit._vote_2_of_3(v_serper, v_ddg, v_llm)

                # If still no majority: conservative fallback
                if final is None:
                    # If judge gave high-confidence stance but below threshold (rare), respect direction softly
                    if stance in ("SUPPORT", "REFUTE") and conf >= 0.60:
                        final = True if stance == "SUPPORT" else False
                    else:
                        # If many trusted hits exist but unclear, do NOT hallucinate:
                        # choose False as conservative default (matches your prior behavior)
                        final = False

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
            # On tool failure, do NOT prune aggressively.
            RealToolkit._cache[cache_key] = True
            return True
