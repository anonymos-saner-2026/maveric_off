"""
Class E baselines - Multi-agent verification families (MAV, BoN-MAV, MAD-Fact, GKMAD)

These baselines follow MaVERiC-style budget constraints (tool calls or cost).
All tool usage is accounted for via TOOLS_CONFIG costs.
"""
from __future__ import annotations

import json
import math
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from src.config import client, GENERATOR_MODEL, JUDGE_MODEL, TOOLS_CONFIG
from src.tools.real_toolkit import RealToolkit
from src.baselines.linear_tool import format_snippets_for_prompt, parse_binary_label
from src.baselines.verification_heavy import extract_atomic_claims


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------
def _tool_cost(tool_name: str, default_cost: float = 5.0) -> float:
    return float(TOOLS_CONFIG.get(tool_name, {}).get("cost", default_cost))


def _choose_tool_type(claim: str) -> str:
    s = (claim or "").lower()
    if re.search(r"\d", s) and re.search(r"[=<>+\-*/]", s):
        return "PYTHON_EXEC"
    if re.search(r"\b(percent|percentage|average|mean|sum|ratio|square|sqrt|root)\b", s):
        return "PYTHON_EXEC"
    return "WEB_SEARCH"


def _single_search_evidence(query: str) -> Tuple[List[Dict[str, str]], float]:
    cost = _tool_cost("WEB_SEARCH", default_cost=5.0)
    try:
        raw = RealToolkit.google_search(query)
        data = json.loads(raw) if raw else {}
    except Exception:
        return [], 0.0

    snippets: List[Dict[str, str]] = []
    for hit in (data.get("serper") or [])[:5]:
        snippets.append(
            {
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "snippet": hit.get("snippet", ""),
                "provider": "serper",
            }
        )
    for hit in (data.get("ddg") or [])[:5]:
        snippets.append(
            {
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "snippet": hit.get("snippet", ""),
                "provider": "ddg",
            }
        )
    return snippets, cost


def _parse_json_block(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        return {}
    return {}


def _claim_priority_score(claim: str) -> float:
    if not claim:
        return 0.0
    s = claim.strip()
    score = min(3.0, len(s) / 60.0)

    if re.search(r"\d", s):
        score += 2.0
    if re.search(r"\b(19\d{2}|20\d{2})\b", s):
        score += 1.5
    if re.search(r"\b(first|largest|smallest|oldest|youngest|highest|lowest|most|least)\b", s, re.I):
        score += 1.0
    if re.search(r"\b[A-Z][a-z]+\b", s) and len(re.findall(r"\b[A-Z][a-z]+\b", s)) >= 2:
        score += 1.0
    if re.search(r"\b(according to|study|report|survey|data)\b", s, re.I):
        score += 0.5
    return float(score)


def _split_budget(total: float, m: int) -> List[float]:
    if m <= 0:
        return []
    base = float(total) / float(m)
    return [base for _ in range(m)]


def _safe_llm_call(prompt: str, model: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
    if client is None:
        return ""
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (res.choices[0].message.content or "").strip()
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# E1) MAV (Multi-Agent Verification)
# -----------------------------------------------------------------------------
class MAVBaseline:
    """
    E1. MAV (Multi-Agent Verification)

    - Decomposes a response into atomic claims
    - Uses multiple Aspect Verifiers (AVs)
    - Aggregates binary approvals per claim under a shared budget
    """

    AV_PROMPT = """You are an aspect verifier for factual claims.

Aspect focus: {aspect}
Strategy: {strategy}

Claim: "{claim}"

Evidence snippets:
{evidence}

Instructions:
- Judge ONLY based on evidence.
- If evidence is missing or irrelevant, answer UNKNOWN.
- Be strict about entity, time, and numerical accuracy.

Output format (STRICT JSON):
{{
  "verdict": "TRUE" or "FALSE" or "UNKNOWN",
  "confidence": 0.0-1.0,
  "evidence_ids": [1,2],
  "rationale": "short reason"
}}
"""

    def __init__(self, num_verifiers: int = 5, max_claims: int = 10, seed: int = 0):
        self.num_verifiers = int(num_verifiers)
        self.max_claims = int(max_claims)
        self._rng = random.Random(seed)
        self.last_votes: Dict[str, Dict[str, int]] = {}
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def _build_aspect_verifiers(self) -> List[Dict[str, str]]:
        aspects = [
            {"aspect": "entities and relations", "strategy": "spot entity mismatches"},
            {"aspect": "dates and timelines", "strategy": "check year and sequence"},
            {"aspect": "numbers and quantities", "strategy": "verify numeric accuracy"},
            {"aspect": "source grounding", "strategy": "prioritize reliable sources"},
            {"aspect": "logical consistency", "strategy": "validate internal logic"},
            {"aspect": "commonsense sanity", "strategy": "detect implausible claims"},
        ]
        if self.num_verifiers <= len(aspects):
            return aspects[: self.num_verifiers]
        out = list(aspects)
        while len(out) < self.num_verifiers:
            out.append(aspects[len(out) % len(aspects)])
        return out

    def _select_claims(self, claims: List[str], budget: float) -> List[str]:
        if not claims:
            return []
        scored = sorted(
            [(c, _claim_priority_score(c)) for c in claims],
            key=lambda x: x[1],
            reverse=True,
        )
        selected: List[str] = []
        remaining = float(budget)
        for c, _ in scored:
            tool = _choose_tool_type(c)
            cost = _tool_cost(tool, default_cost=5.0)
            if cost <= remaining + 1e-12:
                selected.append(c)
                remaining -= cost
        return selected

    def _av_judge(self, claim: str, evidence: List[Dict[str, str]], av: Dict[str, str]) -> Optional[bool]:
        evidence_text = format_snippets_for_prompt(evidence, max_chars=2500)
        prompt = self.AV_PROMPT.format(
            aspect=av["aspect"],
            strategy=av["strategy"],
            claim=claim,
            evidence=evidence_text,
        )
        raw = _safe_llm_call(prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=240)
        data = _parse_json_block(raw)
        verdict = str(data.get("verdict", "")).upper().strip()
        if verdict == "TRUE":
            return True
        if verdict == "FALSE":
            return False
        if verdict == "UNKNOWN":
            return None
        return parse_binary_label(raw)

    def run_mav(
        self,
        question: str,
        response: str,
        budget: float,
        claims: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}
        if claims is None:
            claims = extract_atomic_claims(response or "", max_claims=self.max_claims)
        claims = [c for c in claims if c]
        if not claims:
            claims = [response] if response else [question]

        if question and question not in claims:
            claims = [question] + claims

        avs = self._build_aspect_verifiers()
        total_budget = float(budget)
        budgets = _split_budget(total_budget, len(avs))
        min_tool_cost = min(_tool_cost("WEB_SEARCH", 5.0), _tool_cost("PYTHON_EXEC", 8.0))
        if budgets and max(budgets) < min_tool_cost:
            budgets = [total_budget] + [0.0 for _ in range(len(avs) - 1)]

        approvals: Dict[str, int] = {c: 0 for c in claims}
        checked: Dict[str, int] = {c: 0 for c in claims}
        per_claim_calls: Dict[str, int] = {c: 0 for c in claims}
        per_claim_spent: Dict[str, float] = {c: 0.0 for c in claims}

        for av, b in zip(avs, budgets):
            remaining = float(b)
            selected = self._select_claims(claims, remaining)
            for c in selected:
                tool = _choose_tool_type(c)
                cost = _tool_cost(tool, default_cost=5.0)
                if cost > remaining + 1e-12:
                    continue

                verdict: Optional[bool] = None
                evidence: List[Dict[str, str]] = []
                if tool == "PYTHON_EXEC":
                    try:
                        res = RealToolkit.verify_claim("PYTHON_EXEC", c)
                        verdict = True if res is True else False if res is False else None
                    except Exception:
                        verdict = None
                    self.tool_calls += 1
                    self.budget_spent += cost
                    remaining -= cost
                    per_claim_calls[c] += 1
                    per_claim_spent[c] += cost
                else:
                    evidence, spent = _single_search_evidence(c)
                    if spent <= 0:
                        continue
                    self.tool_calls += 1
                    self.budget_spent += spent
                    remaining -= spent
                    per_claim_calls[c] += 1
                    per_claim_spent[c] += spent
                    verdict = self._av_judge(c, evidence, av)

                checked[c] += 1
                if verdict is True:
                    approvals[c] += 1

        self.last_votes = {
            c: {"approvals": approvals[c], "checked": checked[c]}
            for c in claims
        }
        self.stats = {
            "claims_count": int(len(claims)),
            "tool_calls_per_claim": {k: int(v) for k, v in per_claim_calls.items() if v > 0},
            "budget_spent_per_claim": {k: round(float(v), 2) for k, v in per_claim_spent.items() if v > 0},
            "budget_utilization": round(float(self.budget_spent) / max(1e-9, float(budget)), 3),
        }

        accepted: List[str] = []
        for c in claims:
            if checked[c] <= 0:
                continue
            if approvals[c] >= int(math.ceil(checked[c] / 2.0)):
                accepted.append(c)

        return {
            "claims": claims,
            "accepted": accepted,
            "votes": self.last_votes,
        }

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        response: Optional[str] = None,
        budget: float = 20.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}
        response_text = response if response is not None else claim
        out = self.run_mav(question=claim, response=response_text, budget=budget)
        accepted = set(out.get("accepted", []))
        votes = out.get("votes", {})
        if claim in votes and votes[claim].get("checked", 0) <= 0:
            return None
        if claim in accepted:
            return True
        if claim in votes and votes[claim].get("checked", 0) > 0:
            return False
        return None


# -----------------------------------------------------------------------------
# E2) BoN-MAV (Best-of-n + MAV)
# -----------------------------------------------------------------------------
class BoNMAVBaseline:
    """
    E2. BoN-MAV (Best-of-n + MAV)

    - Samples n candidates
    - Pre-scores with a cheap self-critic
    - Applies MAV to top-k candidates under budget split
    """

    GENERATE_PROMPT = """You are answering a factual question.

Question: {question}

Provide a concise, factual answer.
"""

    SELF_CRITIC_PROMPT = """You are rating the factual plausibility of an answer.

Question: {question}
Answer: {answer}

Return a single number from 0 to 10 indicating plausibility.
Only output the number.
"""

    def __init__(self, n: int = 5, m_verifiers: int = 5, top_k: int = 4, seed: int = 0):
        self.n = int(n)
        self.m_verifiers = int(m_verifiers)
        self.top_k = int(top_k)
        self._rng = random.Random(seed)
        self._mav = MAVBaseline(num_verifiers=m_verifiers, seed=seed)
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def _generate_candidates(self, question: str, n: int) -> List[str]:
        candidates: List[str] = []
        for _ in range(max(1, n)):
            prompt = self.GENERATE_PROMPT.format(question=question)
            txt = _safe_llm_call(prompt, model=GENERATOR_MODEL, temperature=0.8, max_tokens=180)
            if txt:
                candidates.append(txt)
        return candidates

    def _self_critic_score(self, question: str, answer: str) -> float:
        prompt = self.SELF_CRITIC_PROMPT.format(question=question, answer=answer)
        raw = _safe_llm_call(prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=8)
        try:
            return float(raw.strip().split()[0])
        except Exception:
            return 0.0

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        candidates: Optional[List[str]] = None,
        budget: float = 20.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}
        if candidates is None or not candidates:
            candidates = self._generate_candidates(claim, self.n)
        if not candidates:
            return None

        scores = [(a, self._self_critic_score(claim, a)) for a in candidates]
        scores.sort(key=lambda x: x[1], reverse=True)
        topk = [a for a, _ in scores[: max(1, min(self.top_k, len(scores)) )]]

        budget_each = float(budget) / float(len(topk))
        best_score = -1.0
        best_verdict: Optional[bool] = None
        best_answer: Optional[str] = None

        for ans in topk:
            out = self._mav.run_mav(question=claim, response=ans, budget=budget_each)
            self.tool_calls += int(getattr(self._mav, "tool_calls", 0) or 0)
            self.budget_spent += float(getattr(self._mav, "budget_spent", 0.0) or 0.0)
            votes = out.get("votes", {})
            if claim in votes and votes[claim].get("checked", 0) > 0:
                approvals = votes[claim].get("approvals", 0)
                checked = votes[claim].get("checked", 1)
                score = float(approvals) / float(checked)
            else:
                score = 0.0

            verdict = True if claim in out.get("accepted", []) else False if (claim in votes and votes[claim].get("checked", 0) > 0) else None
            if score > best_score:
                best_score = score
                best_verdict = verdict
                best_answer = ans

        self.stats = {
            "candidates_count": int(len(candidates)),
            "topk": int(len(topk)),
            "budget_per_candidate": round(float(budget_each), 2),
            "best_score": round(float(best_score), 3),
            "best_answer_preview": (best_answer or "")[:120],
        }

        return best_verdict


# -----------------------------------------------------------------------------
# E3) MAD-Fact
# -----------------------------------------------------------------------------
class MADFactBaseline:
    """
    E3. MAD-Fact (claim-centric multi-juror debate)
    """

    CLERK_PROMPT = """You are a clerk extracting atomic factual claims.

Question: {question}
Response: {response}

Extract 3-8 atomic factual claims. Each must be verifiable and standalone.
Output ONLY JSON:
{{"claims": ["..."]}}
"""

    JUROR_PROMPT = """You are Juror #{idx} in a fact-checking jury.

Claim: "{claim}"
Question: {question}

Evidence snippets:
{evidence}

Debate so far:
{debate}

Instructions:
- If evidence is insufficient, choose UNSURE.
- If evidence clearly supports, choose SUPPORTED.
- If evidence contradicts, choose REFUTED.

Output ONLY JSON:
{{
  "stance": "SUPPORTED" or "REFUTED" or "UNSURE",
  "evidence_ids": [1,2],
  "rationale": "short"
}}
"""

    JUDGE_PROMPT = """You are the judge. Aggregate jury stances.

Claim: "{claim}"
Stances: {stances}

Rules:
- Majority SUPPORTED => SUPPORTED
- Majority REFUTED => REFUTED
- Otherwise => UNSURE

Output: SUPPORTED, REFUTED, or UNSURE
"""

    def __init__(self, num_jurors: int = 3, rounds: int = 2, max_claims: int = 8, seed: int = 0):
        self.num_jurors = int(num_jurors)
        self.rounds = int(rounds)
        self.max_claims = int(max_claims)
        self._rng = random.Random(seed)
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def _clerk_extract(self, question: str, response: str) -> List[str]:
        prompt = self.CLERK_PROMPT.format(question=question, response=response)
        raw = _safe_llm_call(prompt, model=GENERATOR_MODEL, temperature=0.0, max_tokens=500)
        data = _parse_json_block(raw)
        claims = data.get("claims", []) if isinstance(data, dict) else []
        if not claims:
            claims = extract_atomic_claims(response or "", max_claims=self.max_claims)
        return [str(c).strip() for c in claims if c][: self.max_claims]

    def _juror_step(self, idx: int, claim: str, question: str, evidence: List[Dict[str, str]], debate: str) -> str:
        evidence_text = format_snippets_for_prompt(evidence, max_chars=2200)
        prompt = self.JUROR_PROMPT.format(
            idx=idx,
            claim=claim,
            question=question,
            evidence=evidence_text,
            debate=debate or "[none]",
        )
        return _safe_llm_call(prompt, model=JUDGE_MODEL, temperature=0.2, max_tokens=220)

    def _judge(self, claim: str, stances: List[str]) -> str:
        prompt = self.JUDGE_PROMPT.format(claim=claim, stances=", ".join(stances))
        raw = _safe_llm_call(prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=12)
        up = (raw or "").strip().upper()
        if "SUPPORTED" in up:
            return "SUPPORTED"
        if "REFUTED" in up:
            return "REFUTED"
        return "UNSURE"

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        response: Optional[str] = None,
        budget: float = 20.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}
        response_text = response if response is not None else claim
        claims = self._clerk_extract(claim, response_text)
        if claim not in claims:
            claims = [claim] + claims

        budget_each = float(budget) / max(1, len(claims))
        verdicts: Dict[str, str] = {}

        per_claim_calls: Dict[str, int] = {c: 0 for c in claims}
        per_claim_spent: Dict[str, float] = {c: 0.0 for c in claims}

        for c in claims:
            remaining = float(budget_each)
            debate_history: List[str] = []
            stances = ["UNSURE" for _ in range(self.num_jurors)]

            for _ in range(self.rounds):
                evidence: List[Dict[str, str]] = []
                if remaining >= _tool_cost("WEB_SEARCH", default_cost=5.0):
                    evidence, spent = _single_search_evidence(c)
                    if spent > 0:
                        self.tool_calls += 1
                        self.budget_spent += spent
                        remaining -= spent
                        per_claim_calls[c] += 1
                        per_claim_spent[c] += spent

                round_msgs: List[str] = []
                for j in range(self.num_jurors):
                    raw = self._juror_step(j + 1, c, claim, evidence, "\n".join(debate_history))
                    data = _parse_json_block(raw)
                    stance = str(data.get("stance", "")).upper()
                    if stance not in {"SUPPORTED", "REFUTED", "UNSURE"}:
                        stance = "UNSURE"
                    stances[j] = stance
                    round_msgs.append(f"Juror{j+1}: {stance}")
                debate_history.append(" | ".join(round_msgs))

            verdicts[c] = self._judge(c, stances)

        self.stats = {
            "claims_count": int(len(claims)),
            "tool_calls_per_claim": {k: int(v) for k, v in per_claim_calls.items() if v > 0},
            "budget_spent_per_claim": {k: round(float(v), 2) for k, v in per_claim_spent.items() if v > 0},
            "budget_utilization": round(float(self.budget_spent) / max(1e-9, float(budget)), 3),
        }

        if claim in verdicts:
            if verdicts[claim] == "SUPPORTED":
                return True
            if verdicts[claim] == "REFUTED":
                return False
            return None

        if any(v == "REFUTED" for v in verdicts.values()):
            return False
        if all(v == "SUPPORTED" for v in verdicts.values()):
            return True
        return None


# -----------------------------------------------------------------------------
# E4) GKMAD
# -----------------------------------------------------------------------------
class GKMADBaseline:
    """
    E4. GKMAD - Guided and Knowledgeable Multi-Agent Debate
    """

    GUIDE = [
        "Clarify entities, time, and quantities.",
        "Generate a precise verification query.",
        "Retrieve evidence and cite it.",
        "Argue support or refute based on evidence.",
        "Identify unresolved uncertainties.",
    ]

    DEBATER_PROMPT = """You are the {role} debater.

Guide checklist:
{guide}

Claim: "{claim}"
Question: {question}

Evidence:
{evidence}

Debate memory:
{memory}

Rules:
- Follow the guide steps.
- Cite evidence ids when possible.
- If evidence is insufficient, say so.

Output (STRICT JSON):
{{
  "stance": "SUPPORTED" or "REFUTED" or "UNSURE",
  "evidence_ids": [1,2],
  "argument": "short"
}}
"""

    ADVISOR_PROMPT = """You are the advisor.

Claim: "{claim}"
Question: {question}

Pro message: {pro_msg}
Con message: {con_msg}

Evidence:
{evidence}

Memory:
{memory}

Provide:
1) Key disagreements
2) Missing evidence
3) Next search query (one)

Output JSON:
{{
  "disagreements": "...",
  "missing": "...",
  "next_query": "..."
}}
"""

    FINAL_VERIFY_PROMPT = """You are the final verifier. Ignore rhetoric.

Claim: "{claim}"
Question: {question}

Debate memory:
{memory}

Final evidence:
{evidence}

Decide SUPPORTED, REFUTED, or UNSURE based only on evidence quality.
Output ONLY one of: SUPPORTED, REFUTED, UNSURE
"""

    def __init__(self, rounds: int = 2, seed: int = 0):
        self.rounds = int(rounds)
        self._rng = random.Random(seed)
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats: Dict[str, Any] = {}

    def verify(
        self,
        claim: str,
        transcript: Optional[str] = None,
        response: Optional[str] = None,
        budget: float = 20.0,
    ) -> Optional[bool]:
        self.tool_calls = 0
        self.budget_spent = 0.0
        self.stats = {}
        remaining = float(budget)
        memory: List[str] = []
        guide_text = "\n".join([f"- {g}" for g in self.GUIDE])

        for _ in range(self.rounds):
            evidence: List[Dict[str, str]] = []
            if remaining >= _tool_cost("WEB_SEARCH", default_cost=5.0):
                evidence, spent = _single_search_evidence(claim)
                if spent > 0:
                    self.tool_calls += 1
                    self.budget_spent += spent
                    remaining -= spent

            evidence_text = format_snippets_for_prompt(evidence, max_chars=2200)
            mem_text = "\n".join(memory[-4:]) if memory else "[none]"

            pro_prompt = self.DEBATER_PROMPT.format(
                role="PRO",
                guide=guide_text,
                claim=claim,
                question=claim,
                evidence=evidence_text,
                memory=mem_text,
            )
            con_prompt = self.DEBATER_PROMPT.format(
                role="CON",
                guide=guide_text,
                claim=claim,
                question=claim,
                evidence=evidence_text,
                memory=mem_text,
            )

            pro_msg = _safe_llm_call(pro_prompt, model=JUDGE_MODEL, temperature=0.2, max_tokens=220)
            con_msg = _safe_llm_call(con_prompt, model=JUDGE_MODEL, temperature=0.2, max_tokens=220)

            advisor_prompt = self.ADVISOR_PROMPT.format(
                claim=claim,
                question=claim,
                pro_msg=pro_msg,
                con_msg=con_msg,
                evidence=evidence_text,
                memory=mem_text,
            )
            advisor_msg = _safe_llm_call(advisor_prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=200)

            memory.append(f"PRO: {pro_msg}")
            memory.append(f"CON: {con_msg}")
            memory.append(f"ADVISOR: {advisor_msg}")

        final_evidence: List[Dict[str, str]] = []
        if remaining >= _tool_cost("WEB_SEARCH", default_cost=5.0):
            final_evidence, spent = _single_search_evidence(claim)
            if spent > 0:
                self.tool_calls += 1
                self.budget_spent += spent
                remaining -= spent

        self.stats = {
            "rounds": int(self.rounds),
            "budget_utilization": round(float(self.budget_spent) / max(1e-9, float(budget)), 3),
        }

        final_prompt = self.FINAL_VERIFY_PROMPT.format(
            claim=claim,
            question=claim,
            memory="\n".join(memory[-6:]) if memory else "[none]",
            evidence=format_snippets_for_prompt(final_evidence, max_chars=2200),
        )
        raw = _safe_llm_call(final_prompt, model=JUDGE_MODEL, temperature=0.0, max_tokens=12)
        up = (raw or "").strip().upper()
        if "SUPPORTED" in up:
            return True
        if "REFUTED" in up:
            return False
        return None


__all__ = [
    "MAVBaseline",
    "BoNMAVBaseline",
    "MADFactBaseline",
    "GKMADBaseline",
]
