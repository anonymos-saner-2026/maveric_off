"""
Class C Baselines - Verification-heavy (critic & revision)

This module implements 3 baselines that emphasize iterative verification
and revision cycles without strategic ROI-based claim selection:
- C1. BudgetedCRITICBaseline: Exhaustive-until-budget with graph pruning
- C2. VerifyAndReviseBaseline: Answer → extract claims → verify → revise loops
- C3. RARRBaseline: Retrieval-augmented revision (retrieve → revise loops)

All baselines respect tool budget and use calibrated confidence verification.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from src.config import client, GENERATOR_MODEL, JUDGE_MODEL, TOOLS_CONFIG
from src.tools.real_toolkit import RealToolkit, JudgeResult, _summarize_hits
from src.baselines.linear_tool import (
    retrieve_evidence, 
    format_snippets_for_prompt, 
    parse_binary_label,
    estimate_cost
)

# Try to import ArgumentationGraph for C1
try:
    from src.core.graph import ArgumentationGraph, ArgumentNode
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False
    ArgumentationGraph = None  # type: ignore
    ArgumentNode = None  # type: ignore


# ==============================================================================
# VERIFYCLAIM INTERFACE
# ==============================================================================

@dataclass
class VerifyResult:
    """Result from VerifyClaim function."""
    verdict: Optional[bool]      # True/False/None (abstain)
    confidence: float            # Calibrated confidence [0,1]
    spent: float                 # Actual cost spent
    support_ids: List[int]       # Evidence snippet IDs that support
    refute_ids: List[int]        # Evidence snippet IDs that refute
    rationale: str               # Judge's reasoning


def VerifyClaim(
    claim: str, 
    budget: float, 
    tool_type: str = "WEB_SEARCH"
) -> VerifyResult:
    """
    Unified verification interface with budget tracking and calibrated confidence.
    
    This wraps RealToolkit's verification infrastructure to provide:
    - Ternary verdict (True/False/None)
    - Calibrated confidence score
    - Cost tracking
    - Evidence citations
    
    Args:
        claim: The claim text to verify
        budget: Maximum budget available
        tool_type: "WEB_SEARCH" or "PYTHON_EXEC"
    
    Returns:
        VerifyResult with verdict, confidence, cost, and metadata
    """
    # Check budget
    web_cost = TOOLS_CONFIG.get("WEB_SEARCH", {}).get("cost", 5.0)
    if web_cost > budget:
        return VerifyResult(
            verdict=None,
            confidence=0.0,
            spent=0.0,
            support_ids=[],
            refute_ids=[],
            rationale="Budget insufficient for verification"
        )
    
    try:
        # Step 1: Retrieve evidence
        snippets, search_spent = retrieve_evidence(claim, None, min(budget, web_cost * 2))
        
        if not snippets:
            return VerifyResult(
                verdict=None,
                confidence=0.0,
                spent=search_spent,
                support_ids=[],
                refute_ids=[],
                rationale="No evidence found"
            )
        
        # Step 2: Format evidence for judge
        evidence_lines = _summarize_hits(snippets, max_n=10)
        
        # Step 3: Call calibrated RAG judge
        judge_result: JudgeResult = RealToolkit._rag_judge_with_calibrated_conf(
            clean_fact=claim,
            evidence_lines=evidence_lines,
            evidence_hits=snippets
        )
        
        return VerifyResult(
            verdict=judge_result.verdict,
            confidence=judge_result.final_confidence,
            spent=search_spent,
            support_ids=judge_result.support_ids,
            refute_ids=judge_result.refute_ids,
            rationale=judge_result.rationale
        )
        
    except Exception as e:
        print(f"        ⚠️ VerifyClaim error: {e}")
        return VerifyResult(
            verdict=None,
            confidence=0.0,
            spent=0.0,
            support_ids=[],
            refute_ids=[],
            rationale=f"Verification error: {e}"
        )


# ==============================================================================
# ATOMIC CLAIM EXTRACTOR
# ==============================================================================

EXTRACT_CLAIMS_PROMPT = """You are an expert at identifying factual claims in text.

Extract all ATOMIC FACTUAL CLAIMS from the text below.

## Requirements for Each Claim
1. **Self-contained**: Understandable without additional context
2. **Verifiable**: Can be checked as true or false using external sources
3. **Atomic**: Contains exactly ONE fact (not compound statements)
4. **Specific**: Include names, dates, numbers when present

## Text to Analyze
{text}

## Instructions
- Extract 3-10 claims depending on text complexity
- Each claim should be a complete, standalone sentence
- Do NOT include opinions, predictions, or subjective statements
- Do NOT combine multiple facts into one claim

## Output Format (STRICT JSON)
{{
  "claims": [
    "First atomic factual claim",
    "Second atomic factual claim"
  ]
}}

IMPORTANT: Output ONLY valid JSON, no other text."""


def extract_atomic_claims(text: str, max_claims: int = 10) -> List[str]:
    """
    Extract verifiable atomic claims from text using LLM.
    
    Args:
        text: Text to extract claims from (e.g., LLM's answer)
        max_claims: Maximum number of claims to extract
    
    Returns:
        List of atomic claim strings
    """
    if not text or len(text.strip()) < 20:
        return []
    
    prompt = EXTRACT_CLAIMS_PROMPT.format(text=text[:3000])
    
    try:
        res = client.chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        response = res.choices[0].message.content or ""
        
        # Parse JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            claims = data.get("claims", [])
            if isinstance(claims, list):
                return [str(c).strip() for c in claims if c][:max_claims]
        
        # Fallback: extract numbered items
        lines = response.split('\n')
        claims = []
        for line in lines:
            line = line.strip()
            if re.match(r'^[\d\-\*\•]', line):
                c = re.sub(r'^[\d\.\-\*\•\)]+\s*', '', line).strip()
                if c and len(c) > 15:
                    claims.append(c)
        return claims[:max_claims]
        
    except Exception as e:
        print(f"        ⚠️ Claim extraction error: {e}")
        return []


# ==============================================================================
# GRAPH UTILITIES (for C1)
# ==============================================================================

def prune_node(graph: 'ArgumentationGraph', node_id: str) -> 'ArgumentationGraph':
    """
    Remove a verified-false node and its edges from the graph.
    
    Args:
        graph: The ArgumentationGraph to modify
        node_id: ID of the node to remove
    
    Returns:
        Modified graph (same object)
    """
    if not HAS_GRAPH or graph is None:
        return graph
    
    try:
        graph.remove_node(node_id)
        print(f"        🗑️ Pruned node: {node_id}")
    except Exception as e:
        print(f"        ⚠️ Prune error for {node_id}: {e}")
    
    return graph


def remove_true_true_attacks(
    graph: 'ArgumentationGraph', 
    tau: Dict[str, str]
) -> 'ArgumentationGraph':
    """
    Remove ATTACK edges where both endpoints are verified TRUE.
    
    This is a refinement rule: if two claims are both verified true,
    they cannot logically attack each other.
    
    Args:
        graph: The ArgumentationGraph to modify
        tau: Verification status dict {node_id: "TRUE"|"FALSE"|"UNK"}
    
    Returns:
        Modified graph (same object)
    """
    if not HAS_GRAPH or graph is None:
        return graph
    
    edges_to_remove = []
    
    try:
        for u, v, d in graph.nx_graph.edges(data=True):
            if d.get("type") == "attack":
                if tau.get(u) == "TRUE" and tau.get(v) == "TRUE":
                    edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            graph.nx_graph.remove_edge(u, v)
            print(f"        🔗 Removed TRUE-TRUE attack: {u} → {v}")
            
    except Exception as e:
        print(f"        ⚠️ Remove TRUE-TRUE attacks error: {e}")
    
    return graph


# ==============================================================================
# C1. Budgeted CRITIC Baseline
# ==============================================================================

class BudgetedCRITICBaseline:
    """
    C1. Budgeted CRITIC (Exhaustive-until-budget)
    
    Uses parser output graph to get atomic claims, then:
    1. Initialize working answer
    2. Loop: pick next claim → verify → prune if false → revise answer
    3. Apply light topology refinement (remove TRUE-TRUE attacks)
    
    If no graph provided, falls back to extracting claims from answer.
    """
    
    INIT_ANSWER_PROMPT = """You are a fact-checking expert. Based on the information below, provide an INITIAL assessment.

## Claim to Verify
"{claim}"

## Debate Transcript
{transcript}

## Instructions
1. Analyze the debate carefully
2. Identify the key points made by each side
3. Note which claims seem factual vs. rhetorical
4. Provide a preliminary verdict with reasoning

## Output Format
<reasoning>
Your step-by-step analysis of the debate, identifying key arguments from each side...
</reasoning>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>

<explanation>Brief explanation of your reasoning (2-3 sentences)</explanation>"""

    PICK_NEXT_CLAIM_PROMPT = """You are selecting the next claim to verify from a pool of unverified claims.

## Main Claim Being Evaluated
"{main_claim}"

## Current Working Assessment
{working_answer}

## Claims Pool (not yet verified)
{claims_pool}

## Already Verified
{verified_status}

## Instructions
Select the claim that is MOST CRITICAL to verify next. Consider:
1. Claims that DIRECTLY support or contradict the main verdict
2. Claims with SPECIFIC facts (names, dates, numbers) that can be easily checked
3. Claims that appear to be CENTRAL to the debate's resolution

## Output Format
Selected: [exact claim text to verify]
Reason: [one sentence explaining why this claim is most critical]"""

    REVISE_PROMPT = """You are revising your fact-checking assessment based on NEW verified information.

## Original Claim
"{claim}"

## Debate Transcript (excerpt)
{transcript}

## Your Previous Assessment
{previous_answer}

## Newly Verified Facts
{verified_facts}

## Instructions
1. Review EACH verified fact and its TRUE/FALSE status
2. If ANY verified fact CONTRADICTS your previous verdict, you MUST reconsider
3. If verified facts SUPPORT your previous verdict, strengthen your confidence
4. Be intellectually honest: change your verdict if evidence warrants it

## Important
- Verified FALSE claims should be discounted entirely
- Verified TRUE claims should be weighted heavily
- Update your reasoning chain based on verified facts

## Output Format
<reasoning>
How the new verified facts affect your assessment...
Specifically addressing: [list key verified facts and their impact]
</reasoning>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>

<explanation>Updated assessment incorporating all verified facts</explanation>"""

    def __init__(self, max_verifications: int = 15):
        """
        Args:
            max_verifications: Maximum claims to verify per session
        """
        self.max_verifications = max_verifications
    
    def _get_claims_from_graph(self, graph: 'ArgumentationGraph') -> List[Tuple[str, str]]:
        """Extract (node_id, claim_text) pairs from graph."""
        if not HAS_GRAPH or graph is None:
            return []
        
        claims = []
        for node_id, node in graph.nodes.items():
            if hasattr(node, 'content') and node.content:
                claims.append((node_id, node.content))
        return claims
    
    def _pick_next_claim(
        self, 
        main_claim: str,
        working_answer: str,
        pool: List[Tuple[str, str]], 
        verified: Dict[str, str]
    ) -> Optional[Tuple[str, str]]:
        """Use LLM to pick the next claim to verify."""
        if not pool:
            return None
        
        # Format pools for prompt
        pool_text = "\n".join([f"- [{nid}] {txt[:150]}" for nid, txt in pool[:15]])
        verified_text = "\n".join([
            f"- {nid}: {status}" for nid, status in verified.items()
        ]) if verified else "[None verified yet]"
        
        prompt = self.PICK_NEXT_CLAIM_PROMPT.format(
            main_claim=main_claim,
            working_answer=working_answer[:1500],
            claims_pool=pool_text,
            verified_status=verified_text[:1000]
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            response = res.choices[0].message.content or ""
            
            # Try to match selected claim to pool
            response_lower = response.lower()
            for nid, txt in pool:
                if nid.lower() in response_lower or txt[:50].lower() in response_lower:
                    return (nid, txt)
            
            # Fallback: return first claim
            return pool[0] if pool else None
            
        except Exception as e:
            print(f"        ⚠️ Pick claim error: {e}")
            return pool[0] if pool else None
    
    def verify(
        self, 
        claim: str, 
        transcript: Optional[str] = None, 
        parsed_graph: Optional['ArgumentationGraph'] = None,
        budget: float = 30.0
    ) -> Optional[bool]:
        """
        Verify a claim using Budgeted CRITIC approach.
        
        Args:
            claim: The main claim to verify
            transcript: Optional debate transcript
            parsed_graph: Optional ArgumentationGraph from parser
            budget: Total budget for verification
        
        Returns:
            True/False/None
        """
        budget_remaining = budget
        tau: Dict[str, str] = {}  # Verification status
        graph = parsed_graph
        
        # Step 1: Initialize working answer
        init_prompt = self.INIT_ANSWER_PROMPT.format(
            claim=claim,
            transcript=(transcript or "")[:4000] if transcript else "[No transcript]"
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": init_prompt}],
                temperature=0.3
            )
            working_answer = res.choices[0].message.content or ""
        except Exception as e:
            print(f"        ⚠️ Init answer error: {e}")
            working_answer = f"Evaluating claim: {claim}"
        
        # Step 2: Get claims pool
        if HAS_GRAPH and graph is not None:
            pool = self._get_claims_from_graph(graph)
            tau = {nid: "UNK" for nid, _ in pool}
        else:
            # Fallback: extract claims from working answer + transcript
            combined = f"{working_answer}\n\n{transcript or ''}"
            extracted = extract_atomic_claims(combined, max_claims=12)
            pool = [(f"c{i}", c) for i, c in enumerate(extracted)]
            tau = {nid: "UNK" for nid, _ in pool}
        
        if not pool:
            # No claims to verify, return parsed verdict from initial answer
            return parse_binary_label(working_answer)
        
        # Step 3: Verification loop
        verification_count = 0
        
        while budget_remaining > 0 and pool and verification_count < self.max_verifications:
            # Pick next claim
            selected = self._pick_next_claim(claim, working_answer, pool, tau)
            if not selected:
                break
            
            node_id, claim_text = selected
            
            # Verify claim
            result = VerifyClaim(claim_text, budget_remaining)
            
            if result.spent == 0:
                # Couldn't verify (budget or error)
                pool = [(nid, txt) for nid, txt in pool if nid != node_id]
                continue
            
            budget_remaining -= result.spent
            verification_count += 1
            
            # Update verification status
            if result.verdict is True:
                tau[node_id] = "TRUE"
                print(f"        ✅ Verified TRUE: {claim_text[:60]}...")
            elif result.verdict is False:
                tau[node_id] = "FALSE"
                print(f"        ❌ Verified FALSE: {claim_text[:60]}...")
                # Prune false node from graph
                if HAS_GRAPH and graph is not None:
                    graph = prune_node(graph, node_id)
            else:
                tau[node_id] = "UNK"
                print(f"        ❓ ABSTAIN: {claim_text[:60]}...")
            
            # Light refinement: remove TRUE-TRUE attacks
            if HAS_GRAPH and graph is not None:
                graph = remove_true_true_attacks(graph, tau)
            
            # Remove verified claim from pool
            pool = [(nid, txt) for nid, txt in pool if nid != node_id]
            
            # Revise working answer
            verified_facts = "\n".join([
                f"- [{status}] {nid}: {next((txt for n, txt in self._get_claims_from_graph(graph) if n == nid) if HAS_GRAPH and graph else '', 'claim')[:100]}" 
                if HAS_GRAPH and graph else f"- [{status}] {nid}"
                for nid, status in tau.items() if status != "UNK"
            ])
            
            if verified_facts:
                revise_prompt = self.REVISE_PROMPT.format(
                    claim=claim,
                    transcript=(transcript or "")[:2000] if transcript else "[No transcript]",
                    previous_answer=working_answer[:2000],
                    verified_facts=verified_facts
                )
                
                try:
                    res = client.chat.completions.create(
                        model=GENERATOR_MODEL,
                        messages=[{"role": "user", "content": revise_prompt}],
                        temperature=0.2
                    )
                    working_answer = res.choices[0].message.content or working_answer
                except Exception as e:
                    print(f"        ⚠️ Revise error: {e}")
        
        # Step 4: Return final verdict
        return parse_binary_label(working_answer)


# ==============================================================================
# C2. Iterative Verify-and-Revise Baseline
# ==============================================================================

class VerifyAndReviseBaseline:
    """
    C2. Iterative Verify-and-Revise
    
    Multi-round approach:
    1. Produce initial answer
    2. Extract atomic claims from answer
    3. Verify claims until budget exhausted
    4. Revise answer based on verified findings
    5. Repeat for max_rounds
    """
    
    INIT_ANSWER_PROMPT = """You are a fact-checking expert. Based on the information below, provide your assessment.

## Claim to Verify
"{claim}"

## Context (Debate Transcript)
{transcript}

## Instructions
1. Analyze all available information
2. Identify supporting and contradicting evidence
3. Provide a clear verdict with detailed reasoning

## Output Format
<reasoning>
Your step-by-step analysis...
Include specific claims that support or contradict the main claim.
</reasoning>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>

<key_claims>
List the key factual claims in your reasoning:
1. [Specific claim 1]
2. [Specific claim 2]
...
</key_claims>"""

    REVISE_WITH_VERIFIED_PROMPT = """You are revising your fact-checking answer based on VERIFICATION RESULTS.

## Original Claim
"{claim}"

## Your Current Answer
{current_answer}

## Verification Results
{verified_results}

## Instructions
1. Review EACH verification result carefully
2. ✅ Verified TRUE: These claims are confirmed - incorporate them confidently
3. ❌ Verified FALSE: These claims are WRONG - remove or correct them in your reasoning
4. ❓ Abstain: Uncertain - don't rely heavily on these

## Critical Rules
- If a claim you relied on was verified FALSE, you MUST update your verdict
- If key supporting claims were verified TRUE, strengthen your confidence
- Be explicit about how verification results changed your reasoning

## Output Format
<verified_impact>
How does each verification result affect my assessment?
- [Claim X] was [TRUE/FALSE] → [impact on my reasoning]
...
</verified_impact>

<revised_reasoning>
My updated analysis incorporating verified facts...
</revised_reasoning>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>

<confidence>HIGH, MEDIUM, or LOW</confidence>"""

    def __init__(self, max_rounds: int = 4):
        """
        Args:
            max_rounds: Maximum revision rounds
        """
        self.max_rounds = max_rounds
    
    def verify(
        self, 
        claim: str, 
        transcript: Optional[str] = None, 
        budget: float = 30.0
    ) -> Optional[bool]:
        """
        Verify a claim using iterative verify-and-revise approach.
        
        Args:
            claim: The claim to verify
            transcript: Optional context/debate transcript
            budget: Total budget for verification
        
        Returns:
            True/False/None
        """
        budget_remaining = budget
        
        # Step 1: Generate initial answer
        init_prompt = self.INIT_ANSWER_PROMPT.format(
            claim=claim,
            transcript=(transcript or "")[:4000] if transcript else "[No context]"
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": init_prompt}],
                temperature=0.3
            )
            current_answer = res.choices[0].message.content or ""
        except Exception as e:
            print(f"        ⚠️ Init answer error: {e}")
            return None
        
        # Step 2: Iterative verify-and-revise
        for round_num in range(self.max_rounds):
            if budget_remaining <= 0:
                break
            
            print(f"        📝 Verify-Revise Round {round_num + 1}/{self.max_rounds}")
            
            # Extract claims from current answer
            claims = extract_atomic_claims(current_answer, max_claims=8)
            
            if not claims:
                break
            
            # Verify claims
            verified_results: List[Tuple[str, str, float]] = []
            
            for c in claims:
                if budget_remaining <= 0:
                    break
                
                result = VerifyClaim(c, budget_remaining)
                
                if result.spent == 0:
                    continue
                
                budget_remaining -= result.spent
                
                status = "TRUE" if result.verdict is True else (
                    "FALSE" if result.verdict is False else "ABSTAIN"
                )
                verified_results.append((c, status, result.confidence))
                
                icon = "✅" if result.verdict is True else ("❌" if result.verdict is False else "❓")
                print(f"          {icon} {status}: {c[:60]}... (conf: {result.confidence:.2f})")
            
            if not verified_results:
                break
            
            # Format verified results
            verified_text = "\n".join([
                f"- {'✅' if st == 'TRUE' else ('❌' if st == 'FALSE' else '❓')} [{st}] (conf: {conf:.0%}) {claim_txt[:150]}"
                for claim_txt, st, conf in verified_results
            ])
            
            # Revise answer
            revise_prompt = self.REVISE_WITH_VERIFIED_PROMPT.format(
                claim=claim,
                current_answer=current_answer[:3000],
                verified_results=verified_text
            )
            
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": revise_prompt}],
                    temperature=0.2
                )
                current_answer = res.choices[0].message.content or current_answer
            except Exception as e:
                print(f"        ⚠️ Revise error: {e}")
                break
        
        # Step 3: Return final verdict
        return parse_binary_label(current_answer)


# ==============================================================================
# C3. RARR-style Baseline
# ==============================================================================

class RARRBaseline:
    """
    C3. RARR-style Retrieval-Augmented Revision
    
    Each round:
    1. Retrieve evidence for current answer
    2. Revise answer to be more grounded in evidence
    
    Unlike C2, this focuses on grounding rather than explicit verification.
    """
    
    INIT_ANSWER_PROMPT = """You are a fact-checking expert. Provide an initial assessment of the claim.

## Claim to Verify
"{claim}"

## Context
{transcript}

## Instructions
Give your initial assessment. Be specific about the facts you're relying on.

## Output
<initial_assessment>
Your analysis of the claim...
</initial_assessment>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>"""

    RARR_REVISE_PROMPT = """You are improving a fact-checking answer by grounding it in RETRIEVED EVIDENCE.

## Original Claim to Verify
"{claim}"

## Your Current Answer
{current_answer}

## Retrieved Evidence
{evidence}

## Instructions
1. COMPARE your answer with the retrieved evidence
2. IDENTIFY which parts of your answer are:
   - ✅ SUPPORTED by evidence (cite [1], [2], etc.)
   - ❌ CONTRADICTED by evidence
   - ❓ NOT ADDRESSED by evidence
3. REVISE your answer to be more accurate based on evidence
4. If evidence clearly contradicts your verdict, CHANGE it

## Evidence Quality Notes
- Wikipedia, .gov, .edu sources are highly reliable
- Cross-reference multiple sources when possible
- Be skeptical of single-source claims

## Output Format
<evidence_analysis>
Evidence [1]: [supports/contradicts/irrelevant] because...
Evidence [2]: [supports/contradicts/irrelevant] because...
...
</evidence_analysis>

<grounded_reasoning>
My revised reasoning, citing specific evidence...
- According to [1], ...
- Evidence [3] shows that...
</grounded_reasoning>

<verdict>TRUE or FALSE or UNCERTAIN</verdict>

<grounding_confidence>
How well-grounded is this verdict?
- X facts confirmed by evidence
- Y facts still unverified
</grounding_confidence>"""

    def __init__(self, max_rounds: int = 4):
        """
        Args:
            max_rounds: Maximum revision rounds
        """
        self.max_rounds = max_rounds
    
    def verify(
        self, 
        claim: str, 
        transcript: Optional[str] = None, 
        budget: float = 30.0
    ) -> Optional[bool]:
        """
        Verify a claim using RARR-style retrieval-augmented revision.
        
        Args:
            claim: The claim to verify
            transcript: Optional context
            budget: Total budget for retrieval
        
        Returns:
            True/False/None
        """
        budget_remaining = budget
        
        # Step 1: Generate initial answer
        init_prompt = self.INIT_ANSWER_PROMPT.format(
            claim=claim,
            transcript=(transcript or "")[:3000] if transcript else "[No context]"
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": init_prompt}],
                temperature=0.3
            )
            current_answer = res.choices[0].message.content or ""
        except Exception as e:
            print(f"        ⚠️ Init answer error: {e}")
            return None
        
        # Step 2: Iterative retrieve-and-revise
        for round_num in range(self.max_rounds):
            if budget_remaining <= 0:
                break
            
            print(f"        🔄 RARR Round {round_num + 1}/{self.max_rounds}")
            
            # Retrieve evidence for current answer
            # Use both the claim and key parts of current answer as queries
            query = f"{claim} {current_answer[:200]}"
            snippets, spent = retrieve_evidence(query, transcript, budget_remaining)
            budget_remaining -= spent
            
            if not snippets:
                print(f"          ⚠️ No evidence retrieved")
                continue
            
            print(f"          📚 Retrieved {len(snippets)} evidence snippets")
            
            # Format evidence
            evidence_text = format_snippets_for_prompt(snippets, max_chars=4000)
            
            # Revise answer with evidence
            revise_prompt = self.RARR_REVISE_PROMPT.format(
                claim=claim,
                current_answer=current_answer[:2500],
                evidence=evidence_text
            )
            
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": revise_prompt}],
                    temperature=0.2
                )
                new_answer = res.choices[0].message.content or ""
                
                # Check if answer meaningfully changed
                if len(new_answer) > 100:
                    current_answer = new_answer
                    
            except Exception as e:
                print(f"        ⚠️ Revise error: {e}")
                break
        
        # Step 3: Return final verdict
        return parse_binary_label(current_answer)


# ==============================================================================
# CONVENIENCE EXPORTS
# ==============================================================================

__all__ = [
    # Baselines
    "BudgetedCRITICBaseline",
    "VerifyAndReviseBaseline",
    "RARRBaseline",
    # Utilities
    "VerifyClaim",
    "VerifyResult",
    "extract_atomic_claims",
    "prune_node",
    "remove_true_true_attacks",
]
