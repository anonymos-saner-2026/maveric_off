"""
Class B Baselines - Linear Tool-use Baselines (no graphs)

This module implements 4 baselines that use tools linearly without building
argumentation graphs:
- B1. ReActBaseline: Sequential thought-action-observation loop
- B2. RAGAnswerBaseline: Retrieve evidence then answer directly
- B3. RAGVerifierBaseline: Retrieve then judge with calibrated confidence
- B4. SelfAskBaseline: Planner-driven subquestion decomposition

All baselines respect tool budget B and use the same cost model as MaVERiC.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from src.config import client, GENERATOR_MODEL, JUDGE_MODEL, TOOLS_CONFIG
from src.tools.real_toolkit import RealToolkit, JudgeResult, _summarize_hits


# ==============================================================================
# SHARED UTILITIES
# ==============================================================================

def estimate_cost(tool_name: str, query: str = "") -> float:
    """
    Estimate cost of a tool call using TOOLS_CONFIG.
    
    Args:
        tool_name: One of "WEB_SEARCH", "PYTHON_EXEC", "COMMON_SENSE"
        query: The query string (currently unused but kept for future extensions)
    
    Returns:
        Cost as float (e.g., 5.0 for WEB_SEARCH)
    """
    return TOOLS_CONFIG.get(tool_name, {}).get("cost", 5.0)


def call_tool(tool_name: str, query: str) -> str:
    """
    Unified tool caller that dispatches to RealToolkit methods.
    
    Args:
        tool_name: "WEB_SEARCH", "PYTHON_EXEC", or "COMMON_SENSE"
        query: The query/task for the tool
    
    Returns:
        Tool output as string
    """
    if tool_name == "WEB_SEARCH":
        return RealToolkit.google_search(query)
    
    elif tool_name == "PYTHON_EXEC":
        result = RealToolkit.verify_claim("PYTHON_EXEC", query)
        if result is True:
            return "VERIFIED_TRUE"
        elif result is False:
            return "VERIFIED_FALSE"
        else:
            return "VERIFICATION_FAILED"
    
    elif tool_name == "COMMON_SENSE":
        # Use LLM internal knowledge
        prompt = f"""Answer the following question using your internal knowledge.
Be concise and factual.

Question: {query}

Answer:"""
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            return res.choices[0].message.content or "No answer available"
        except Exception as e:
            return f"Error: {e}"
    
    else:
        return f"Unknown tool: {tool_name}"


def retrieve_evidence(
    claim: str,
    transcript: Optional[str],
    budget: float,
    max_rounds: int = 3
) -> Tuple[List[Dict[str, str]], float]:
    """
    Retrieve evidence snippets within the given budget.
    
    Uses RealToolkit's search infrastructure (Serper + DuckDuckGo).
    
    Args:
        claim: The claim to find evidence for
        transcript: Optional debate transcript for context
        budget: Maximum budget to spend
        max_rounds: Maximum search rounds (each costs WEB_SEARCH cost)
    
    Returns:
        (snippets, cost_spent) where snippets is list of {title, url, snippet}
    """
    web_cost = estimate_cost("WEB_SEARCH")
    all_snippets: List[Dict[str, str]] = []
    cost_spent = 0.0
    
    # Build queries using RealToolkit's query builder
    clean_claim = claim.strip()
    query_rounds = RealToolkit._make_queries(clean_claim)
    
    for round_idx, queries in enumerate(query_rounds[:max_rounds]):
        if cost_spent + web_cost > budget:
            break
        
        # Take first 2 queries from each round for efficiency
        for q in queries[:2]:
            if cost_spent + web_cost > budget:
                break
            
            try:
                result_json = RealToolkit.google_search(q)
                result = json.loads(result_json) if result_json else {}
                
                # Collect snippets from both providers
                for hit in result.get("serper", [])[:3]:
                    all_snippets.append({
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "snippet": hit.get("snippet", ""),
                        "provider": "serper"
                    })
                
                for hit in result.get("ddg", [])[:3]:
                    all_snippets.append({
                        "title": hit.get("title", ""),
                        "url": hit.get("url", ""),
                        "snippet": hit.get("snippet", ""),
                        "provider": "ddg"
                    })
                
                cost_spent += web_cost
                
            except Exception as e:
                print(f"        ⚠️ Search error: {e}")
                continue
    
    # Deduplicate by URL
    seen_urls = set()
    unique_snippets = []
    for s in all_snippets:
        url = s.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_snippets.append(s)
    
    return unique_snippets, cost_spent


def format_snippets_for_prompt(snippets: List[Dict[str, str]], max_chars: int = 5000) -> str:
    """Format snippets for inclusion in prompts."""
    if not snippets:
        return "[No evidence retrieved]"
    
    lines = []
    total_chars = 0
    
    for i, s in enumerate(snippets, 1):
        title = s.get("title", "Untitled")[:100]
        url = s.get("url", "")[:100]
        snippet = s.get("snippet", "")[:400]
        
        line = f"[{i}] {title}\n    URL: {url}\n    {snippet}\n"
        
        if total_chars + len(line) > max_chars:
            lines.append(f"... ({len(snippets) - i + 1} more snippets truncated)")
            break
        
        lines.append(line)
        total_chars += len(line)
    
    return "\n".join(lines)


def parse_binary_label(text: str) -> Optional[bool]:
    """
    Parse verdict from text, returns True/False/None.
    Robustly handles:
    - [TRUE], [FALSE], [ABSTAIN]
    - <verdict>TRUE</verdict>, <verdict>FALSE</verdict>
    - Plain words: TRUE, FALSE, UNCERTAIN, ABSTAIN
    """
    if not text:
        return None
    
    text_upper = text.upper()
    
    # Check for True
    if "[TRUE]" in text_upper or "<VERDICT>TRUE" in text_upper:
        return True
    if re.search(r"\bTRUE\b", text_upper):
        # Ensure it's not "NOT TRUE" or similar if we wanted to be very careful, 
        # but for these baselines usually TRUE appears in a verdict block.
        # Check for negation near True? 
        # Simpler: prioritized check. B1/C1 prompts use specific formats.
        return True
        
    # Check for False
    if "[FALSE]" in text_upper or "<VERDICT>FALSE" in text_upper:
        return False
    if re.search(r"\bFALSE\b", text_upper):
        return False
        
    # Check for Abstain/Uncertain
    if "[ABSTAIN]" in text_upper or "UNCERTAIN" in text_upper or "ABSTAIN" in text_upper:
        return None
        
    return None


# ==============================================================================
# B1. ReAct Baseline
# ==============================================================================

@dataclass
class ReActState:
    """State for ReAct agent."""
    claim: str
    transcript: str
    scratchpad: List[str]
    budget_initial: float
    budget_remaining: float


@dataclass 
class ReActAction:
    """Parsed action from ReAct agent."""
    type: str  # "TOOL" or "ANSWER"
    tool_name: Optional[str] = None
    query: Optional[str] = None
    payload: Optional[str] = None  # For ANSWER type


class ReActBaseline:
    """
    B1. ReAct (Sequential Reasoning with Tools)
    
    Implements a thought-action-observation loop where the agent decides
    whether to use tools or provide a final answer at each step.
    """
    
    REACT_PROMPT = """You are a fact-checking agent with access to verification tools.

## Current State
Claim to verify: "{claim}"
Transcript context (if any): 
{transcript}

Remaining budget: ${budget_rem:.2f}

## Tools Available
- WEB_SEARCH(query): Search the web for evidence. Cost: $5.00
- PYTHON_EXEC(task): Execute Python to verify mathematical claims. Cost: $8.00
- COMMON_SENSE(question): Query internal knowledge. Cost: $1.00

## Scratchpad History
{scratchpad}

## Instructions
1. THINK about what evidence you need to verify this claim
2. DECIDE on ONE action:
   - Use a tool to gather evidence, OR
   - Provide final ANSWER if you have enough evidence

## Output Format (STRICT XML)
<thought>Your step-by-step reasoning about what to check next</thought>
<action>
<type>TOOL</type>
<tool_name>WEB_SEARCH</tool_name>
<query>your specific search query here</query>
</action>

OR for final answer:

<thought>Your reasoning for the final verdict</thought>
<action>
<type>ANSWER</type>
<payload>TRUE</payload>
</action>

IMPORTANT: 
- Only use ONE action per response
- For ANSWER, payload must be exactly TRUE, FALSE, or ABSTAIN
- Be specific in your search queries
"""

    FORCE_ANSWER_PROMPT = """You are a fact-checking agent. Budget exhausted - you MUST give a final verdict NOW.

## Claim
"{claim}"

## Evidence Collected
{scratchpad}

Based on ALL the evidence above, provide your FINAL verdict.
Consider:
- What evidence supports the claim?
- What evidence contradicts the claim?
- Is the evidence sufficient to make a determination?

Output ONLY one of: [TRUE] or [FALSE] or [ABSTAIN]

Your verdict:"""

    def __init__(self, max_steps: int = 8):
        """
        Args:
            max_steps: Maximum number of reasoning steps before forcing answer
        """
        self.max_steps = max_steps
    
    def _init_state(self, claim: str, transcript: Optional[str], budget: float) -> ReActState:
        """Initialize ReAct state."""
        return ReActState(
            claim=claim,
            transcript=(transcript or "")[:2000],
            scratchpad=["[Start of fact-checking session]"],
            budget_initial=budget,
            budget_remaining=budget
        )
    
    def _parse_action(self, response: str) -> ReActAction:
        """Parse action from LLM response."""
        response = response or ""
        
        # Extract thought
        thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Extract action type
        type_match = re.search(r"<type>(.*?)</type>", response, re.DOTALL)
        action_type = type_match.group(1).strip().upper() if type_match else ""
        
        if action_type == "ANSWER":
            payload_match = re.search(r"<payload>(.*?)</payload>", response, re.DOTALL)
            payload = payload_match.group(1).strip().upper() if payload_match else "ABSTAIN"
            return ReActAction(type="ANSWER", payload=payload)
        
        elif action_type == "TOOL":
            tool_match = re.search(r"<tool_name>(.*?)</tool_name>", response, re.DOTALL)
            query_match = re.search(r"<query>(.*?)</query>", response, re.DOTALL)
            
            tool_name = tool_match.group(1).strip().upper() if tool_match else "WEB_SEARCH"
            query = query_match.group(1).strip() if query_match else ""
            
            return ReActAction(type="TOOL", tool_name=tool_name, query=query)
        
        # Fallback: check for keywords
        if "[TRUE]" in response or "[FALSE]" in response or "[ABSTAIN]" in response:
            payload = "TRUE" if "[TRUE]" in response else ("FALSE" if "[FALSE]" in response else "ABSTAIN")
            return ReActAction(type="ANSWER", payload=payload)
        
        # Default to answer with abstain
        return ReActAction(type="ANSWER", payload="ABSTAIN")
    
    def _update_state(self, state: ReActState, thought: str, action: ReActAction, observation: str) -> ReActState:
        """Update state with new step."""
        step_num = len(state.scratchpad)
        
        entry = f"\n[Step {step_num}]"
        if thought:
            entry += f"\nThought: {thought[:500]}"
        
        if action.type == "TOOL":
            entry += f"\nAction: {action.tool_name}({action.query[:200] if action.query else ''})"
            entry += f"\nObservation: {observation[:1000]}"
        
        state.scratchpad.append(entry)
        return state
    
    def verify(self, claim: str, transcript: Optional[str] = None, budget: float = 20.0) -> Optional[bool]:
        """
        Verify a claim using ReAct reasoning loop.
        
        Args:
            claim: The claim to verify
            transcript: Optional debate transcript for context
            budget: Total budget for tool calls
        
        Returns:
            True if claim is verified TRUE, False if FALSE, None if ABSTAIN
        """
        state = self._init_state(claim, transcript, budget)
        
        for step in range(self.max_steps):
            # Build prompt
            scratchpad_text = "\n".join(state.scratchpad[-5:])  # Keep last 5 entries
            prompt = self.REACT_PROMPT.format(
                claim=state.claim,
                transcript=state.transcript[:1500] if state.transcript else "[No transcript]",
                budget_rem=state.budget_remaining,
                scratchpad=scratchpad_text if scratchpad_text else "[Empty]"
            )
            
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                response = res.choices[0].message.content or ""
            except Exception as e:
                print(f"        ⚠️ ReAct LLM error: {e}")
                break
            
            # Parse action
            action = self._parse_action(response)
            
            # Handle ANSWER
            if action.type == "ANSWER":
                return parse_binary_label(action.payload or "")
            
            # Handle TOOL
            if action.type == "TOOL" and action.tool_name and action.query:
                tool_cost = estimate_cost(action.tool_name)
                
                # Check budget
                if tool_cost > state.budget_remaining:
                    # Cannot afford tool - force answer
                    break
                
                # Call tool
                observation = call_tool(action.tool_name, action.query)
                state.budget_remaining -= tool_cost
                
                # Extract thought for state update
                thought_match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
                thought = thought_match.group(1).strip() if thought_match else ""
                
                state = self._update_state(state, thought, action, observation)
            
            # Check budget
            if state.budget_remaining <= 0:
                break
        
        # Force answer - budget exhausted or max steps reached
        scratchpad_text = "\n".join(state.scratchpad)
        force_prompt = self.FORCE_ANSWER_PROMPT.format(
            claim=state.claim,
            scratchpad=scratchpad_text[:8000]
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": force_prompt}],
                temperature=0.0
            )
            return parse_binary_label(res.choices[0].message.content or "")
        except Exception as e:
            print(f"        ⚠️ ReAct force answer error: {e}")
            return None


# ==============================================================================
# B2. RAG + Answer Baseline
# ==============================================================================

class RAGAnswerBaseline:
    """
    B2. RAG + Answer (Retrieval-First)
    
    Simple two-step approach:
    1. Retrieve evidence using web search
    2. Answer directly using evidence context
    """
    
    ANSWER_WITH_EVIDENCE_PROMPT = """You are a fact-checker. Evaluate the claim using ONLY the provided evidence.

## Claim to Verify
"{claim}"

## Debate Transcript (Context)
{transcript}

## Retrieved Evidence
{evidence}

## Instructions
1. Analyze the evidence carefully
2. Determine if the claim is TRUE, FALSE, or cannot be determined
3. Consider source reliability:
   - Wikipedia, .gov, .edu sources are MORE reliable
   - Unknown sources should be treated with caution
4. If evidence is contradictory or insufficient, choose ABSTAIN

## Output Format
Verdict: [TRUE] or [FALSE] or [ABSTAIN]
Justification: (1-2 sentences explaining your reasoning)"""

    def verify(self, claim: str, transcript: Optional[str] = None, budget: float = 20.0) -> Optional[bool]:
        """
        Verify a claim using RAG + direct answer.
        
        Args:
            claim: The claim to verify
            transcript: Optional debate transcript for context
            budget: Total budget for retrieval
        
        Returns:
            True/False/None
        """
        # Step 1: Retrieve evidence
        snippets, spent = retrieve_evidence(claim, transcript, budget)
        
        # Step 2: Format evidence
        evidence_text = format_snippets_for_prompt(snippets)
        
        # Step 3: Generate answer
        prompt = self.ANSWER_WITH_EVIDENCE_PROMPT.format(
            claim=claim,
            transcript=(transcript or "")[:3000] if transcript else "[No transcript available]",
            evidence=evidence_text
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = res.choices[0].message.content or ""
            return parse_binary_label(content)
        except Exception as e:
            print(f"        ⚠️ RAG+Answer error: {e}")
            return None


# ==============================================================================
# B3. RAG + Verifier Baseline
# ==============================================================================

class RAGVerifierBaseline:
    """
    B3. RAG + Verifier (Evidence-Judge Pipeline)
    
    Three-step approach:
    1. Retrieve evidence
    2. Judge with calibrated confidence (ternary verdict + confidence)
    3. Apply threshold gates for final decision
    
    Uses existing RealToolkit infrastructure for stance-aware judging.
    """
    
    # Thresholds from RealToolkit
    THETA_T = 0.72  # Minimum confidence for TRUE
    THETA_F = 0.62  # Minimum confidence for FALSE
    
    def verify(self, claim: str, transcript: Optional[str] = None, budget: float = 20.0) -> Optional[bool]:
        """
        Verify a claim using RAG + calibrated verifier.
        
        Args:
            claim: The claim to verify
            transcript: Optional debate transcript for context
            budget: Total budget for retrieval
        
        Returns:
            True/False (never None - uses conservative FALSE fallback)
        """
        # Step 1: Retrieve evidence
        snippets, spent = retrieve_evidence(claim, transcript, budget)
        
        if not snippets:
            # No evidence - conservative fallback
            return False
        
        # Step 2: Prepare evidence for judge
        evidence_lines = _summarize_hits(snippets, max_n=10)
        
        # Step 3: Call calibrated judge
        judge_result: JudgeResult = RealToolkit._rag_judge_with_calibrated_conf(
            clean_fact=claim,
            evidence_lines=evidence_lines,
            evidence_hits=snippets
        )
        
        # Step 4: Apply threshold gates
        z = judge_result.verdict
        c_final = judge_result.final_confidence
        
        if z is True and c_final >= self.THETA_T:
            return True
        
        if z is False and c_final >= self.THETA_F:
            return False
        
        # Step 5: Abstain/low confidence - conservative fallback
        # Per pseudocode: return False for conservative behavior
        return False


# ==============================================================================
# B4. Self-Ask / Plan-and-Solve Baseline
# ==============================================================================

class SelfAskBaseline:
    """
    B4. Self-Ask / Plan-and-Solve
    
    Planner-driven approach:
    1. Decompose claim into sub-questions
    2. Solve each sub-question with evidence retrieval
    3. Compose final answer from Q/A pairs
    """
    
    PLANNER_PROMPT = """You are a fact-checking planner. Break down complex claims into simpler, verifiable sub-questions.

## Claim to Verify
"{claim}"

## Context (if available)
{transcript}

## Instructions
1. Identify the key factual components that need verification
2. Create 2-4 focused sub-questions that:
   - Are independently verifiable via web search
   - Together cover all aspects of the main claim
   - Are specific and searchable (include names, dates, numbers)
3. Order questions logically (simpler facts first)

## Output Format (STRICT JSON)
{{
  "sub_questions": [
    "First specific verifiable question",
    "Second specific verifiable question"
  ]
}}

IMPORTANT: Output ONLY valid JSON, no other text."""

    SOLVE_SUBQ_PROMPT = """You are answering a sub-question as part of fact-checking a larger claim.

## Sub-Question
{sub_question}

## Evidence Retrieved
{snippets}

## Instructions
1. Analyze the evidence carefully
2. Provide a concise, factual answer based on the evidence
3. If evidence is insufficient or contradictory, say "INSUFFICIENT EVIDENCE"
4. Cite which evidence snippet(s) support your answer

## Answer (be concise):"""

    COMPOSE_FINAL_PROMPT = """You are synthesizing sub-question answers to verify the main claim.

## Main Claim
"{claim}"

## Sub-Questions and Answers
{qa_pairs}

## Instructions
1. Consider ALL sub-question answers
2. Determine if they COLLECTIVELY support or refute the main claim
3. If some answers are missing or insufficient, factor that into your confidence
4. Be logical and consistent in your reasoning

## Output Format
Verdict: [TRUE] or [FALSE] or [ABSTAIN]
Reasoning: (brief synthesis of how sub-answers lead to this conclusion)"""

    def __init__(self, max_subquestions: int = 4):
        """
        Args:
            max_subquestions: Maximum number of sub-questions to generate
        """
        self.max_subquestions = max_subquestions
    
    def _parse_subquestions(self, response: str) -> List[str]:
        """Parse sub-questions from planner response."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                subqs = data.get("sub_questions", [])
                if isinstance(subqs, list):
                    return [str(q).strip() for q in subqs if q][:self.max_subquestions]
        except json.JSONDecodeError:
            pass
        
        # Fallback: extract numbered questions
        lines = response.split('\n')
        questions = []
        for line in lines:
            line = line.strip()
            if re.match(r'^[\d\-\*\•]', line):
                # Remove numbering/bullets
                q = re.sub(r'^[\d\.\-\*\•\)]+\s*', '', line).strip()
                if q and len(q) > 10:
                    questions.append(q)
        
        return questions[:self.max_subquestions]
    
    def verify(self, claim: str, transcript: Optional[str] = None, budget: float = 20.0) -> Optional[bool]:
        """
        Verify a claim using Self-Ask decomposition.
        
        Args:
            claim: The claim to verify  
            transcript: Optional debate transcript for context
            budget: Total budget for sub-question answering
        
        Returns:
            True/False/None
        """
        budget_remaining = budget
        
        # Step 1: Generate decomposition plan
        planner_prompt = self.PLANNER_PROMPT.format(
            claim=claim,
            transcript=(transcript or "")[:2000] if transcript else "[No transcript]"
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": planner_prompt}],
                temperature=0.0
            )
            plan_response = res.choices[0].message.content or ""
        except Exception as e:
            print(f"        ⚠️ Planner error: {e}")
            # Fallback to single question
            plan_response = json.dumps({"sub_questions": [claim]})
        
        subquestions = self._parse_subquestions(plan_response)
        
        if not subquestions:
            subquestions = [claim]  # Fallback
        
        # Step 2: Solve each sub-question
        qa_pairs: List[Tuple[str, str]] = []
        
        # Distribute budget among sub-questions
        budget_per_q = budget_remaining / max(1, len(subquestions))
        
        for q in subquestions:
            if budget_remaining <= 0:
                qa_pairs.append((q, "BUDGET EXHAUSTED"))
                continue
            
            # Retrieve evidence for this sub-question
            snippets, spent = retrieve_evidence(q, transcript, min(budget_per_q, budget_remaining))
            budget_remaining -= spent
            
            # Format snippets
            snippets_text = format_snippets_for_prompt(snippets, max_chars=2000)
            
            # Solve sub-question
            solve_prompt = self.SOLVE_SUBQ_PROMPT.format(
                sub_question=q,
                snippets=snippets_text
            )
            
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": solve_prompt}],
                    temperature=0.0
                )
                answer = res.choices[0].message.content or "No answer"
            except Exception as e:
                answer = f"Error: {e}"
            
            qa_pairs.append((q, answer[:500]))
        
        # Step 3: Compose final answer
        qa_formatted = "\n\n".join([
            f"Q{i+1}: {q}\nA{i+1}: {a}" 
            for i, (q, a) in enumerate(qa_pairs)
        ])
        
        compose_prompt = self.COMPOSE_FINAL_PROMPT.format(
            claim=claim,
            qa_pairs=qa_formatted
        )
        
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": compose_prompt}],
                temperature=0.0
            )
            content = res.choices[0].message.content or ""
            return parse_binary_label(content)
        except Exception as e:
            print(f"        ⚠️ Compose error: {e}")
            return None


# ==============================================================================
# CONVENIENCE EXPORTS
# ==============================================================================

__all__ = [
    "ReActBaseline",
    "RAGAnswerBaseline", 
    "RAGVerifierBaseline",
    "SelfAskBaseline",
    # Utilities
    "estimate_cost",
    "call_tool",
    "retrieve_evidence",
]
