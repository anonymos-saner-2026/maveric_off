from __future__ import annotations
import json
import re
from typing import List, Dict, Any, Optional
from src.config import client, GENERATOR_MODEL, JUDGE_MODEL

class Baseline:
    """Base class for all evaluation baselines."""
    def verify(self, claim: str) -> Optional[bool]:
        raise NotImplementedError

class CoTBaseline(Baseline):
    """
    A1. Single-agent Chain-of-Thought.
    Purely internal reasoning with a single forward pass.
    """
    def verify(self, claim: str) -> Optional[bool]:
        prompt = f"""
        Role: Fact-Checking Expert (Zero-Tool Mode).
        Task: Verify the claim below using your internal knowledge and logic.
        
        Step 1: Think step-by-step through the logical evidence for and against this claim.
        Step 2: Provide a final verdict.
        
        Claim: "{claim}"
        
        Output format:
        <thought>
        ... step-by-step reasoning ...
        </thought>
        Verdict: [TRUE] or [FALSE] or [ABSTAIN]
        """
        try:
            res = client.chat.completions.create(
                model=GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = res.choices[0].message.content
            if "[TRUE]" in content: return True
            if "[FALSE]" in content: return False
            return None
        except Exception as e:
            print(f"      ⚠️ CoT Baseline Error: {e}")
            return None

class SelfConsistencyBaseline(Baseline):
    """
    A2. Self-consistency.
    Majority vote over K independent CoT samples.
    """
    def __init__(self, k: int = 5):
        self.k = k
        self.cot = CoTBaseline()

    def verify(self, claim: str) -> Optional[bool]:
        votes = []
        for i in range(self.k):
            # Use temp > 0 for diverse sampling
            prompt = f"""
            Role: Fact-Checking Expert. Verify step-by-step.
            Claim: "{claim}"
            Output: <thought>...</thought> Verdict: [TRUE] or [FALSE] or [ABSTAIN]
            """
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7  # Sampling diversity
                )
                v_text = res.choices[0].message.content
                if "[TRUE]" in v_text: votes.append(True)
                elif "[FALSE]" in v_text: votes.append(False)
            except:
                pass
        
        if not votes: return None
        true_votes = sum(1 for v in votes if v is True)
        return true_votes > (len(votes) / 2)

class MADBaseline(Baseline):
    """
    A3. Multi-Agent Debate (No tools).
    Simulates a multi-round debate among 3 agents followed by a majority vote.
    """
    def __init__(self, num_rounds: int = 2):
        self.num_rounds = num_rounds
        self.agent_profiles = [
            {"name": "Agent_Expert", "prompt": "You are a methodical logical expert. Search for truth."},
            {"name": "Agent_Skeptic", "prompt": "You are a critical thinker. Question the evidence and find flaws."},
            {"name": "Agent_Analyst", "prompt": "You are a balanced analyst. Synthesize points and clarify nuance."}
        ]

    def verify(self, claim: str) -> Optional[bool]:
        history = [f"Debate Topic: {claim}"]
        
        for r in range(self.num_rounds):
            for agent in self.agent_profiles:
                prompt = f"""
                Background: {agent['prompt']}
                Chat History:
                {chr(10).join(history)}
                
                YOUR TURN: Provide a short argument regarding the claim. Be logical and direct.
                """
                try:
                    res = client.chat.completions.create(
                        model=GENERATOR_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    msg = f"{agent['name']}: {res.choices[0].message.content.strip()}"
                    history.append(msg)
                except:
                    continue

        # Final voting round
        verdicts = []
        for agent in self.agent_profiles:
            vote_prompt = f"""
            Based on the debate below, what is your FINAL verdict on the claim: "{claim}"?
            
            DEBATE:
            {chr(10).join(history)}
            
            Output strictly: [TRUE] or [FALSE] or [ABSTAIN]
            """
            try:
                res = client.chat.completions.create(
                    model=GENERATOR_MODEL,
                    messages=[{"role": "user", "content": vote_prompt}],
                    temperature=0.0
                )
                txt = res.choices[0].message.content
                if "[TRUE]" in txt: verdicts.append(True)
                elif "[FALSE]" in txt: verdicts.append(False)
            except:
                pass

        if not verdicts: return None
        t_cnt = sum(1 for v in verdicts if v is True)
        return t_cnt > (len(verdicts) / 2)

class JudgeOnlyBaseline(Baseline):
    """
    A4. Judge-based Debate (no tool).
    Takes a debate transcript and decides the outcome without tool verification.
    """
    def verify(self, claim: str, transcript: Optional[str] = None) -> Optional[bool]:
        if not transcript:
            # Fallback to CoT if no transcript provided
            return CoTBaseline().verify(claim)
            
        prompt = f"""
        Role: Senior Fact-Checking Judge.
        Task: You are evaluating a debate about the claim: "{claim}"
        
        Transcript:
        --- START ---
        {transcript[:10000]}
        --- END ---
        
        Instructions:
        1. Evaluate the logical consistency of arguments.
        2. Identify which side is more supported by objective reasoning (NOT rhetoric or length).
        3. Note that some participants might be colluding or lying. Rely on logic.
        
        Output format:
        <reasoning> ... </reasoning>
        Final Verdict: [TRUE] or [FALSE] or [ABSTAIN]
        """
        try:
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = res.choices[0].message.content
            if "[TRUE]" in content: return True
            if "[FALSE]" in content: return False
            return None
        except Exception as e:
            print(f"      ⚠️ Judge Baseline Error: {e}")
            return None
