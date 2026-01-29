import os
import sys
import json
import types

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines import class_f
from src.core.graph import ArgumentationGraph, ArgumentNode


def _fake_safe_json_loads(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def test_f1_mad_filter_drops_outliers(monkeypatch):
    transcript = """
Alice: I support it. VERDICT: TRUE
Bob: I agree. VERDICT: TRUE
Charlie: This is wrong. VERDICT: FALSE
Dave: Yes true. VERDICT: TRUE
"""
    baseline = class_f.F1MADAdversaryFilterBaseline(drop_m=1, disagreement_mode="pairwise", tie_break="skeptical", use_llm_extractor=False)
    verdict = baseline.verify("Claim", transcript=transcript, graph=None, budget=0.0)

    assert verdict is True
    assert baseline.stats.get("agents_dropped") == 1


def test_f1_majority_tie_break_skeptical(monkeypatch):
    transcript = """
Alice: VERDICT: TRUE
Bob: VERDICT: FALSE
"""
    baseline = class_f.F1MADAdversaryFilterBaseline(drop_m=0, disagreement_mode="diff_majority", tie_break="skeptical", use_llm_extractor=False)
    verdict = baseline.verify("Claim", transcript=transcript, graph=None, budget=0.0)

    assert verdict is False


def test_f2_evidence_requirement_requires_citations(monkeypatch):
    def _fake_judge_draft(*_args, **_kwargs):
        return {"support_claims": ["Claim A"], "refute_claims": []}

    def _fake_retrieve(*_args, **_kwargs):
        snippets = [{"title": "t", "url": "u", "snippet": "s"}]
        return snippets, 5.0

    def _fake_rag_judge(*_args, **_kwargs):
        return types.SimpleNamespace(
            verdict=True,
            llm_confidence=0.8,
            evidence_confidence=0.8,
            final_confidence=0.8,
            support_ids=[],
            refute_ids=[],
            rationale="No citations",
        )

    monkeypatch.setattr(class_f.F2EvidenceRequirementBaseline, "_judge_draft", lambda self, *_args, **_kwargs: _fake_judge_draft())
    monkeypatch.setattr(class_f, "retrieve_evidence", _fake_retrieve)
    monkeypatch.setattr(class_f.RealToolkit, "_rag_judge_with_calibrated_conf", _fake_rag_judge)

    baseline = class_f.F2EvidenceRequirementBaseline(per_claim_cap=1)
    verdict = baseline.verify("Claim A", transcript="", graph=None, budget=10.0)

    assert verdict is False


def test_f2_accepts_with_support_citation(monkeypatch):
    def _fake_judge_draft(*_args, **_kwargs):
        return {"support_claims": ["Claim A"], "refute_claims": []}

    def _fake_retrieve(*_args, **_kwargs):
        snippets = [{"title": "t", "url": "u", "snippet": "s"}]
        return snippets, 5.0

    def _fake_rag_judge(*_args, **_kwargs):
        return types.SimpleNamespace(
            verdict=True,
            llm_confidence=0.9,
            evidence_confidence=0.9,
            final_confidence=0.9,
            support_ids=[1],
            refute_ids=[],
            rationale="Cites evidence",
        )

    monkeypatch.setattr(class_f.F2EvidenceRequirementBaseline, "_judge_draft", lambda self, *_args, **_kwargs: _fake_judge_draft())
    monkeypatch.setattr(class_f, "retrieve_evidence", _fake_retrieve)
    monkeypatch.setattr(class_f.RealToolkit, "_rag_judge_with_calibrated_conf", _fake_rag_judge)

    baseline = class_f.F2EvidenceRequirementBaseline(per_claim_cap=1)
    verdict = baseline.verify("Claim A", transcript="", graph=None, budget=10.0)

    assert verdict is True


def test_f3_graph_consistency_gate_triggers(monkeypatch):
    g = ArgumentationGraph()
    a = ArgumentNode(id="A1", content="Claim A", speaker="x", is_verified=True, ground_truth=True, verification_cost=1.0)
    b = ArgumentNode(id="A2", content="Claim B", speaker="y", is_verified=True, ground_truth=True, verification_cost=1.0)
    g.add_node(a)
    g.add_node(b)
    g.add_attack("A1", "A2")

    solver = class_f.F3GraphConsistencyGatingSolver(graph=g, budget=1.0)

    def _fake_get_tool_and_cost(_node):
        return "COMMON_SENSE", 0.1

    def _fake_verify_node(node, _tool, _cost):
        node.is_verified = True
        node.ground_truth = True
        return True

    monkeypatch.setattr(solver, "_get_tool_and_cost", _fake_get_tool_and_cost)
    monkeypatch.setattr(solver, "_verify_node", _fake_verify_node)
    final_ext, verdict = solver.run()

    assert solver.gate_triggered is True
    assert verdict is False
