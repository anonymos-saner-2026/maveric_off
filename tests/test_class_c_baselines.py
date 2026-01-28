import os
import sys
import types

import pytest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines import verification_heavy as vh


class _DummyResponse:
    def __init__(self, content: str):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


def _make_dummy_client(responses):
    responses = list(responses)

    def _create(*_args, **_kwargs):
        if not responses:
            raise AssertionError("No more dummy responses configured")
        return _DummyResponse(responses.pop(0))

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )
    )


def _make_repeating_client(content: str):
    def _create(*_args, **_kwargs):
        return _DummyResponse(content)

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=_create)
        )
    )


def test_verifyclaim_budget_insufficient(monkeypatch):
    monkeypatch.setattr(vh, "TOOLS_CONFIG", {"WEB_SEARCH": {"cost": 5.0}})
    result = vh.VerifyClaim("Some claim", budget=1.0)

    assert result.verdict is None
    assert result.spent == 0.0
    assert "Budget" in result.rationale


def test_budgeted_critic_fallback_without_claims(monkeypatch):
    dummy_client = _make_dummy_client(["<verdict>TRUE</verdict>"])
    monkeypatch.setattr(vh, "client", dummy_client)
    monkeypatch.setattr(vh, "extract_atomic_claims", lambda *_args, **_kwargs: [])

    baseline = vh.BudgetedCRITICBaseline(max_verifications=2)
    verdict = baseline.verify("Test claim", transcript=None, parsed_graph=None, budget=10.0)

    assert verdict is True


def test_rarr_handles_no_evidence(monkeypatch):
    dummy_client = _make_dummy_client(["<verdict>TRUE</verdict>"])
    monkeypatch.setattr(vh, "client", dummy_client)
    monkeypatch.setattr(vh, "retrieve_evidence", lambda *_args, **_kwargs: ([], 0.0))

    baseline = vh.RARRBaseline(max_rounds=2)
    verdict = baseline.verify("Test claim", transcript=None, budget=5.0)

    assert verdict is True


def test_class_c_sanity_realistic_topics(monkeypatch):
    topics = [
        (
            "Vaccines reduce the risk of severe COVID-19 outcomes.",
            "Debater A: Multiple studies show vaccines lower hospitalization rates.\n"
            "Debater B: Some vaccinated people still get sick, so the effect is unclear."
        ),
        (
            "The Great Wall of China is visible from space with the naked eye.",
            "Debater A: Astronauts can see the wall from orbit.\n"
            "Debater B: It is a myth; it is not visible without aid."
        ),
        (
            "Renewable energy costs have declined significantly over the last decade.",
            "Debater A: Solar and wind prices have dropped a lot.\n"
            "Debater B: Costs vary by region and remain high in some places."
        ),
    ]

    dummy_client = _make_repeating_client("<verdict>TRUE</verdict>")
    monkeypatch.setattr(vh, "client", dummy_client)
    monkeypatch.setattr(vh, "extract_atomic_claims", lambda *_args, **_kwargs: ["Claim A", "Claim B"])
    monkeypatch.setattr(
        vh,
        "VerifyClaim",
        lambda *_args, **_kwargs: vh.VerifyResult(
            verdict=True,
            confidence=0.9,
            spent=5.0,
            support_ids=[1],
            refute_ids=[],
            rationale="Stubbed"
        )
    )
    monkeypatch.setattr(
        vh,
        "retrieve_evidence",
        lambda *_args, **_kwargs: ([{"title": "t", "url": "u", "snippet": "s"}], 0.0)
    )

    critic = vh.BudgetedCRITICBaseline(max_verifications=1)
    verify_revise = vh.VerifyAndReviseBaseline(max_rounds=1)
    rarr = vh.RARRBaseline(max_rounds=1)

    for claim, transcript in topics:
        assert critic.verify(claim, transcript=transcript, parsed_graph=None, budget=10.0) is True
        assert verify_revise.verify(claim, transcript=transcript, budget=10.0) is True
        assert rarr.verify(claim, transcript=transcript, budget=10.0) is True
