import os
import sys
import json

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines import class_e


def _fake_google_search(_query: str) -> str:
    return json.dumps({
        "serper": [{"title": "t", "url": "u", "snippet": "s"}],
        "ddg": []
    })


def test_mav_baseline_sanity(monkeypatch):
    if class_e.client is None:
        pytest.skip("LLM client not configured")
    monkeypatch.setattr(class_e, "extract_atomic_claims", lambda *_args, **_kwargs: ["Claim A"])
    monkeypatch.setattr(class_e.RealToolkit, "google_search", _fake_google_search)

    baseline = class_e.MAVBaseline(num_verifiers=3)
    verdict = baseline.verify("Earth orbits the Sun.", budget=15.0)

    assert baseline.tool_calls > 0
    assert baseline.budget_spent > 0
    assert verdict in {True, False, None}


def test_bon_mav_baseline_sanity(monkeypatch):
    if class_e.client is None:
        pytest.skip("LLM client not configured")
    monkeypatch.setattr(class_e, "extract_atomic_claims", lambda *_args, **_kwargs: ["Claim A"])
    monkeypatch.setattr(class_e.RealToolkit, "google_search", _fake_google_search)

    baseline = class_e.BoNMAVBaseline(n=3, m_verifiers=3, top_k=2)
    verdict = baseline.verify("Water boils at 100 C at sea level.", budget=20.0)

    assert baseline.tool_calls > 0
    assert baseline.budget_spent > 0
    assert verdict in {True, False, None}


def test_mad_fact_baseline_sanity(monkeypatch):
    if class_e.client is None:
        pytest.skip("LLM client not configured")
    monkeypatch.setattr(class_e.RealToolkit, "google_search", _fake_google_search)

    baseline = class_e.MADFactBaseline(num_jurors=3, rounds=2)
    verdict = baseline.verify("The Moon orbits Earth.", budget=10.0)

    assert baseline.tool_calls >= 0
    assert verdict in {True, False, None}


def test_gkmad_baseline_sanity(monkeypatch):
    if class_e.client is None:
        pytest.skip("LLM client not configured")
    monkeypatch.setattr(class_e.RealToolkit, "google_search", _fake_google_search)

    baseline = class_e.GKMADBaseline(rounds=2)
    verdict = baseline.verify("Paris is the capital of France.", budget=10.0)

    assert baseline.tool_calls > 0
    assert verdict in {True, False, None}


def test_mav_budget_too_small_returns_none(monkeypatch):
    if class_e.client is None:
        pytest.skip("LLM client not configured")
    monkeypatch.setattr(class_e, "extract_atomic_claims", lambda *_args, **_kwargs: ["Claim A"])
    monkeypatch.setattr(class_e.RealToolkit, "google_search", _fake_google_search)

    baseline = class_e.MAVBaseline(num_verifiers=3)
    verdict = baseline.verify("Claim A", budget=1.0)

    assert verdict is None
    assert baseline.tool_calls == 0
