import os
import sys

import pytest


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.baselines import verification_heavy as vh


LIVE_TOPICS = [
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


def _require_live_clients():
    if vh.client is None:
        pytest.fail("OpenAI client is not initialized. Set OPENAI_API_KEY and OPENAI_BASE_URL if needed.")
    if not os.getenv("OPENAI_API_KEY"):
        pytest.fail("OPENAI_API_KEY is missing for live tests.")


def _is_verdict(value):
    return value is True or value is False or value is None


def test_class_c_live_sanity():
    _require_live_clients()

    critic = vh.BudgetedCRITICBaseline(max_verifications=1)
    verify_revise = vh.VerifyAndReviseBaseline(max_rounds=1)
    rarr = vh.RARRBaseline(max_rounds=1)

    for claim, transcript in LIVE_TOPICS:
        c1 = critic.verify(claim, transcript=transcript, parsed_graph=None, budget=5.0)
        c2 = verify_revise.verify(claim, transcript=transcript, budget=5.0)
        c3 = rarr.verify(claim, transcript=transcript, budget=5.0)

        assert _is_verdict(c1)
        assert _is_verdict(c2)
        assert _is_verdict(c3)
