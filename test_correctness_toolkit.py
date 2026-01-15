# test_correctness_toolkit.py
# Focused correctness tests for tools/real_toolkit.py patches A/B/C.
#
# How to run:
#   python test_correctness_toolkit.py
#
# Assumptions:
# - Your patched toolkit is at: tools/real_toolkit.py
# - It defines class RealToolkit (as in your patched code).
#
# This test script DOES NOT call the internet.
# It monkeypatches:
# - RealToolkit.google_search
# - RealToolkit._llm_rag_judge
# - RealToolkit._llm_common_sense_vote
# - RealToolkit._llm_relation_judge
#
# What it verifies:
# A) stance-aware heuristic: hoax + refute cues must NOT become TRUE from anchor matches
# B) decision order: heuristic weak/None must NOT short-circuit; RAG judge must run
# C) edge verify: prune only when FALSE with high confidence, otherwise keep edge


import sys
sys.path.append("src")
import traceback
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# Import patched toolkit
# -----------------------------
try:
    # If your file is "tools/real_toolkit.py"
    from tools.real_toolkit import RealToolkit  # type: ignore
except Exception as e:
    print("❌ Cannot import RealToolkit from tools.real_toolkit.")
    print("   Please ensure file path is tools/real_toolkit.py and importable.")
    raise


# -----------------------------
# Mini test harness
# -----------------------------
class TestFailure(Exception):
    pass


def assert_true(cond: bool, msg: str):
    if not cond:
        raise TestFailure(msg)


def assert_eq(a, b, msg: str):
    if a != b:
        raise TestFailure(f"{msg} | got={a} expected={b}")


def section(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# -----------------------------
# Monkeypatch helpers
# -----------------------------
class MonkeyPatch:
    def __init__(self):
        self._saved = []

    def setattr(self, obj, name: str, value):
        old = getattr(obj, name)
        self._saved.append((obj, name, old))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)
        self._saved.clear()


# -----------------------------
# Fake search payloads
# -----------------------------
def mk_payload(query: str, serper: List[Dict[str, str]], ddg: List[Dict[str, str]]) -> str:
    import json

    return json.dumps({"query": query, "serper": serper, "ddg": ddg, "ts_ms": 0}, ensure_ascii=False)


def fake_hits_refute_moon_hoax() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    # Trusted-ish domains with refute cues
    serper = [
        {
            "title": "Moon landing conspiracy theories - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Moon_landing_conspiracy_theories",
            "snippet": "Moon landing conspiracy theories claim that the Apollo moon landings were fabricated; these claims have been debunked.",
            "provider": "serper",
        },
        {
            "title": "Apollo moon landing hoax claims are a myth - Britannica",
            "url": "https://www.britannica.com/story/were-the-moon-landings-faked",
            "snippet": "The notion that the Moon landings were faked is a myth; there is no evidence for a hoax.",
            "provider": "serper",
        },
    ]
    ddg = [
        {
            "title": "NASA: Moon landing facts",
            "url": "https://www.nasa.gov/specials/apollo50th/",
            "snippet": "NASA history of Apollo missions and evidence returned; addresses misconceptions and debunked hoax claims.",
            "provider": "ddg",
        }
    ]
    return serper, ddg


def fake_hits_affirm_apollo11_samples() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    serper = [
        {
            "title": "Apollo 11 - Wikipedia",
            "url": "https://en.wikipedia.org/wiki/Apollo_11",
            "snippet": "Apollo 11 was the spaceflight that first landed humans on the Moon.",
            "provider": "serper",
        },
        {
            "title": "NASA Apollo 11 Mission Overview",
            "url": "https://www.nasa.gov/mission_pages/apollo/missions/apollo11.html",
            "snippet": "The mission was flown by NASA and astronauts were involved in the event.",
            "provider": "serper",
        },
    ]
    ddg = []
    return serper, ddg


def fake_hits_ambiguous_anchor_trap() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    # Contains anchor words but does NOT affirm claim; should not lead to TRUE by anchor-match.
    serper = [
        {
            "title": "Discussion of theory that Apollo was staged - example",
            "url": "https://en.wikipedia.org/wiki/Moon_landing_conspiracy_theories",
            "snippet": "This article discusses the theory that Apollo was staged; various viewpoints are mentioned regarding the event.",
            "provider": "serper",
        }
    ]
    ddg = []
    return serper, ddg


# -----------------------------
# Tests
# -----------------------------
def test_A_heuristic_stance_no_anchor_true(mp: MonkeyPatch):
    """
    A) If snippets contain refute cues and claim is hoax-like, heuristic must not yield TRUE.
    Ideally heuristic strong FALSE triggers early FALSE.
    """
    section("TEST A: Heuristic stance-aware (no anchor-match TRUE on hoax/refute cues)")

    serper, ddg = fake_hits_refute_moon_hoax()

    # direct call into heuristic function with "combined hits"
    claim = "Claim: The Apollo Moon landings were a coordinated hoax staged by NASA."
    clean = RealToolkit._distill_claim(claim)

    verdict, strength = RealToolkit._evidence_heuristic_verdict(clean, serper + ddg)
    print(f"heuristic verdict={verdict} strength={strength}")

    # Must NOT be TRUE. Strongly prefer FALSE.
    assert_true(verdict is not True, "Heuristic incorrectly returned TRUE for hoax claim with refute cues.")
    assert_true(verdict is False and strength == "strong", "Expected strong FALSE for hoax+refute cues.")


def test_B_no_short_circuit_must_call_rag_judge(mp: MonkeyPatch):
    """
    B) If heuristic is weak/None, toolkit must call RAG judge (must not short-circuit).
    We'll craft an ambiguous case where heuristic returns None, then ensure judge is invoked.
    """
    section("TEST B: No short-circuit (must call RAG judge unless strong heuristic)")

    calls = {"judge": 0, "search": 0}

    # Fake google_search: always returns ambiguous hits that produce heuristic None (hoaxy => None)
    serper, ddg = fake_hits_ambiguous_anchor_trap()
    payload = mk_payload("q", serper, ddg)

    def fake_google_search(q: str) -> str:
        calls["search"] += 1
        return payload

    # Fake rag judge: return FALSE with high confidence to see final follows judge
    def fake_rag_judge(clean_fact: str, evidence_lines: str):
        calls["judge"] += 1
        return (False, 0.9, [], [1], "Refuted in evidence")

    # Also avoid common sense fallback interfering
    def fake_common_sense(clean_fact: str):
        return None

    mp.setattr(RealToolkit, "google_search", staticmethod(fake_google_search))
    mp.setattr(RealToolkit, "_llm_rag_judge", staticmethod(fake_rag_judge))
    mp.setattr(RealToolkit, "_llm_common_sense_vote", staticmethod(fake_common_sense))

    claim = "Claim: The Apollo Moon landings were a coordinated hoax staged by NASA."
    out = RealToolkit.verify_claim("WEB_SEARCH", claim)
    print(f"verify_claim output={out}, judge_calls={calls['judge']}, search_calls={calls['search']}")

    assert_true(calls["judge"] >= 1, "RAG judge was not called when heuristic was not strong.")
    assert_eq(out, False, "Final verdict should follow judge FALSE in this test.")


def test_B_strong_heuristic_can_short_circuit_false(mp: MonkeyPatch):
    """
    B) Heuristic should be allowed to short-circuit only for STRONG cases.
    For hoax+refute cues, strong FALSE should return without calling judge.
    """
    section("TEST B2: Strong heuristic FALSE may short-circuit (no judge call)")

    calls = {"judge": 0}

    serper, ddg = fake_hits_refute_moon_hoax()
    payload = mk_payload("q", serper, ddg)

    def fake_google_search(q: str) -> str:
        return payload

    def fake_rag_judge(clean_fact: str, evidence_lines: str):
        calls["judge"] += 1
        return (True, 0.9, [1], [], "Should not be called")

    mp.setattr(RealToolkit, "google_search", staticmethod(fake_google_search))
    mp.setattr(RealToolkit, "_llm_rag_judge", staticmethod(fake_rag_judge))

    claim = "Claim: The Apollo Moon landings were a coordinated hoax staged by NASA."
    out = RealToolkit.verify_claim("WEB_SEARCH", claim)
    print(f"verify_claim output={out}, judge_calls={calls['judge']}")

    assert_eq(calls["judge"], 0, "Judge should not be called if strong heuristic already decided.")
    assert_eq(out, False, "Strong heuristic refutation should yield FALSE.")


def test_B_true_requires_confidence_gate(mp: MonkeyPatch):
    """
    B) When judge returns TRUE but with low confidence (<0.62), toolkit should NOT verify TRUE (conservative).
    """
    section("TEST B3: TRUE requires confidence gate (low-conf TRUE should not pass)")

    calls = {"judge": 0}

    serper, ddg = fake_hits_affirm_apollo11_samples()
    payload = mk_payload("q", serper, ddg)

    def fake_google_search(q: str) -> str:
        return payload

    def fake_rag_judge(clean_fact: str, evidence_lines: str):
        calls["judge"] += 1
        return (True, 0.55, [1], [], "Weak support")

    mp.setattr(RealToolkit, "google_search", staticmethod(fake_google_search))
    mp.setattr(RealToolkit, "_llm_rag_judge", staticmethod(fake_rag_judge))
    mp.setattr(RealToolkit, "_llm_common_sense_vote", staticmethod(lambda _: None))

    claim = "Claim: Apollo 11 returned lunar rock samples to Earth."
    out = RealToolkit.verify_claim("WEB_SEARCH", claim)
    print(f"verify_claim output={out}, judge_calls={calls['judge']}")

    assert_true(calls["judge"] >= 1, "Judge should be called in this scenario.")
    assert_eq(out, False, "Low-confidence TRUE should not be accepted as verified TRUE.")


def test_C_edge_verify_prune_only_when_confident_false(mp: MonkeyPatch):
    """
    C) verify_support/verify_attack should only return False (prune) when relation judge says FALSE with conf>=0.75.
    Otherwise, it should return True (keep edge).
    """
    section("TEST C: Edge verify uses ABSTAIN + prune only when confident FALSE")

    # Case 1: FALSE at 0.80 => should prune => function returns False
    def rel_false_high(a: str, b: str, mode: str):
        return ("FALSE", 0.80)

    mp.setattr(RealToolkit, "_llm_relation_judge", staticmethod(rel_false_high))
    out1 = RealToolkit.verify_support("A1", "B1")
    out2 = RealToolkit.verify_attack("A1", "B1")
    print(f"FALSE@0.80 => support={out1}, attack={out2}")
    assert_eq(out1, False, "verify_support should prune on FALSE with high confidence.")
    assert_eq(out2, False, "verify_attack should prune on FALSE with high confidence.")

    # Case 2: FALSE at 0.60 => should NOT prune => returns True
    def rel_false_low(a: str, b: str, mode: str):
        return ("FALSE", 0.60)

    mp.setattr(RealToolkit, "_llm_relation_judge", staticmethod(rel_false_low))
    out3 = RealToolkit.verify_support("A2", "B2")
    out4 = RealToolkit.verify_attack("A2", "B2")
    print(f"FALSE@0.60 => support={out3}, attack={out4}")
    assert_eq(out3, True, "verify_support should NOT prune on FALSE with low confidence.")
    assert_eq(out4, True, "verify_attack should NOT prune on FALSE with low confidence.")

    # Case 3: ABSTAIN => keep edge => returns True
    def rel_abstain(a: str, b: str, mode: str):
        return ("ABSTAIN", 0.0)

    mp.setattr(RealToolkit, "_llm_relation_judge", staticmethod(rel_abstain))
    out5 = RealToolkit.verify_support("A3", "B3")
    out6 = RealToolkit.verify_attack("A3", "B3")
    print(f"ABSTAIN => support={out5}, attack={out6}")
    assert_eq(out5, True, "verify_support should keep edge on ABSTAIN.")
    assert_eq(out6, True, "verify_attack should keep edge on ABSTAIN.")

    # Case 4: TRUE => keep edge => returns True
    def rel_true(a: str, b: str, mode: str):
        return ("TRUE", 0.9)

    mp.setattr(RealToolkit, "_llm_relation_judge", staticmethod(rel_true))
    out7 = RealToolkit.verify_support("A4", "B4")
    out8 = RealToolkit.verify_attack("A4", "B4")
    print(f"TRUE => support={out7}, attack={out8}")
    assert_eq(out7, True, "verify_support should keep edge on TRUE.")
    assert_eq(out8, True, "verify_attack should keep edge on TRUE.")


def run_all():
    mp = MonkeyPatch()
    failures = 0

    # IMPORTANT: clear toolkit cache to avoid cross-test contamination
    if hasattr(RealToolkit, "_cache"):
        RealToolkit._cache.clear()

    tests = [
        test_A_heuristic_stance_no_anchor_true,
        test_B_no_short_circuit_must_call_rag_judge,
        test_B_strong_heuristic_can_short_circuit_false,
        test_B_true_requires_confidence_gate,
        test_C_edge_verify_prune_only_when_confident_false,
    ]

    for t in tests:
        try:
            # Clear cache before each test for determinism
            if hasattr(RealToolkit, "_cache"):
                RealToolkit._cache.clear()

            mp.undo()
            t(mp)
            print("✅ PASS")
        except TestFailure as e:
            failures += 1
            print("❌ FAIL:", str(e))
        except Exception as e:
            failures += 1
            print("❌ ERROR:", repr(e))
            traceback.print_exc()

    mp.undo()

    print("\n" + "-" * 70)
    if failures == 0:
        print("✅ ALL TESTS PASSED")
        return 0
    print(f"❌ {failures} TEST(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(run_all())
