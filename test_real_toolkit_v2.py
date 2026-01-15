# test_real_toolkit_live_v2.py
"""
Harder LIVE integration tests for RealToolkit.

Run:
    python test_real_toolkit_live_v2.py

Notes:
- This will spend tokens and may use Serper/DDG.
- It uses retries for stability.
"""

import time
import json
from typing import Callable, Optional, Tuple, List

from src.tools.real_toolkit import RealToolkit, PythonSandbox
from src.config import SERPER_API_KEY


# -----------------------------
# Assertion helpers
# -----------------------------
def _ok(msg: str) -> None:
    print(f"[OK] {msg}")

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def _fail(msg: str) -> None:
    raise AssertionError(msg)

def assert_true(name: str, cond: bool) -> None:
    if not cond:
        _fail(f"[{name}] expected True, got False")
    _ok(name)

def assert_false(name: str, cond: bool) -> None:
    if cond:
        _fail(f"[{name}] expected False, got True")
    _ok(name)

def assert_is_bool(name: str, x) -> None:
    if not isinstance(x, bool):
        _fail(f"[{name}] expected bool, got {type(x)}: {x}")
    _ok(f"{name} (bool)")

def assert_json_list(name: str, s: str) -> None:
    try:
        obj = json.loads(s)
        if not isinstance(obj, list):
            _fail(f"[{name}] expected JSON list, got {type(obj)}")
        _ok(f"{name} (json list, len={len(obj)})")
    except Exception as e:
        _fail(f"[{name}] invalid JSON: {e}\nRaw: {s[:300]}")


# -----------------------------
# Retry wrapper (LLM/search may be noisy)
# -----------------------------
def retry_bool(
    fn: Callable[[], bool],
    expected: bool,
    name: str,
    tries: int = 3,
    sleep_s: float = 0.8,
) -> bool:
    last = None
    for i in range(tries):
        last = fn()
        if isinstance(last, bool) and last == expected:
            _ok(f"{name} (try {i+1}/{tries})")
            return last
        time.sleep(sleep_s)
    _fail(f"[{name}] failed after {tries} tries, last={last}")


# -----------------------------
# Tests
# -----------------------------
def test_python_sandbox_more():
    # Allowed code should run and return FINAL_RESULT
    out = PythonSandbox.run("FINAL_RESULT = 'VERIFIED_TRUE'")
    assert_true("python_sandbox_simple_ok", "VERIFIED_TRUE" in out)

    # Block common bypass attempts
    out = PythonSandbox.run("__import__('os')\nFINAL_RESULT='VERIFIED_TRUE'")
    assert_true("python_sandbox_blocks___import__", "Security Block" in out)

    out = PythonSandbox.run("FINAL_RESULT = eval('2+2')")
    assert_true("python_sandbox_blocks_eval", "Security Block" in out)

    _ok("test_python_sandbox_more")


def test_google_search_nonempty_when_possible():
    """
    Try to get non-empty search results. If your environment blocks DDG
    and SERPER_API_KEY is missing/invalid, it may legitimately return [].
    In that case we WARN instead of FAIL.
    """
    query = "Eiffel Tower location"
    s = RealToolkit.google_search(query)
    assert_json_list("google_search_json_v2", s)

    results = json.loads(s)
    if len(results) == 0:
        if not SERPER_API_KEY:
            _warn("google_search returned empty list and SERPER_API_KEY is empty. DDG may be blocked in this environment.")
        else:
            _warn("google_search returned empty list even though SERPER_API_KEY is set. Could be quota/timeout.")
    else:
        # Basic sanity check fields if any result exists
        r0 = results[0]
        assert_true("google_search_result_has_snippet", isinstance(r0.get("snippet", ""), str))
        _ok("google_search_nonempty_when_possible")


def test_attack_support_consistency_constraints():
    """
    Hard constraints we expect:
    - For a supportive pair: support True, attack False
    - For a contradictory pair: attack True, support False
    - For direct contradiction, attack(A,B) and attack(B,A) should both be True (often)
    """
    RealToolkit._cache.clear()

    supportive_A = "Paris is the capital city of France."
    supportive_B = "France's capital is Paris."

    retry_bool(lambda: RealToolkit.verify_support(supportive_A, supportive_B), True, "support_true_supportive_pair", tries=3)
    retry_bool(lambda: RealToolkit.verify_attack(supportive_A, supportive_B), False, "attack_false_supportive_pair", tries=3)

    contradict_A = "Paris is not the capital of France."
    contradict_B = "Paris is the capital of France."

    retry_bool(lambda: RealToolkit.verify_attack(contradict_A, contradict_B), True, "attack_true_contradiction_pair_1", tries=3)
    retry_bool(lambda: RealToolkit.verify_support(contradict_A, contradict_B), False, "support_false_contradiction_pair_1", tries=3)

    # Symmetry check for strong contradiction (may still be noisy, so retry)
    retry_bool(lambda: RealToolkit.verify_attack(contradict_B, contradict_A), True, "attack_true_contradiction_pair_2", tries=3)

    _ok("test_attack_support_consistency_constraints")


def test_verify_claim_python_exec_batch():
    """
    Batch of math/date facts that SHOULD be deterministic under PYTHON_EXEC.
    If your verify_claim sometimes falls back to final judge, retry helps.

    We keep the set moderate to avoid too many calls.
    """
    RealToolkit._cache.clear()

    cases: List[Tuple[str, bool]] = [
        ("2 + 2 equals 4.", True),
        ("2 + 2 equals 5.", False),
        ("17 * 19 equals 323.", True),
        ("17 * 19 equals 322.", False),
        ("2020 was a leap year.", True),
        ("1900 was a leap year.", False),   # 1900 is NOT leap (century not divisible by 400)
        ("2000 was a leap year.", True),    # 2000 IS leap (divisible by 400)
        ("The square root of 144 is 12.", True),
        ("The square root of 144 is 13.", False),
    ]

    for i, (claim, expected) in enumerate(cases, 1):
        retry_bool(lambda c=claim: RealToolkit.verify_claim("PYTHON_EXEC", c), expected, f"py_exec_batch_{i}", tries=3)

    _ok("test_verify_claim_python_exec_batch")


def test_verify_claim_web_search_myths_and_facts():
    """
    Web search tests are inherently noisier, but these are stable:
    - capital of France (true/false)
    - Great Wall visible from Moon (myth -> FALSE)
    """
    RealToolkit._cache.clear()

    retry_bool(lambda: RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Paris."), True, "ws_fact_true_capital", tries=3)
    retry_bool(lambda: RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Berlin."), False, "ws_fact_false_capital", tries=3)

    # A classic myth. Many sources call it a myth; verdict should be FALSE.
    retry_bool(
        lambda: RealToolkit.verify_claim("WEB_SEARCH", "The Great Wall of China is visible from the Moon with the naked eye."),
        False,
        "ws_myth_false_great_wall_moon",
        tries=3
    )

    _ok("test_verify_claim_web_search_myths_and_facts")


def test_cache_stress_repeat_and_size():
    """
    Stress cache a bit:
    - Repeat same claim multiple times
    - Ensure cache size does not grow after the first call
    """
    RealToolkit._cache.clear()
    claim = "17 * 19 equals 323."

    sz0 = len(RealToolkit._cache)
    out1 = RealToolkit.verify_claim("PYTHON_EXEC", claim)
    assert_is_bool("cache_stress_first_bool", out1)
    sz1 = len(RealToolkit._cache)

    for _ in range(5):
        outk = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        assert_true("cache_stress_same_result", outk == out1)

    sz2 = len(RealToolkit._cache)
    assert_true("cache_stress_not_grow_after_repeats", sz2 == sz1 and sz1 >= sz0)
    _ok(f"cache sizes: before={sz0}, after_first={sz1}, after_repeats={sz2}")


def run_all():
    print("=== RealToolkit LIVE integration tests v2 (hard) ===")
    test_python_sandbox_more()
    test_google_search_nonempty_when_possible()
    test_attack_support_consistency_constraints()
    test_verify_claim_python_exec_batch()
    test_verify_claim_web_search_myths_and_facts()
    test_cache_stress_repeat_and_size()
    print("\nAll RealToolkit LIVE v2 tests passed ✅")


if __name__ == "__main__":
    run_all()
