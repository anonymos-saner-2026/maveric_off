# test_real_toolkit_live.py
"""
Integration tests for RealToolkit using REAL APIs (OpenAI + search).

Run:
    python test_real_toolkit_live.py

What it tests (live):
- PythonSandbox safety blocks
- verify_attack / verify_support consistency
- verify_claim with PYTHON_EXEC (deterministic math)
- verify_claim with WEB_SEARCH (stable facts)
- Cache behavior (RealToolkit._cache should not grow on repeated identical calls)
"""

import json
import time

from src.tools.real_toolkit import RealToolkit, PythonSandbox
from src.config import SERPER_API_KEY


# -----------------------------
# Simple assertion helpers
# -----------------------------
def _ok(msg):
    print(f"[OK] {msg}")

def _warn(msg):
    print(f"[WARN] {msg}")

def _fail(msg):
    raise AssertionError(msg)

def assert_true(name, cond):
    if not cond:
        _fail(f"[{name}] expected True, got False")
    _ok(name)

def assert_false(name, cond):
    if cond:
        _fail(f"[{name}] expected False, got True")
    _ok(name)

def assert_is_bool(name, x):
    if not isinstance(x, bool):
        _fail(f"[{name}] expected bool, got {type(x)}: {x}")
    _ok(f"{name} (bool)")

def assert_json_list(name, s):
    try:
        obj = json.loads(s)
        if not isinstance(obj, list):
            _fail(f"[{name}] expected JSON list, got {type(obj)}")
        _ok(f"{name} (json list, len={len(obj)})")
    except Exception as e:
        _fail(f"[{name}] invalid JSON: {e}\nRaw: {s[:300]}")


# -----------------------------
# Retry wrapper for non-deterministic LLM outputs
# -----------------------------
def retry_bool(fn, expected: bool, name: str, tries: int = 3, sleep_s: float = 0.8):
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
def test_python_sandbox_blocks():
    out = PythonSandbox.run("import os\nFINAL_RESULT='VERIFIED_TRUE'")
    assert_true("python_sandbox_blocks_import", "Security Block" in out)

    out = PythonSandbox.run("while True:\n    pass\nFINAL_RESULT='VERIFIED_TRUE'")
    assert_true("python_sandbox_blocks_while_true", "Security Block" in out)

    out = PythonSandbox.run("FINAL_RESULT = 'VERIFIED_TRUE'")
    assert_true("python_sandbox_allows_simple", "VERIFIED_TRUE" in out)


def test_google_search_returns_json():
    if not SERPER_API_KEY:
        _warn("SERPER_API_KEY is empty. google_search will fallback to DDG, which can be flaky in some environments.")
    s = RealToolkit.google_search("Eiffel Tower location")
    assert_json_list("google_search_json", s)


def test_verify_attack_support_basic():
    RealToolkit._cache.clear()

    # Attack should be TRUE
    def _attack_true():
        return RealToolkit.verify_attack(
            "Paris is not the capital of France.",
            "Paris is the capital of France."
        )
    retry_bool(_attack_true, True, "verify_attack_true_basic", tries=3)

    # Irrelevant should be FALSE
    def _attack_false():
        return RealToolkit.verify_attack(
            "Bananas are yellow.",
            "Paris is the capital of France."
        )
    retry_bool(_attack_false, False, "verify_attack_false_irrelevant", tries=3)

    # Support should be TRUE
    def _support_true():
        return RealToolkit.verify_support(
            "Paris is the capital city of France.",
            "France's capital is Paris."
        )
    retry_bool(_support_true, True, "verify_support_true_basic", tries=3)

    # Support should be FALSE for contradiction/irrelevance
    def _support_false():
        return RealToolkit.verify_support(
            "Berlin is the capital of France.",
            "France's capital is Paris."
        )
    retry_bool(_support_false, False, "verify_support_false_basic", tries=3)


def test_verify_claim_python_exec_deterministic():
    RealToolkit._cache.clear()

    # These are the most reliable because they should resolve via PythonSandbox
    def _py_true():
        return RealToolkit.verify_claim("PYTHON_EXEC", "2 + 2 equals 4.")
    retry_bool(_py_true, True, "verify_claim_python_exec_true", tries=3)

    def _py_false():
        return RealToolkit.verify_claim("PYTHON_EXEC", "2 + 2 equals 5.")
    retry_bool(_py_false, False, "verify_claim_python_exec_false", tries=3)


def test_verify_claim_web_search_stable_facts():
    RealToolkit._cache.clear()

    # Web search can still be noisy, but these are extremely stable.
    # Use retries to reduce variance.
    def _ws_true():
        return RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Paris.")
    retry_bool(_ws_true, True, "verify_claim_web_search_true", tries=3)

    def _ws_false():
        return RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Berlin.")
    retry_bool(_ws_false, False, "verify_claim_web_search_false", tries=3)


def test_cache_does_not_grow_on_repeat():
    RealToolkit._cache.clear()

    # Pick a deterministic task to minimize any randomness in outputs
    claim = "2 + 2 equals 4."

    sz0 = len(RealToolkit._cache)
    out1 = RealToolkit.verify_claim("PYTHON_EXEC", claim)
    assert_is_bool("cache_repeat_first_call_returns_bool", out1)
    sz1 = len(RealToolkit._cache)

    out2 = RealToolkit.verify_claim("PYTHON_EXEC", claim)
    assert_is_bool("cache_repeat_second_call_returns_bool", out2)
    sz2 = len(RealToolkit._cache)

    assert_true("cache_repeat_same_result", out1 == out2)
    assert_true("cache_repeat_cache_not_grow", sz2 == sz1 and sz1 >= sz0)
    _ok(f"cache sizes: before={sz0}, after_first={sz1}, after_second={sz2}")


def run_all():
    print("=== RealToolkit LIVE integration tests ===")
    test_python_sandbox_blocks()
    test_google_search_returns_json()
    test_verify_attack_support_basic()
    test_verify_claim_python_exec_deterministic()
    test_verify_claim_web_search_stable_facts()
    test_cache_does_not_grow_on_repeat()
    print("\nAll RealToolkit LIVE tests passed ✅")


if __name__ == "__main__":
    run_all()
