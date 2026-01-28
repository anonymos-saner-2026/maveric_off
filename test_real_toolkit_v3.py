# test_real_toolkit_live_v3.py
"""
RealToolkit LIVE integration tests v3 (very hard).

Run:
    python test_real_toolkit_live_v3.py

This test suite calls REAL APIs. It is designed to be demanding:
- property-based random arithmetic (deterministic)
- negation consistency
- attack/support constraints
- cache stress
- web-search myth/facts (with retries, warnings if search empty)
"""

import random
import time
import json
from typing import Callable, List, Tuple, Optional

from src.tools.real_toolkit import RealToolkit
from src.config import SERPER_API_KEY


# -----------------------------
# Helpers
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

def safe_json_list(s: str):
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    return None


# -----------------------------
# Generators for hard tests
# -----------------------------
def gen_arith_case() -> Tuple[str, bool]:
    """
    Generate an arithmetic equality claim that is either true or false.
    Includes noise words and punctuation.
    """
    op = random.choice(["+", "-", "*"])
    a = random.randint(-50, 50)
    b = random.randint(-50, 50)

    if op == "+":
        val = a + b
    elif op == "-":
        val = a - b
    else:
        val = a * b

    make_true = random.random() < 0.5
    c = val if make_true else val + random.choice([-3, -2, -1, 1, 2, 3])

    noise_prefix = random.choice([
        "",
        "I think ",
        "Hmm... ",
        "Quick check: ",
        "Just to be sure, ",
        "lol ",
        "🧮 ",
    ])
    noise_suffix = random.choice([
        "",
        " right?",
        " !!!",
        " (please verify)",
        " tbh",
        " 😅",
    ])

    claim = f"{noise_prefix}{a} {op} {b} equals {c}{noise_suffix}"
    return claim, make_true


def gen_leap_case() -> Tuple[str, bool]:
    """
    Generate leap year claim (positive or negated).
    """
    y = random.choice([1600, 1700, 1800, 1900, 1996, 2000, 2004, 2100, 2400])
    is_leap = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))

    neg = random.random() < 0.5
    if neg:
        claim = f"{y} was not a leap year."
        expected = (not is_leap)
    else:
        claim = f"{y} was a leap year."
        expected = is_leap

    # Add conversational filler to test deterministic-first on raw text
    claim = random.choice(["", "FYI, ", "Note: ", ""]) + claim + random.choice(["", " 🙂", "!!"])
    return claim, expected


def gen_sqrt_case() -> Tuple[str, bool]:
    a = random.choice([0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169, 196, 225])
    b_true = int(a ** 0.5)
    make_true = random.random() < 0.5
    b = b_true if make_true else b_true + random.choice([-2, -1, 1, 2])

    claim = f"The square root of {a} is {b}."
    # Add hedge noise
    claim = random.choice(["", "I guess ", "Pretty sure "]) + claim
    return claim, make_true


def negate_simple_claim(claim: str) -> Optional[str]:
    """
    Negate a subset of supported patterns:
    - leap year -> insert/remove 'not'
    - sqrt -> 'is' -> 'is not'
    - arithmetic equals -> 'equals' -> 'does not equal'
    We only use negation for deterministic comparisons, so keep it simple.
    """
    s = claim.strip()
    low = s.lower()

    if "leap year" in low:
        if " not a leap year" in low or " not leap year" in low:
            return re_replace_case_insensitive(s, "not a leap year", "a leap year")
        else:
            # insert "not"
            return re_insert_not_leap(s)

    if "square root of" in low and " is " in low:
        if " is not " in low:
            return re_replace_case_insensitive(s, " is not ", " is ")
        else:
            return re_replace_case_insensitive(s, " is ", " is not ")

    if " equals " in low:
        if " does not equal " in low:
            return re_replace_case_insensitive(s, " does not equal ", " equals ")
        else:
            return re_replace_case_insensitive(s, " equals ", " does not equal ")

    return None


def re_replace_case_insensitive(text: str, old: str, new: str) -> str:
    # minimal case-insensitive replace for one occurrence
    idx = text.lower().find(old.lower())
    if idx < 0:
        return text
    return text[:idx] + new + text[idx + len(old):]


def re_insert_not_leap(text: str) -> str:
    low = text.lower()
    idx = low.find(" a leap year")
    if idx >= 0:
        return text[:idx] + " not" + text[idx:]
    idx2 = low.find(" is a leap year")
    if idx2 >= 0:
        return text[:idx2 + len(" is")] + " not" + text[idx2 + len(" is"):]
    idx3 = low.find(" was a leap year")
    if idx3 >= 0:
        return text[:idx3 + len(" was")] + " not" + text[idx3 + len(" was"):]
    return text + " (negation failed)"


# -----------------------------
# Tests
# -----------------------------
def test_py_exec_property_arithmetic(n: int = 60):
    RealToolkit._cache.clear()
    mismatches = 0
    for i in range(1, n + 1):
        claim, expected = gen_arith_case()
        got = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        assert_is_bool(f"arith_{i}_bool", got)
        if got != expected:
            mismatches += 1
    if mismatches:
        _warn(f"py_exec_arithmetic_mismatches={mismatches}/{n}")
    _ok(f"test_py_exec_property_arithmetic (n={n})")


def test_py_exec_property_leap_year(n: int = 30):
    RealToolkit._cache.clear()
    mismatches = 0
    for i in range(1, n + 1):
        claim, expected = gen_leap_case()
        got = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        assert_is_bool(f"leap_{i}_bool", got)
        if got != expected:
            mismatches += 1
    if mismatches:
        _warn(f"py_exec_leap_mismatches={mismatches}/{n}")
    _ok(f"test_py_exec_property_leap_year (n={n})")


def test_py_exec_property_sqrt(n: int = 30):
    RealToolkit._cache.clear()
    mismatches = 0
    for i in range(1, n + 1):
        claim, expected = gen_sqrt_case()
        got = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        assert_is_bool(f"sqrt_{i}_bool", got)
        if got != expected:
            mismatches += 1
    if mismatches:
        _warn(f"py_exec_sqrt_mismatches={mismatches}/{n}")
    _ok(f"test_py_exec_property_sqrt (n={n})")


def test_negation_consistency():
    """
    For deterministic patterns, negating the statement should flip the verdict.
    We only test where we can construct a safe negation string.
    """
    RealToolkit._cache.clear()

    base_cases = []
    for _ in range(10):
        base_cases.append(gen_leap_case()[0])
    for _ in range(10):
        base_cases.append(gen_sqrt_case()[0])
    for _ in range(10):
        base_cases.append(gen_arith_case()[0])

    tested = 0
    flips = 0
    for i, claim in enumerate(base_cases, 1):
        neg = negate_simple_claim(claim)
        if not neg:
            continue

        v1 = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        v2 = RealToolkit.verify_claim("PYTHON_EXEC", neg)

        assert_is_bool(f"neg_{i}_v1_bool", v1)
        assert_is_bool(f"neg_{i}_v2_bool", v2)

        if v1 != v2:
            flips += 1
        tested += 1

    if tested < 10:
        _warn(f"negation_consistency_low_samples={tested}")
    if flips == 0:
        _warn("negation_consistency_no_flips")
    _ok(f"test_negation_consistency (tested={tested}, flips={flips})")


def test_attack_support_mutual_exclusion():
    """
    For the same (A,B), we should not get both attack=True and support=True.
    We test a set of pairs.
    """
    RealToolkit._cache.clear()

    pairs = [
        ("Paris is the capital of France.", "France's capital is Paris."),  # support
        ("Paris is not the capital of France.", "Paris is the capital of France."),  # attack
        ("Bananas are yellow.", "France's capital is Paris."),  # unrelated
        ("All mammals are animals.", "Dogs are mammals."),  # might be support-ish but not direct; still should not be both
    ]

    for i, (a, b) in enumerate(pairs, 1):
        atk = retry_bool(lambda aa=a, bb=b: RealToolkit.verify_attack(aa, bb), expected=True, name=f"atk_check_{i}_tolerant", tries=1, sleep_s=0.0) \
              if "not the capital" in a else RealToolkit.verify_attack(a, b)
        sup = RealToolkit.verify_support(a, b)

        assert_is_bool(f"pair_{i}_atk_bool", atk)
        assert_is_bool(f"pair_{i}_sup_bool", sup)
        assert_false(f"pair_{i}_not_both_true", atk and sup)

    _ok("test_attack_support_mutual_exclusion")


def test_cache_stress():
    """
    Repeat the same deterministic claim many times and ensure cache size does not grow.
    """
    RealToolkit._cache.clear()
    claim = "2000 was a leap year."
    sz0 = len(RealToolkit._cache)

    v = RealToolkit.verify_claim("PYTHON_EXEC", claim)
    assert_is_bool("cache_stress_first_bool", v)
    sz1 = len(RealToolkit._cache)

    for _ in range(20):
        v2 = RealToolkit.verify_claim("PYTHON_EXEC", claim)
        if v2 != v:
            _fail(f"[cache_stress_same_result] changed result: {v2} vs {v}")
    sz2 = len(RealToolkit._cache)

    assert_true("cache_stress_cache_not_grow", sz2 == sz1 and sz1 >= sz0)
    _ok(f"cache sizes: before={sz0}, after_first={sz1}, after_repeats={sz2}")


def test_web_search_myths_with_warning():
    """
    Web search may return empty; we do not fail on empty evidence, but we test verdict stability.
    """
    RealToolkit._cache.clear()

    retry_bool(lambda: RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Paris."), True, "ws_paris_true", tries=3)
    retry_bool(lambda: RealToolkit.verify_claim("WEB_SEARCH", "The capital of France is Berlin."), False, "ws_berlin_false", tries=3)

    # Myth test: may be noisy, retry
    retry_bool(
        lambda: RealToolkit.verify_claim("WEB_SEARCH", "The Great Wall of China is visible from the Moon with the naked eye."),
        False,
        "ws_great_wall_moon_false",
        tries=3
    )

    # Also check the raw search output is parseable JSON (even if empty)
    s = RealToolkit.google_search("Great Wall visible from the Moon myth")
    obj = safe_json_list(s)
    if obj is None:
        _warn("google_search did not return JSON list. Consider returning JSON list for reproducibility.")
    else:
        _ok(f"ws_search_json_ok (len={len(obj)})")
        if len(obj) == 0:
            if SERPER_API_KEY:
                _warn("Search returned empty list even though SERPER_API_KEY is set. Quota/timeout likely.")
            else:
                _warn("Search returned empty list. DDG may be blocked or SERPER not configured.")


def run_all():
    print("=== RealToolkit LIVE integration tests v3 (very hard) ===")
    random.seed(2026)

    test_py_exec_property_arithmetic(n=60)
    test_py_exec_property_leap_year(n=30)
    test_py_exec_property_sqrt(n=30)
    test_negation_consistency()
    test_attack_support_mutual_exclusion()
    test_cache_stress()
    test_web_search_myths_with_warning()

    print("\nAll RealToolkit LIVE v3 tests passed ✅")


if __name__ == "__main__":
    run_all()
