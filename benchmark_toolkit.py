# test_real_toolkit_benchmark.py
# Integration benchmark for RealToolkit.verify_claim (WEB_SEARCH / PYTHON_EXEC)
#
# Usage:
#   python test_real_toolkit_benchmark.py
#   python test_real_toolkit_benchmark.py --suite hoax
#   python test_real_toolkit_benchmark.py --suite sanity
#   python test_real_toolkit_benchmark.py --suite all --max 999
#
# Notes:
# - Requires your project imports to work: from src.tools.real_toolkit import RealToolkit
# - WEB_SEARCH will call external search + OpenAI judge -> can be slow and rate-limited.
# - "Expect 100%" is only realistic for sanity(PYTHON_EXEC). WEB_SEARCH may have residual noise.

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Literal

# Adjust import path if needed:
# sys.path.append(".")
from src.tools.real_toolkit import RealToolkit  # noqa


ToolType = Literal["PYTHON_EXEC", "WEB_SEARCH", "COMMON_SENSE"]


@dataclass
class ClaimCase:
    id: str
    tool: ToolType
    claim: str
    gold: bool
    tags: List[str]


@dataclass
class CaseResult:
    case: ClaimCase
    pred: Optional[bool]
    ok: bool
    latency_s: float
    error: Optional[str] = None


# ----------------------------
# Suites
# ----------------------------

def sanity_suite() -> List[ClaimCase]:
    # Deterministic tier0 should pass these with ~100% (and fast).
    cases = [
        ClaimCase("sanity_leap_2000", "PYTHON_EXEC", "2000 was a leap year.", True, ["sanity", "leap"]),
        ClaimCase("sanity_leap_1900", "PYTHON_EXEC", "1900 was a leap year.", False, ["sanity", "leap"]),
        ClaimCase("sanity_arith_1", "PYTHON_EXEC", "17 * 19 equals 323", True, ["sanity", "arith"]),
        ClaimCase("sanity_arith_2", "PYTHON_EXEC", "17 * 19 equals 322", False, ["sanity", "arith"]),
        ClaimCase("sanity_sqrt_1", "PYTHON_EXEC", "The square root of 144 is 12.", True, ["sanity", "sqrt"]),
        ClaimCase("sanity_sqrt_2", "PYTHON_EXEC", "The square root of 144 is 13.", False, ["sanity", "sqrt"]),
        ClaimCase("sanity_pct_1", "PYTHON_EXEC", "25% of 200 equals 50", True, ["sanity", "percent"]),
        ClaimCase("sanity_pct_2", "PYTHON_EXEC", "25% of 200 equals 40", False, ["sanity", "percent"]),
        ClaimCase("sanity_cmp_1", "PYTHON_EXEC", "3 > 2", True, ["sanity", "compare"]),
        ClaimCase("sanity_cmp_2", "PYTHON_EXEC", "3 < 2", False, ["sanity", "compare"]),
        # Negation normalization
        ClaimCase("sanity_sqrt_neg", "PYTHON_EXEC", "The square root of 16 is not 4.", False, ["sanity", "sqrt"]),
        ClaimCase("sanity_leap_neg", "PYTHON_EXEC", "1900 is not a leap year.", True, ["sanity", "leap"]),
        ClaimCase("sanity_div_1", "PYTHON_EXEC", "8 / 2 equals 4", True, ["sanity", "arith"]),
        ClaimCase("sanity_div_2", "PYTHON_EXEC", "8 / 2 equals 5", False, ["sanity", "arith"]),
    ]
    return cases


def hoax_suite() -> List[ClaimCase]:
    # All gold=False. Goal: minimize false positives.
    # You can extend to 50 easily by adding more lines.
    hoax_claims = [
        ("hoax_apollo_1", "The Apollo 11 Moon landing was staged and filmed on a soundstage.", False),
        ("hoax_apollo_2", "NASA faked all Moon landings; astronauts never walked on the Moon.", False),
        ("hoax_flat_1", "The Earth is flat and NASA hides the truth.", False),
        ("hoax_flat_2", "There is an ice wall around Earth that prevents people from falling off.", False),
        ("hoax_vax_1", "COVID-19 vaccines contain microchips to track people.", False),
        ("hoax_vax_2", "Vaccines cause autism and this has been proven.", False),
        ("hoax_chemtrails_1", "Chemtrails are sprayed by governments to control the population.", False),
        ("hoax_5g_1", "5G towers caused COVID-19 outbreaks.", False),
        ("hoax_qanon_1", "A secret cabal controls the world and harvests children for adrenochrome.", False),
        ("hoax_area51_1", "Area 51 contains captured extraterrestrials and hidden alien technology.", False),
        ("hoax_lizard_1", "World leaders are shape-shifting lizard people.", False),
        ("hoax_holocaust_1", "The Holocaust never happened.", False),
        ("hoax_climate_1", "Climate change is a hoax invented by scientists for funding.", False),
        ("hoax_moon_rocks_1", "Moon rocks are fake and were manufactured on Earth to deceive the public.", False),
        ("hoax_antivax_1", "Vaccines are designed to sterilize people as part of a population control plan.", False),
        ("hoax_ancient_aliens_1", "The pyramids were built by aliens because humans could not build them.", False),
        ("hoax_birds_1", "Birds are government drones used for surveillance.", False),
        ("hoax_dinosaurs_1", "Dinosaurs never existed; fossils were planted by scientists.", False),
        ("hoax_mindcontrol_1", "HAARP is used to control human minds through radio waves.", False),
        ("hoax_wtc_1", "All evidence proves the World Trade Center attacks were a controlled demolition by the government.", False),
        ("hoax_medicine_1", "Drinking bleach cures viral infections and governments hide this cure.", False),
        ("hoax_space_1", "Space does not exist; it is an illusion created by governments.", False),
        ("hoax_finance_1", "A secret global banking family controls every government and all wars.", False),
    ]
    cases = [
        ClaimCase(cid, "WEB_SEARCH", text, gold, ["hoax", "conspiracy"])
        for (cid, text, gold) in hoax_claims
    ]
    return cases


def general_suite() -> List[ClaimCase]:
    # Mixed suite for rough signal. (WEB_SEARCH is inherently non-deterministic.)
    # Keep claims simple and unambiguous to reduce evaluation ambiguity.
    cases = [
        # Likely TRUE
        ClaimCase("gen_true_earth_sun", "WEB_SEARCH", "The Earth orbits the Sun.", True, ["general", "science"]),
        ClaimCase("gen_true_water", "WEB_SEARCH", "Water freezes at 0 degrees Celsius at standard atmospheric pressure.", True, ["general", "science"]),
        ClaimCase("gen_true_olympics_2024", "WEB_SEARCH", "The 2024 Summer Olympics were held in Paris.", True, ["general", "sports"]),
        # Likely FALSE
        ClaimCase("gen_false_olympics_tokyo_2024", "WEB_SEARCH", "The 2024 Summer Olympics were held in Tokyo.", False, ["general", "sports"]),
        ClaimCase("gen_false_olympics_la_2024", "WEB_SEARCH", "The 2024 Summer Olympics were held in Los Angeles.", False, ["general", "sports"]),
        ClaimCase("gen_false_speed_sound", "WEB_SEARCH", "The speed of sound in air at room temperature is about 3,000 meters per second.", False, ["general", "science"]),
        # PYTHON_EXEC mixed
        ClaimCase("gen_py_arith_true", "PYTHON_EXEC", "123 + 456 equals 579", True, ["general", "arith"]),
        ClaimCase("gen_py_arith_false", "PYTHON_EXEC", "123 + 456 equals 580", False, ["general", "arith"]),
    ]
    return cases


def build_suite(which: str) -> List[ClaimCase]:
    suites = {
        "sanity": sanity_suite,
        "hoax": hoax_suite,
        "general": general_suite,
        "all": lambda: sanity_suite() + hoax_suite() + general_suite(),
    }
    if which not in suites:
        raise ValueError(f"Unknown suite '{which}'. Choose from: {list(suites.keys())}")
    return suites[which]()


# ----------------------------
# Runner
# ----------------------------

def run_case(case: ClaimCase, quiet: bool = True) -> CaseResult:
    t0 = time.time()
    err = None
    try:
        # verify_claim prints logs; if you want silence, redirect stdout outside (not done here).
        pred_raw = RealToolkit.verify_claim(case.tool, case.claim)
        pred = bool(pred_raw) if pred_raw is not None else None
    except Exception as e:
        pred = None
        err = repr(e)
    dt = time.time() - t0
    ok = (pred == case.gold) if pred is not None else False
    if not quiet:
        print(f"[{case.id}] tool={case.tool} gold={case.gold} pred={pred} ok={ok} ({dt:.2f}s)")
    return CaseResult(case=case, pred=pred, ok=ok, latency_s=dt, error=err)


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    if not results:
        return {"n": 0}

    n = len(results)
    ok = sum(1 for r in results if r.ok)
    acc = ok / n

    lat = [r.latency_s for r in results]
    lat_mean = statistics.mean(lat) if lat else 0.0
    lat_p50 = statistics.median(lat) if lat else 0.0
    lat_p90 = sorted(lat)[int(0.9 * (len(lat) - 1))] if len(lat) >= 2 else (lat[0] if lat else 0.0)

    by_tool: Dict[str, Dict[str, Any]] = {}
    for tool in sorted(set(r.case.tool for r in results)):
        rs = [r for r in results if r.case.tool == tool]
        t_ok = sum(1 for r in rs if r.ok)
        by_tool[tool] = {
            "n": len(rs),
            "acc": (t_ok / len(rs)) if rs else 0.0,
            "lat_mean": statistics.mean([r.latency_s for r in rs]) if rs else 0.0,
        }

    # Hoax suite FPR: gold=False but pred=True
    hoax = [r for r in results if "hoax" in (r.case.tags or [])]
    hoax_n = len(hoax)
    hoax_fp = sum(1 for r in hoax if (r.case.gold is False and r.pred is True))
    hoax_fpr = (hoax_fp / hoax_n) if hoax_n else None

    return {
        "n": n,
        "accuracy": acc,
        "lat_mean_s": lat_mean,
        "lat_p50_s": lat_p50,
        "lat_p90_s": lat_p90,
        "by_tool": by_tool,
        "hoax_n": hoax_n,
        "hoax_false_positive": hoax_fp,
        "hoax_fpr": hoax_fpr,
    }


def print_failures(results: List[CaseResult], max_show: int = 50) -> None:
    fails = [r for r in results if not r.ok]
    if not fails:
        print("\n✅ No failing cases.")
        return

    print(f"\n❌ Failing cases ({len(fails)}):")
    for i, r in enumerate(fails[:max_show], start=1):
        c = r.case
        print("-" * 80)
        print(f"{i}) id={c.id} tool={c.tool} tags={c.tags}")
        print(f"   gold={c.gold} pred={r.pred} latency={r.latency_s:.2f}s error={r.error}")
        print(f"   claim: {c.claim}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=str, default="all", choices=["sanity", "hoax", "general", "all"])
    ap.add_argument("--max", type=int, default=999, help="limit number of cases")
    ap.add_argument("--shuffle", action="store_true", help="shuffle cases (useful to mix WEB_SEARCH load)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true", help="less per-case prints")
    args = ap.parse_args()

    cases = build_suite(args.suite)
    if args.shuffle:
        import random
        random.seed(args.seed)
        random.shuffle(cases)

    cases = cases[: max(0, args.max)]

    print(f"\n=== RealToolkit Benchmark ===")
    print(f"Suite: {args.suite} | Cases: {len(cases)}")
    print(f"TRUE gate (final_conf): {RealToolkit.JUDGE_TRUE_MIN_FINAL_CONF} | FALSE gate: {RealToolkit.JUDGE_FALSE_MIN_FINAL_CONF}")
    print(f"Edge prune FALSE conf: {RealToolkit.EDGE_PRUNE_FALSE_CONF}")
    print("=" * 32)

    results: List[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        if not args.quiet:
            print(f"\n[{idx}/{len(cases)}] {case.id} ({case.tool}) gold={case.gold}")
        r = run_case(case, quiet=args.quiet)
        results.append(r)

    s = summarize(results)
    print("\n=== Summary ===")
    print(json.dumps(s, indent=2, ensure_ascii=False))

    print_failures(results, max_show=50)

    # Exit code: non-zero if any fail (CI-friendly)
    any_fail = any(not r.ok for r in results)
    if any_fail:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
