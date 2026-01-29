#!/usr/bin/env python3
"""
run_eval.py
Main evaluation script for running MaVERiC on datasets.

Key patch (TruthfulQA):
- TruthfulQA is NOT a binary True/False task on the question.
- We evaluate pairwise: truthful_answer vs false_answer.
- Predicted_label=True means we picked the truthful answer (gold_label always True in this pairwise setup).

Key patch (Cache):
- Prefer DebateGraph-v1 cache:
    cache/debate_graphs/<dataset>/DebateGraph-v1.jsonl
- If cached graph exists for statement_id, load it.
- If missing, generate+parse; optionally append to cache for future runs.

Usage:
  python run_eval.py --method maveric --dataset truthfulqa --budget 8 --max_samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.core.graph import ArgumentationGraph, ArgumentNode  # noqa: E402
from src.core.solver import MaVERiCSolver  # noqa: E402
from src.baselines.selector_ablation import (  # noqa: E402
    D1RandomRefineSolver,
    D2UncertaintyRefineSolver,
    D3CentralityRefineSolver,
    D4DistanceRefineSolver,
    D5ProxyOnlyRefineSolver,
)
from src.baselines.class_e import (  # noqa: E402
    MAVBaseline,
    BoNMAVBaseline,
    MADFactBaseline,
    GKMADBaseline,
)
from src.baselines.class_f import (  # noqa: E402
    F1MADAdversaryFilterBaseline,
    F2EvidenceRequirementBaseline,
    F3GraphConsistencyGatingSolver,
)
from src.core.baselines import _build_pseudo_transcript_from_graph  # noqa: E402
from dataset_loader import load_dataset_by_name  # noqa: E402

from src.agents.debater import generate_debate  # noqa: E402
from src.agents.parser import parse_debate  # noqa: E402


# -------------------------
# Cache helpers
# -------------------------
def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()

def _load_jsonl_as_index(path: str, key_field: str = "statement_id") -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return idx
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                k = str(rec.get(key_field, "")).strip()
                if k:
                    idx[k] = rec
            except Exception:
                continue
    return idx

def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def record_to_graph(rec: Dict[str, Any]) -> ArgumentationGraph:
    g = ArgumentationGraph()

    for nd in rec.get("nodes", []) or []:
        node = ArgumentNode(
            id=str(nd.get("id")),
            content=str(nd.get("content", "")),
            speaker=str(nd.get("speaker", "UNK")),
            is_verified=False,
            ground_truth=None,
            verification_cost=float(nd.get("verification_cost", 1.0) or 1.0),
            tool_type=str(nd.get("tool_type", "AUTO")),
        )
        g.add_node(node)

    for ed in rec.get("edges", []) or []:
        u = str(ed.get("u"))
        v = str(ed.get("v"))
        t = str(ed.get("type", "attack"))
        if hasattr(g, "nx_graph") and g.nx_graph is not None:
            g.nx_graph.add_edge(u, v, type=t)

    root_id = rec.get("root_id")
    if root_id:
        g.root_id_override = str(root_id)

    return g

def graph_to_record(
    g: ArgumentationGraph,
    dataset: str,
    sample_id: str,
    statement_id: str,
    pair_role: Optional[str],
    statement: str,
    debate_text: str,
) -> Dict[str, Any]:
    nodes_out: List[Dict[str, Any]] = []
    for nid, n in getattr(g, "nodes", {}).items():
        nodes_out.append(
            {
                "id": str(nid),
                "content": str(getattr(n, "content", "")),
                "speaker": str(getattr(n, "speaker", "UNK")),
                "tool_type": str(getattr(n, "tool_type", "AUTO")),
                "verification_cost": float(getattr(n, "verification_cost", 1.0) or 1.0),
            }
        )

    edges_out: List[Dict[str, Any]] = []
    nxg = getattr(g, "nx_graph", None)
    if nxg is not None:
        for u, v, d in nxg.edges(data=True):
            edges_out.append({"u": str(u), "v": str(v), "type": str((d or {}).get("type", "attack"))})

    root_id = None
    if hasattr(g, "find_semantic_root"):
        try:
            root_id = g.find_semantic_root()
        except Exception:
            root_id = None

    return {
        "version": "DebateGraph-v1",
        "dataset": dataset,
        "sample_id": sample_id,
        "statement_id": statement_id,
        "pair_role": pair_role,
        "statement": statement,
        "conversation_ref": {
            "version": "DebateConversation-v1",
            "statement_id": statement_id,
            "sha1_debate_text": _sha1(debate_text),
        },
        "parse_meta": {"parser": "src.agents.parser.parse_debate", "timestamp": _now_iso()},
        "root_id": root_id,
        "nodes": nodes_out,
        "edges": edges_out,
        "atomic_claims": [{"id": x["id"], "text": x["content"]} for x in nodes_out],
    }


# -------------------------
# Graph construction (with cache)
# -------------------------
def _ensure_statement_root(g: ArgumentationGraph, claim_id: str, claim: str) -> None:
    root_id = f"{claim_id}_root"
    if root_id not in g.nodes:
        g.add_node(
            ArgumentNode(
                id=root_id,
                content=f"Claim: {claim}",
                speaker="Claimer",
                is_verified=False,
                ground_truth=None,
                verification_cost=1.0,
                tool_type="AUTO",
            )
        )
    g.root_id_override = root_id
    g.claim = claim


def create_debate_graph_from_claim(
    claim: str,
    claim_id: str,
    dataset_name: str,
    graph_cache: Dict[str, Dict[str, Any]],
    use_graph_cache: bool,
    append_cache_on_miss: bool,
    graph_cache_path: str,
    pair_role: Optional[str] = None,
) -> ArgumentationGraph:
    """
    Create a debate graph by:
      1) loading cached DebateGraph-v1 if available
      2) else generate_debate + parse_debate
      3) fallback to single root node if generation/parsing fails
    """
    if use_graph_cache and (claim_id in graph_cache):
        g_cached = record_to_graph(graph_cache[claim_id])
        _ensure_statement_root(g_cached, claim_id, claim)
        return g_cached

    debate_text = generate_debate(topic=claim, num_liars=2)

    if not debate_text:
        g = ArgumentationGraph()
        _ensure_statement_root(g, claim_id, claim)
        return g

    try:
        g = parse_debate(debate_text)
    except Exception:
        g = None

    if g is None:
        g = ArgumentationGraph()

    _ensure_statement_root(g, claim_id, claim)

    # optionally append to cache
    if append_cache_on_miss and graph_cache_path:
        rec = graph_to_record(
            g=g,
            dataset=dataset_name,
            sample_id=str(claim_id.split("_")[0]) if "_" in claim_id else claim_id,
            statement_id=claim_id,
            pair_role=pair_role,
            statement=claim,
            debate_text=debate_text,
        )
        _append_jsonl(graph_cache_path, rec)
        graph_cache[claim_id] = rec

    return g


def _safe_get_dataset_name(sample: Dict[str, Any], fallback: str) -> str:
    for k in ("dataset", "dataset_name", "source"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return (fallback or "").strip().lower()


# -------------------------
# TruthfulQA helpers
# -------------------------
def truthfulqa_extract_pair(sample: Dict[str, Any]) -> Tuple[str, str]:
    truth = (
        sample.get("truthful_statement")
        or sample.get("best_answer")
        or sample.get("correct_answer")
        or sample.get("truth")
        or sample.get("answer_true")
        or sample.get("truthful_answer")
    )
    false = (
        sample.get("false_statement")
        or sample.get("incorrect_answer")
        or sample.get("false_answer")
        or sample.get("myth_answer")
        or sample.get("answer_false")
        or sample.get("false_answer_text")
        or sample.get("incorrect_answer_text")
    )
    if not truth or not false:
        keys = ", ".join(sorted(sample.keys()))
        raise ValueError(
            "TruthfulQA requires a truthful answer AND a false answer for pairwise evaluation.\n"
            "Expected keys like: best_answer + incorrect_answer (or similar).\n"
            f"Got keys: {keys}"
        )
    return str(truth).strip(), str(false).strip()


# -------------------------
# Solver runner
# -------------------------
def _apply_tool_costs(graph: ArgumentationGraph, tool_costs: Optional[Dict[str, float]]) -> None:
    if not tool_costs:
        return

    for node in graph.nodes.values():
        tool = str(getattr(node, "tool_type", "AUTO") or "AUTO").upper()
        if tool in tool_costs:
            node.verification_cost = float(tool_costs[tool])


def _build_solver(
    baseline: str,
    graph: ArgumentationGraph,
    budget: float,
    tool_costs: Optional[Dict[str, float]] = None,
):
    key = str(baseline or "maveric").lower().strip()
    if key in {"maveric", "default", ""}:
        return MaVERiCSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d1_random", "d1"}:
        return D1RandomRefineSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d2_uncertainty", "d2"}:
        return D2UncertaintyRefineSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d3_centrality", "d3", "d3_pagerank"}:
        return D3CentralityRefineSolver(
            graph=graph,
            budget=budget,
            kind="pagerank",
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d3_degree"}:
        return D3CentralityRefineSolver(
            graph=graph,
            budget=budget,
            kind="degree",
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d3_betweenness"}:
        return D3CentralityRefineSolver(
            graph=graph,
            budget=budget,
            kind="betweenness",
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d4_distance", "d4"}:
        return D4DistanceRefineSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"d5_proxy", "d5", "d5_proxy_only", "proxy_only"}:
        return D5ProxyOnlyRefineSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )
    if key in {"f3_graph_gate", "f3_graph_consistency", "f3"}:
        return F3GraphConsistencyGatingSolver(
            graph=graph,
            budget=budget,
            topk_counterfactual=8,
            delta_support_to_root=0.8,
            tool_costs=tool_costs,
        )

    raise ValueError(f"Unknown baseline: {baseline}")


def _build_class_e_baseline(baseline: str):
    key = str(baseline or "").lower().strip()
    if key in {"e1_mav", "mav", "mav_baseline"}:
        return MAVBaseline()
    if key in {"e2_bon_mav", "bon_mav", "bon-mav"}:
        return BoNMAVBaseline()
    if key in {"e3_mad_fact", "mad_fact", "mad-fact"}:
        return MADFactBaseline()
    if key in {"e4_gkmad", "gkmad"}:
        return GKMADBaseline()
    return None


def _build_class_f_baseline(baseline: str):
    key = str(baseline or "").lower().strip()
    if key in {"f1_mad_filter", "f1"}:
        return F1MADAdversaryFilterBaseline()
    if key in {"f2_judge_evidence", "f2"}:
        return F2EvidenceRequirementBaseline()
    return None


def run_solver_on_statement(
    statement: str,
    statement_id: str,
    budget: float,
    dataset_name: str,
    graph_cache: Dict[str, Dict[str, Any]],
    use_graph_cache: bool,
    append_cache_on_miss: bool,
    graph_cache_path: str,
    verbose: bool = False,
    pair_role: Optional[str] = None,
    tool_costs: Optional[Dict[str, float]] = None,
    baseline: str = "maveric",
) -> Dict[str, Any]:
    if verbose:
        print(f"    🧩 Building graph for: {statement[:120]}... (id={statement_id})")

    baseline_obj = _build_class_e_baseline(baseline)
    if baseline_obj is not None:
        start = time.time()
        verdict = baseline_obj.verify(
            claim=statement,
            transcript=None,
            budget=float(budget),
        )
        budget_used = float(getattr(baseline_obj, "budget_spent", 0.0) or 0.0)
        tool_calls_total = int(getattr(baseline_obj, "tool_calls", 0) or 0)
        baseline_metrics = getattr(baseline_obj, "stats", None)
        elapsed = time.time() - start

        vt = 1 if verdict is True else 0
        vf = 1 if verdict is False else 0

        return {
            "statement": statement,
            "verdict": None if verdict is None else bool(verdict),
            "y_direct": None,
            "root_in_ext": False,
            "unverified": verdict is None,
            "verified_true": int(vt),
            "verified_false": int(vf),
            "score": 0.0,
            "final_ext_size": 0,
            "budget_used": round(float(budget_used), 2),
            "budget_left": round(max(0.0, float(budget) - float(budget_used)), 2),
            "tool_calls_total": int(tool_calls_total),
            "baseline_metrics": baseline_metrics,
            "pruned": 0,
            "edges_removed": 0,
            "edges_removed_false_refine": 0,
            "edges_removed_pruned": 0,
            "error": None,
            "runtime_s": round(elapsed, 2),
        }

    graph = create_debate_graph_from_claim(
        claim=statement,
        claim_id=statement_id,
        dataset_name=dataset_name,
        graph_cache=graph_cache,
        use_graph_cache=use_graph_cache,
        append_cache_on_miss=append_cache_on_miss,
        graph_cache_path=graph_cache_path,
        pair_role=pair_role,
    )
    _apply_tool_costs(graph, tool_costs)

    baseline_f = _build_class_f_baseline(baseline)
    if baseline_f is not None:
        start = time.time()
        transcript = _build_pseudo_transcript_from_graph(graph)
        verdict = baseline_f.verify(
            claim=statement,
            transcript=transcript,
            graph=graph,
            budget=float(budget),
        )
        budget_used = float(getattr(baseline_f, "budget_spent", 0.0) or 0.0)
        tool_calls_total = int(getattr(baseline_f, "tool_calls", 0) or 0)
        baseline_metrics = getattr(baseline_f, "stats", None)
        elapsed = time.time() - start

        vt = 1 if verdict is True else 0
        vf = 1 if verdict is False else 0

        return {
            "statement": statement,
            "verdict": None if verdict is None else bool(verdict),
            "y_direct": None,
            "root_in_ext": False,
            "unverified": verdict is None,
            "verified_true": int(vt),
            "verified_false": int(vf),
            "score": 0.0,
            "final_ext_size": 0,
            "budget_used": round(float(budget_used), 2),
            "budget_left": round(max(0.0, float(budget) - float(budget_used)), 2),
            "tool_calls_total": int(tool_calls_total),
            "baseline_metrics": baseline_metrics,
            "pruned": 0,
            "edges_removed": 0,
            "edges_removed_false_refine": 0,
            "edges_removed_pruned": 0,
            "error": None,
            "runtime_s": round(elapsed, 2),
        }

    solver = _build_solver(
        baseline=baseline,
        graph=graph,
        budget=budget,
        tool_costs=tool_costs,
    )

    try:
        final_ext, verdict = solver.run()

        y_direct = getattr(solver, "y_direct", None)
        root_id = getattr(solver, "root_id", None)
        root_in_ext = bool(root_id and root_id in final_ext)
        unverified = verdict is None or bool(getattr(solver, "verify_error", False))

        vt = len(getattr(solver, "verified_true_ids", set()) or set())
        vf = len(getattr(solver, "verified_false_ids", set()) or set())

        score_fn = getattr(solver, "pairwise_score", None)
        score_val = score_fn(final_ext) if callable(score_fn) else 0.0
        score = float(score_val) if isinstance(score_val, (int, float)) else 0.0

        # NEW: refinement stats if solver exposes them
        pruned = int(getattr(solver, "pruned_count", 0) or 0)
        edges_removed = int(getattr(solver, "edges_removed_count", 0) or 0)
        edges_removed_false_refine = int(getattr(solver, "edges_removed_false_refine_count", 0) or 0)
        edges_removed_pruned = int(getattr(solver, "edges_removed_prune_count", 0) or 0)

        error = None

    except Exception as e:
        final_ext, verdict = set(), None
        y_direct, root_in_ext, vt, vf, score = None, False, 0, 0, -1e9
        pruned, edges_removed, edges_removed_false_refine, edges_removed_pruned = 0, 0, 0, 0
        unverified = True
        error = str(e)

    budget_left = float(getattr(solver, "budget", 0.0) or 0.0)
    budget_used = max(0.0, float(budget) - budget_left)

    tool_calls_total = int(getattr(solver, "tool_calls", 0) or 0)

    return {
        "statement": statement,
        "verdict": None if verdict is None else bool(verdict),
        "y_direct": y_direct,
        "root_in_ext": bool(root_in_ext),
        "unverified": bool(unverified),
        "verified_true": int(vt),
        "verified_false": int(vf),
        "score": float(score),
        "final_ext_size": int(len(final_ext)),
        "budget_used": round(budget_used, 2),
        "budget_left": round(budget_left, 2),
        "tool_calls_total": tool_calls_total,
        "pruned": int(pruned),
        "edges_removed": int(edges_removed),
        "edges_removed_false_refine": int(edges_removed_false_refine),
        "edges_removed_pruned": int(edges_removed_pruned),
        "error": error,
    }


# -------------------------
# Pairwise decision (robust)
# -------------------------
import math
from typing import Dict as _Dict, Any as _Any

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _tanh_scale(x: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return math.tanh(float(x) / float(scale))

def _robust_conf_proxy(res: _Dict[str, _Any]) -> float:
    if res.get("error"):
        return 0.0
    y_direct = res.get("y_direct", None)
    verdict = bool(res.get("verdict", False))
    vt = int(res.get("verified_true", 0) or 0)
    vf = int(res.get("verified_false", 0) or 0)
    net = vt - vf
    cov = vt + vf
    root_in_ext = bool(res.get("root_in_ext", False))
    tool_calls = float(res.get("tool_calls_total", 0) or 0)
    budget_used = float(res.get("budget_used", 0.0) or 0.0)

    conf = 0.50
    if y_direct is True:
        conf += 0.25
    elif y_direct is False:
        conf -= 0.35

    conf += 0.10 if verdict else -0.10
    conf += 0.18 * _tanh_scale(net, scale=3.0)
    conf += 0.10 * _tanh_scale(cov, scale=6.0)
    if root_in_ext:
        conf += 0.05
    conf -= 0.06 * _tanh_scale(tool_calls, scale=10.0)
    conf -= 0.06 * _tanh_scale(budget_used, scale=10.0)
    return _clamp(conf, 0.0, 1.0)

def _rank_tuple(res: _Dict[str, _Any]) -> tuple:
    has_error = 1 if res.get("error") else 0
    y_direct = res.get("y_direct", None)
    if y_direct is True:
        ycat = 2
    elif y_direct is False:
        ycat = 0
    else:
        ycat = 1

    conf = _robust_conf_proxy(res)
    vt = int(res.get("verified_true", 0) or 0)
    vf = int(res.get("verified_false", 0) or 0)
    net = vt - vf
    tool_calls = float(res.get("tool_calls_total", 0) or 0)
    budget_used = float(res.get("budget_used", 0.0) or 0.0)
    ext = int(res.get("final_ext_size", 0) or 0)

    return (-has_error, ycat, conf, net, -tool_calls, -budget_used, -ext)

def truthfulqa_decide_pair_robust(truth_res: _Dict[str, _Any], false_res: _Dict[str, _Any]) -> Optional[bool]:
    if bool(truth_res.get("unverified")) and bool(false_res.get("unverified")):
        return None

    rt = _rank_tuple(truth_res)
    rf = _rank_tuple(false_res)
    if rt > rf:
        return True
    if rf > rt:
        return False
    t_calls = int(truth_res.get("tool_calls_total", 0) or 0)
    f_calls = int(false_res.get("tool_calls_total", 0) or 0)
    if t_calls != f_calls:
        return t_calls < f_calls
    return True


# -------------------------
# Main sample runner
# -------------------------
def run_maveric_on_sample(
    sample: Dict[str, Any],
    budget: float,
    dataset_name: str,
    graph_cache: Dict[str, Dict[str, Any]],
    use_graph_cache: bool,
    append_cache_on_miss: bool,
    graph_cache_path: str,
    verbose: bool = False,
    tool_costs: Optional[Dict[str, float]] = None,
    baseline: str = "maveric",
) -> Dict[str, Any]:
    sample_id = str(sample.get("id", "unknown"))
    ds = _safe_get_dataset_name(sample, dataset_name)

    start_time = time.time()

    if ds == "truthfulqa":
        question = str(sample.get("claim") or sample.get("question") or "").strip()
        truth_stmt, false_stmt = truthfulqa_extract_pair(sample)

        if verbose:
            print(f"\n[{sample_id}] TruthfulQA (pairwise)")
            print(f"  Q: {question[:120]}")
            print(f"  T: {truth_stmt[:120]}")
            print(f"  F: {false_stmt[:120]}")

        b_each = max(0.5, budget / 2.0)

        truth_res = run_solver_on_statement(
            statement=truth_stmt,
            statement_id=sample_id + "_T",
            budget=b_each,
            dataset_name=ds,
            graph_cache=graph_cache,
            use_graph_cache=use_graph_cache,
            append_cache_on_miss=append_cache_on_miss,
            graph_cache_path=graph_cache_path,
            verbose=verbose,
            pair_role="truth",
            tool_costs=tool_costs,
            baseline=baseline,
        )
        false_res = run_solver_on_statement(
            statement=false_stmt,
            statement_id=sample_id + "_F",
            budget=b_each,
            dataset_name=ds,
            graph_cache=graph_cache,
            use_graph_cache=use_graph_cache,
            append_cache_on_miss=append_cache_on_miss,
            graph_cache_path=graph_cache_path,
            verbose=verbose,
            pair_role="false",
            tool_costs=tool_costs,
            baseline=baseline,
        )

        predicted_label = truthfulqa_decide_pair_robust(truth_res, false_res)
        gold_label = True
        unverified_pair = predicted_label is None

        runtime_s = time.time() - start_time
        truth_ext = int(truth_res.get("final_ext_size", 0) or 0)
        false_ext = int(false_res.get("final_ext_size", 0) or 0)

        return {
            "sample_id": sample_id,
            "dataset": ds,
            "baseline": baseline,
            "claim": question,
            "truth_statement": truth_stmt,
            "false_statement": false_stmt,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "unverified": bool(unverified_pair),
            "correct": (predicted_label == gold_label) if predicted_label is not None else False,
            "budget_used": round(float(truth_res["budget_used"]) + float(false_res["budget_used"]), 2),
            "tool_calls": {"TOTAL": int(truth_res["tool_calls_total"]) + int(false_res["tool_calls_total"])},
            "refinement_stats": {
                "pruned": int(truth_res.get("pruned", 0)) + int(false_res.get("pruned", 0)),
                "edges_removed": int(truth_res.get("edges_removed", 0)) + int(false_res.get("edges_removed", 0)),
                "edges_removed_false_refine": int(truth_res.get("edges_removed_false_refine", 0))
                + int(false_res.get("edges_removed_false_refine", 0)),
                "edges_removed_pruned": int(truth_res.get("edges_removed_pruned", 0)) + int(false_res.get("edges_removed_pruned", 0)),
                "sgs_size": max(truth_ext, false_ext),
                "truth_final_ext_size": truth_ext,
                "false_final_ext_size": false_ext,
            },
            "runtime_s": round(runtime_s, 2),
            "subruns": {"truth": truth_res, "false": false_res},
        }

    # Default binary
    claim = str(sample.get("claim") or sample.get("question") or "").strip()
    gold_label = bool(sample.get("label", False))

    if verbose:
        print(f"\n[{sample_id}] {ds} (binary)")
        print(f"  Claim: {claim[:140]}")
        print(f"  Gold: {gold_label}")

    graph = create_debate_graph_from_claim(
        claim=claim,
        claim_id=sample_id,
        dataset_name=ds,
        graph_cache=graph_cache,
        use_graph_cache=use_graph_cache,
        append_cache_on_miss=append_cache_on_miss,
        graph_cache_path=graph_cache_path,
        pair_role=None,
    )
    _apply_tool_costs(graph, tool_costs)

    solver = _build_solver(
        baseline=baseline,
        graph=graph,
        budget=budget,
        tool_costs=tool_costs,
    )

    final_ext, verdict = set(), False
    error = None

    try:
        final_ext, verdict = solver.run()
        predicted_label = None if verdict is None else bool(verdict)
        if getattr(solver, "verify_error", False):
            predicted_label = None
    except Exception as e:
        predicted_label = None
        final_ext = set()
        error = str(e)

    runtime_s = time.time() - start_time
    budget_left = getattr(solver, "budget", 0.0)
    budget_used = max(0.0, budget - float(budget_left))
    tool_calls = {"TOTAL": int(getattr(solver, "tool_calls", 0) or 0)}

    refinement_stats = {
        "pruned": int(getattr(solver, "pruned_count", 0) or 0),
        "sgs_size": int(len(final_ext)),
        "edges_removed": int(getattr(solver, "edges_removed_count", 0) or 0),
        "edges_removed_false_refine": int(getattr(solver, "edges_removed_false_refine_count", 0) or 0),
        "edges_removed_pruned": int(getattr(solver, "edges_removed_prune_count", 0) or 0),
    }

    result = {
        "sample_id": sample_id,
        "dataset": ds,
        "baseline": baseline,
        "claim": claim,
        "gold_label": gold_label,
        "predicted_label": predicted_label,
        "unverified": predicted_label is None,
        "correct": (predicted_label == gold_label) if predicted_label is not None else False,
        "budget_used": round(budget_used, 2),
        "tool_calls": tool_calls,
        "refinement_stats": refinement_stats,
        "runtime_s": round(runtime_s, 2),
    }
    if error:
        result["error"] = error

    return result


# -------------------------
# Main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluation on MaVERiC")
    parser.add_argument("--method", type=str, default="maveric", help="Method name")
    parser.add_argument(
        "--baseline",
        type=str,
        default="maveric",
        help=(
            "Baseline: maveric, d1_random, d2_uncertainty, d3_pagerank, d3_degree, "
            "d3_betweenness, d4_distance, d5_proxy, e1_mav, e2_bon_mav, e3_mad_fact, "
            "e4_gkmad, f1_mad_filter, f2_judge_evidence, f3_graph_gate"
        ),
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--verbose", action="store_true")

    # Cache args
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--use_graph_cache", action="store_true", help="Use DebateGraph-v1 cache if present")
    parser.add_argument("--append_cache_on_miss", action="store_true", help="Append graph to cache if missing")
    parser.add_argument("--tool_costs", type=str, default="", help="JSON string to override tool costs")

    args = parser.parse_args()

    ds_name = str(args.dataset).lower().strip()
    graph_cache_path = os.path.join(args.cache_dir, "debate_graphs", ds_name, "DebateGraph-v1.jsonl")

    graph_cache: Dict[str, Dict[str, Any]] = {}
    if args.use_graph_cache and os.path.exists(graph_cache_path):
        graph_cache = _load_jsonl_as_index(graph_cache_path, key_field="statement_id")

    print("=" * 70)
    print("MaVERiC Evaluation")
    print("=" * 70)
    print(f"Method: {args.method}")
    print(f"Baseline: {args.baseline}")
    print(f"Dataset: {args.dataset}")
    print(f"Budget: {args.budget}")
    print(f"Max samples: {args.max_samples}")
    print(f"Graph cache: {'ON' if args.use_graph_cache else 'OFF'} | path={graph_cache_path} | loaded={len(graph_cache)}")
    print("=" * 70)

    samples = load_dataset_by_name(args.dataset, max_samples=args.max_samples)

    tool_costs = None
    if args.tool_costs:
        try:
            tool_costs = json.loads(args.tool_costs)
        except Exception:
            print("⚠️ Invalid --tool_costs JSON; ignoring override")
            tool_costs = None
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    print(f"\nLoaded {len(samples)} samples\n")

    results: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, 1):
        sid = str(sample.get("id", f"sample_{idx:04d}"))
        print(f"[{idx}/{len(samples)}] Processing {sid}...")

        result = run_maveric_on_sample(
            sample=sample,
            budget=float(args.budget),
            dataset_name=ds_name,
            graph_cache=graph_cache,
            use_graph_cache=bool(args.use_graph_cache),
            append_cache_on_miss=bool(args.append_cache_on_miss),
            graph_cache_path=graph_cache_path,
            verbose=bool(args.verbose),
            tool_costs=tool_costs,
            baseline=str(args.baseline),
        )
        results.append(result)

        verified = [r for r in results if not r.get("unverified")]
        correct_so_far = sum(1 for r in verified if bool(r.get("correct")))
        accuracy_so_far = correct_so_far / max(1, len(verified))
        print(f"  Accuracy so far: {accuracy_so_far:.2%} ({correct_so_far}/{len(verified)})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{args.method}_{args.dataset}.jsonl"
    output_path = os.path.join(args.output_dir, output_filename)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    verified = [r for r in results if not r.get("unverified")]
    final_acc = sum(1 for r in verified if bool(r.get("correct"))) / max(1, len(verified))

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"Results saved to: {output_path}")
    print(f"\nFinal Accuracy: {final_acc:.2%}")
    print(f"Total samples: {len(results)} (verified: {len(verified)})")
    
    # --- Auto-Summarize ---
    print("\n" + "=" * 70)
    print("Generating Summary Report...")
    print("=" * 70)
    
    try:
        from summarize import print_summary
        import contextlib
        from io import StringIO
        
        # 1. Print to console
        print_summary(results, output_path)
        
        # 2. Write to summary file
        summary_path = output_path.replace(".jsonl", "_summary.txt")
        
        # Capture the output of print_summary
        buf = StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary(results, output_path)
        summary_text = buf.getvalue()
        
        with open(summary_path, "w", encoding="utf-8") as f_sum:
            f_sum.write(summary_text)
            
        print(f"\n✅ Detailed summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"⚠️ Failed to auto-summarize: {e}")
        print("You can try running summarize.py manually.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
