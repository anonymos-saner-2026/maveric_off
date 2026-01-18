#!/usr/bin/env python3
"""
build_debate_graph_dataset.py

Build 2-stage caches:
1) DebateConversation-v1 (raw debate text)
2) DebateGraph-v1 (parsed ArgumentationGraph: nodes/edges/root/atomic claims)

Usage:
  python build_debate_graph_dataset.py --dataset truthfulqa --max_samples 50 --budget_hint 8 \
    --cache_dir cache --num_liars 2 --seed 3407

Notes:
- This script depends on:
  - dataset_loader.load_dataset_by_name
  - src.agents.debater.generate_debate
  - src.agents.parser.parse_debate
  - src.core.graph.ArgumentationGraph, ArgumentNode
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dataset_loader import load_dataset_by_name  # noqa: E402
from src.agents.debater import generate_debate  # noqa: E402
from src.agents.parser import parse_debate      # noqa: E402
from src.core.graph import ArgumentationGraph, ArgumentNode  # noqa: E402


# -------------------------
# Helpers: TruthfulQA pair extraction
# -------------------------
def _safe_get_dataset_name(sample: Dict[str, Any], fallback: str) -> str:
    for k in ("dataset", "dataset_name", "source"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().lower()
    return (fallback or "").strip().lower()

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

def _now_iso() -> str:
    # timezone-naive is OK, but we keep ISO for readability
    return datetime.now().isoformat(timespec="seconds")

def _sha1(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


# -------------------------
# Graph serialization / deserialization
# -------------------------
def graph_to_record(
    g: ArgumentationGraph,
    dataset: str,
    sample_id: str,
    statement_id: str,
    pair_role: Optional[str],
    statement: str,
    debate_sha1: str,
) -> Dict[str, Any]:
    # nodes
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

    # edges
    edges_out: List[Dict[str, Any]] = []
    nxg = getattr(g, "nx_graph", None)
    if nxg is not None:
        for u, v, d in nxg.edges(data=True):
            edges_out.append({"u": str(u), "v": str(v), "type": str((d or {}).get("type", "attack"))})

    # root
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
            "sha1_debate_text": debate_sha1,
        },
        "parse_meta": {"parser": "src.agents.parser.parse_debate", "timestamp": _now_iso()},
        "root_id": root_id,
        "nodes": nodes_out,
        "edges": edges_out,
        "atomic_claims": [{"id": x["id"], "text": x["content"]} for x in nodes_out],
    }

def record_to_graph(rec: Dict[str, Any]) -> ArgumentationGraph:
    g = ArgumentationGraph()

    # add nodes
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

    # add edges
    for ed in rec.get("edges", []) or []:
        u = str(ed.get("u"))
        v = str(ed.get("v"))
        t = str(ed.get("type", "attack"))
        if hasattr(g, "nx_graph") and g.nx_graph is not None:
            g.nx_graph.add_edge(u, v, type=t)

    # provide root finder fallback
    root_id = rec.get("root_id")
    if root_id:
        g.root_id_override = str(root_id)

    return g


# -------------------------
# Cache IO
# -------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

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
                k = str(rec.get(key_field, ""))
                if k:
                    idx[k] = rec
            except Exception:
                continue
    return idx

def _append_jsonl(path: str, rec: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# -------------------------
# Build pipeline
# -------------------------
def build_for_statement(
    dataset: str,
    sample_id: str,
    statement_id: str,
    pair_role: Optional[str],
    original: Dict[str, Any],
    statement: str,
    num_liars: int,
    seed: int,
    convo_path: str,
    graph_path: str,
    convo_index: Dict[str, Dict[str, Any]],
    graph_index: Dict[str, Dict[str, Any]],
    regenerate: bool,
) -> None:
    # Skip if cached
    if (not regenerate) and (statement_id in convo_index) and (statement_id in graph_index):
        return

    # 1) Generate debate (raw)
    debate_text = generate_debate(topic=statement, num_liars=int(num_liars))
    if not debate_text:
        debate_text = ""  # keep empty; parser may fail later

    debate_sha1 = _sha1(debate_text)

    convo_rec = {
        "version": "DebateConversation-v1",
        "dataset": dataset,
        "sample_id": sample_id,
        "statement_id": statement_id,
        "pair_role": pair_role,
        "original": original,
        "statement": statement,
        "generator_meta": {
            "model": "UNKNOWN",  # fill if you want
            "num_agents": None,  # fill if you want
            "num_liars": int(num_liars),
            "seed": int(seed),
            "timestamp": _now_iso(),
            "code_version": None,
        },
        "debate_text": debate_text,
    }

    # write conversation cache (overwrite-in-index style: append + update idx)
    if regenerate or (statement_id not in convo_index):
        _append_jsonl(convo_path, convo_rec)
        convo_index[statement_id] = convo_rec

    # 2) Parse -> graph cache
    if debate_text:
        try:
            g = parse_debate(debate_text)
        except Exception:
            g = None
    else:
        g = None

    # Fallback if parse fails
    if g is None:
        gg = ArgumentationGraph()
        root = ArgumentNode(
            id=f"{statement_id}_root",
            content=f"Claim: {statement}",
            speaker="Claimer",
            is_verified=False,
            ground_truth=None,
            verification_cost=1.0,
            tool_type="AUTO",
        )
        gg.add_node(root)
        gg.root_id_override = f"{statement_id}_root"
        g = gg

    graph_rec = graph_to_record(
        g=g,
        dataset=dataset,
        sample_id=sample_id,
        statement_id=statement_id,
        pair_role=pair_role,
        statement=statement,
        debate_sha1=debate_sha1,
    )

    if regenerate or (statement_id not in graph_index):
        _append_jsonl(graph_path, graph_rec)
        graph_index[statement_id] = graph_rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--max_samples", type=int, default=50)
    ap.add_argument("--cache_dir", type=str, default="cache")
    ap.add_argument("--num_liars", type=int, default=2)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--regenerate", action="store_true", help="Regenerate even if cache exists")
    args = ap.parse_args()

    dataset = str(args.dataset).lower().strip()
    cache_dir = str(args.cache_dir).strip()

    # cache locations
    convo_dir = os.path.join(cache_dir, "debate_conversations", dataset)
    graph_dir = os.path.join(cache_dir, "debate_graphs", dataset)
    convo_path = os.path.join(convo_dir, "DebateConversation-v1.jsonl")
    graph_path = os.path.join(graph_dir, "DebateGraph-v1.jsonl")

    _ensure_dir(convo_dir)
    _ensure_dir(graph_dir)

    convo_index = _load_jsonl_as_index(convo_path, key_field="statement_id")
    graph_index = _load_jsonl_as_index(graph_path, key_field="statement_id")

    print(f"[Cache] Conversation: {convo_path} (loaded={len(convo_index)})")
    print(f"[Cache] Graph:        {graph_path} (loaded={len(graph_index)})")

    samples = load_dataset_by_name(dataset, max_samples=int(args.max_samples))
    if not samples:
        print("ERROR: No samples loaded!")
        sys.exit(1)

    built = 0
    for i, sample in enumerate(samples, 1):
        sample_id = str(sample.get("id", f"sample_{i:04d}"))
        ds = _safe_get_dataset_name(sample, dataset)

        if ds == "truthfulqa":
            q = str(sample.get("claim") or sample.get("question") or "").strip()
            t_stmt, f_stmt = truthfulqa_extract_pair(sample)

            original = {
                "question": q,
                "truth_statement": t_stmt,
                "false_statement": f_stmt,
            }

            # statement ids
            tid = sample_id + "_T"
            fid = sample_id + "_F"

            build_for_statement(
                dataset=ds,
                sample_id=sample_id,
                statement_id=tid,
                pair_role="truth",
                original=original,
                statement=t_stmt,
                num_liars=int(args.num_liars),
                seed=int(args.seed),
                convo_path=convo_path,
                graph_path=graph_path,
                convo_index=convo_index,
                graph_index=graph_index,
                regenerate=bool(args.regenerate),
            )
            build_for_statement(
                dataset=ds,
                sample_id=sample_id,
                statement_id=fid,
                pair_role="false",
                original=original,
                statement=f_stmt,
                num_liars=int(args.num_liars),
                seed=int(args.seed),
                convo_path=convo_path,
                graph_path=graph_path,
                convo_index=convo_index,
                graph_index=graph_index,
                regenerate=bool(args.regenerate),
            )
            built += 2
        else:
            claim = str(sample.get("claim") or sample.get("question") or "").strip()
            original = {"claim": claim}
            build_for_statement(
                dataset=ds,
                sample_id=sample_id,
                statement_id=sample_id,
                pair_role=None,
                original=original,
                statement=claim,
                num_liars=int(args.num_liars),
                seed=int(args.seed),
                convo_path=convo_path,
                graph_path=graph_path,
                convo_index=convo_index,
                graph_index=graph_index,
                regenerate=bool(args.regenerate),
            )
            built += 1

        if i % 10 == 0:
            print(f"Processed {i}/{len(samples)} samples...")

    print(f"Done. Newly processed statements: ~{built}")
    print(f"Conversation cache size now (in-memory index): {len(convo_index)}")
    print(f"Graph cache size now (in-memory index): {len(graph_index)}")


if __name__ == "__main__":
    main()
