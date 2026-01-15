# main_experiments.py
# Updated version with robust normalization + correct metrics under pruning + safer parallel behavior.
# Key fixes:
# - Normalize grounded extension sets to set[str] node IDs before IoU and summaries.
# - Confidence computed on alive nodes in nx_graph (not stale graph.nodes view).
# - Node reduction computed from nx_graph node counts (not graph.nodes dict).
# - Reset node states (is_verified + ground_truth) for fair solver runs.
# - Optional caching for oracle verification and semantic judge to reduce duplicated calls.
# - Avoid heuristic tool_calls estimation; use solver.tool_calls if available, else 0.

import copy
import datetime
import os
import sys
import threading
import concurrent.futures
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-GUI backend for headless runs
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

# Internal modules
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate
from src.core.solver import MaVERiCSolver
from src.core.baselines import RandomSolver, CRITICSolver, MADSolver
from src.tools.real_toolkit import RealToolkit
from src.config import client, JUDGE_MODEL


# ==========================================
# 1. CONFIG
# ==========================================
MAX_WORKERS = 3
NUM_LIARS = 5

TOPICS = [
    "Does the Great Wall of China appear visible to the naked eye from the Moon?",
    "Do bulls get angry specifically because of the color red in matador capes?",
    "Do goldfish strictly have a memory span of only three seconds?",
    "Did Napoleon Bonaparte have a height significantly below the average Frenchman of his time?",
    "Did the Vikings wear horned helmets during battle as commonly depicted?",
    "Did humans and non-avian dinosaurs coexist on Earth at the same time?",
    "Is Mount Everest the tallest mountain on Earth when measured from the center of the Earth?",
    "Is the dark side of the Moon permanently in darkness and never receives sunlight?",
    "Did Albert Einstein fail his mathematics class during his school years?",
    "Does tryptophan in turkey meat act as the primary cause of sleepiness after Thanksgiving dinner?",
]

BUDGET_LIMIT = 30.0

# Optional knobs to reduce cost / instability
USE_SEMANTIC_JUDGE = True
SEMANTIC_JUDGE_METHODS = {"MAD", "Random", "CRITIC", "MaVERiC"}  # or {"MaVERiC"} to reduce calls
ORACLE_VERIFY_TOOL = "WEB_SEARCH"
ORACLE_MAX_NODES: Optional[int] = None  # None = verify all nodes; else verify only first N nodes


# ==========================================
# 2. THREADING UTILITIES
# ==========================================
print_lock = threading.Lock()

oracle_cache: Dict[str, bool] = {}
oracle_lock = threading.Lock()

semantic_cache: Dict[Tuple[str, str], float] = {}
semantic_lock = threading.Lock()


class ThreadSafeLogger(object):
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, message: str):
        with self.lock:
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log.flush()


# ==========================================
# 3. METRICS & NORMALIZATION
# ==========================================
def normalize_node_id_set(nodes_like: Any, graph=None) -> Set[str]:
    """
    Normalize:
      - set/list of str node_ids
      - set/list of Node objects (must have .id)
    into a set[str].

    If graph is provided, filters out IDs not present in graph.nodes / graph.nx_graph.
    """
    if not nodes_like:
        return set()

    out: Set[str] = set()
    if isinstance(nodes_like, (set, list, tuple)):
        iterable = nodes_like
    else:
        iterable = [nodes_like]

    for x in iterable:
        if x is None:
            continue
        if isinstance(x, str):
            out.add(x)
        elif hasattr(x, "id"):
            try:
                out.add(str(x.id))
            except Exception:
                continue
        else:
            # unknown type, ignore
            continue

    if graph is not None:
        alive_ids = set()
        try:
            alive_ids = set(graph.nx_graph.nodes())
        except Exception:
            alive_ids = set()
        # Keep if in graph.nodes or in nx_graph
        out = {nid for nid in out if (nid in getattr(graph, "nodes", {})) or (nid in alive_ids)}

    return out


def get_graph_text_summary(node_id_set: Iterable[str], graph=None) -> str:
    """
    Build a single string summary from node IDs using graph lookup.
    """
    if not node_id_set:
        return ""
    if graph is None or not hasattr(graph, "nodes"):
        return ""

    content_list: List[str] = []
    for nid in node_id_set:
        try:
            if nid in graph.nodes:
                content_list.append(str(graph.nodes[nid].content))
        except Exception:
            continue

    # stable ordering for caching / judge determinism
    return " ".join(sorted(content_list))


def llm_semantic_judge(system_answer_text: str, ground_truth_text: str) -> float:
    """
    Returns 1.0 if system answer semantically agrees with reference answer on the core fact, else 0.0.
    Cached by (system_text, gt_text).
    """
    sys_t = (system_answer_text or "").strip()
    gt_t = (ground_truth_text or "").strip()
    if not sys_t or not gt_t:
        return 0.0

    key = (sys_t, gt_t)
    with semantic_lock:
        if key in semantic_cache:
            return semantic_cache[key]

    prompt = f"""
Role: Objective Fact-Checking Judge.
Reference Answer (Ground Truth): "{gt_t}"
System Answer: "{sys_t}"
Task: Does the System Answer semantically AGREE with the Reference Answer regarding the core fact?
Reply STRICTLY with 'YES' or 'NO'.
"""
    try:
        res = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        score = 1.0 if "YES" in res.choices[0].message.content.upper() else 0.0
    except Exception:
        score = 0.0

    with semantic_lock:
        semantic_cache[key] = score
    return score


def oracle_verify_claim(claim: str) -> bool:
    """
    Tool-based oracle verification with caching.
    """
    c = (claim or "").strip()
    if not c:
        return False

    with oracle_lock:
        if c in oracle_cache:
            return oracle_cache[c]

    y = bool(RealToolkit.verify_claim(ORACLE_VERIFY_TOOL, c))

    with oracle_lock:
        oracle_cache[c] = y
    return y


# ==========================================
# 4. CORE WORKER
# ==========================================
def process_single_topic(idx: int, topic: str) -> List[Dict[str, Any]]:
    topic_results: List[Dict[str, Any]] = []

    try:
        # --- PHASE 1: GENERATE & PARSE ---
        raw_text = generate_debate(topic, num_liars=NUM_LIARS)
        base_graph = parse_debate(raw_text)

        try:
            initial_node_count = int(base_graph.nx_graph.number_of_nodes())
        except Exception:
            initial_node_count = len(getattr(base_graph, "nodes", {}))

        # --- ORACLE GRAPH (tool-based) ---
        oracle_graph = copy.deepcopy(base_graph)
        nodes_to_check = list(oracle_graph.nodes.values())

        if ORACLE_MAX_NODES is not None:
            nodes_to_check = nodes_to_check[: int(ORACLE_MAX_NODES)]

        for n in nodes_to_check:
            try:
                n.ground_truth = oracle_verify_claim(n.content)
            except Exception:
                n.ground_truth = False

            if not n.ground_truth:
                try:
                    oracle_graph.remove_node(n.id)
                except Exception:
                    pass

        gt_ids = normalize_node_id_set(oracle_graph.get_grounded_extension(), oracle_graph)
        gt_text = get_graph_text_summary(gt_ids, oracle_graph)

        # --- PHASE 2: RUN SOLVERS ---
        solvers = [
            ("MAD", MADSolver),
            ("Random", RandomSolver),
            ("CRITIC", CRITICSolver),
            ("MaVERiC", MaVERiCSolver),
        ]

        for method_name, SolverClass in solvers:
            env_graph = copy.deepcopy(base_graph)

            # reset states for fairness
            for n in env_graph.nodes.values():
                n.is_verified = False
                n.ground_truth = None

            solver = SolverClass(env_graph, BUDGET_LIMIT)

            # run solver (supports either returning set or (set, verdict))
            result = solver.run()
            if isinstance(result, tuple) and len(result) == 2:
                extension_like, verdict_bool = result
            else:
                extension_like, verdict_bool = result, None

            ext_ids = normalize_node_id_set(extension_like, env_graph)

            # spent
            spent = 0.0
            if hasattr(solver, "budget"):
                try:
                    spent = float(BUDGET_LIMIT) - float(solver.budget)
                except Exception:
                    spent = 0.0
            spent = max(0.0, spent)

            # tool calls (do not guess)
            tool_calls = getattr(solver, "tool_calls", 0)
            try:
                tool_calls = int(tool_calls)
            except Exception:
                tool_calls = 0

            # confidence: verified among alive nodes in nx_graph
            try:
                alive_ids = set(env_graph.nx_graph.nodes())
            except Exception:
                alive_ids = set(env_graph.nodes.keys())

            alive_nodes = [env_graph.nodes[nid] for nid in alive_ids if nid in env_graph.nodes]
            total_alive = len(alive_nodes)
            verified_alive = sum(1 for n in alive_nodes if getattr(n, "is_verified", False))
            conf_score = (verified_alive / total_alive) if total_alive > 0 else 0.0

            # IoU on IDs
            inter = len(ext_ids & gt_ids)
            union = len(ext_ids | gt_ids)
            graph_iou = (inter / union) if union > 0 else 1.0

            # semantic judge (optional)
            if USE_SEMANTIC_JUDGE and (method_name in SEMANTIC_JUDGE_METHODS):
                sys_text = get_graph_text_summary(ext_ids, env_graph)
                sem_acc = llm_semantic_judge(sys_text, gt_text)
            else:
                sem_acc = float("nan")

            # reduction computed on nx_graph
            try:
                final_node_count = int(env_graph.nx_graph.number_of_nodes())
            except Exception:
                final_node_count = len(env_graph.nodes)

            reduction = (
                ((initial_node_count - final_node_count) / initial_node_count) * 100.0
                if initial_node_count > 0
                else 0.0
            )

            with print_lock:
                verdict_str = "FACT" if verdict_bool is True else ("MYTH" if verdict_bool is False else "UNC")
                sem_str = f"{sem_acc*100:.0f}%" if (sem_acc == sem_acc) else "NA"  # NaN check
                print(
                    f"   🏁 [Topic {idx}] {method_name:<8} | SemAcc: {sem_str:<4} | "
                    f"GraphIoU: {graph_iou:.2f} | Verdict: {verdict_str:<4} | Cost: ${spent:5.2f} | Tools: {tool_calls}"
                )

            topic_results.append(
                {
                    "Topic": topic,
                    "Method": method_name,
                    "Graph_IoU": graph_iou,
                    "Semantic_Acc": sem_acc,
                    "Cost ($)": spent,
                    "Confidence": conf_score,
                    "Tool_Calls": tool_calls,
                    "Node_Reduction (%)": reduction,
                    "Final_Verdict": verdict_bool,
                    "Initial_Nodes": initial_node_count,
                    "Final_Nodes": final_node_count,
                }
            )

    except Exception as e:
        with print_lock:
            print(f"❌ Error on Topic {idx}: {e}")
            import traceback
            traceback.print_exc()

    return topic_results


# ==========================================
# 5. MAIN
# ==========================================
def main():
    os.makedirs("runs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = ThreadSafeLogger(f"runs/exp_parallel_{timestamp}.log")

    print("=" * 80)
    print(
        f"🚀 MaVERiC EXPERIMENT | Topics: {len(TOPICS)} | Workers: {MAX_WORKERS} | "
        f"Budget: {BUDGET_LIMIT} | SemanticJudge: {USE_SEMANTIC_JUDGE}"
    )
    print("=" * 80)

    final_results: List[Dict[str, Any]] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_topic, i + 1, topic): (i + 1, topic)
            for i, topic in enumerate(TOPICS)
        }
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(TOPICS), desc="Processing Topics"):
            idx, topic = futures[future]
            try:
                result = future.result()
                if result:
                    final_results.extend(result)
            except Exception as e:
                with print_lock:
                    print(f"❌ Future failed on Topic {idx}: {topic}\n   Error: {e}")

    if not final_results:
        print("❌ No results collected.")
        return

    df = pd.DataFrame(final_results)

    csv_path = f"runs/results_parallel_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")

    print("\n📊 FINAL AGGREGATE METRICS (mean):")
    cols = ["Graph_IoU", "Semantic_Acc", "Cost ($)", "Tool_Calls", "Confidence", "Node_Reduction (%)"]
    summary = df.groupby("Method")[cols].mean(numeric_only=True)
    print(summary)

    # Plots (best effort)
    try:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        sns.barplot(data=df, x="Method", y="Semantic_Acc")
        plt.title("Semantic Accuracy")
        plt.ylim(0, 1.05)

        plt.subplot(1, 3, 2)
        sns.barplot(data=df, x="Method", y="Cost ($)")
        plt.title("Avg. Cost per Topic")

        plt.subplot(1, 3, 3)
        sns.scatterplot(data=df, x="Cost ($)", y="Semantic_Acc", hue="Method", style="Method", s=80)
        plt.title("Cost vs Semantic Accuracy")

        plt.tight_layout()
        plot_path = f"runs/plot_parallel_{timestamp}.png"
        plt.savefig(plot_path, dpi=200)
        print(f"✅ Plot saved to: {plot_path}")
    except Exception as e:
        print(f"⚠️ Error plotting: {e}")


if __name__ == "__main__":
    main()
