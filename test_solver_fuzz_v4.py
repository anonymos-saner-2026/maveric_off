# test_solver_unit_v3_fuzz.py
# Randomized stress (fuzz) tests for MaVERiC Solver v2.
# Deterministic (seeded), no real API calls.

import random
from typing import Optional
import sys
import traceback
from copy import deepcopy

# ---------------------------
# Imports with fallbacks
# ---------------------------
def _import_graph():
    try:
        from src.core.graph import ArgumentationGraph, ArgumentNode
        return ArgumentationGraph, ArgumentNode
    except Exception:
        from src.graph import ArgumentationGraph, ArgumentNode
        return ArgumentationGraph, ArgumentNode

def _import_solver_module():
    try:
        import src.core.solver as solver_mod
        return solver_mod
    except Exception:
        import src.core.solver_v2 as solver_mod
        return solver_mod


ArgumentationGraph, ArgumentNode = _import_graph()
solver_mod = _import_solver_module()


# ---------------------------
# Minimal harness
# ---------------------------
def _ok(msg):
    print(f"[OK] {msg}")

def _fail(msg):
    raise AssertionError(msg)

def _assert(cond, msg):
    if not cond:
        _fail(msg)

def _assert_eq(a, b, msg):
    if a != b:
        _fail(f"{msg} | got={a} expected={b}")


# ---------------------------
# Fake RealToolkit
# ---------------------------
class FakeRealToolkit:
    claim_truth = {}         # claim_text -> bool
    attack_truth = {}        # (att_text, tgt_text) -> bool
    support_truth = {}       # (src_text, tgt_text) -> bool

    calls_verify_claim = 0
    calls_verify_attack = 0
    calls_verify_support = 0

    @staticmethod
    def reset():
        FakeRealToolkit.claim_truth = {}
        FakeRealToolkit.attack_truth = {}
        FakeRealToolkit.support_truth = {}
        FakeRealToolkit.calls_verify_claim = 0
        FakeRealToolkit.calls_verify_attack = 0
        FakeRealToolkit.calls_verify_support = 0

    @staticmethod
    def verify_claim(tool_type: str, claim: str) -> Optional[bool]:
        FakeRealToolkit.calls_verify_claim += 1
        return bool(FakeRealToolkit.claim_truth.get(claim, True))

    @staticmethod
    def verify_attack(attacker_content: str, target_content: str) -> bool:
        FakeRealToolkit.calls_verify_attack += 1
        return bool(FakeRealToolkit.attack_truth.get((attacker_content, target_content), True))

    @staticmethod
    def verify_support(source_content: str, target_content: str) -> bool:
        FakeRealToolkit.calls_verify_support += 1
        return bool(FakeRealToolkit.support_truth.get((source_content, target_content), False))


solver_mod.RealToolkit = FakeRealToolkit


# ---------------------------
# Helpers
# ---------------------------
EDGE_TYPES = ["attack", "support"]

def make_node(nid, content, tool_type="AUTO", cost=0.0):
    return ArgumentNode(
        id=nid,
        content=content,
        speaker="X",
        is_verified=False,
        ground_truth=None,
        verification_cost=float(cost) if cost else 0.0,
        tool_type=tool_type,
    )

def make_solver(g, budget, **kwargs):
    return solver_mod.MaVERiCSolver(graph=g, budget=budget, **kwargs)


def snapshot_graph(g):
    # Snapshot for mutation checks
    nodes = sorted(list(g.nx_graph.nodes()))
    edges = sorted([(u, v, (d or {}).get("type")) for u, v, d in g.nx_graph.edges(data=True)])
    return (tuple(nodes), tuple(edges))


def random_claim(rng, i):
    # mix math-like and web-like
    if rng.random() < 0.35:
        a = rng.randint(-50, 50)
        b = rng.randint(-50, 50)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            c = a + b if rng.random() < 0.7 else a + b + rng.choice([-2, -1, 1, 2])
        elif op == "-":
            c = a - b if rng.random() < 0.7 else a - b + rng.choice([-2, -1, 1, 2])
        else:
            c = a * b if rng.random() < 0.7 else a * b + rng.choice([-3, -2, -1, 1, 2, 3])
        return f"🧮 {a} {op} {b} equals {c}."
    elif rng.random() < 0.15:
        y = rng.choice([1600, 1700, 1800, 1900, 1996, 2000, 2004, 2100, 2400])
        claim_true = ((y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0)))
        # flip sometimes
        if rng.random() < 0.3:
            claim_true = not claim_true
        return f"{y} was {'a' if claim_true else 'not a'} leap year."
    else:
        # web-ish
        topics = ["capital of France", "Great Wall Moon", "height of Everest", "Nobel prize", "speed of light"]
        t = rng.choice(topics)
        return f"Fact check: {t} (variant {i})."


def random_tool_type_for_node(rng, content):
    # Sometimes force tool type to test tool override behavior
    if "equals" in content or "leap year" in content:
        # sometimes wrong on purpose
        return rng.choice(["AUTO", "COMMON_SENSE", "PYTHON_EXEC"])
    return rng.choice(["AUTO", "WEB_SEARCH", "COMMON_SENSE"])


def random_cost(rng, tool_type):
    # enforce mix of affordable/unaffordable nodes
    if tool_type == "PYTHON_EXEC":
        return rng.choice([0.0, 1.0, 2.0, 3.0])
    if tool_type == "WEB_SEARCH":
        return rng.choice([0.0, 5.0, 10.0, 50.0, 100.0])
    return rng.choice([0.0, 0.5, 1.0, 2.0])


def random_edge_type(rng):
    return "support" if rng.random() < 0.25 else "attack"


def generate_random_graph(rng, n_nodes):
    g = ArgumentationGraph()
    node_ids = [f"N{i}" for i in range(n_nodes)]

    # Create nodes
    for i, nid in enumerate(node_ids):
        content = random_claim(rng, i)
        tool_type = random_tool_type_for_node(rng, content)
        cost = random_cost(rng, tool_type)
        g.add_node(make_node(nid, content, tool_type=tool_type, cost=cost))

        # assign deterministic truth for verify_claim:
        # math-like claims -> compute truth from expression sometimes
        # web-ish -> random truth
        if "equals" in content and any(op in content for op in [" + ", " - ", " * "]):
            # parse "a op b equals c"
            try:
                parts = content.replace("🧮", "").replace(".", "").strip().split()
                a = int(parts[0]); op = parts[1]; b = int(parts[2]); c = int(parts[4])
                if op == "+": truth = (a + b == c)
                elif op == "-": truth = (a - b == c)
                else: truth = (a * b == c)
            except Exception:
                truth = True
        elif "leap year" in content:
            try:
                y = int(content.split()[0])
                is_leap = (y % 4 == 0) and ((y % 100 != 0) or (y % 400 == 0))
                if "not a leap year" in content:
                    truth = (not is_leap)
                else:
                    truth = is_leap
            except Exception:
                truth = True
        else:
            truth = (rng.random() < 0.6)

        FakeRealToolkit.claim_truth[content] = truth

    # Add edges with some probability (include cycles)
    p_edge = min(0.35, 3.0 / max(1, n_nodes))
    for u in node_ids:
        for v in node_ids:
            if u == v:
                continue
            if rng.random() < p_edge:
                et = random_edge_type(rng)
                if et == "attack":
                    g.add_attack(u, v)
                else:
                    g.add_support(u, v)

                # define deterministic validity checks for refinement:
                # some attacks are invalid and should be pruned/converted
                u_txt = g.nodes[u].content
                v_txt = g.nodes[v].content

                # Make ~20% attacks invalid; ~50% of invalid are actually support
                if et == "attack":
                    if rng.random() < 0.20:
                        FakeRealToolkit.attack_truth[(u_txt, v_txt)] = False
                        if rng.random() < 0.50:
                            FakeRealToolkit.support_truth[(u_txt, v_txt)] = True
                        else:
                            FakeRealToolkit.support_truth[(u_txt, v_txt)] = False
                    else:
                        FakeRealToolkit.attack_truth[(u_txt, v_txt)] = True

                if et == "support":
                    # some supports invalid
                    FakeRealToolkit.support_truth[(u_txt, v_txt)] = (rng.random() < 0.85)

    return g


# ---------------------------
# Invariants to check
# ---------------------------
def assert_graph_edge_types_valid(g):
    for u, v, d in g.nx_graph.edges(data=True):
        t = (d or {}).get("type")
        _assert(t in {"attack", "support"}, f"invalid edge type: {t} on {u}->{v}")

def assert_nodes_dict_sync(g):
    # All nx nodes must exist in g.nodes dict
    for nid in g.nx_graph.nodes():
        _assert(nid in g.nodes, f"nx has node {nid} but g.nodes missing it")

def assert_budget_non_negative(solver):
    _assert(solver.budget >= -1e-9, f"budget went negative: {solver.budget}")

def assert_verdict_is_bool(verdict):
    _assert(verdict is None or isinstance(verdict, bool), f"verdict must be bool/None, got {type(verdict)}")

def assert_final_ext_subset_of_nodes(g, ext):
    for nid in ext:
        _assert(nid in g.nodes, f"extension node missing from dict: {nid}")

def assert_no_attack_and_support_duplicate_same_pair(g):
    # since nx.DiGraph only keeps one edge per (u,v), this should hold implicitly
    # still check type not None
    for u, v in g.nx_graph.edges():
        d = g.nx_graph.get_edge_data(u, v) or {}
        _assert(d.get("type") in {"attack", "support"}, "edge must have type")

def assert_tool_calls_bound(solver, initial_budget):
    # tool_calls cannot exceed number of verifications feasible by cheapest cost (0.5 by default)
    # allow slack due to custom node costs; just ensure not crazy
    _assert(solver.tool_calls <= 10_000, "tool_calls exploded unexpectedly")
    _assert(solver.tool_calls >= 0, "tool_calls negative??")

def assert_flagged_adversaries_exist_in_graph_dict(solver):
    for nid in solver.flagged_adversaries:
        _assert(nid in solver.graph.nodes, f"flagged adversary missing in graph.nodes: {nid}")


# ---------------------------
# Tests
# ---------------------------
def test_fuzz_run_many_trials(seed=1337, trials=200):
    rng = random.Random(seed)

    # Monkeypatch pagerank to deterministic uniform to avoid numeric instability
    orig_pagerank = solver_mod.nx.pagerank
    def uniform_pagerank(G, alpha=0.85):
        nodes = list(G.nodes())
        if not nodes:
            return {}
        val = 1.0 / len(nodes)
        return {n: val for n in nodes}
    solver_mod.nx.pagerank = uniform_pagerank

    try:
        for t in range(trials):
            FakeRealToolkit.reset()

            n_nodes = rng.randint(6, 25)
            g = generate_random_graph(rng, n_nodes)

            snap_before = snapshot_graph(g)
            init_budget = rng.choice([1.0, 2.0, 5.0, 10.0, 25.0, 50.0])

            s = make_solver(
                g,
                budget=init_budget,
                topk_counterfactual=rng.choice([5, 10, 15, 25]),
                delta_root=rng.choice([5.0, 10.0, 20.0, 50.0]),
                delta_adv=rng.choice([1.0, 2.0, 3.0]),
                delta_support_to_root=rng.choice([1.0, 2.5, 5.0]),
            )

            # Run
            final_ext, verdict = s.run()

            # Invariants
            assert_budget_non_negative(s)
            assert_verdict_is_bool(verdict)
            assert_final_ext_subset_of_nodes(g, final_ext)
            assert_graph_edge_types_valid(g)
            assert_nodes_dict_sync(g)
            assert_no_attack_and_support_duplicate_same_pair(g)
            assert_tool_calls_bound(s, init_budget)
            assert_flagged_adversaries_exist_in_graph_dict(s)

            # Re-running get_grounded_extension should not mutate graph
            snap_mid = snapshot_graph(g)
            _ = g.get_grounded_extension()
            snap_after = snapshot_graph(g)
            _assert_eq(snap_mid, snap_after, "get_grounded_extension mutated graph during fuzz")

            # Soft sanity: graph should have not gained brand new node ids
            before_nodes = set(snap_before[0])
            after_nodes = set(snap_after[0])
            _assert(after_nodes.issubset(before_nodes), "graph gained new nodes unexpectedly")

            if (t + 1) % 25 == 0:
                print(f"  ... fuzz progress {t+1}/{trials}")

        _ok(f"fuzz_run_many_trials (seed={seed}, trials={trials})")

    finally:
        solver_mod.nx.pagerank = orig_pagerank


def test_fuzz_run_live_schema(seed=2025, trials=60):
    rng = random.Random(seed)

    # deterministic uniform pagerank
    orig_pagerank = solver_mod.nx.pagerank
    def uniform_pagerank(G, alpha=0.85):
        nodes = list(G.nodes())
        if not nodes:
            return {}
        val = 1.0 / len(nodes)
        return {n: val for n in nodes}
    solver_mod.nx.pagerank = uniform_pagerank

    try:
        for t in range(trials):
            FakeRealToolkit.reset()
            n_nodes = rng.randint(6, 18)
            g = generate_random_graph(rng, n_nodes)
            init_budget = rng.choice([2.0, 5.0, 10.0, 20.0])

            s = make_solver(g, budget=init_budget, topk_counterfactual=rng.choice([5, 10, 15]))

            final_ext, verdict = s.run()

            _assert(isinstance(final_ext, set), "run should return a set extension")
            assert_graph_edge_types_valid(s.graph)
            assert_budget_non_negative(s)

            if (t + 1) % 15 == 0:
                print(f"  ... run_live fuzz progress {t+1}/{trials}")

        _ok(f"fuzz_run_live_schema (seed={seed}, trials={trials})")

    finally:
        solver_mod.nx.pagerank = orig_pagerank


# ---------------------------
# Runner
# ---------------------------
def run_all():
    test_fuzz_run_many_trials(seed=1337, trials=200)
    test_fuzz_run_live_schema(seed=2025, trials=60)
    print("\nAll solver v2 fuzz tests passed ✅")


if __name__ == "__main__":
    try:
        run_all()
    except Exception as e:
        print(f"[FAIL] fuzz: {e}")
        traceback.print_exc()
        sys.exit(1)
