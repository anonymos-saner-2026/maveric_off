# graph_test_v3.py
"""
Hard tests for ArgumentationGraph semantics and helpers.

Run with:
    python graph_test_v3.py
"""

import random
from copy import deepcopy

from src.core.graph import ArgumentationGraph, ArgumentNode


def assert_set_equal(name, got, expected):
    if got != expected:
        raise AssertionError(f"[{name}] expected {expected}, got {got}")
    print(f"[OK] {name}: {got}")


# ============================================================
# Basic tests (từ v2, giữ lại để regression không vỡ)
# ============================================================

def test_simple_attack_chain():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_attack("A", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    assert_set_equal("simple_attack_chain_attack_only", ext_attack_only, {"A"})

    ext_shield = g.get_grounded_extension(use_shield=True)
    assert_set_equal("simple_attack_chain_shield", ext_shield, {"A"})


def test_mutual_attack_cycle():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_attack("A", "B")
    g.add_attack("B", "A")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    assert_set_equal("mutual_attack_cycle_attack_only", ext_attack_only, set())

    ext_shield = g.get_grounded_extension(use_shield=True)
    assert_set_equal("mutual_attack_cycle_shield", ext_shield, set())


def test_shield_with_verified_supporter():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))
    g.add_node(ArgumentNode("D", "D", "speaker", is_verified=True, ground_truth=True))

    g.add_attack("C", "B")
    g.add_support("D", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    if "B" in ext_attack_only:
        raise AssertionError("[shield_with_verified_supporter_attack_only] B should not be accepted")
    print(f"[OK] shield_with_verified_supporter_attack_only: extension = {ext_attack_only}")

    ext_shield = g.get_grounded_extension(use_shield=True)
    if "B" not in ext_shield:
        raise AssertionError("[shield_with_verified_supporter_shield] B should be accepted under shield semantics")
    print(f"[OK] shield_with_verified_supporter_shield: extension = {ext_shield}")


def test_shield_requires_verification():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))
    g.add_node(ArgumentNode("D", "D", "speaker", is_verified=False, ground_truth=True))

    g.add_attack("C", "B")
    g.add_support("D", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    if "B" in ext_attack_only:
        raise AssertionError("[shield_requires_verification_attack_only] B should not be accepted")
    print(f"[OK] shield_requires_verification_attack_only: extension = {ext_attack_only}")

    ext_shield = g.get_grounded_extension(use_shield=True)
    if "B" in ext_shield:
        raise AssertionError("[shield_requires_verification_shield] B should not be accepted when supporter is unverified")
    print(f"[OK] shield_requires_verification_shield: extension = {ext_shield}")


def test_attacker_verified_false_is_ignored():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker", is_verified=True, ground_truth=False))

    g.add_attack("C", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    if "B" not in ext_attack_only:
        raise AssertionError("[attacker_verified_false_attack_only] B should be accepted when attacker is verified False")
    print(f"[OK] attacker_verified_false_attack_only: extension = {ext_attack_only}")

    ext_shield = g.get_grounded_extension(use_shield=True)
    if "B" not in ext_shield:
        raise AssertionError("[attacker_verified_false_shield] B should be accepted when attacker is verified False")
    print(f"[OK] attacker_verified_false_shield: extension = {ext_shield}")


def test_remove_node_sync():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_attack("A", "B")

    g.remove_node("A")

    if "A" in g.nx_graph.nodes or "A" in g.nodes:
        raise AssertionError("[remove_node_sync] A should be removed from both graph and nodes dict")
    if "B" not in g.nx_graph.nodes:
        raise AssertionError("[remove_node_sync] B should still be present")
    print("[OK] remove_node_sync")


def test_attack_subgraph_only_attack_edges():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))

    g.add_attack("A", "B")
    g.add_support("C", "B")

    g_attack = g.get_attack_subgraph()

    assert_set_equal(
        "attack_subgraph_nodes",
        set(g_attack.nodes()),
        set(g.nx_graph.nodes())
    )

    for u, v in g_attack.edges():
        if not g.nx_graph.has_edge(u, v):
            raise AssertionError("[attack_subgraph_only_attack_edges] edge not in original graph")
        if g.nx_graph[u][v].get("type") != "attack":
            raise AssertionError("[attack_subgraph_only_attack_edges] non-attack edge leaked into attack subgraph")

    print("[OK] attack_subgraph_only_attack_edges: edges =", list(g_attack.edges()))


def test_get_shielded_nodes():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker", is_verified=True, ground_truth=True))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker", is_verified=True, ground_truth=False))
    g.add_node(ArgumentNode("D", "D", "speaker"))

    g.add_support("A", "B")
    g.add_support("C", "D")

    shielded = g.get_shielded_nodes()
    assert_set_equal("get_shielded_nodes", shielded, {"B"})


def test_three_level_chain_with_branching():
    g = ArgumentationGraph()
    for nid in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(ArgumentNode(nid, nid, "speaker"))

    g.add_attack("A", "B")
    g.add_attack("B", "C")
    g.add_attack("D", "C")
    g.add_support("E", "C")
    g.add_support("F", "C")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    assert_set_equal(
        "three_level_chain_attack_only_phase1",
        ext_attack_only,
        {"A", "D", "E", "F"}
    )

    ext_shield = g.get_grounded_extension(use_shield=True)
    assert_set_equal(
        "three_level_chain_shield_phase1",
        ext_shield,
        {"A", "D", "E", "F"}
    )

    g.nodes["E"].is_verified = True
    g.nodes["E"].ground_truth = True
    g.nodes["F"].is_verified = True
    g.nodes["F"].ground_truth = True

    ext_shield2 = g.get_grounded_extension(use_shield=True)
    if "C" not in ext_shield2:
        raise AssertionError("[three_level_chain_shield_phase2] C should be accepted under shield semantics")
    print(f"[OK] three_level_chain_shield_phase2: extension = {ext_shield2}")


def test_mixed_verified_attackers_and_supporters():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker", is_verified=True, ground_truth=False))
    g.add_node(ArgumentNode("D", "D", "speaker"))
    g.add_node(ArgumentNode("E", "E", "speaker", is_verified=True, ground_truth=True))

    g.add_attack("C", "B")
    g.add_attack("D", "B")
    g.add_support("E", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    if "B" in ext_attack_only:
        raise AssertionError("[mixed_verified_attackers_supporters_attack_only] B should not be accepted (active attacker D)")
    print(f"[OK] mixed_verified_attackers_supporters_attack_only: extension = {ext_attack_only}")

    ext_shield = g.get_grounded_extension(use_shield=True)
    if "B" not in ext_shield:
        raise AssertionError("[mixed_verified_attackers_supporters_shield] B should be accepted under shield semantics")
    print(f"[OK] mixed_verified_attackers_supporters_shield: extension = {ext_shield}")


def test_collusive_cluster_vs_truth():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("T", "Truth", "speaker"))
    g.add_node(ArgumentNode("L1", "L1", "speaker"))
    g.add_node(ArgumentNode("L2", "L2", "speaker"))
    g.add_node(ArgumentNode("S", "S", "speaker"))

    g.add_attack("L1", "T")
    g.add_attack("L2", "T")
    g.add_support("L1", "L2")
    g.add_support("L2", "L1")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    if "T" in ext_attack_only:
        raise AssertionError("[collusive_cluster_vs_truth_attack_only_phase1] T should not be accepted")
    print(f"[OK] collusive_cluster_vs_truth_attack_only_phase1: extension = {ext_attack_only}")

    g.add_support("S", "T")
    g.nodes["S"].is_verified = True
    g.nodes["S"].ground_truth = True

    ext_shield = g.get_grounded_extension(use_shield=True)
    if "T" in ext_shield:
        raise AssertionError("[collusive_cluster_vs_truth_shield_phase2] T should not be accepted yet (supporters < attackers)")
    print(f"[OK] collusive_cluster_vs_truth_shield_phase2: extension = {ext_shield}")

    g.add_node(ArgumentNode("S2", "S2", "speaker", is_verified=True, ground_truth=True))
    g.add_support("S2", "T")

    ext_shield2 = g.get_grounded_extension(use_shield=True)
    if "T" not in ext_shield2:
        raise AssertionError("[collusive_cluster_vs_truth_shield_phase3] T should be accepted under shield semantics")
    print(f"[OK] collusive_cluster_vs_truth_shield_phase3: extension = {ext_shield2}")


def test_long_chain_propagation():
    g = ArgumentationGraph()
    for nid in ["A", "B", "C", "D"]:
        g.add_node(ArgumentNode(nid, nid, "speaker"))

    g.add_attack("A", "B")
    g.add_attack("B", "C")
    g.add_attack("C", "D")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    assert_set_equal("long_chain_propagation_attack_only", ext_attack_only, {"A", "C"})

    ext_shield = g.get_grounded_extension(use_shield=True)
    assert_set_equal("long_chain_propagation_shield", ext_shield, {"A", "C"})


# ============================================================
# Hard / random / property-based tests (v3)
# ============================================================

def _snapshot_graph(g: ArgumentationGraph):
    nodes = set(g.nx_graph.nodes())
    edges = set(
        (u, v, tuple(sorted(d.items())))
        for u, v, d in g.nx_graph.edges(data=True)
    )
    return nodes, edges


def test_grounded_extension_does_not_mutate_graph():
    """
    get_grounded_extension must not mutate the underlying nx_graph.
    """
    g = ArgumentationGraph()
    for nid in ["A", "B", "C"]:
        g.add_node(ArgumentNode(nid, nid, "speaker"))
    g.add_attack("A", "B")
    g.add_support("C", "B")

    before_nodes, before_edges = _snapshot_graph(g)
    _ = g.get_grounded_extension(use_shield=False)
    after_nodes, after_edges = _snapshot_graph(g)

    if before_nodes != after_nodes or before_edges != after_edges:
        raise AssertionError("[graph_not_mutated_attack_only] Graph was mutated by get_grounded_extension(use_shield=False)")

    before_nodes, before_edges = _snapshot_graph(g)
    _ = g.get_grounded_extension(use_shield=True)
    after_nodes, after_edges = _snapshot_graph(g)

    if before_nodes != after_nodes or before_edges != after_edges:
        raise AssertionError("[graph_not_mutated_shield] Graph was mutated by get_grounded_extension(use_shield=True)")

    print("[OK] grounded_extension_does_not_mutate_graph")


def _random_argument_graph(num_nodes: int, p_attack: float, p_support: float) -> ArgumentationGraph:
    """
    Generate a random small graph with random verification states.
    """
    g = ArgumentationGraph()
    ids = [f"A{i}" for i in range(num_nodes)]
    for nid in ids:
        # Random verification state
        is_verified = random.random() < 0.3
        if is_verified:
            gt = random.choice([True, False])
        else:
            gt = None
        g.add_node(ArgumentNode(nid, nid, "speaker", is_verified=is_verified, ground_truth=gt))

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            r = random.random()
            if r < p_attack:
                g.add_attack(ids[i], ids[j])
            elif r < p_attack + p_support:
                g.add_support(ids[i], ids[j])

    return g


def test_attack_only_vs_attack_subgraph_random():
    """
    For arbitrary random graphs, attack-only grounded extension on the full graph
    should match grounded extension computed on an attack-only graph constructed
    from the same nodes and attack edges.
    """
    random.seed(42)
    for trial in range(50):
        g = _random_argument_graph(num_nodes=5, p_attack=0.3, p_support=0.3)

        ext_full = g.get_grounded_extension(use_shield=False)

        # Build attack-only graph clone
        g2 = ArgumentationGraph()
        for nid, node in g.nodes.items():
            # Deep copy of node state is not strictly necessary, but clearer
            g2.add_node(ArgumentNode(
                id=node.id,
                content=node.content,
                speaker=node.speaker,
                is_verified=node.is_verified,
                ground_truth=node.ground_truth,
                verification_cost=node.verification_cost,
                tool_type=node.tool_type,
            ))
        for u, v, d in g.nx_graph.edges(data=True):
            if d.get("type") == "attack":
                g2.add_attack(u, v)

        ext_attack_only = g2.get_grounded_extension(use_shield=False)

        if ext_full != ext_attack_only:
            raise AssertionError(
                f"[attack_only_vs_attack_subgraph_random] mismatch on trial {trial}: "
                f"full={ext_full}, attack_only={ext_attack_only}"
            )

    print("[OK] attack_only_vs_attack_subgraph_random (50 trials)")


def test_shield_vs_no_shield_when_no_verification_random():
    """
    When no node is verified (is_verified=False for all) and ground_truth=None,
    then shield should never apply. In that case, use_shield=True and
    use_shield=False must produce the same grounded extension.
    """
    random.seed(123)
    for trial in range(50):
        g = _random_argument_graph(num_nodes=6, p_attack=0.35, p_support=0.35)
        # Overwrite all verification state to "unverified"
        for node in g.nodes.values():
            node.is_verified = False
            node.ground_truth = None

        ext_no_shield = g.get_grounded_extension(use_shield=False)
        ext_shield = g.get_grounded_extension(use_shield=True)

        if ext_no_shield != ext_shield:
            raise AssertionError(
                f"[shield_vs_no_shield_when_no_verification_random] mismatch on trial {trial}: "
                f"no_shield={ext_no_shield}, shield={ext_shield}"
            )

    print("[OK] shield_vs_no_shield_when_no_verification_random (50 trials)")


def test_idempotent_grounded_extension_random():
    """
    Calling get_grounded_extension multiple times on the same graph
    should always return the same result (deterministic semantics).
    """
    random.seed(999)
    for trial in range(30):
        g = _random_argument_graph(num_nodes=6, p_attack=0.4, p_support=0.3)

        ext1 = g.get_grounded_extension(use_shield=False)
        ext2 = g.get_grounded_extension(use_shield=False)
        ext3 = g.get_grounded_extension(use_shield=True)
        ext4 = g.get_grounded_extension(use_shield=True)

        if ext1 != ext2:
            raise AssertionError(
                f"[idempotent_grounded_extension_random_attack_only] ext1 != ext2 on trial {trial}: "
                f"{ext1} vs {ext2}"
            )
        if ext3 != ext4:
            raise AssertionError(
                f"[idempotent_grounded_extension_random_shield] ext3 != ext4 on trial {trial}: "
                f"{ext3} vs {ext4}"
            )

    print("[OK] idempotent_grounded_extension_random (30 trials)")


def test_duplicate_support_edges_no_effect():
    """
    Adding duplicate or redundant support edges from the same verified-true supporter
    should not change the grounded extension.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))
    g.add_node(ArgumentNode("S", "S", "speaker", is_verified=True, ground_truth=True))

    g.add_attack("C", "B")
    g.add_support("S", "B")

    ext_before = g.get_grounded_extension(use_shield=True)

    # Add "duplicate" support (networkx will not add a second parallel edge,
    # but this tests that re-calling add_support does not break semantics)
    g.add_support("S", "B")

    ext_after = g.get_grounded_extension(use_shield=True)

    if ext_before != ext_after:
        raise AssertionError(
            f"[duplicate_support_edges_no_effect] extension changed after duplicate support: "
            f"before={ext_before}, after={ext_after}"
        )

    print("[OK] duplicate_support_edges_no_effect")


# ============================================================
# Runner
# ============================================================

def run_all_tests():
    # Basic from v2
    test_simple_attack_chain()
    test_mutual_attack_cycle()
    test_shield_with_verified_supporter()
    test_shield_requires_verification()
    test_attacker_verified_false_is_ignored()
    test_remove_node_sync()
    test_attack_subgraph_only_attack_edges()
    test_get_shielded_nodes()
    test_three_level_chain_with_branching()
    test_mixed_verified_attackers_and_supporters()
    test_collusive_cluster_vs_truth()
    test_long_chain_propagation()

    # Hard / random
    test_grounded_extension_does_not_mutate_graph()
    test_attack_only_vs_attack_subgraph_random()
    test_shield_vs_no_shield_when_no_verification_random()
    test_idempotent_grounded_extension_random()
    test_duplicate_support_edges_no_effect()

    print("\nAll graph v3 tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
