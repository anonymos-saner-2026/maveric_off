# graph_test.py
"""
Quick tests for ArgumentationGraph semantics and helpers.

Run with:
    python graph_test.py
"""

from src.core.graph import ArgumentationGraph, ArgumentNode


def assert_set_equal(name, got, expected):
    if got != expected:
        raise AssertionError(f"[{name}] expected {expected}, got {got}")
    print(f"[OK] {name}: {got}")


def test_simple_attack_chain():
    """
    A -> B
    Grounded extension (attack-only, shield off) should accept A only.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_attack("A", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    assert_set_equal("simple_attack_chain_attack_only", ext_attack_only, {"A"})

    # Shield on but no supporters, so same result
    ext_shield = g.get_grounded_extension(use_shield=True)
    assert_set_equal("simple_attack_chain_shield", ext_shield, {"A"})


def test_mutual_attack_cycle():
    """
    A <-> B
    No unattacked nodes, grounded extension should be empty.
    """
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
    """
    C -> B, D --support--> B
    D is verified True.

    - Without shield: B is attacked, D and C unattacked.
      Grounded extension contains {C, D}, but not B.
    - With shield: supporters_true(B) = [D], attackers(B) = [C], so
      len(supporters_true) >= len(attackers) and B is accepted.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))
    g.add_node(ArgumentNode("D", "D", "speaker", is_verified=True, ground_truth=True))

    g.add_attack("C", "B")
    g.add_support("D", "B")

    ext_attack_only = g.get_grounded_extension(use_shield=False)
    # In attack-only mode, B is attacked, C and D have no attackers.
    # Both C and D are accepted, B is defeated.
    # We only assert that B is not in the extension.
    if "B" in ext_attack_only:
        raise AssertionError("[shield_with_verified_supporter_attack_only] B should not be accepted")
    print(f"[OK] shield_with_verified_supporter_attack_only: extension = {ext_attack_only}")

    ext_shield = g.get_grounded_extension(use_shield=True)
    # With shield, B must be accepted as well.
    if "B" not in ext_shield:
        raise AssertionError("[shield_with_verified_supporter_shield] B should be accepted under shield semantics")
    print(f"[OK] shield_with_verified_supporter_shield: extension = {ext_shield}")


def test_shield_requires_verification():
    """
    C -> B, D --support--> B
    D is NOT verified. Shield should NOT apply.

    So B remains attacked and should not be accepted (under shield or attack-only).
    """
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
    """
    C -> B, but C has been verified False.

    Then the attack from C should be ignored, so B is effectively unattacked.
    Grounded extension must include B.
    """
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
    """
    remove_node should update both nx_graph and nodes dict.
    """
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
    """
    get_attack_subgraph should only contain attack edges,
    and have the same node set as the original graph.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker"))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker"))

    g.add_attack("A", "B")
    g.add_support("C", "B")

    g_attack = g.get_attack_subgraph()

    # Nodes must match
    assert_set_equal(
        "attack_subgraph_nodes",
        set(g_attack.nodes()),
        set(g.nx_graph.nodes())
    )

    # Only attack edges
    for u, v in g_attack.edges():
        if not g.nx_graph.has_edge(u, v):
            raise AssertionError("[attack_subgraph_only_attack_edges] edge not in original graph")
        if g.nx_graph[u][v].get("type") != "attack":
            raise AssertionError("[attack_subgraph_only_attack_edges] non-attack edge leaked into attack subgraph")

    print("[OK] attack_subgraph_only_attack_edges: edges =", list(g_attack.edges()))


def test_get_shielded_nodes():
    """
    get_shielded_nodes should return exactly nodes with at least one
    verified-true supporter.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("A", "A", "speaker", is_verified=True, ground_truth=True))
    g.add_node(ArgumentNode("B", "B", "speaker"))
    g.add_node(ArgumentNode("C", "C", "speaker", is_verified=True, ground_truth=False))
    g.add_node(ArgumentNode("D", "D", "speaker"))

    # A supports B (A is verified True -> B shielded)
    g.add_support("A", "B")
    # C supports D (C is verified False -> D not shielded)
    g.add_support("C", "D")

    shielded = g.get_shielded_nodes()
    assert_set_equal("get_shielded_nodes", shielded, {"B"})


def run_all_tests():
    test_simple_attack_chain()
    test_mutual_attack_cycle()
    test_shield_with_verified_supporter()
    test_shield_requires_verification()
    test_attacker_verified_false_is_ignored()
    test_remove_node_sync()
    test_attack_subgraph_only_attack_edges()
    test_get_shielded_nodes()
    print("\nAll graph tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
