# graph_test.py
"""
Extended tests for ArgumentationGraph semantics and helpers.

Run with:
    python graph_test.py
"""

from src.core.graph import ArgumentationGraph, ArgumentNode


def assert_set_equal(name, got, expected):
    if got != expected:
        raise AssertionError(f"[{name}] expected {expected}, got {got}")
    print(f"[OK] {name}: {got}")


# ============================================================
# Basic tests (v1) – vẫn giữ lại để không vỡ gì
# ============================================================

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


# ============================================================
# Heavier / more complex tests (v2)
# ============================================================

def test_three_level_chain_with_branching():
    """
    A -> B -> C
    D -> C
    E supports C (verified True only in second phase).

    Case 1: no supporters verified -> C should not be accepted.
    Case 2: E verified True and a second supporter F verified True
            so that supporters_true(C) >= attackers(C),
            then C should be accepted under shield semantics.
    """
    # Base graph
    g = ArgumentationGraph()
    for nid in ["A", "B", "C", "D", "E", "F"]:
        g.add_node(ArgumentNode(nid, nid, "speaker"))

    g.add_attack("A", "B")
    g.add_attack("B", "C")
    g.add_attack("D", "C")
    g.add_support("E", "C")
    g.add_support("F", "C")

    # Phase 1: no one verified, shield does nothing
    ext_attack_only = g.get_grounded_extension(use_shield=False)
    # Attack-only semantics: A, D, E, F unattacked in first round.
    # B and C get defeated. Extension should contain {A, D, E, F}.
    assert_set_equal(
        "three_level_chain_attack_only_phase1",
        ext_attack_only,
        {"A", "D", "E", "F"}
    )

    ext_shield = g.get_grounded_extension(use_shield=True)
    # Shield on but no verified supporters -> same as attack-only
    assert_set_equal(
        "three_level_chain_shield_phase1",
        ext_shield,
        {"A", "D", "E", "F"}
    )

    # Phase 2: E and F are verified True -> C gets 2 verified supporters
    g.nodes["E"].is_verified = True
    g.nodes["E"].ground_truth = True
    g.nodes["F"].is_verified = True
    g.nodes["F"].ground_truth = True

    ext_shield2 = g.get_grounded_extension(use_shield=True)
    # Now C should be shielded against B and D:
    # supporters_true(C) = {E, F}, attackers(C) = {B, D}
    if "C" not in ext_shield2:
        raise AssertionError("[three_level_chain_shield_phase2] C should be accepted under shield semantics")
    print(f"[OK] three_level_chain_shield_phase2: extension = {ext_shield2}")


def test_mixed_verified_attackers_and_supporters():
    """
    B is attacked by C and D, and supported by E.

    - C is verified False (attack from C ignored)
    - D is unverified (active attacker)
    - E is verified True (active supporter)

    Attackers(B) = {D}, supporters_true(B) = {E}.

    With shield off: B has attacker D -> not accepted.
    With shield on: shield applies, so B should be accepted.
    """
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
    """
    Toy model of "collusive liars vs truth":

    Nodes:
      T  - truthful argument (we mark verified True)
      L1, L2 - liars, support each other and attack T

    Edges:
      L1 <-> L2 (support)
      L1 -> T (attack)
      L2 -> T (attack)

    Without any verification:
      - Cycle between L1 and L2, both attack T.
      - No unattacked node, so grounded extension is empty.

    After marking T as verified True:

      - Attacks from L1, L2 are still active, but if we additionally add
        a supporter S that is verified True and supports T, shield can
        rescue T.

    This test checks that:
      1) In pure attack-only semantics, cluster can kill T.
      2) With a strong verified-true supporter, shield can bring T in.
    """
    g = ArgumentationGraph()
    g.add_node(ArgumentNode("T", "Truth", "speaker"))
    g.add_node(ArgumentNode("L1", "L1", "speaker"))
    g.add_node(ArgumentNode("L2", "L2", "speaker"))
    g.add_node(ArgumentNode("S", "S", "speaker"))

    # Collusive structure
    g.add_attack("L1", "T")
    g.add_attack("L2", "T")
    g.add_support("L1", "L2")
    g.add_support("L2", "L1")

    # Phase 1: no verification
    ext_attack_only = g.get_grounded_extension(use_shield=False)
    # A 3-node structure where T is attacked by L1 and L2,
    # and L1,L2 have no attackers. They should be accepted,
    # T defeated.
    if "T" in ext_attack_only:
        raise AssertionError("[collusive_cluster_vs_truth_attack_only_phase1] T should not be accepted")
    print(f"[OK] collusive_cluster_vs_truth_attack_only_phase1: extension = {ext_attack_only}")

    # Phase 2: add a verified-true supporter S -> T
    g.add_support("S", "T")
    g.nodes["S"].is_verified = True
    g.nodes["S"].ground_truth = True

    ext_shield = g.get_grounded_extension(use_shield=True)
    # Now T has one verified-true supporter and 2 attackers.
    # With current rule len(supporters_true) >= len(attackers),
    # T is still not shielded (1 < 2). So we assert T still not in.
    if "T" in ext_shield:
        raise AssertionError("[collusive_cluster_vs_truth_shield_phase2] T should not be accepted yet (supporters < attackers)")
    print(f"[OK] collusive_cluster_vs_truth_shield_phase2: extension = {ext_shield}")

    # Phase 3: add a second verified-true supporter S2 -> T
    g.add_node(ArgumentNode("S2", "S2", "speaker", is_verified=True, ground_truth=True))
    g.add_support("S2", "T")

    ext_shield2 = g.get_grounded_extension(use_shield=True)
    # Now supporters_true(T) = {S, S2}, attackers(T) = {L1, L2},
    # so len(supp) == len(att) and T should be shielded.
    if "T" not in ext_shield2:
        raise AssertionError("[collusive_cluster_vs_truth_shield_phase3] T should be accepted under shield semantics")
    print(f"[OK] collusive_cluster_vs_truth_shield_phase3: extension = {ext_shield2}")


def test_long_chain_propagation():
    """
    Longer chain to check multi-round propagation:

      A -> B -> C -> D

    No supports, no verification.

    Attack-only semantics:

      - Round 1: A unattacked -> accepted, B defeated.
      - Round 2: D has attacker C, C has no attackers (since B removed).
                 So C unattacked -> accepted, D defeated.
      - Extension should be {A, C}.
    """
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
# Runner
# ============================================================

def run_all_tests():
    # Basic
    test_simple_attack_chain()
    test_mutual_attack_cycle()
    test_shield_with_verified_supporter()
    test_shield_requires_verification()
    test_attacker_verified_false_is_ignored()
    test_remove_node_sync()
    test_attack_subgraph_only_attack_edges()
    test_get_shielded_nodes()

    # Heavier
    test_three_level_chain_with_branching()
    test_mixed_verified_attackers_and_supporters()
    test_collusive_cluster_vs_truth()
    test_long_chain_propagation()

    print("\nAll graph tests passed ✅")


if __name__ == "__main__":
    run_all_tests()
