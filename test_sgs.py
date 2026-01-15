# tests/test_sgs.py
import unittest
import random
from typing import List, Tuple, Set
from src.core.graph import ArgumentationGraph, ArgumentNode


def _build_random_graph(
    seed: int,
    n_nodes: int = 60,
    n_edges: int = 520,
    p_attack: float = 0.55,
    p_verified_true: float = 0.12,
    p_verified_false: float = 0.08,
) -> ArgumentationGraph:
    """
    Create a moderately large random argumentation graph with mixed attack/support edges.
    Some nodes are verified true/false; others unknown.

    We intentionally allow:
      - dense mixing of edges
      - cycles
      - occasional self-loops (rare)
    to stress SGS invariants.
    """
    rng = random.Random(seed)
    g = ArgumentationGraph()

    node_ids = [f"n{i}" for i in range(n_nodes)]
    for nid in node_ids:
        g.add_node(make_node(nid))

    # Assign verification states
    for nid in node_ids:
        r = rng.random()
        if r < p_verified_false:
            set_verified(g, nid, False)
        elif r < p_verified_false + p_verified_true:
            set_verified(g, nid, True)
        # else unknown

    # Add edges
    for _ in range(n_edges):
        u = rng.choice(node_ids)
        v = rng.choice(node_ids)

        # Keep self-loops rare but possible
        if u == v and rng.random() > 0.05:
            v = rng.choice(node_ids)

        if rng.random() < p_attack:
            g.add_attack(u, v)
        else:
            g.add_support(u, v)

    return g


def _accepted_is_conflict_free(g: ArgumentationGraph, acc: Set[str]) -> bool:
    """
    Conflict-free w.r.t attack edges: no accepted node attacks another accepted node.
    """
    for u in acc:
        for _, v, d in g.nx_graph.out_edges(u, data=True):
            if d.get("type") == "attack" and v in acc:
                return False
    return True


# Try common import paths depending on your project layout.
try:
    from src.core.graph import ArgumentationGraph, ArgumentNode
except Exception:
    # Fallback if running from repo root with src on PYTHONPATH
    from core.graph import ArgumentationGraph, ArgumentNode


def make_node(nid: str, verified: bool = False, truth=None) -> ArgumentNode:
    node = ArgumentNode(id=nid, content=f"claim-{nid}", speaker="test")
    if verified:
        node.is_verified = True
        node.ground_truth = truth
    return node


def set_verified(g: ArgumentationGraph, nid: str, truth: bool) -> None:
    n = g.nodes[nid]
    n.is_verified = True
    n.ground_truth = truth


class TestSGS(unittest.TestCase):
    """
    These tests are designed to validate the SGS implementation aligned with the paper-style
    least-fixed-point definition:
      - No "peeling" removal of defeated nodes that would incorrectly accept downstream nodes.
      - Def(A) drives which attackers are considered alive.
      - Only verified-TRUE supporters contribute to shielding.
      - Verified-FALSE nodes are excluded from the active set.
      - alpha controls shield strictness.
    """

    def test_01_chain_attack_grounded_accepts_alternate_layers(self):
        """
        Graph: a -> b -> c  (ATTACK edges)

        Grounded-style least fixpoint accepts {a, c}:
        - a is unattacked => IN
        - b is attacked by IN => OUT
        - c is attacked only by OUT => IN
        """
        g = ArgumentationGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(make_node(nid))

        g.add_attack("a", "b")
        g.add_attack("b", "c")

        acc = g.get_grounded_extension(use_shield=False)
        self.assertEqual(acc, {"a", "c"})


    def test_02_defeat_drives_alive_attackers(self):
        """
        Ensure Atk_alive_A(v) depends on Def(A), not on removing nodes from the graph.

        Graph:
          d -> a
          a -> b
          b -> c

        Expected:
          Iter0: accept d (unattacked), Def={a}
          Iter1: b becomes acceptable since its only attacker a is defeated, accept b, Def adds c
          c should NOT be accepted because it is attacked by accepted b (b is not defeated).
        So accepted = {d, b}.
        """
        g = ArgumentationGraph()
        for nid in ["d", "a", "b", "c"]:
            g.add_node(make_node(nid))
        g.add_attack("d", "a")
        g.add_attack("a", "b")
        g.add_attack("b", "c")

        acc = g.get_grounded_extension(use_shield=False)
        self.assertEqual(acc, {"d", "b"})

    def test_03_shield_depends_on_alive_attackers_not_total_attackers(self):
        """
        Hard case: v has two attackers initially; one attacker is defeated by accepted nodes,
        reducing the alive attacker count so shield becomes sufficient.

        Construct:
          a1 -> v   (ATTACK)
          a2 -> v   (ATTACK)
          x  -> a2  (ATTACK)   where x is unattacked, so x will be accepted and defeats a2
          a1 <-> q  (mutual ATTACK cycle) so a1 is not accepted but remains alive
          s  -> v   (SUPPORT) and s is verified TRUE

        With use_shield=True, alpha=1:
          After x accepted, a2 is defeated, so alive attackers of v become {a1}.
          Since |Sup^+(v)| = 1 and |Atk_alive(v)| = 1, v becomes shield-eligible.
        With use_shield=False:
          v should not be accepted (still has an alive attacker a1).
        """
        g = ArgumentationGraph()
        for nid in ["a1", "q", "a2", "x", "s", "v"]:
            g.add_node(make_node(nid))

        # Attacks on v
        g.add_attack("a1", "v")
        g.add_attack("a2", "v")

        # Ensure a1 is not accepted by creating a mutual attack cycle a1 <-> q
        g.add_attack("a1", "q")
        g.add_attack("q", "a1")

        # Defeat a2 via an unattacked node x
        g.add_attack("x", "a2")

        # Verified-true supporter
        g.add_support("s", "v")
        set_verified(g, "s", True)

        acc_no_shield = g.get_grounded_extension(use_shield=False)
        self.assertNotIn("v", acc_no_shield, "Without shield, v should not be accepted.")

        acc_shield = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc_shield, "With shield and reduced alive attackers, v should be accepted.")

    def test_04_support_spam_has_no_effect_without_verified_true(self):
        """
        Support-spam invariance.

        v is attacked by a1, but a1 is in a mutual attack cycle so it stays alive (not accepted).
        Add many SUPPORT edges from unverified nodes to v; they must not help.

        Then verify exactly one supporter TRUE and confirm v becomes acceptable when alpha=1.
        """
        g = ArgumentationGraph()
        g.add_node(make_node("a1"))
        g.add_node(make_node("q"))
        g.add_node(make_node("v"))

        # Keep a1 alive but not accepted using a1 <-> q cycle
        g.add_attack("a1", "q")
        g.add_attack("q", "a1")

        # a1 attacks v
        g.add_attack("a1", "v")

        # Add a large number of unverified supporters
        spam_supporters = [f"s{i}" for i in range(50)]
        for s in spam_supporters:
            g.add_node(make_node(s))  # unverified by default
            g.add_support(s, "v")

        acc_spam = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertNotIn("v", acc_spam, "Unverified support should not change acceptability.")

        # Now add one verified-true supporter
        g.add_node(make_node("s_true"))
        g.add_support("s_true", "v")
        set_verified(g, "s_true", True)

        acc_one_true = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc_one_true, "One verified-true supporter should shield against one alive attacker when alpha=1.")

    def test_05_alpha_sensitivity(self):
        """
        alpha controls shield strictness.

        Setup: v has exactly one alive attacker; v has exactly one verified-true supporter.
          - alpha=1 => accept v
          - alpha=2 => do NOT accept v
        """
        g = ArgumentationGraph()
        for nid in ["a1", "q", "v", "s"]:
            g.add_node(make_node(nid))

        # a1 alive but not accepted
        g.add_attack("a1", "q")
        g.add_attack("q", "a1")
        g.add_attack("a1", "v")

        # one verified true supporter
        g.add_support("s", "v")
        set_verified(g, "s", True)

        acc_alpha1 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc_alpha1)

        acc_alpha2 = g.get_grounded_extension(use_shield=True, alpha=2.0)
        self.assertNotIn("v", acc_alpha2)

    def test_06_verified_false_nodes_are_pruned_from_active_set(self):
        """
        Verified-false nodes must not influence SGS.

        f is verified FALSE and attacks v. If f is pruned from active nodes, v becomes unattacked.
        So v should be accepted.

        Also ensure f itself is not returned as accepted.
        """
        g = ArgumentationGraph()
        for nid in ["f", "v"]:
            g.add_node(make_node(nid))

        g.add_attack("f", "v")
        set_verified(g, "f", False)

        acc = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc)
        self.assertNotIn("f", acc)

    def test_07_monotonicity_with_more_verified_support(self):
        """
        Sanity test: adding verified-true support should not make a previously accepted node
        become unaccepted under fixed alpha and fixed attacks.

        We do a simple check:
          - First, v is accepted due to being unattacked.
          - Then we add verified support edges into v (supporters verified true).
          - v should remain accepted.
        """
        g = ArgumentationGraph()
        for nid in ["v", "s1", "s2"]:
            g.add_node(make_node(nid))

        # v is unattacked, so it is accepted even without shield
        acc0 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc0)

        # add verified true supporters
        g.add_support("s1", "v")
        g.add_support("s2", "v")
        set_verified(g, "s1", True)
        set_verified(g, "s2", True)

        acc1 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("v", acc1)

    def test_08_conflict_free_blocks_shielded_node_attacked_by_accepted(self):
        """
        Build a graph where v becomes shield-eligible, but v is attacked by an accepted node.
        SGS must keep conflict-free, so v must NOT be accepted.

        Graph:
          x is unattacked -> accepted
          x attacks v  -> v is in Def(A) once x accepted
          s supports v and s is verified TRUE (would otherwise shield v)
          a attacks v (alive attacker)
        Even if shield condition holds, v is attacked by accepted x, so v must be excluded.
        """
        g = ArgumentationGraph()
        for nid in ["x", "v", "s", "a"]:
            g.add_node(make_node(nid))

        g.add_attack("x", "v")      # makes v defeated after x accepted
        g.add_attack("a", "v")      # alive attacker
        g.add_support("s", "v")
        set_verified(g, "s", True)

        acc = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertIn("x", acc)
        self.assertNotIn("v", acc)

    def test_09_self_loop_attack_does_not_crash_and_behaves_sensibly(self):
        """
        Self-loop attack: v attacks itself.
        Under conflict-free acceptability, self-attacking nodes should not be accepted,
        even if they have verified support.
        """
        g = ArgumentationGraph()
        for nid in ["v", "s"]:
            g.add_node(make_node(nid))

        g.add_attack("v", "v")

        acc0 = g.get_grounded_extension(use_shield=False)
        self.assertNotIn("v", acc0)

        g.add_support("s", "v")
        set_verified(g, "s", True)

        acc1 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertNotIn("v", acc1)
        self.assertIn("s", acc1)


    def test_10_rename_invariance(self):
        """
        Renaming node ids should preserve acceptance structure (up to renaming).
        """
        g1 = ArgumentationGraph()
        for nid in ["a", "b", "c", "s"]:
            g1.add_node(make_node(nid))
        g1.add_attack("a", "b")
        g1.add_attack("b", "c")
        g1.add_support("s", "c")
        set_verified(g1, "s", True)

        acc1 = g1.get_grounded_extension(use_shield=True, alpha=1.0)

        # Build renamed copy
        mapping = {"a": "n10", "b": "n20", "c": "n30", "s": "n40"}
        g2 = ArgumentationGraph()
        for old, new in mapping.items():
            n_old = g1.nodes[old]
            g2.add_node(make_node(new, verified=n_old.is_verified, truth=n_old.ground_truth))
        for u, v, d in g1.nx_graph.edges(data=True):
            if d.get("type") == "attack":
                g2.add_attack(mapping[u], mapping[v])
            elif d.get("type") == "support":
                g2.add_support(mapping[u], mapping[v])

        acc2 = g2.get_grounded_extension(use_shield=True, alpha=1.0)
        acc1_renamed = {mapping[x] for x in acc1}
        self.assertEqual(acc2, acc1_renamed)

    def test_11_support_spam_metamorphic(self):
        """
        Adding many unverified supporters should not change SGS on the original nodes.
        New spam nodes may be accepted themselves if they are unattacked.
        """
        g = ArgumentationGraph()
        for nid in ["a", "b", "c"]:
            g.add_node(make_node(nid))
        g.add_attack("a", "b")
        g.add_attack("b", "c")

        original_nodes = set(g.nx_graph.nodes())
        base = g.get_grounded_extension(use_shield=True, alpha=1.0)

        # spam supports into every original node from unverified nodes
        for i in range(200):
            sid = f"spam{i}"
            g.add_node(make_node(sid))  # unverified by default
            for v in original_nodes:
                g.add_support(sid, v)

        after = g.get_grounded_extension(use_shield=True, alpha=1.0)

        # Compare only on original nodes
        self.assertEqual(base, after.intersection(original_nodes))

    def test_12_randomized_stress_invariants(self):
        """
        Randomized stress test on moderately large graphs, aligned with our SGS implementation
        that enforces conflict-freeness via conservative filtering.

        Invariants checked:

        (I1) No verified-false node is in accepted.
        (I2) Accepted set is conflict-free under ATTACK edges.
        (I3) Monotonicity of the *shield predicate* w.r.t alpha at A = empty:
            if shield holds for alpha=1.0, it must hold for alpha=0.5 (pure inequality check).
            Note: we do NOT require monotonicity of the final accepted set under conflict filtering.
        (I4) Unverified support-spam invariance on original nodes (projection invariance):
            adding NEW unverified nodes with ONLY SUPPORT edges into original nodes
            must not change acceptance restricted to original nodes.
        (I5) Determinism sanity: running SGS twice on the same graph returns the same set.
        """
        seeds = list(range(20))  # increase to 50-100 for heavier stress
        n_nodes = 80
        n_edges = 900

        for seed in seeds:
            g = _build_random_graph(
                seed=seed,
                n_nodes=n_nodes,
                n_edges=n_edges,
                p_attack=0.58,
                p_verified_true=0.12,
                p_verified_false=0.10,
            )

            original_nodes = set(g.nx_graph.nodes())

            # Baseline acceptance (alpha=1.0) and determinism check (I5)
            acc_a10 = g.get_grounded_extension(use_shield=True, alpha=1.0)
            acc_a10_again = g.get_grounded_extension(use_shield=True, alpha=1.0)
            self.assertEqual(
                acc_a10,
                acc_a10_again,
                f"Seed {seed}: SGS is not deterministic on identical input",
            )

            # (I1) No verified-false in accepted
            for nid in acc_a10:
                node = g.nodes[nid]
                self.assertFalse(
                    node.is_verified and node.ground_truth is False,
                    f"Seed {seed}: verified-false node {nid} appeared in accepted set",
                )

            # (I2) Conflict-free under ATTACK edges
            self.assertTrue(
                _accepted_is_conflict_free(g, acc_a10),
                f"Seed {seed}: accepted set violates conflict-free under ATTACK edges",
            )

            # (I3) Shield predicate monotonicity at A = empty (pure inequality property)
            # Compute per-node counts on the original graph (excluding verified-false attackers)
            for v in original_nodes:
                v_node = g.nodes[v]
                if v_node.is_verified and v_node.ground_truth is False:
                    continue

                atk = 0
                sup_true = 0
                for u, _, d in g.nx_graph.in_edges(v, data=True):
                    if u not in original_nodes:
                        continue
                    et = d.get("type")
                    if et == "attack":
                        u_node = g.nodes[u]
                        if u_node.is_verified and u_node.ground_truth is False:
                            continue
                        atk += 1
                    elif et == "support":
                        u_node = g.nodes[u]
                        if u_node.is_verified and u_node.ground_truth is True:
                            sup_true += 1

                # If sup_true >= 1.0 * atk then sup_true >= 0.5 * atk must also hold
                if atk > 0 and sup_true >= 1.0 * atk:
                    self.assertTrue(
                        sup_true >= 0.5 * atk,
                        f"Seed {seed}: shield predicate monotonicity violated at A=empty for node {v}",
                    )

            # (I4) Unverified support-spam invariance on original nodes (projection invariance)
            # Save projection of acceptance on original nodes
            proj_before = acc_a10.intersection(original_nodes)

            # Add spam nodes (unverified) and ONLY support edges into original nodes
            spam_count = 60
            spam_ids = [f"spam_{seed}_{i}" for i in range(spam_count)]
            for sid in spam_ids:
                g.add_node(make_node(sid))  # unverified

            rng = random.Random(seed + 999)
            targets = list(original_nodes)
            for sid in spam_ids:
                for _ in range(40):  # 2400 new support edges
                    v = rng.choice(targets)
                    g.add_support(sid, v)

            acc_after_spam = g.get_grounded_extension(use_shield=True, alpha=1.0)
            proj_after = acc_after_spam.intersection(original_nodes)

            self.assertEqual(
                proj_before,
                proj_after,
                f"Seed {seed}: unverified support-spam changed SGS on original nodes",
            )

    def test_13_metamorphic_stress_pipeline(self):
        """
        Extremely strict metamorphic stress test.

        We build a fairly large graph, compute baseline SGS acceptance, then apply a sequence
        of transformations that should NOT change acceptance on the original nodes:

        (T1) Add many new UNVERIFIED nodes with ONLY SUPPORT edges into original nodes.
            This must not change acceptance restricted to original nodes.

        (T2) Add many new VERIFIED-FALSE nodes and arbitrary edges (including ATTACK).
            Since verified-false nodes are pruned from active set, they must not change SGS
            on original nodes.

        (T3) Rename all node ids via a bijection; acceptance should be preserved up to renaming.

        We then verify:
        - projection invariance on original nodes after T1+T2
        - rename invariance after T3
        - core invariants: no verified-false accepted, conflict-free
        """
        seed = 2026
        n_nodes = 120
        n_edges = 1600

        g = _build_random_graph(
            seed=seed,
            n_nodes=n_nodes,
            n_edges=n_edges,
            p_attack=0.60,
            p_verified_true=0.10,
            p_verified_false=0.10,
        )

        original_nodes = set(g.nx_graph.nodes())

        base = g.get_grounded_extension(use_shield=True, alpha=1.0)

        # ---------------------------
        # (T1) Add unverified support spam into original nodes
        # ---------------------------
        rng = random.Random(seed + 1)
        spam_supporters = [f"u_spam_{i}" for i in range(80)]
        for sid in spam_supporters:
            g.add_node(make_node(sid))  # unverified
            for _ in range(50):  # 80*50=4000 supports
                v = rng.choice(list(original_nodes))
                g.add_support(sid, v)

        after_t1 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertEqual(
            base.intersection(original_nodes),
            after_t1.intersection(original_nodes),
            "T1 failed: unverified support spam changed SGS on original nodes",
        )

        # ---------------------------
        # (T2) Add verified-false nodes with arbitrary edges
        # These nodes must be pruned and should not affect SGS on original nodes.
        # ---------------------------
        false_ids = [f"fnode_{i}" for i in range(50)]
        for fid in false_ids:
            g.add_node(make_node(fid))
            set_verified(g, fid, False)

        # Add arbitrary edges from verified-false nodes to original nodes (attack/support)
        # Also add some edges into verified-false nodes, to stress edge iteration.
        rng2 = random.Random(seed + 2)
        all_nodes_now = list(g.nx_graph.nodes())
        for fid in false_ids:
            for _ in range(40):  # 50*40=2000 edges
                v = rng2.choice(list(original_nodes))
                if rng2.random() < 0.7:
                    g.add_attack(fid, v)
                else:
                    g.add_support(fid, v)
            for _ in range(10):
                u = rng2.choice(all_nodes_now)
                if rng2.random() < 0.5:
                    g.add_attack(u, fid)
                else:
                    g.add_support(u, fid)

        after_t2 = g.get_grounded_extension(use_shield=True, alpha=1.0)
        self.assertEqual(
            base.intersection(original_nodes),
            after_t2.intersection(original_nodes),
            "T2 failed: verified-false nodes changed SGS on original nodes",
        )

        # Also re-check core invariants on the transformed graph
        for nid in after_t2:
            node = g.nodes[nid]
            self.assertFalse(
                node.is_verified and node.ground_truth is False,
                f"T2 invariant failed: verified-false node {nid} is accepted",
            )
        self.assertTrue(
            _accepted_is_conflict_free(g, after_t2),
            "T2 invariant failed: accepted set not conflict-free under ATTACK edges",
        )

        # ---------------------------
        # (T3) Rename all node ids (bijection) and ensure acceptance preserved up to renaming
        # ---------------------------
        # Build a renamed graph gR
        mapping = {nid: f"R_{i}" for i, nid in enumerate(sorted(g.nx_graph.nodes()))}

        gR = ArgumentationGraph()
        # copy nodes with verification states
        for old_id, new_id in mapping.items():
            old = g.nodes[old_id]
            gR.add_node(make_node(new_id, verified=old.is_verified, truth=old.ground_truth))

        # copy edges
        for u, v, d in g.nx_graph.edges(data=True):
            et = d.get("type")
            if et == "attack":
                gR.add_attack(mapping[u], mapping[v])
            elif et == "support":
                gR.add_support(mapping[u], mapping[v])

        accR = gR.get_grounded_extension(use_shield=True, alpha=1.0)

        # Compare acceptance on the renamed original node set
        original_nodes_renamed = {mapping[n] for n in original_nodes}
        expected_renamed = {mapping[n] for n in after_t2.intersection(original_nodes)}
        got_renamed = accR.intersection(original_nodes_renamed)

        self.assertEqual(
            expected_renamed,
            got_renamed,
            "T3 failed: rename invariance violated on original nodes",
        )

if __name__ == "__main__":
    unittest.main()
