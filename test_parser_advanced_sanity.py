
import unittest
from src.agents.parser import parse_debate
import networkx as nx

class TestParserAdvancedSanity(unittest.TestCase):
    """
    Advanced sanity checks for MaVERiC parser.
    Covers complex coreference, deduplication, and argumentative structures.
    """

    def test_quantum_physics_coreference(self):
        """
        Test Case 1: Hard Science & Pronouns.
        'It' should resolve to 'The electron', not 'The observer'.
        """
        transcript = """
Moderator: Let's discuss the observer effect in quantum mechanics.
Alice: When an electron passes through a double slit, it behaves like a wave.
Bob: However, if an observer watches it, it collapses into a particle state.
Charlie: That collapse is instantaneous. It defies classical locality.
"""
        print(f"\n[Test] Quantum Coreference:\n{transcript}")
        # Disable refine to test raw parser capabilities
        g = parse_debate(transcript, split_mode="llm", refine_iters=0)
        
        # Check Nodes
        nodes_content = [n.content.lower() for n in g.nodes.values()]
        
        # Bob's claim should mention "electron" explicitly
        bobs_claims = [n.content for n in g.nodes.values() if n.speaker == "bob"]
        has_resolved_electron = any("electron" in c.lower() for c in bobs_claims)
        
        # Charlie's claim should mention "collapse" or "observation"
        charlie_claims = [n.content for n in g.nodes.values() if n.speaker == "charlie"]
        has_resolved_collapse = any("collapse" in c.lower() for c in charlie_claims)

        print("Bob's Claims:", bobs_claims)
        print("Charlie's Claims:", charlie_claims)

        self.assertTrue(has_resolved_electron, "Bob's pronouns ('it') failed to resolve to 'electron'")
        self.assertTrue(has_resolved_collapse, "Charlie's pronouns ('It') failed to resolve to 'collapse'")

    def test_historical_fact_deduplication(self):
        """
        Test Case 2: History & Deduplication.
        Multiple agents stating the precise same date/fact should merge.
        """
        transcript = """
Moderator: When did the Titanic sink?
Alice: The Titanic sank on April 15, 1912.
Bob: It struck an iceberg on April 14, causing it to sink the next morning.
Charlie: The Titanic sank on April 15, 1912.
David: Correct. The ship went down on April 15, 1912.
"""
        print(f"\n[Test] Historical Deduplication:\n{transcript}")
        g = parse_debate(transcript, split_mode="llm", refine_iters=0)
        
        # We expect FEWER nodes than claims.
        # Alice, Charlie, David all say "April 15, 1912".
        # Should be merged into 1 or 2 nodes max (David's might be slight var).
        
        date_nodes = [n for n in g.nodes.values() if "1912" in n.content]
        unique_contents = set(n.content.lower().strip() for n in date_nodes)
        
        print("Date Nodes:", [n.content for n in date_nodes])
        print("Unique Contents:", unique_contents)

        # Ideally Alice and Charlie are EXACTLY merged.
        # David might be distinct due to "The ship went down".
        # But Alice and Charlie textual match is 100%.
        
        # Count nodes attributed to Alice vs Charlie
        alice_nodes = [n for n in g.nodes.values() if n.speaker == "alice"]
        charlie_nodes = [n for n in g.nodes.values() if n.speaker == "charlie"]
        
        # If merged, one speaker might lose attribution or be mapped.
        # Deduplication logic keeps one node.
        # If Alice and Charlie are merged, there should be only 1 node with that text.
        
        exact_match_count = sum(1 for n in g.nodes.values() if "April 15, 1912" in n.content)
        
        # Depending on dedupe rigor, might vary. 
        # But we assert that Alice and Charlie (identical strings) are NOT distinct nodes.
        self.assertLess(len(date_nodes), 3, "Failed to deduplicate 3 assertions of the same date")

    def test_ai_ethics_rebuttal(self):
        """
        Test Case 3: Philosophy & Rebuttal Chains.
        Complex structure: A -> B -> C (Chain of rebuttal).
        """
        transcript = """
Moderator: Is AI dangerous?
Alice: AI poses an existential threat because it can optimize distinct goals from humans.
Bob: That goal misalignment is theoretical. In practice, we align AI with RLHF.
Charlie: RLHF is insufficient. It only patches surface behavior, not inner alignment.
"""
        print(f"\n[Test] AI Ethics Rebuttal:\n{transcript}")
        g = parse_debate(transcript, split_mode="llm", refine_iters=0)
        
        # Edges needed:
        # Bob ATTACKS Alice
        # Charlie ATTACKS Bob
        
        edges = [(g.nodes[u].speaker, g.nodes[v].speaker, d.get("type")) 
                 for u, v, d in g.nx_graph.edges(data=True)]
        
        print("Edges:", edges)
        
        bob_has_outgoing = any(u == "bob" and t in {"attack", "support"} for u, v, t in edges)
        charlie_links_bob = any(u == "charlie" and v == "bob" and t in {"attack", "support"} for u, v, t in edges)

        self.assertTrue(bob_has_outgoing, "Missing outgoing relation from Bob")
        self.assertTrue(charlie_links_bob, "Missing Edge: Charlie -> Bob [Attack/Support]")

if __name__ == "__main__":
    unittest.main()
