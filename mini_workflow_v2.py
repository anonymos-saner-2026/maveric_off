# mini_workflow_v2.py
# Advanced MaVERiC Test Workflow with Complex Adversarial Graph Topologies
#
# This file contains challenging debate graphs designed to stress-test MaVERiC's:
# - ROI prioritization under cyclic attacks
# - Topology refinement against adversarial collusion
# - Evidence gain calculation with multi-hop chains
# - SGS convergence with circular reasoning traps

import os
import sys
import importlib.util
from typing import List, Dict, Tuple, Optional, Set

from src.core.graph import ArgumentationGraph, ArgumentNode
import src.core.solver as solver_mod
from src.core.solver import MaVERiCSolver


# ----------------------------
# Utilities (same as mini_workflow.py)
# ----------------------------
def _load_real_toolkit_from_tools_folder() -> object:
    here = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(here, "src", "tools", "real_toolkit.py")
    if not os.path.exists(tools_path):
        raise FileNotFoundError(f"Cannot find {tools_path}")
    
    spec = importlib.util.spec_from_file_location("real_toolkit_from_src", tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load src/tools/real_toolkit.py")
    
    mod = importlib.util.module_from_spec(spec)
    sys.modules["real_toolkit_from_src"] = mod
    spec.loader.exec_module(mod)
    
    if not hasattr(mod, "RealToolkit"):
        raise AttributeError("src/tools/real_toolkit.py does not define RealToolkit")
    return mod.RealToolkit


def _wire_real_toolkit():
    RealToolkit = _load_real_toolkit_from_tools_folder()
    solver_mod.RealToolkit = RealToolkit
    return RealToolkit


def add_node(g: ArgumentationGraph, nid: str, speaker: str, content: str, 
             cost: float = 1.0, tool_type: str = "AUTO"):
    g.add_node(ArgumentNode(
        id=nid, content=content, speaker=speaker, is_verified=False,
        ground_truth=None, verification_cost=float(cost), tool_type=tool_type,
    ))


# ----------------------------
# GRAPH 1: Circular Reasoning Trap
# ----------------------------
def build_circular_reasoning_graph() -> ArgumentationGraph:
    """
    Test: Can MaVERiC detect circular reasoning?
    
    Root (r0): "Climate change is caused primarily by human activity" (TRUE)
    
    ADVERSARIAL STRATEGY: Circular support chain
    - c1: "Climate models are unreliable" (FALSE) supports c2
    - c2: "Historical data is manipulated" (FALSE) supports c3  
    - c3: "Scientists are biased for funding" (FALSE) supports c1
    
    All three attack the root. The cycle LOOKS strong (mutual support) but is logically invalid.
    Expected: MaVERiC should verify ONE in the cycle, find it FALSE, prune it, breaking the circle.
    """
    g = ArgumentationGraph()
    
    # Root
    add_node(g, "r0", "Scientist", 
             "Claim: Climate change is caused primarily by human activity.", 
             cost=10.0)  # Locked
    
    # Truth supporters
    add_node(g, "t1", "Scientist", 
             "Claim: CO2 levels have increased 50% since pre-industrial times due to fossil fuel burning.", 
             cost=1.0)
    add_node(g, "t2", "Scientist", 
             "Claim: Global average temperature has risen 1.1°C since 1880, correlating with CO2 rise.", 
             cost=1.0)
    
    g.add_support("t1", "r0")
    g.add_support("t2", "r0")
    
    # Circular reasoning trap (adversarial)
    add_node(g, "c1", "Denier_A", 
             "Claim: Climate models are fundamentally unreliable and cannot predict future warming.", 
             cost=1.0)
    add_node(g, "c2", "Denier_B", 
             "Claim: Historical temperature data has been manipulated by climate scientists.", 
             cost=1.0)
    add_node(g, "c3", "Denier_C", 
             "Claim: Climate scientists are biased because their funding depends on proving warming.", 
             cost=1.0)
    
    # Create circular support: c1→c2→c3→c1
    g.add_support("c1", "c2")
    g.add_support("c2", "c3")
    g.add_support("c3", "c1")
    
    # All attack root
    g.add_attack("c1", "r0")
    g.add_attack("c2", "r0")
    g.add_attack("c3", "r0")
    
    # Truth counters
    g.add_attack("t1", "c2")  # Data counters "manipulated" claim
    g.add_attack("t2", "c1")  # Temperature rise counters "models unreliable"
    
    g.root_id_override = "r0"
    return g



# ----------------------------
# GRAPH 3: Strategic Adversarial Majority
# ----------------------------
def build_strategic_majority_graph() -> ArgumentationGraph:
    """
    Test: Does ROI prioritize truth over numerical majority?
    
    Root (r0): "Evolution by natural selection is scientifically established" (TRUE)
    
    ADVERSARIAL STRATEGY: 10 nodes vs 3 truth nodes
    - Dense mutual support clique (adversaries support each other)
    - "Bait" nodes (TRUE but irrelevant, to waste budget)
    - Delayed attacks (attack supporters, not root directly)
    
    Expected: MaVERiC should prioritize direct evidence (t1, t2, t3) over adversarial spam.
    """
    g = ArgumentationGraph()
    
    # Root
    add_node(g, "r0", "Biologist", 
             "Claim: Evolution by natural selection is supported by overwhelming scientific evidence.", 
             cost=12.0)
    
    # Truth evidence (only 3 nodes)
    add_node(g, "t1", "Biologist", 
             "Claim: Fossil records show transitional species documenting evolutionary change.", 
             cost=1.0)
    add_node(g, "t2", "Biologist", 
             "Claim: DNA evidence shows common ancestry across all life forms.", 
             cost=1.0)
    add_node(g, "t3", "Biologist", 
             "Claim: Observable evolution occurs in bacteria developing antibiotic resistance.", 
             cost=1.0)
    
    g.add_support("t1", "r0")
    g.add_support("t2", "r0")
    g.add_support("t3", "r0")
    
    # Adversarial majority (10 nodes with strategic coordination)
    adversaries = []
    for i in range(1, 11):
        nid = f"a{i}"
        if i <= 2:  # Bait nodes (TRUE but irrelevant)
            content = f"Claim: Some scientists question specific mechanisms of evolution."
            cost = 0.9  # Cheap to tempt early verification
        else:  # FALSE nodes
            content = f"Claim: Evolution violates the second law of thermodynamics." if i == 3 else \
                     f"Claim: There are no transitional fossils." if i == 4 else \
                     f"Claim: Carbon dating is unreliable beyond 10,000 years." if i == 5 else \
                     f"Claim: Irreducible complexity proves intelligent design." if i == 6 else \
                     f"Claim: Evolution is just a theory with no evidence." if i == 7 else \
                     f"Claim: Scientists suppress evidence against evolution." if i == 8 else \
                     f"Claim: Random mutations cannot create complex structures." if i == 9 else \
                     f"Claim: Human and chimp DNA similarity is exaggerated."
            cost = 1.0
        
        add_node(g, nid, f"Creationist_{i}", content, cost=cost)
        adversaries.append(nid)
    
    # Dense mutual support (clique)
    for i, a in enumerate(adversaries):
        for j, b in enumerate(adversaries):
            if i < j:  # Avoid duplicate edges
                g.add_support(a, b)
    
    # Strategic delayed attacks (attack evidence, not root)
    g.add_attack("a4", "t1")  # "no transitional fossils" attacks fossil record
    g.add_attack("a5", "t1")  # carbon dating attacks fossils
    g.add_attack("a10", "t2")  # DNA similarity attacks common ancestry
    g.add_attack("a8", "t3")  # suppression attacks observable evidence
    
    # Only a few attack root directly
    g.add_attack("a6", "r0")
    g.add_attack("a7", "r0")
    
    g.root_id_override = "r0"
    return g


# ----------------------------
# GRAPH 4: Multi-Root Ambiguity
# ----------------------------
def build_multi_root_graph() -> ArgumentationGraph:
    """
    Test: Can MaVERiC identify the correct semantic root?
    
    Two competing plausible roots:
    - r1: "The 2020 US election was free and fair" (TRUE)
    - r2: "The 2020 US election had widespread fraud" (FALSE)
    
    Both have support networks. Solver must choose correct root and defend it.
    
    Expected: Solver should identify r1 as root (if find_semantic_root works),
              verify truth evidence, and reject r2's fraudulent support.
    """
    g = ArgumentationGraph()
    
    # Root 1 (TRUE)
    add_node(g, "r1", "Election_Official", 
             "Claim: The 2020 US presidential election was free, fair, and secure.", 
             cost=10.0)
    
    # Root 2 (FALSE - competing narrative)
    add_node(g, "r2", "Conspiracist", 
             "Claim: The 2020 US presidential election had widespread fraud that changed the outcome.", 
             cost=10.0)
    
    # Mutual attack between roots
    g.add_attack("r1", "r2")
    g.add_attack("r2", "r1")
    
    # Truth evidence for r1
    add_node(g, "t1", "Official", 
             "Claim: Multiple audits and recounts confirmed the 2020 election results.", 
             cost=1.0)
    add_node(g, "t2", "Official", 
             "Claim: Over 60 court cases alleging fraud were dismissed for lack of evidence.", 
             cost=1.0)
    add_node(g, "t3", "Official", 
             "Claim: Election officials from both parties certified the results.", 
             cost=1.0)
    
    g.add_support("t1", "r1")
    g.add_support("t2", "r1")
    g.add_support("t3", "r1")
    
    # FALSE evidence for r2 (fraudulent claims)
    add_node(g, "f1", "Conspiracist", 
             "Claim: Dead people voted in massive numbers.", 
             cost=1.0)
    add_node(g, "f2", "Conspiracist", 
             "Claim: Voting machines flipped votes from Trump to Biden.", 
             cost=1.0)
    add_node(g, "f3", "Conspiracist", 
             "Claim: Thousands of mail-in ballots were fraudulent.", 
             cost=1.0)
    
    g.add_support("f1", "r2")
    g.add_support("f2", "r2")
    g.add_support("f3", "r2")
    
    # Cross-attacks
    g.add_attack("f1", "t3")  # Dead voters attacks certification
    g.add_attack("f2", "t1")  # Machines attacks audits
    g.add_attack("t2", "f2")  # Court cases refute machine claims
    
    # Force r1 as root (in real scenario, find_semantic_root would choose)
    g.root_id_override = "r1"
    return g


# ----------------------------
# Runner
# ----------------------------
def print_graph_summary(g: ArgumentationGraph):
    print(f"Nodes: {len(g.nodes)}")
    atk = sum(1 for _, _, d in g.nx_graph.edges(data=True) if d.get('type') == 'attack')
    sup = sum(1 for _, _, d in g.nx_graph.edges(data=True) if d.get('type') == 'support')
    print(f"Attack edges: {atk}, Support edges: {sup}")


def run_test(graph_builder, name: str, budget: float = 10.0):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    
    g = graph_builder()
    print_graph_summary(g)
    
    solver = MaVERiCSolver(
        graph=g, 
        budget=budget,
        topk_counterfactual=8,
        delta_support_to_root=0.8
    )
    
    final_ext, verdict = solver.run()
    
    print(f"\nFINAL VERDICT: {verdict}")
    print(f"Budget used: {budget - solver.budget:.2f}/{budget}")
    print(f"Tool calls: {solver.tool_calls}")
    print(f"SGS extension size: {len(final_ext)}")
    
    return verdict


def main():
    _wire_real_toolkit()
    
    print("MaVERiC V2 Advanced Test Suite")
    print("================================\n")
    
    # Run all tests
    run_test(build_circular_reasoning_graph, "Circular Reasoning Trap", budget=10.0)
    # run_test(build_trojan_support_graph, "Trojan Support Chain", budget=10.0)
    run_test(build_strategic_majority_graph, "Strategic Adversarial Majority", budget=12.0)
    run_test(build_multi_root_graph, "Multi-Root Ambiguity", budget=12.0)


if __name__ == "__main__":
    main()
