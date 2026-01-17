# test_solver_live_smoke.py
from src.core.graph import ArgumentationGraph, ArgumentNode
from src.core.solver import MaVERiCSolver

def make_graph():
    g = ArgumentationGraph()
    g.add_node(ArgumentNode(id="A1", content="2 + 2 equals 4.", speaker="S"))
    g.add_node(ArgumentNode(id="A2", content="2 + 2 equals 5.", speaker="L"))
    g.add_attack("A2", "A1")
    return g

if __name__ == "__main__":
    g = make_graph()

    # force root to A1 for deterministic test
    g.root_id_override = "A1"

    solver = MaVERiCSolver(graph=g, budget=5.0, topk_counterfactual=10)
    ext, verdict = solver.run()

    print("Final extension:", ext)
    print("Verdict:", verdict)
    print("Tool calls:", solver.tool_calls)
