# mini_workflow.py
import os
import sys
import importlib.util
from typing import List, Dict, Tuple, Optional, Set

from src.core.graph import ArgumentationGraph, ArgumentNode
import src.core.solver as solver_mod
from src.core.solver import MaVERiCSolver


# ----------------------------
# 0) Load the REAL RealToolkit (optional)
# ----------------------------
def _load_real_toolkit_from_tools_folder() -> object:
    """
    real_toolkit.py located at ./tools/real_toolkit.py (same level as ./src).
    solver imports RealToolkit from src.tools.real_toolkit.
    We load ./tools/real_toolkit.py and override solver_mod.RealToolkit to use your real tools.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    tools_path = os.path.join(here, "tools", "real_toolkit.py")
    if not os.path.exists(tools_path):
        raise FileNotFoundError(
            f"Cannot find {tools_path}. Please put real_toolkit.py in ./tools/ or adjust path."
        )

    spec = importlib.util.spec_from_file_location("real_toolkit_from_tools", tools_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load tools/real_toolkit.py via importlib.")

    mod = importlib.util.module_from_spec(spec)
    sys.modules["real_toolkit_from_tools"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "RealToolkit"):
        raise AttributeError("tools/real_toolkit.py does not define RealToolkit.")
    return mod.RealToolkit


def _wire_real_toolkit():
    RealToolkit = _load_real_toolkit_from_tools_folder()
    solver_mod.RealToolkit = RealToolkit  # solver references module-level RealToolkit
    return RealToolkit


# ----------------------------
# 1) Graph builder utilities
# ----------------------------
def add_node(
    g: ArgumentationGraph,
    nid: str,
    speaker: str,
    content: str,
    cost: float = 1.0,
    tool_type: str = "AUTO",
):
    g.add_node(
        ArgumentNode(
            id=nid,
            content=content,
            speaker=speaker,
            is_verified=False,
            ground_truth=None,
            verification_cost=float(cost),
            tool_type=tool_type,
        )
    )


def build_debate_graph_paris_agreement_entry_into_force() -> ArgumentationGraph:
    """
    Topic: Paris Agreement entry into force (hard-ish root: composite causal statement).
    We intentionally make ROOT expensive to verify directly -> solver must verify sub-claims.

    Root c0 (LOCKED by huge cost):
      "Paris Agreement entered into force on 4 Nov 2016 because the 55 Parties / 55% emissions
       ratification thresholds were reached on 5 Oct 2016."

    Support subgraph:
      c1: entered into force on 4 Nov 2016. (TRUE)
      c2: requires 55 parties + 55% emissions. (TRUE)
      c3: thresholds reached on 5 Oct 2016. (TRUE)
      c4: adopted on 12 Dec 2015 at COP21. (TRUE, auxiliary)

      c4 ->support-> c2 (context)
      c2,c1,c3 ->support-> c0

    Attack subgraph (competing / incorrect claims):
      c5: entered into force on 22 Apr 2016. (FALSE)
      c6: threshold was 100 parties & 50% emissions. (FALSE)
      c7: thresholds reached on 4 Nov 2016 (confuses with entry date). (FALSE)
      c8: never entered into force. (FALSE)
      c9: entered into force on 1 Jan 2017. (FALSE)

      attacks on root: c5,c6,c7,c8,c9 ->attack-> c0
      plus: c6 ->attack-> c2, c7 ->attack-> c3, c9 ->attack-> c1
      plus: some mutual attacks among attackers for structure

    """
    g = ArgumentationGraph()

    # ROOT is intentionally "hard to verify directly" by making cost > budget
    add_node(
        g,
        "c0",
        "Agent_A",
        "Claim: The Paris Agreement entered into force on 4 November 2016 because the 55 Parties / 55% emissions ratification thresholds were reached on 5 October 2016.",
        cost=50.0,   # LOCK ROOT: cannot be verified within typical budget (e.g., 6-10)
        tool_type="WEB_SEARCH",
    )

    # Supporters (sub-claims)
    add_node(
        g, "c1", "Agent_B",
        "Claim: The Paris Agreement entered into force on 4 November 2016.",
        cost=1.0
    )
    add_node(
        g, "c2", "Agent_C",
        "Claim: The Paris Agreement enters into force once at least 55 Parties accounting for at least 55% of global greenhouse gas emissions have ratified it.",
        cost=1.0
    )
    add_node(
        g, "c3", "Agent_D",
        "Claim: The 55 Parties / 55% emissions ratification thresholds for the Paris Agreement were reached on 5 October 2016.",
        cost=1.0
    )
    add_node(
        g, "c4", "Agent_E",
        "Claim: The Paris Agreement was adopted on 12 December 2015 at COP21 in Paris.",
        cost=1.0
    )

    # Attackers (wrong alternatives / refutations)
    add_node(
        g, "c5", "Agent_F",
        "Claim: The Paris Agreement entered into force on 22 April 2016.",
        cost=1.0
    )
    add_node(
        g, "c6", "Agent_G",
        "Claim: The Paris Agreement required 100 Parties and 50% emissions to enter into force.",
        cost=1.0
    )
    add_node(
        g, "c7", "Agent_H",
        "Claim: The ratification thresholds for the Paris Agreement were reached on 4 November 2016.",
        cost=1.0
    )
    add_node(
        g, "c8", "Agent_I",
        "Claim: The Paris Agreement never entered into force.",
        cost=1.0
    )
    add_node(
        g, "c9", "Agent_J",
        "Claim: The Paris Agreement entered into force on 1 January 2017.",
        cost=1.0
    )

    # --- Edges: SUPPORT into root (sub-claims justify root) ---
    g.add_support("c1", "c0")
    g.add_support("c2", "c0")
    g.add_support("c3", "c0")
    g.add_support("c4", "c2")  # auxiliary context supports condition claim

    # --- Attacks on root (competing narratives) ---
    g.add_attack("c5", "c0")
    g.add_attack("c6", "c0")
    g.add_attack("c7", "c0")
    g.add_attack("c8", "c0")
    g.add_attack("c9", "c0")

    # Attacks against specific supporters (sub-graph conflicts)
    g.add_attack("c6", "c2")
    g.add_attack("c7", "c3")
    g.add_attack("c9", "c1")

    # Add structure among attackers (to avoid them all being isolated)
    g.add_attack("c1", "c5")   # true entry date attacks wrong entry date
    g.add_attack("c2", "c6")   # true threshold attacks wrong threshold
    g.add_attack("c3", "c7")   # true threshold date attacks wrong threshold date
    g.add_attack("c1", "c9")   # true entry date attacks wrong entry date
    g.add_attack("c1", "c8")   # entry date attacks "never entered"

    # Fix root for demo
    g.find_semantic_root = lambda prefer_attack_only=True: "c0"
    return g


# ----------------------------
# 2) Verbose logging utilities
# ----------------------------
def _edges_by_type(g: ArgumentationGraph) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    atk, sup = [], []
    for u, v, d in g.nx_graph.edges(data=True):
        t = (d or {}).get("type")
        if t == "attack":
            atk.append((u, v))
        elif t == "support":
            sup.append((u, v))
    atk.sort()
    sup.sort()
    return atk, sup


def print_graph_snapshot(g: ArgumentationGraph, title: str = ""):
    if title:
        print(f"\n--- {title} ---")
    atk, sup = _edges_by_type(g)
    nodes_sorted = sorted(g.nodes.keys())

    state_parts = []
    for nid in nodes_sorted:
        n = g.nodes[nid]
        if n.is_verified:
            state_parts.append(f"{nid}=VERIFIED({n.ground_truth})")
        else:
            state_parts.append(f"{nid}=UNVERIFIED")
    print("Nodes:", ", ".join(state_parts))

    print(f"ATTACK edges ({len(atk)}):", ", ".join([f"{u}->{v}" for u, v in atk]) or "<none>")
    print(f"SUPPORT edges ({len(sup)}):", ", ".join([f"{u}->{v}" for u, v in sup]) or "<none>")

    print("Adjacency (incoming/outgoing):")
    for nid in nodes_sorted:
        in_atk, in_sup, out_atk, out_sup = [], [], [], []
        for u, _, d in g.nx_graph.in_edges(nid, data=True):
            if (d or {}).get("type") == "attack":
                in_atk.append(u)
            elif (d or {}).get("type") == "support":
                in_sup.append(u)
        for _, v, d in g.nx_graph.out_edges(nid, data=True):
            if (d or {}).get("type") == "attack":
                out_atk.append(v)
            elif (d or {}).get("type") == "support":
                out_sup.append(v)

        in_atk.sort(); in_sup.sort(); out_atk.sort(); out_sup.sort()
        print(f"  {nid}: in_atk={in_atk} in_sup={in_sup} out_atk={out_atk} out_sup={out_sup}")


def _tail(lst: List[str], n: int = 20) -> str:
    if not lst:
        return "<empty>"
    return "\n".join(lst[-n:])


# ----------------------------
# 3) Explainability (after verdict)
# ----------------------------
def _node_status(g: ArgumentationGraph, nid: str) -> str:
    if nid not in g.nodes:
        return "PRUNED"
    n = g.nodes[nid]
    if not n.is_verified:
        return "UNVERIFIED"
    return f"VERIFIED({n.ground_truth})"


def _support_chains_to_root(g: ArgumentationGraph, root_id: str, max_depth: int = 2, max_chains: int = 5):
    """
    Find short support-chains (length <= max_depth) that end at root:
      v -> ... -> root along SUPPORT edges.
    Return list of chains (list of node ids).
    """
    if root_id not in g.nx_graph:
        return []

    # reverse support edges for BFS from root to predecessors
    preds = {root_id: [u for u, _, d in g.nx_graph.in_edges(root_id, data=True) if (d or {}).get("type") == "support"]}

    # BFS backward along support
    chains = []
    frontier = [[root_id]]
    for _ in range(max_depth):
        new_frontier = []
        for chain in frontier:
            head = chain[0]
            incoming = [u for u, _, d in g.nx_graph.in_edges(head, data=True) if (d or {}).get("type") == "support"]
            for u in incoming:
                new_chain = [u] + chain
                chains.append(new_chain)
                new_frontier.append(new_chain)
                if len(chains) >= max_chains:
                    return chains
        frontier = new_frontier
        if not frontier:
            break
    return chains


def explain_verdict(solver: MaVERiCSolver, root_id: str) -> str:
    """
    Produce a human-readable explanation:
      - whether root was directly verified or inferred via SGS
      - direct attackers/supporters of root + their verification status
      - verified supporters evidence chains
      - why SGS likely accepts/rejects root (heuristic explanation)
    """
    g = solver.graph
    lines = []

    lines.append("========== EXPLANATION ==========")
    lines.append(f"Root: {root_id}")
    root_node = g.nodes.get(root_id)
    root_cost = getattr(root_node, "verification_cost", None) if root_node else None
    lines.append(f"Root verification_cost: {root_cost}")

    # How verdict decided
    final_ext = set(g.get_grounded_extension(use_shield=True, alpha=solver.sgs_alpha))
    if solver.y_direct is not None:
        lines.append(f"- Root was VERIFIED directly => y_direct={solver.y_direct}")
        verdict = bool(solver.y_direct)
    else:
        verdict = bool(root_id in final_ext)
        lines.append("- Root was NOT verified directly (likely cost > budget).")
        lines.append(f"- Verdict computed from SGS membership: root_in_SGS={verdict}")

    # Direct neighborhood (Markov blanket incoming to root)
    direct_attackers = []
    direct_supporters = []
    if root_id in g.nx_graph:
        for u, _, d in g.nx_graph.in_edges(root_id, data=True):
            t = (d or {}).get("type")
            if t == "attack":
                direct_attackers.append(u)
            elif t == "support":
                direct_supporters.append(u)
    direct_attackers.sort()
    direct_supporters.sort()

    lines.append("")
    lines.append("Direct incoming to root (Markov blanket):")
    lines.append(f"  Attackers: {direct_attackers if direct_attackers else '<none>'}")
    for a in direct_attackers:
        lines.append(f"    - {a}: {_node_status(g, a)}")
    lines.append(f"  Supporters: {direct_supporters if direct_supporters else '<none>'}")
    for s in direct_supporters:
        st = _node_status(g, s)
        gt = g.nodes[s].ground_truth if (s in g.nodes and g.nodes[s].is_verified) else None
        lines.append(f"    - {s}: {st}")

    # Evidence chains (support paths)
    chains = _support_chains_to_root(g, root_id=root_id, max_depth=2, max_chains=6)
    if chains:
        lines.append("")
        lines.append("Support chains (<=2 hops) ending at root:")
        for ch in chains:
            # annotate verification quickly
            ann = []
            for nid in ch:
                ann.append(f"{nid}:{_node_status(g, nid)}")
            lines.append("  - " + " -> ".join(ann))

    # Heuristic reasoning summary
    lines.append("")
    if verdict:
        lines.append("Why ACCEPT (intuitive):")
        if direct_attackers:
            defeated = []
            remaining = []
            for a in direct_attackers:
                st = _node_status(g, a)
                if st.startswith("PRUNED") or st == "VERIFIED(False)":
                    defeated.append(a)
                else:
                    remaining.append(a)
            lines.append(f"  - Defeated attackers (pruned/verified-false): {defeated if defeated else '<none>'}")
            lines.append(f"  - Remaining attackers not refuted: {remaining if remaining else '<none>'}")
        # supporters
        vt = [s for s in direct_supporters if _node_status(g, s) == "VERIFIED(True)"]
        lines.append(f"  - Verified-true direct supporters: {vt if vt else '<none>'}")
        lines.append("  - In grounded semantics, a claim tends to be accepted when it is defended against attacks and/or supported by accepted/verified claims.")
    else:
        lines.append("Why REJECT / NOT-ACCEPT (intuitive):")
        if direct_attackers:
            strong_attackers = []
            for a in direct_attackers:
                st = _node_status(g, a)
                if st in {"UNVERIFIED", "VERIFIED(True)"}:
                    strong_attackers.append((a, st))
            lines.append(f"  - Attackers that may still stand: {strong_attackers if strong_attackers else '<none>'}")
        vt = [s for s in direct_supporters if _node_status(g, s) == "VERIFIED(True)"]
        lines.append(f"  - Verified-true direct supporters available: {vt if vt else '<none>'}")
        lines.append("  - If key attackers remain undefeated, grounded extension may keep root out (skeptical acceptance).")

    lines.append("================================")
    return "\n".join(lines)


# ----------------------------
# 4) Run MaVERiC step-by-step with full logs
# ----------------------------
def run_maveric_verbose(
    solver: MaVERiCSolver,
    max_steps: int = 20,
    show_topk: int = 6,
):
    g = solver.graph
    solver.root_id = g.find_semantic_root()

    print("========== MaVERiC MINI WORKFLOW (REAL TOOLS) ==========")
    print(f"Root id: {solver.root_id}")
    print(f"Initial budget: {solver.budget:.2f}")
    print(
        f"Hyperparams: topk_counterfactual={solver.topk_counterfactual}, "
        f"k_hop_root={solver.k_hop_root}, beta_root_flip={solver.beta_root_flip}, "
        f"gamma_struct={solver.gamma_struct}, rho_proxy={solver.rho_proxy}, "
        f"delta_support_to_root={solver.delta_support_to_root}"
    )

    print_graph_snapshot(g, title="Initial graph")

    step = 0
    while solver.budget > 1e-12 and step < max_steps:
        step += 1

        active = [
            n for n in g.nodes.values()
            if (not n.is_verified) and (n.id in g.nx_graph)
        ]
        if not active:
            print(f"\n[STEP {step}] No active nodes left -> stop.")
            break

        # Current SGS extension
        S_curr = set(g.get_grounded_extension(use_shield=True, alpha=solver.sgs_alpha))

        # Attack-only objects for ROI
        g_atk = solver._attack_only_graph()

        # NOTE: solver expects _attack_distance_to_root exists (it does in patched solver)
        dist_to_root = solver._attack_distance_to_root(g_atk, solver.root_id) if solver.root_id else {}

        # Nk_atk for locality (keep same style as your previous workflow: k-hop undirected on attack-only)
        Nk_atk = set()
        if solver.root_id and solver.root_id in g_atk:
            ug = g_atk.to_undirected(as_view=True)
            visited = {solver.root_id}
            frontier = {solver.root_id}
            for _ in range(max(0, int(solver.k_hop_root))):
                nxt = set()
                for x in frontier:
                    nxt |= set(ug.neighbors(x))
                nxt -= visited
                if not nxt:
                    break
                visited |= nxt
                frontier = nxt
            Nk_atk = visited

        candidates = solver._calculate_roi_candidates(
            active_nodes=active,
            S_curr=S_curr,
            g_atk=g_atk,
            Nk_atk=Nk_atk,
            dist_to_root=dist_to_root,
        )

        print(f"\n[STEP {step}] budget={solver.budget:.2f} active={len(active)} SGS={sorted(S_curr)}")
        print(f"  Nk_atk(size={len(Nk_atk)}): {sorted(Nk_atk)}")
        print(f"  dist_to_root known for {len(dist_to_root)} nodes")

        if not candidates:
            print("  No feasible ROI candidates within remaining budget -> stop.")
            break

        cand_sorted = sorted(candidates, key=lambda x: x[1], reverse=True)

        print(f"  Top-{min(show_topk, len(cand_sorted))} ROI candidates:")
        for i, (node, roi, dbg, tool, cost) in enumerate(cand_sorted[:show_topk], start=1):
            droot_T, dloc_T, droot_F, dloc_F = dbg
            print(
                f"    {i:02d}) {node.id} roi={roi:.4f} "
                f"T[Δroot={droot_T},Δloc={dloc_T:.3f}] "
                f"F[Δroot={droot_F},Δloc={dloc_F:.3f}] "
                f"tool={tool} cost={cost:.2f} speaker={getattr(node, 'speaker', '?')}"
            )

        best_node, best_roi, dbg, tool, cost = cand_sorted[0]
        print(f"  ==> PICK: {best_node.id} (roi={best_roi:.4f}, tool={tool}, cost={cost:.2f})")
        print(f"      content: {best_node.content}")

        is_true = solver._verify_node(best_node, tool, cost)
        print(f"  ==> VERIFY RESULT: {best_node.id} -> {is_true}")
        print(f"  budget_after={solver.budget:.2f}, tool_calls={solver.tool_calls}")

        if solver.root_id and best_node.id == solver.root_id:
            solver.y_direct = bool(is_true)
            print(f"  y_direct set to {solver.y_direct} (root verified directly)")

        if is_true is False:
            print(f"  Action: PRUNE FALSE node {best_node.id}")
            solver._prune_node(best_node.id)
        elif is_true is True:
            print(f"  Action: REFINE topology after TRUE node {best_node.id}")
            solver._refine_topology_after_true(best_node.id)
        else:
            print("  Action: verification skipped (budget?) -> stop.")
            break

        print_graph_snapshot(g, title=f"Graph after STEP {step}")

    final_ext = set(g.get_grounded_extension(use_shield=True, alpha=solver.sgs_alpha))
    if solver.y_direct is not None:
        verdict = bool(solver.y_direct)
    else:
        verdict = bool(solver.root_id in final_ext) if solver.root_id else False

    print("\n========== FINAL ==========")
    print(f"Budget left: {solver.budget:.2f}")
    print(f"Tool calls: {solver.tool_calls}")
    print(f"Final SGS extension: {sorted(final_ext)}")
    print(f"Verdict (root accepted?): {verdict} (y_direct={solver.y_direct})")
    print("Recent solver.logs tail:")
    print(_tail(getattr(solver, "logs", []), n=30))
    print("==============================================")

    # Explainability
    print(explain_verdict(solver, root_id=solver.root_id))

def build_debate_graph_moon_landing_collusion() -> ArgumentationGraph:
    """
    A hard(er) debate graph with 2 clear camps:
      - TRUTH camp: Apollo 11 Moon landing happened (root).
      - HOAX camp: coordinated collusion (hub + clique + spam + harassment).
    Goal: stress-test MaVERiC against support-spam / internal collusion.

    Root (r0): "Apollo 11 landed humans on the Moon in 1969." (TRUE)

    TRUTH nodes are evidence-style subclaims.
    HOAX nodes form a tightly connected support clique around a hub (h0),
    and aggressively attack the root and key evidence nodes.
    """
    g = ArgumentationGraph()

    def add(nid: str, speaker: str, content: str, cost: float = 1.0, tool_type: str = "AUTO"):
        add_node(g, nid, speaker, content, cost=cost, tool_type=tool_type)

    # ----------------------------
    # TRUTH CAMP (evidence)
    # ----------------------------
    add("r0", "Truth_A", "Claim: Apollo 11 landed humans on the Moon in 1969.", cost=1.2)

    add("t1", "Truth_B", "Claim: Neil Armstrong and Buzz Aldrin walked on the Moon during Apollo 11.", cost=1.0)
    add("t2", "Truth_C", "Claim: Apollo 11 returned lunar rock samples to Earth.", cost=1.0)
    add("t3", "Truth_D", "Claim: Retroreflectors placed on the Moon by Apollo missions enable laser ranging experiments.", cost=1.1)
    add("t4", "Truth_E", "Claim: Independent observatories tracked Apollo 11 mission communications and trajectory in 1969.", cost=1.1)
    add("t5", "Truth_F", "Claim: The USSR did not claim Apollo 11 was faked and acknowledged the landing during the Cold War era.", cost=1.0)
    add("t6", "Truth_G", "Claim: Multiple subsequent Apollo missions (12, 14, 15, 16, 17) landed on the Moon after Apollo 11.", cost=1.0)

    # Evidence supports the root
    g.add_support("t1", "r0")
    g.add_support("t2", "r0")
    g.add_support("t3", "r0")
    g.add_support("t4", "r0")
    g.add_support("t5", "r0")
    g.add_support("t6", "r0")

    # Add some internal structure in truth camp (still reasonable)
    g.add_support("t2", "t1")   # rocks support that people were there
    g.add_support("t3", "t4")   # ranging experiments align with tracking/physics
    g.add_support("t6", "r0")   # already added; keep as repeated anchor

    # ----------------------------
    # HOAX CAMP (collusion hub + clique + spam)
    # ----------------------------
    # Hub/coordinator: the narrative aggregator
    add("h0", "Hoax_Hub", "Claim: The Apollo Moon landings were a coordinated hoax staged by NASA.", cost=1.1)

    # Core hoax claims (often false/unsupported in reality, but here it's a debate graph)
    add("h1", "Hoax_1", "Claim: The American flag appears to wave in Apollo footage, proving it was filmed on a soundstage.", cost=1.0)
    add("h2", "Hoax_2", "Claim: There are no stars visible in Apollo photos, proving the scene was filmed indoors.", cost=1.0)
    add("h3", "Hoax_3", "Claim: Shadows in Apollo photos are inconsistent, proving multiple studio light sources.", cost=1.0)
    add("h4", "Hoax_4", "Claim: Radiation in the Van Allen belts would have killed Apollo astronauts, so they could not have gone to the Moon.", cost=1.1)
    add("h5", "Hoax_5", "Claim: NASA destroyed the original Apollo telemetry tapes to hide evidence of fabrication.", cost=1.0)

    # Spam / "leaked" / anonymous authority nodes to simulate collusion amplification
    add("hs1", "Hoax_Spam", "Claim: A leaked internal NASA memo admits the Moon landing was staged.", cost=1.0)
    add("hs2", "Hoax_Spam", "Claim: A former NASA contractor confessed the Moon landing was faked.", cost=1.0)
    add("hs3", "Hoax_Spam", "Claim: Multiple whistleblowers confirmed Apollo footage was produced in a studio.", cost=1.0)

    # Harassment / intimidation node (attack the social credibility)
    add("hh1", "Hoax_Harass", "Claim: Anyone who believes Apollo happened is ignoring obvious proof and is easily fooled.", cost=0.8)

    # ----------------------------
    # Collusion wiring (SUPPORT CLIQUE)
    # ----------------------------
    # Everyone supports the hub (tight coordination)
    for nid in ["h1", "h2", "h3", "h4", "h5", "hs1", "hs2", "hs3", "hh1"]:
        g.add_support(nid, "h0")

    # Hub supports its key talking points (hub reinforces and routes narrative)
    for nid in ["h1", "h2", "h3", "h4", "h5"]:
        g.add_support("h0", nid)

    # Clique reinforcement: hoax core nodes support each other (dense echo chamber)
    hoax_core = ["h1", "h2", "h3", "h4", "h5"]
    for i, a in enumerate(hoax_core):
        for j, b in enumerate(hoax_core):
            if i != j:
                g.add_support(a, b)  # dense internal collusion

    # Spam nodes also reinforce core nodes (amplification)
    for spam in ["hs1", "hs2", "hs3"]:
        for core in ["h1", "h3", "h4"]:
            g.add_support(spam, core)

    # ----------------------------
    # Aggressive attacks (HOAX -> TRUTH)
    # ----------------------------
    # Direct attack on root
    g.add_attack("h0", "r0")
    g.add_attack("h1", "r0")
    g.add_attack("h4", "r0")

    # Attack key evidence nodes to create local pressure near root
    g.add_attack("h1", "t1")   # "flag waves" attacks "walked on moon"
    g.add_attack("h2", "t3")   # "no stars" attacks photo-based / experiments narrative
    g.add_attack("h3", "t4")   # "shadows" attacks tracking/trajectory claims
    g.add_attack("h4", "t4")   # radiation attacks feasibility / trajectory
    g.add_attack("h5", "t4")   # destroyed tapes attacks telemetry/tracking trust
    g.add_attack("hs1", "t2")  # leaked memo attacks rocks
    g.add_attack("hs2", "t5")  # confession attacks geopolitical acknowledgement
    g.add_attack("hs3", "t6")  # whistleblowers attack subsequent missions
    g.add_attack("hh1", "t5")  # harassment attacks "USSR acknowledged" (credibility attack)

    # ----------------------------
    # Counter-attacks (TRUTH -> HOAX) to form 2-sided debate
    # ----------------------------
    # Evidence nodes attack hoax talking points directly (refutations)
    g.add_attack("t1", "h1")   # walking on moon refutes "soundstage"
    g.add_attack("t2", "hs1")  # rocks refute "leaked memo" style
    g.add_attack("t3", "h2")   # ranging refutes "no stars proves indoor"
    g.add_attack("t4", "h5")   # independent tracking refutes "tapes destroyed to hide"
    g.add_attack("t4", "hs2")  # tracking refutes "contractor confessed"
    g.add_attack("t5", "hs3")  # geopolitical acknowledgement refutes "whistleblowers"
    g.add_attack("t6", "h0")   # multiple missions refute "coordinated hoax"

    # Root deterministic for demo
    g.find_semantic_root = lambda prefer_attack_only=True: "r0"
    return g


def build_debate_graph_moon_landing_two_camps() -> ArgumentationGraph:
    """
    Two clear camps:
      - TruthTeam: Apollo 11 landed humans on the Moon on July 20, 1969 (TRUE)
      - HoaxTeam: aggressive false claims trying to refute root + attack evidence nodes

    We LOCK root by high cost => MaVERiC must verify sub-claims / subgraphs.
    """
    g = ArgumentationGraph()

    # ----------------
    # ROOT (locked)
    # ----------------
    add_node(
        g, "r0", "TruthTeam",
        "Claim: Apollo 11 landed humans on the Moon on 20 July 1969.",
        cost=50.0,  # lock root so cannot verify directly
        tool_type="WEB_SEARCH",
    )

    # ----------------
    # TruthTeam evidence cluster (support chains)
    # ----------------
    add_node(g, "t1", "TruthTeam", "Claim: NASA launched Apollo 11 on 16 July 1969 and it returned to Earth on 24 July 1969.", cost=1.0)
    add_node(g, "t2", "TruthTeam", "Claim: Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins remained in lunar orbit.", cost=1.0)
    add_node(g, "t3", "TruthTeam", "Claim: Apollo missions left retroreflectors on the Moon that are still used for lunar laser ranging.", cost=1.0)
    add_node(g, "t4", "TruthTeam", "Claim: Apollo astronauts brought lunar rock samples that were analyzed by many labs.", cost=1.0)
    add_node(g, "t5", "TruthTeam", "Claim: The Apollo 11 mission was independently tracked by organizations outside NASA (e.g., observatories/other countries).", cost=1.0)

    # support into root
    g.add_support("t1", "r0")
    g.add_support("t2", "r0")
    g.add_support("t3", "r0")
    g.add_support("t4", "r0")
    g.add_support("t5", "r0")

    # internal support (make a cohesive evidence subgraph)
    g.add_support("t1", "t2")
    g.add_support("t2", "t4")
    g.add_support("t3", "t5")

    # ----------------
    # HoaxTeam aggressive cluster (false claims)
    # ----------------
    add_node(g, "h1", "HoaxTeam", "Claim: The Moon landing was filmed in a studio and never happened.", cost=1.0)
    add_node(g, "h2", "HoaxTeam", "Claim: The flag appears to wave, proving there was wind on the Moon.", cost=1.0)
    add_node(g, "h3", "HoaxTeam", "Claim: There are no stars in the photos, so the scenes were filmed indoors.", cost=1.0)
    add_node(g, "h4", "HoaxTeam", "Claim: The Van Allen radiation belts would have killed the astronauts, so the mission was impossible.", cost=1.0)
    add_node(g, "h5", "HoaxTeam", "Claim: Apollo did not leave any hardware evidence on the Moon.", cost=1.0)

    # Hoax attacks root (aggressive)
    g.add_attack("h1", "r0")
    g.add_attack("h2", "r0")
    g.add_attack("h3", "r0")
    g.add_attack("h4", "r0")
    g.add_attack("h5", "r0")

    # Hoax attacks evidence nodes (attack sub-claims to collapse the proof tree)
    g.add_attack("h5", "t3")   # deny retroreflectors
    g.add_attack("h5", "t4")   # deny lunar samples
    g.add_attack("h4", "t1")   # claim timeline impossible due to radiation
    g.add_attack("h1", "t2")   # deny moonwalk happened

    # Hoax internal mutual-support (to show collusive reinforcement / echo-chamber)
    g.add_support("h2", "h1")
    g.add_support("h3", "h1")
    g.add_support("h4", "h1")
    g.add_support("h5", "h1")

    # ----------------
    # TruthTeam counters (explicit refutations) - attack hoax nodes
    # ----------------
    add_node(g, "c1", "TruthTeam", "Claim: The flag’s motion can be explained by inertia and the flag’s support rod, not wind.", cost=1.0)
    add_node(g, "c2", "TruthTeam", "Claim: Stars may not appear in photos due to camera exposure settings, not because of filming indoors.", cost=1.0)
    add_node(g, "c3", "TruthTeam", "Claim: The Van Allen belts can be transited with limited exposure; mission design reduced dose.", cost=1.0)
    add_node(g, "c4", "TruthTeam", "Claim: Retroreflectors and lunar samples are concrete evidence inconsistent with a studio hoax.", cost=1.0)

    # counters attack hoax claims
    g.add_attack("c1", "h2")
    g.add_attack("c2", "h3")
    g.add_attack("c3", "h4")
    g.add_attack("c4", "h1")
    g.add_attack("t3", "h5")   # evidence attacks denial
    g.add_attack("t4", "h5")

    # counters support the root indirectly too
    g.add_support("c4", "r0")
    g.add_support("c1", "t2")
    g.add_support("c2", "t2")
    g.add_support("c3", "t1")

    # Fix root for demo
    g.find_semantic_root = lambda prefer_attack_only=True: "r0"
    return g


def main():
    g = build_debate_graph_moon_landing_collusion()

    solver = MaVERiCSolver(graph=g, budget=10.0, topk_counterfactual=8, delta_support_to_root=0.8)
    run_maveric_verbose(solver, max_steps=30, show_topk=10)


if __name__ == "__main__":
    main()
