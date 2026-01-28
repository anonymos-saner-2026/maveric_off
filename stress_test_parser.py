
import logging
from collections import Counter
from src.agents import parser
from src.agents.parser import parse_debate, _needs_atomic_split
from src.agents.parser_llm import parse_debate_llm
import networkx as nx
import time

# Transcript: Nuclear Energy Debate
# Designed to be LONG, COMPLEX, and REPETITIVE.
# Characters: Alice (Pro-Nuclear), Bob (Anti-Nuclear), Charlie (Neutral/Economist), David (Environmentalist)
TRANSCRIPTS = {
    "nuclear": """
Moderator: Today we debate: Should the world double down on Nuclear Energy?

Alice: Nuclear energy is essential for a carbon-free future. It provides baseload power that renewables like solar and wind cannot match.
Bob: That is a dangerous gamble. Nuclear waste remains radioactive for thousands of years. We have no safe way to store it.
Alice: Actually, modern storage solutions like deep geological repositories are safe. Finland is already building one.
Charlie: The cost is the real issue. Building a nuclear plant takes 10-15 years and costs billions. Solar is much cheaper now.
David: But solar takes up too much land. Nuclear has a tiny land footprint compared to solar farms.
Bob: It's not just land. It is about safety! Chernobyl and Fukushima showed us that accidents are catastrophic.
Alice: Fukushima was decades-old technology. Modern Gen IV reactors are meltdown-proof. They rely on passive safety systems.
Bob: No technology is fail-safe. Human error always exists. If a meltdown happens, it destroys the region for centuries.
Charlie: Let's talk about economics again. The LCOE (Levelized Cost of Energy) for nuclear is $150/MWh. Solar is under $40/MWh.
Alice: LCOE ignores system costs. You need batteries for solar. If you add backup batteries, solar becomes expensive.
David: Batteries are getting cheaper too. But Alice is right about density. We cannot power a city with just rooftops.
Bob: Nuclear waste is the dealbreaker. It is toxic forever. We are leaving a poison legacy for our children.
Alice: Nuclear waste is actually recyclable. France recycles 96% of its spent fuel.
Bob: Reprocessing creates plutonium, which is a proliferation risk. It increases the chance of nuclear war.
Charlie: Proliferation is a political issue, not technological. But the timeline is fatal. We need to cut emissions NOW, not in 15 years.
Alice: SMRs (Small Modular Reactors) can be built faster. They can be factory-assembled.
Bob: SMRs are unproven. They are just a paper design. None have been commercially deployed at scale.
Alice: NuScale has a design approved by the NRC. It is real progress.
David: We should keep existing plants open but not build new ones. That is the middle ground.
Charlie: Keeping old plants is cost-effective. Building new ones is financial suicide.
Alice: Nuclear is the only way to decarbonize industry. Hydrogen produced by nuclear heat can replace fossil fuels in steelmaking.
Bob: Green hydrogen can be made from wind too. We don't need dangerous reactors for that.
Alice: Wind is intermittent. Industry needs 24/7 heat. Nuclear provides 24/7 heat.
Charlie: The market has spoken. Investors are funding renewables, not nuclear. Nuclear requires massive government subsidies.
Alice: All energy is subsidized. Oil and gas get subsidies. Renewables get tax credits. Nuclear deserves support too.
""",
    "social_media": """
Moderator: Topic: Does social media harm teen mental health?

Alice: Social media increases anxiety and depression in teenagers.
Bob: That is overstated. Some studies find no clear causal link.
Alice: But time spent online replaces sleep, which worsens mental health.
Charlie: Teens also face school pressure. Social media is not the only factor.
David: Heavy use correlates with self-esteem issues, especially for girls.
Bob: Correlation is not causation. Teens with anxiety might use social media more.
Alice: Platforms are designed to be addictive, which makes quitting difficult.
Charlie: There are benefits too. Social media helps teens find support communities.
David: But online harassment and comparison are real harms.
Bob: Parental guidance reduces risk. The impact is not uniform.
""",
    "ai_jobs": """
Moderator: Topic: Will AI eliminate more jobs than it creates?

Alice: Automation will replace many routine jobs in manufacturing and services.
Bob: New industries will create jobs, just like past technological revolutions.
Alice: The speed of AI adoption is faster than reskilling can keep up.
Charlie: Job displacement is real, but productivity gains can raise wages.
David: Some workers will be left behind without safety nets.
Bob: AI tools can augment workers rather than replace them.
Alice: Small businesses may not afford AI, widening inequality.
Charlie: Policy can smooth transitions through training programs.
David: Short-term pain is likely even if long-term gains exist.
Bob: History shows net job growth over time.
""",
    "nutrition": """
Moderator: Topic: Is a high-protein diet safe for most adults?

Alice: High-protein diets help preserve muscle mass and support weight loss.
Bob: Excess protein can strain the kidneys in people with existing kidney disease.
Charlie: Most healthy adults can tolerate higher protein without harm.
David: Many high-protein diets are low in fiber, which harms gut health.
Bob: Fiber deficiency increases constipation and gut inflammation.
Alice: Lean protein sources can be paired with vegetables to avoid fiber issues.
Charlie: The evidence is mixed, but moderation is recommended.
""",
    "remote_work": """
Moderator: Topic: Is remote work better for productivity?

Alice: Remote work reduces commute time, which increases focus.
Bob: Remote work harms collaboration and knowledge sharing.
Charlie: Hybrid models balance focus time with collaboration.
David: Remote work can improve work-life balance but blur boundaries.
Bob: Some employees feel isolated, lowering morale.
Alice: Teams can use async tools to mitigate collaboration issues.
Charlie: Productivity gains depend on role and management practices.
""",
    "education": """
Moderator: Topic: Does standardized testing improve education?

Alice: Standardized tests provide objective benchmarks across schools.
Bob: Teaching to the test narrows the curriculum and reduces creativity.
Charlie: Tests reveal achievement gaps that need policy attention.
David: High-stakes tests increase student anxiety and stress.
Alice: Without testing, it is harder to hold schools accountable.
Bob: Alternative assessments can measure critical thinking better.
Charlie: Some accountability is necessary, but tests should be limited.
""",
    "climate_policy": """
Moderator: Topic: Should carbon taxes be the primary climate policy?

Alice: Carbon taxes efficiently reduce emissions by pricing pollution.
Bob: Carbon taxes are regressive and hurt low-income households.
Charlie: Rebates can offset regressive impacts for households.
David: Regulations provide certainty that emissions will fall.
Alice: A predictable price signal drives clean investment.
Bob: Some industries will relocate if taxes rise.
Charlie: Border adjustments can reduce leakage.
""",
}

def analyze_graph(g):
    print("\n" + "="*50)
    print("GRAPH METRICS REPORT")
    print("="*50)

    # 1. Node Count
    num_nodes = len(g.nodes)
    print(f"Total Nodes: {num_nodes}")
    if num_nodes < 10:
        print("⚠️ WARNING: Node count surprisingly low for such a long debate.")
    elif num_nodes > 50:
        print("⚠️ WARNING: Node count very high. Possible fragmentation.")
    else:
        print("✅ Node count falls in expected range (10-50).")

    # 2. Average Word Count (Atomicity)
    total_words = sum(len(n.content.split()) for n in g.nodes.values())
    avg_words = total_words / num_nodes if num_nodes else 0
    print(f"Avg Words/Node: {avg_words:.1f}")
    if avg_words > 20:
        print("⚠️ WARNING: Nodes seem verbose. Atomic splitting might be weak.")

    # 3. Connectivity
    num_edges = g.nx_graph.number_of_edges()
    print(f"Total Edges: {num_edges}")
    if num_edges < num_nodes / 2:
        print("⚠️ WARNING: Very sparse graph.")

    # 4. Degree stats
    in_deg = dict(g.nx_graph.in_degree())
    out_deg = dict(g.nx_graph.out_degree())
    no_in = [nid for nid, deg in in_deg.items() if deg == 0]
    no_out = [nid for nid, deg in out_deg.items() if deg == 0]
    print(f"Nodes with no in-edge: {len(no_in)}")
    print(f"Nodes with no out-edge: {len(no_out)}")

    # 5. Component stats
    import networkx as nx
    wcc = list(nx.weakly_connected_components(g.nx_graph))
    sizes = sorted([len(c) for c in wcc], reverse=True)
    print(f"Weakly connected components: {len(wcc)}")
    if sizes:
        print(f"Top component sizes: {sizes[:5]}")

    # 6. Coreference Check
    pronoun_nodes = [n.content for n in g.nodes.values() if " it " in f" {n.content.lower()} "]
    if pronoun_nodes:
        print(f"\n❌ Coreference Issues Found ({len(pronoun_nodes)} nodes contain 'it'):")
        for c in pronoun_nodes[:3]:
            print(f"   - {c}")
    else:
        print("\n✅ Coreference Check Passed (No 'it' found).")

    # 7. Atomicity violation check
    atomic_offenders = [n.content for n in g.nodes.values() if _needs_atomic_split(n.content)]
    print(f"Atomicity warnings: {len(atomic_offenders)}")
    for c in atomic_offenders[:3]:
        print(f"   - {c}")

    # 8. Edge type balance
    attack_edges = sum(1 for _, _, d in g.nx_graph.edges(data=True) if d.get("type") == "attack")
    support_edges = sum(1 for _, _, d in g.nx_graph.edges(data=True) if d.get("type") == "support")
    total_edges = g.nx_graph.number_of_edges()
    if total_edges:
        attack_ratio = attack_edges / total_edges
    else:
        attack_ratio = 0.0
    print(f"Attack edges: {attack_edges}")
    print(f"Support edges: {support_edges}")
    print(f"Attack ratio: {attack_ratio:.2f}")

    # 9. Orphan nodes (no in-edge)
    orphans = [nid for nid, deg in in_deg.items() if deg == 0 and nid != "A1"]
    print(f"Orphan nodes (no in-edge, not root): {len(orphans)}")

    # 10. Edge direction sanity (negation conflict)
    negations = {"no", "not", "never", "cannot", "can't", "without"}
    neg_edges = 0
    for u, v, d in g.nx_graph.edges(data=True):
        src = g.nodes[u].content.lower()
        dst = g.nodes[v].content.lower()
        src_neg = any(w in src.split() for w in negations)
        dst_neg = any(w in dst.split() for w in negations)
        if src_neg != dst_neg and d.get("type") == "support":
            neg_edges += 1
    print(f"Potential support/negation conflicts: {neg_edges}")

    # 11. Edge density + reciprocity
    if num_nodes > 1:
        density = num_edges / (num_nodes * (num_nodes - 1))
    else:
        density = 0.0
    reciprocal = sum(1 for u, v in g.nx_graph.edges() if g.nx_graph.has_edge(v, u))
    print(f"Edge density: {density:.3f}")
    print(f"Reciprocal edges: {reciprocal}")

    # 12. Degree distribution
    degree_counts = Counter(dict(g.nx_graph.degree()).values())
    print(f"Degree distribution (top 5): {degree_counts.most_common(5)}")

    # 13. Top hubs by in-degree
    top_in = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top in-degree nodes: {top_in}")

    # 14. Top hubs by out-degree
    top_out = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top out-degree nodes: {top_out}")

    # 15. Edge type contradictions (both attack/support between same pair)
    contradict_pairs = 0
    for u, v, d in g.nx_graph.edges(data=True):
        if g.nx_graph.has_edge(u, v):
            edge_type = g.nx_graph[u][v].get("type")
            if edge_type and edge_type != d.get("type"):
                contradict_pairs += 1
    print(f"Contradicting edge types (same pair): {contradict_pairs}")

    # 16. Edge span by turn distance (approx)
    print("Turn-span proxy (id distance) stats:")
    distances = []
    for u, v in g.nx_graph.edges():
        if not (u.startswith("A") and v.startswith("A")):
            continue
        try:
            u_id = int(u.strip("A").split("_")[0])
            v_id = int(v.strip("A").split("_")[0])
        except ValueError:
            continue
        distances.append(abs(u_id - v_id))
    if distances:
        distances.sort()
        print(f"  min={distances[0]}, median={distances[len(distances)//2]}, max={distances[-1]}")

    contents = sorted([n.content for n in g.nodes.values()])

    # 17. Semantic coherence via LLM
    try:
        summary_prompt = f"""
ROLE: Graph Quality Auditor.
TASK: Evaluate the debate graph quality using the node list and metrics below.
Provide JSON: {{"coherence":1-5, "coverage":1-5, "edge_quality":1-5, "attack_bias":1-5, "orphan_severity":1-5, "notes":"..."}}.

Metrics:
- nodes={num_nodes}
- edges={num_edges}
- attack_ratio={attack_ratio:.2f}
- orphan_count={len(orphans)}
- wcc={len(wcc)}

Nodes:
{contents[:30]}
"""
        data = parser._chat_json([{"role": "user", "content": summary_prompt}])
        print(f"Semantic scores: {data}")
    except Exception:
        print("Semantic scores: LLM unavailable")

    # 18. Semantic edge audit (sample)
    edge_data = None
    try:
        sample_edges = []
        for u, v, d in list(g.nx_graph.edges(data=True))[:20]:
            sample_edges.append({
                "from": u,
                "to": v,
                "type": d.get("type"),
                "from_text": g.nodes[u].content,
                "to_text": g.nodes[v].content,
            })
        edge_prompt = f"""
ROLE: Edge Semantic Auditor.
TASK: For each edge, judge if the type is correct and suggest a fix if incorrect.
Return JSON: {{"results": [{{"from":ID,"to":ID,"label":"correct|incorrect","fix":"keep|flip|remove"}}]}}.

Edges:
{sample_edges}
"""
        edge_data = parser._chat_json([{"role": "user", "content": edge_prompt}])
        print(f"Edge audit: {edge_data}")
        if isinstance(edge_data, dict) and isinstance(edge_data.get("results"), list):
            results = edge_data["results"]
            total = len(results)
            incorrect = sum(1 for r in results if r.get("label") == "incorrect")
            print(f"Edge audit incorrect ratio: {incorrect}/{total}")
    except Exception:
        print("Edge audit: LLM unavailable")

    # 19. Quality gate summary
    try:
        audit_ratio = None
        if isinstance(edge_data, dict) and isinstance(edge_data.get("results"), list):
            results = edge_data["results"]
            incorrect = sum(1 for r in results if r.get("label") == "incorrect")
            audit_ratio = (incorrect / len(results)) if results else None
        gate = parser.quality_gate(g, audit_incorrect_ratio=audit_ratio)
        print(f"Quality gate: {gate}")
    except Exception:
        print("Quality gate: unavailable")

    # 20. Sample nodes
    print("\n--- Top 5 Nodes (Sample) ---")
    for c in contents[:5]:
        print(f"- {c}")
    print("\n--- Top 5 Nodes (Sample) ---")
    for c in contents[:5]:
        print(f"- {c}")

    return g



def main():
    print("Starting Parser Stress Test (Standard Parser)...")
    for name, transcript in TRANSCRIPTS.items():
        print("\n" + "#" * 60)
        print(f"CASE: {name}")
        start_time = time.time()
        # Use standard parser (now defaults to hybrid mode)
        g = parse_debate(transcript)
        end_time = time.time()
        print(f"Parsing completed in {end_time - start_time:.2f} seconds.")
        analyze_graph(g)


if __name__ == "__main__":
    main()
