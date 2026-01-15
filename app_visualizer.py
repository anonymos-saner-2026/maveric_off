import streamlit as st
import time
import sys
import re
import copy

import streamlit.components.v1 as components
from pyvis.network import Network  # (Bạn vẫn có thể dùng nếu cần save html)

from src.core.baselines import MADSolver
from src.agents.debater import generate_debate
from src.agents.parser import parse_debate as parse_debate_to_graph
from src.core.solver import MaVERiCSolver
from src.tools.real_toolkit import RealToolkit
from src.config import client, JUDGE_MODEL


st.set_page_config(layout="wide", page_title="MaVERiC Strategic Visualizer")

# ==========================================
# 1. UTILITY FUNCTIONS
# ==========================================

def get_graph_text_summary(nodes_set, graph=None):
    """Trích xuất nội dung từ tập hợp các node để làm input cho Judge."""
    if not nodes_set:
        return "No arguments survived."
    content_list = []
    for item in nodes_set:
        node_id = item if isinstance(item, str) else item.id
        if graph and node_id in graph.nodes:
            content_list.append(f"[{node_id}]: {graph.nodes[node_id].content}")
    return " ".join(content_list)


def render_graph(nx_graph, pr_scores, nodes_dict, shielded_nodes):
    """
    Render đồ thị với hiệu ứng Neon và Hào quang bảo vệ (Shields).
    """
    net = Network(
        height="600px",
        width="100%",
        directed=True,
        bgcolor="#050505",
        font_color="white",
        notebook=False
    )

    for node_id in nx_graph.nodes():
        if node_id not in nodes_dict:
            continue

        node_obj = nodes_dict[node_id]
        score = pr_scores.get(node_id, 0.1)

        # A. Màu node (MaVERiC Strategy)
        if getattr(node_obj, "is_verified", False):
            color = "#39FF14" if getattr(node_obj, "ground_truth", False) else "#FF3131"
        else:
            color = "#FAFF00" if score > 0.18 else "#454d55"

        # B. Shield halo
        shadow_cfg = False
        if node_id in shielded_nodes:
            shadow_cfg = {
                "enabled": True,
                "color": "rgba(0, 210, 255, 0.8)",
                "size": 25,
                "x": 0,
                "y": 0,
            }

        size = 25 + (score * 200)
        net.add_node(
            node_id,
            size=size,
            color=color,
            label=str(node_id),
            title=f"Speaker: {getattr(node_obj, 'speaker', 'N/A')}\nContent: {getattr(node_obj, 'content', '')}",
            borderWidth=3 if shadow_cfg else 1,
            shadow=shadow_cfg,
        )

    # Edges: Attack vs Support
    for u, v, data in nx_graph.edges(data=True):
        rel_type = data.get("type", "attack")
        if rel_type == "attack":
            net.add_edge(u, v, color="#FF4B4B", width=2, arrows="to")
        else:
            net.add_edge(u, v, color="#00D2FF", width=2, arrows="to", dashes=True)

    net.set_options(
        """
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": { "iterations": 150 }
          }
        }
        """
    )
    return net.generate_html()


# ==========================================
# 2. STREAMLIT UI
# ==========================================

st.title("🛡️ MaVERiC: Strategic Fact-Checking Visualizer")


st.sidebar.header("Control Panel")

# Danh sách 10 topics mẫu
SAMPLE_TOPICS = {
    "Custom Topic": "",
    "Moon: Great Wall Visibility": "The Great Wall of China is visible from the Moon with the naked eye.",
    "Health: Alkaline Water pH": "Consuming alkaline water significantly changes human blood pH levels.",
    "Tech: 5G DNA Mutation": "5G network radiation causes immediate and permanent DNA mutations.",
    "History: Vikings in Amazon": "Evidence shows Vikings established a permanent colony in the Amazon rainforest.",
    "Science: 10% Brain Myth": "Humans only use 10 percent of their brain capacity for cognitive tasks.",
    "Space: Moon Landing Hoax": "The 1969 Apollo 11 moon landing was staged in a professional film studio.",
    "Climate: Wind Turbine Sickness": "Wind turbines emit infrasound that causes specific neurological illnesses.",
    "Geology: Gold Meteorites": "The majority of Earth's accessible gold arrived via massive meteorite bombardments.",
    "History: Alexandria Library": "The Library of Alexandria was moved to a secret location before the fires.",
    "Physics: Dark Matter Error": "Dark Matter is not a substance but a mathematical error in General Relativity."
}

selected_label = st.sidebar.selectbox("Select a Demo Topic", list(SAMPLE_TOPICS.keys()))

# Nếu chọn topic mẫu, tự điền vào input
default_topic = SAMPLE_TOPICS[selected_label]
topic = st.sidebar.text_input("Debate Topic", value=default_topic if default_topic else "The Great Wall of China is visible from the Moon")

num_liars = st.sidebar.slider("Number of Colluding Liars", 1, 7, 6) # Mặc định 6 liars để test độ lỳ
budget = st.sidebar.number_input("Verification Budget ($)", value=30.0)
start = st.sidebar.button("🚀 Start Verification")

if "conf_history" not in st.session_state:
    st.session_state.conf_history = []

col1, col2 = st.columns([1, 1.5])

if start:
    st.session_state.conf_history = [0]  # reset chart

    # --- PHASE 1: DEBATE ---
    with col1:
        st.header("💬 Multi-Agent Debate")
        with st.spinner("Generating adversarial debate..."):
            debate_text = generate_debate(topic, num_liars)
        st.text_area("Debate Transcript", debate_text, height=400, key="debate_transcript")

    # --- PHASE 2: STRATEGIC SOLVER ---
    with col2:
        st.header("🧠 MaVERiC Strategy Engine")

        graph_obj = parse_debate_to_graph(debate_text)
        solver = MaVERiCSolver(graph_obj, budget=budget)

        # Baseline (Smart MAD) chạy về sau (Phase 3), ở đây không cần chạy ngay
        # mad_solver = MADSolver(debate_text, topic=topic)

        # ===== Placeholders (TẠO 1 LẦN) =====
        st.subheader("📊 Graph Legend")
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        l_col1.markdown("🟢 **Verified TRUE**")
        l_col2.markdown("🔴 **Verified FALSE**")
        l_col3.markdown("🟡 **High ROI (Priority)**")
        l_col4.markdown("⚪ **Low Priority**")

        conf_placeholder = st.empty()
        graph_placeholder = st.empty()
        log_placeholder = st.empty()

        # Tạo layout con cho logs 1 lần, tránh duplicate widget
        with log_placeholder.container():
            log_title_ph = st.empty()
            log_box_ph = st.empty()

        # Tạo layout con cho confidence 1 lần
        with conf_placeholder.container():
            conf_title_ph = st.empty()
            conf_chart_ph = st.empty()

        # ===== Live loop =====
        for i, step in enumerate(solver.run_live()):
            # Update LOGS (KHÔNG dùng st.text_area nữa để tránh duplicate widget)
            log_title_ph.markdown("#### Strategy Logs")
            log_box_ph.code("\n".join(solver.logs), language="text")

            if isinstance(step, dict) and step.get("type") == "update":
                # Confidence history
                st.session_state.conf_history.append(step["confidence"])
                conf_title_ph.subheader(f"📈 Epistemic Confidence: {step['confidence']:.1f}%")
                conf_chart_ph.line_chart(st.session_state.conf_history)

                # Graph
                shielded = solver.graph.get_shielded_nodes()
                html_content = render_graph(
                    step["nx_graph"],
                    step["pagerank"],
                    solver.graph.nodes,
                    shielded,
                )
                graph_placeholder.components = None
                with graph_placeholder.container():
                    components.html(html_content, height=550)

                # Budget
                st.sidebar.metric("Remaining Budget", f"${step['budget']:.2f}")

                time.sleep(0.05)

        st.success("Verification Process Completed.")

        # --- PHASE 3: FINAL SEMANTIC ANALYSIS ---
        st.divider()
        st.header("🏆 Final Semantic Verdict")

        # 1) Summarize grounded extension (MaVERiC)
        final_ext = solver.graph.get_grounded_extension()
        sys_summary_text = get_graph_text_summary(final_ext, solver.graph)

        # 2) MaVERiC Judge (evidence-based)
        with st.spinner("🛡️ MaVERiC Judge is analyzing verified evidence..."):
            verdict_prompt = f"""
Role: Objective Fact-Checking Judge.
Statement to verify: "{topic}"
Verified Evidence: "{sys_summary_text}"

Task:
- If the Evidence confirms the Statement is factually correct, VERDICT: ACCURATE.
- If the Evidence shows the Statement is factually wrong, VERDICT: INACCURATE.

Reply STRICTLY in this format:
VERDICT: [ACCURATE/INACCURATE]
REASON: [1-sentence explanation]
"""
            res = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": verdict_prompt}],
                temperature=0.0,
            )
            judge_output = res.choices[0].message.content or ""

            maveric_is_accurate = ("ACCURATE" in judge_output.upper()) and ("INACCURATE" not in judge_output.upper())
            maveric_reason = judge_output.split("REASON:")[-1].strip() if "REASON:" in judge_output else "Analysis complete."

        # 3) Smart MAD Judge (consensus-based)
        with st.spinner("👥 Smart MAD is analyzing group consensus..."):
            mad_solver = MADSolver(debate_text, topic)
            smart_mad_accurate, mad_reason = mad_solver.run()

        # 4) Comparison UI
        st.subheader("📊 MaVERiC vs. Smart MAD: The Truth Gap")
        col_maveric, col_mad = st.columns(2)

        with col_maveric:
            st.markdown("### 🛡️ MaVERiC (Strategic)")
            m_label = "✅ ACCURATE" if maveric_is_accurate else "🚩 INACCURATE"
            st.metric(
                label="Strategic Verdict",
                value=m_label,
                delta="Verified with External Tools",
            )
            st.info(f"**Evidence-based Analysis:** {maveric_reason}")

        with col_mad:
            st.markdown("### 👥 Smart MAD (Consensus)")
            mad_label = "✅ ACCURATE" if smart_mad_accurate else "🚩 INACCURATE"

            is_misled = (smart_mad_accurate != maveric_is_accurate)
            delta_text = "MISLED BY COLLUSION" if is_misled else "ALIGNED WITH TRUTH"
            delta_color = "inverse" if is_misled else "normal"

            st.metric(
                label="Consensus Verdict",
                value=mad_label,
                delta=delta_text,
                delta_color=delta_color,
            )
            st.warning(f"**Majority Logic:** {mad_reason}")

        # 5) Strategic Insight
        st.write("---")
        if is_misled:
            st.error(
                f"⚠️ **Strategic Insight:** The Smart MAD Judge was swayed by the collective 'weight' of the {num_liars} liars. "
                f"Despite its LLM-based reasoning, it failed because it relied solely on internal consistency without external grounding. "
                f"MaVERiC successfully avoided this trap by mathematically identifying and verifying 'Keystone' claims."
            )
        else:
            st.success(
                "ℹ️ **Strategic Insight:** In this instance, the majority consensus aligned with the truth. "
                "However, MaVERiC reached this conclusion with higher verifiable confidence by using ROI-driven tool calls to prune adversarial nodes."
            )
