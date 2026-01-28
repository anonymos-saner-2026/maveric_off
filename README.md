# MaVERiC (Multi-Agent Verification & Reasoning in Context)

MaVERiC is a research prototype for **robust multi-agent debate and verification**. It orchestrates a team of AI agents to debate complex topics, parses their arguments into a structured **Argumentation Graph**, and uses a dedicated **Solver** (MaVERiC) to verify claims and determine the truth.

The system is designed to simulate adversarial environments where misinformation ("Team Myth") attempts to overwhelm the truth ("Team Truth"), and uses graph-based reasoning (SGS: Semantics-based Graph Summary) + ROI-based tool usage to verify facts efficiently under a budget.

## 🚀 Key Features

*   **7-Agent Debate System**: Simulates realistic debates with specialized personas (e.g., "The Fact-Checker", "The Fabricator", "The Accommodator").
*   **Argumentation Graph**: Parses debates into atomic claim nodes connected by `SUPPORT` or `ATTACK` relations.
*   **MaVERiC Solver**:
    *   **SGS (Semantics-based Graph Summary)**: Computes grounded extensions to identify accepted arguments.
    *   **ROI-driven Verification**: Optimizes tool usage (Web Search, Python) based on *Return on Investment* to verify the most critical nodes first.
    *   **Topology Refinement**: Prunes invalid edges (e.g., attacks on verified truths) to maintain graph consistency.
*   **Tool Integration**:
    *   **Google Search** (via Serper): For open-domain fact-checking.
    *   **Python Execution**: For deterministic logic (math, dates).
    *   **LLM Judges**: For semantic verification and "common sense" checks.

## 📦 Installation

### Prerequisites
*   Python 3.9+
*   `pip` or `conda`

### Setup

1.  **Clone the repository** (if applicable):
    ```bash
    git clone https://github.com/anonymos-saner-2026/maveric_off.git
    cd maveric_off
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If `requirements.txt` is missing, core dependencies are: `openai`, `networkx`, `pandas`, `tqdm`, `matplotlib`, `seaborn`, `python-dotenv`, `requests`.*

3.  **Configuration**:
    Create a `.env` file in the root directory with your API keys:
    ```env
    OPENAI_API_KEY=sk-...
    SERPER_API_KEY=...    # For Google Search tool
    OPENAI_BASE_URL=...   # Optional, defaults to https://api.yescale.io/v1 or standard OpenAI
    ```

## 🛠 Usage

### 1. Run the Main Experiment
To run the full pipeline (Debate -> Parse -> Verify) on a set of built-in topics:

```bash
python main_experiment.py
```

**What happens:**
*   Generates a debate for each topic (configured in `main_experiment.py`).
*   Parses the debate into a graph.
*   Runs multiple solvers (`MAD`, `Random`, `CRITIC`, `MaVERiC`) for comparison.
*   Saves outputs to `runs/` (CSV results, plots, logs).

### 2. Run the Mini Workflow (Quick Demo)
For a single-topic demonstration of the MaVERiC solver's step-by-step verification logic:

```bash
python mini_workflow.py
```

### 3. Run Tests
To verify the correctness of the toolkit and logic:
```bash
python test_correctness_toolkit.py
```

## 📂 Project Structure

```
├── main_experiment.py       # Entry point for batch experiments
├── mini_workflow.py         # Single-topic demo script
├── test_correctness_toolkit.py # Unit tests for toolkit logic
├── src/
│   ├── agents/
│   │   ├── debater.py       # LLM Agent debate generation
│   │   └── parser.py        # Parses text -> Argumentation Graph
│   ├── core/
│   │   ├── solver.py        # MaVERiC Solver logic (ROI, SGS)
│   │   ├── graph.py         # ArgumentationGraph data structure
│   │   └── baselines.py     # Baseline solvers (MAD, CRITIC, Random)
│   ├── tools/
│   │   └── real_toolkit.py  # Wrapper for Web Search, Python, & LLM Judges
│   └── config.py            # Global configuration (keys, models, profiles)
└── runs/                    # Output directory for logs and results
```

## 🧠 Solvers Overview

*   **MaVERiC**: The core proposed method. Uses graph semantics (SGS) to identify the "Grounded Extension" (accepted arguments) and uses an ROI function to select the most impactful nodes to verify with external tools interactively.
*   **MAD (Maximum Arbitrary Degree)**: A baseline that selects the "truth" based on simple majority voting (node degree).
*   **CRITIC**: Verification based on LLM self-critique/feedback without graph topology.
*   **Random**: Randomly selects nodes to verify (baseline).

## 📎 Appendix: Class E Baselines (Defaults)

The following defaults are used for Class E baselines (MAV, BoN-MAV, MAD-Fact, GKMAD):

- **E1 MAV (Multi-Agent Verification)**
  - `num_verifiers=5`, `max_claims=10`
  - Budget split across verifiers; if split falls below tool cost, all budget is assigned to one verifier.
- **E2 BoN-MAV (Best-of-n + MAV)**
  - `n=5`, `m_verifiers=5`, `top_k=4`
  - Stage 1 self-critic filters candidates; Stage 2 runs MAV on each top-k with equal budget share.
- **E3 MAD-Fact**
  - `num_jurors=3`, `rounds=2`, `max_claims=8`
  - Budget split across claims; per-claim tools used at most once per round.
- **E4 GKMAD**
  - `rounds=2`
  - Guided debate with advisor + final verifier; tool usage is budgeted per round and at final step.

## ⚠️ Notes
*   **Costs**: Running experiments uses LLM tokens (OpenAI) and Search API credits (Serper). Check `src/config.py` and `TOOL_COSTS` in `src/core/solver.py` for cost estimates.
*   **Determinism**: Python execution is sandboxed but relies on `exec()`. Review `src/tools/real_toolkit.py` for security boundaries.
