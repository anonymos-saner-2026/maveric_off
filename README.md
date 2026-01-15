# MaVERiC 

MaVERiC is a research prototype for strategic fact-checking under budget constraints.
It simulates adversarial multi-agent debates, parses them into an argumentation graph,
and verifies a subset of claims using external tools to decide which arguments survive.

## Why this project
- Simulate collusive debate behavior and test robustness against majority consensus.
- Allocate limited verification budget using graph-aware ROI heuristics.
- Compare MaVERiC against baselines like Random, CRITIC (sequential), and MAD (consensus).

## Core idea
1. Generate a multi-agent debate with a configurable number of colluding liars.
2. Parse the transcript into atomic claims and attack/support relations.
3. Use MaVERiC to pick high-impact claims to verify (tool calls cost budget).
4. Prune false claims and recompute grounded extensions to reach a verdict.

## Repository structure
- `maveric_ijcai/src/agents/debater.py`: debate generator (LLM prompts + agent profiles).
- `maveric_ijcai/src/agents/parser.py`: parses debate into an argumentation graph.
- `maveric_ijcai/src/core/graph.py`: argumentation graph + grounded semantics + shielding.
- `maveric_ijcai/src/core/solver.py`: MaVERiC strategic solver.
- `maveric_ijcai/src/core/baselines.py`: Random, CRITIC, MAD, and ReAct baselines.
- `maveric_ijcai/src/tools/real_toolkit.py`: verification tools (web search + python sandbox).
- `maveric_ijcai/main_experiment.py`: batch experiments and metrics.
- `maveric_ijcai/app_visualizer.py`: Streamlit visualizer for live runs.
- `maveric_ijcai/test_real_toolkit*.py`: live integration tests.

## Requirements
- Python 3.10+
- OpenAI-compatible API key (or compatible proxy)
- Optional: Serper API key for Google search

Install dependencies:
```bash
pip install -r maveric_ijcai/requirements.txt
```

Environment variables (use `.env`):
```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.yescale.io/v1
SERPER_API_KEY=...
```

## Run a batch experiment
```bash
python maveric_ijcai/main_experiment.py
```

This will:
- generate debates,
- build graphs,
- run MaVERiC and baselines,
- report semantic accuracy and graph IoU.

## Run the visualizer
```bash
streamlit run maveric_ijcai/app_visualizer.py
```

The UI shows:
- live graph updates,
- verification logs,
- budget usage,
- MaVERiC vs MAD verdict comparison.

## Tests (live)
These tests call real APIs and may incur cost:
```bash
python maveric_ijcai/test_real_toolkit.py
python maveric_ijcai/test_real_toolkit_v2.py
python maveric_ijcai/test_real_toolkit_v3.py
```

## Notes
- Large data files under `maveric_ijcai/src/data/` are ignored by default.
- This is a research prototype; outputs depend on model quality and API availability.
