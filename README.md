# MaVERiC (Multi-Agent Verification and Reasoning in Context)

MaVERiC is a research prototype for robust multi-agent debate and verification.
It generates debates, parses them into an argumentation graph, and verifies claims
under a tool budget using a graph-aware solver. The repository also includes a
suite of baseline families for fair comparisons.

## Quick Start

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Configure API keys in `.env`

```env
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...
OPENAI_BASE_URL=...  # optional
```

3) Run evaluation

```bash
python experiments/p02_evaluation_harness/run_eval.py --dataset truthfulqa --budget 8 --max_samples 50 --baseline maveric
```

## System Overview

End-to-end pipeline:

1) Debate generation: `src/agents/debater.py`
2) Graph parsing: `src/agents/parser.py`
3) Verification + refinement: `src/core/solver.py`
4) Evaluation harness: `experiments/p02_evaluation_harness/run_eval.py`

### Debate generation (agents)

`src/agents/debater.py` produces a 7-agent debate with configurable liar/truther ratio.
Roles are defined in `src/config.py` under `AGENTS_PROFILES`.

### Parsing into a graph

`src/agents/parser.py` converts the debate into an `ArgumentationGraph`:

- Stage A1: LLM parse of candidate arguments (no relations)
- Stage A2: atomic split to ensure one claim per node
- Stage A3: relation extraction (attack/support)
- Stage B: self-refine loop with edge-only patch ops
- Stage C: deterministic guardrails and relevance checks

Graph structure is defined in `src/core/graph.py`.
Edges are typed as `attack` or `support`.

### Verification tools

`src/tools/real_toolkit.py` implements retrieval, Python verification, and a calibrated
RAG judge for claim verification. It is the shared tool layer for MaVERiC and most baselines.

### MaVERiC solver

`src/core/solver.py` implements the main ROI-driven solver. Key components:

- Verification state `tau(v)` stored on nodes
- Root detection with attack-only centrality and optional LLM tiebreaker
- ROI two-stage selection (proxy shortlist + counterfactual impact)
- SGS grounded extension with evidence gating
- Topology refinement after verifications

Important solver settings (default):

- `topk_counterfactual=25`
- `sgs_require_evidence=True`
- `rho_proxy=0.6`, `gamma_struct=0.8`
- `delta_root=1.5`, `delta_adv=0.5`, `delta_support_to_root=0.5`

## Baselines

Baselines live under `src/baselines/` and follow the same tool budget model.

### Class B (linear tool use)

`src/baselines/linear_tool.py`

- B1 ReActBaseline
- B2 RAGAnswerBaseline
- B3 RAGVerifierBaseline
- B4 SelfAskBaseline

### Class C (verification-heavy)

`src/baselines/verification_heavy.py`

- C1 BudgetedCRITICBaseline
- C2 VerifyAndReviseBaseline
- C3 RARRBaseline

### Class D (selector ablations, share MaVERiC refinement)

`src/baselines/selector_ablation.py`

- D1 RandomSelector + refine
- D2 UncertaintySelector + refine
- D3 CentralitySelector + refine (pagerank/degree/betweenness)
- D4 DistanceToRootSelector + refine
- D5 ProxyOnlySelector + refine

### Class E (multi-agent verification families)

`src/baselines/class_e.py`

- E1 MAVBaseline (multi-aspect verifiers)
- E2 BoNMAVBaseline (best-of-n + MAV)
- E3 MADFactBaseline (multi-juror debate per claim)
- E4 GKMADBaseline (guided debate + advisor + final verifier)

Defaults used in Class E:

- E1 MAV: `num_verifiers=5`, `max_claims=10`
- E2 BoN-MAV: `n=5`, `m_verifiers=5`, `top_k=4`
- E3 MAD-Fact: `num_jurors=3`, `rounds=2`, `max_claims=8`
- E4 GKMAD: `rounds=2`

Notes:
- For Class E, metrics are exposed via `baseline.stats` and returned in `run_eval` as `baseline_metrics`.
- Budget is applied to tool calls only (LLM generation is not budgeted by default).

## Evaluation Harness

`experiments/p02_evaluation_harness/run_eval.py` loads datasets, builds graphs, and
executes the chosen baseline. It supports TruthfulQA pairwise evaluation.

Example:

```bash
python experiments/p02_evaluation_harness/run_eval.py \
  --dataset truthfulqa \
  --budget 8 \
  --max_samples 50 \
  --baseline maveric
```

Supported `--baseline` values:

- `maveric`
- `d1_random`, `d2_uncertainty`, `d3_pagerank`, `d3_degree`, `d3_betweenness`, `d4_distance`, `d5_proxy`
- `e1_mav`, `e2_bon_mav`, `e3_mad_fact`, `e4_gkmad`

Tool cost override:

```bash
--tool_costs '{"WEB_SEARCH": 5.0, "PYTHON_EXEC": 8.0, "COMMON_SENSE": 1.0}'
```

Outputs:

- Results: `results/<timestamp>_<method>_<dataset>.jsonl`
- Summary: `results/<timestamp>_<method>_<dataset>_summary.txt`

## Budgeting and Metrics

Tools and default costs are configured in `src/config.py` and `src/core/solver.py`.
The evaluation harness records:

- Budget used/left
- Tool call counts
- Refinement stats (for graph-based solvers)
- Runtime per sample

Class E baselines include additional metrics in `baseline_metrics`:

- claims_count
- tool_calls_per_claim
- budget_spent_per_claim
- budget_utilization

## Tests

Run baseline tests:

```bash
pytest tests/test_class_c_baselines.py
pytest tests/test_class_e_baselines.py
```

Note: Class E tests call the real LLM client when configured; they are skipped if
`OPENAI_API_KEY` is missing.

## Project Layout

```
src/
  agents/                 # debate generation and parsing
  baselines/              # baseline families
  core/                   # graph + solver
  tools/                  # retrieval, python exec, judge
experiments/              # evaluation harness
tests/                    # unit tests
```

## Practical Tips

- Use `--use_graph_cache` for large runs to save parsing time.
- For smoke tests, lower `--max_samples` and budget.
- Class E baselines are compute-heavy; reduce rounds or jurors when sanity testing.
