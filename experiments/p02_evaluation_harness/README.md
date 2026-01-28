x# Phase 2 Evaluation Harness

Standardized evaluation pipeline for running MaVERiC on benchmark datasets with comprehensive metrics collection.

## Setup

The harness is located in `experiments/p02_evaluation_harness/` with the following structure:

```
p02_evaluation_harness/
├── README.md               # This file
├── dataset_loader.py       # Dataset loading (TruthfulQA, FEVER)
├── metrics.py              # Metrics calculation utilities
├── run_eval.py             # Main evaluation script
├── summarize.py            # Results summarization
└── results/                # Output directory for results
```

## Usage

### 1. Run Evaluation

```bash
cd experiments/p02_evaluation_harness

# Basic usage
python run_eval.py --dataset truthfulqa --budget 10 --max_samples 50

# All options
python run_eval.py \
  --method maveric \
  --dataset truthfulqa \
  --budget 10.0 \
  --max_samples 50 \
  --output_dir results \
  --verbose
```

**Arguments**:
- `--method`: Method name (default: "maveric")
- `--dataset`: Dataset name ("truthfulqa" or "fever")
- `--budget`: Budget per sample (default: 10.0)
- `--max_samples`: Maximum samples to evaluate (default: 50)
- `--output_dir`: Output directory (default: "results")
- `--verbose`: Verbose output

**Output**: Results saved to `results/TIMESTAMP_METHOD_DATASET.jsonl`

### 2. Summarize Results

```bash
python summarize.py results/20260116_143022_maveric_truthfulqa.jsonl
```

**Output**: Comprehensive summary including:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- Budget usage statistics
- Tool call breakdown
- Refinement statistics
- Error analysis

## Output Format

Results are saved in JSONL format (one JSON object per line):

```json
{
  "sample_id": "truthfulqa_0001",
  "claim": "What is the capital of France?",
  "gold_label": true,
  "predicted_label": true,
  "correct": true,
  "budget_used": 8.5,
  "tool_calls": {"WEB_SEARCH": 3, "PYTHON_EXEC": 2},
  "refinement_stats": {"pruned": 2, "edges_removed": 3, "sgs_size": 5},
  "runtime_s": 45.2
}
```

## Datasets

### TruthfulQA
- Questions with true/false labels
- Source: HuggingFace `truthful_qa`
- Fallback: Minimal hand-crafted dataset

### FEVER
- Claims with SUPPORTS/REFUTES labels
- Source: HuggingFace `fever`
- Binary classification (SUPPORTS=True, REFUTES=False)

## Metrics

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: Harmonic mean of Precision and Recall
- **Budget Stats**: Avg, Std, Min, Max budget usage
- **Tool Call Breakdown**: Count and percentage per tool
- **Refinement Stats**: Pruned nodes, removed edges, SGS size

## Example Workflow

```bash
# 1. Run evaluation on 10 samples (quick test)
python run_eval.py --dataset truthfulqa --max_samples 10

# 2. Summarize results
python summarize.py results/20260116_143022_maveric_truthfulqa.jsonl

# 3. Run full evaluation (200 samples)
python run_eval.py --dataset truthfulqa --max_samples 200 --budget 12

# 4. Summarize and export to CSV
python summarize.py results/20260116_150000_maveric_truthfulqa.jsonl --export_csv summary.csv
```

## Notes

- First run may download datasets from HuggingFace
- Fallback datasets used if download fails
- Results are appended to new files (no overwrite)
- Use `--verbose` for detailed per-sample output
