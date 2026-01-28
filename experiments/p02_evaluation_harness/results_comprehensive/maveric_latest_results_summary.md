# Maveric latest evaluation results (2026-01-28)

Latest run per dataset from `experiments/p02_evaluation_harness/results_comprehensive`.

## Quick comparison

| Dataset | Samples | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| copheme | 30 | 0.5333 | 0.0000 | 0.0000 | 0.0000 |
| fever | 30 | 0.7667 | 0.8333 | 0.8696 | 0.8511 |
| truthfulqa | 30 | 0.8333 | 1.0000 | 0.8333 | 0.9091 |
| scifact | 15 | 0.8667 | 0.8462 | 1.0000 | 0.9167 |
| hover | 15 | 0.4667 | 0.6667 | 0.4000 | 0.5000 |

## copheme (20260128_183334)

Source: `experiments/p02_evaluation_harness/results_comprehensive/20260128_183334_maveric_copheme_summary.txt`

| Metric | Value |
| --- | --- |
| Samples (verified) | 30 (30) |
| Accuracy | 0.5333 (16/30) |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1 | 0.0000 |
| Confusion Matrix | TP 0, FP 14, FN 0, TN 16 |
| Avg Budget Used | 7.05 ± 1.49 |
| Min Budget | 1.00 |
| Max Budget | 8.00 |
| Avg Tool Calls | 6.37 ± 2.18 |
| Total Tool Calls | 191 |
| Avg Runtime | 151.96s per sample |
| Total Runtime | 4558.79s |
| Avg Nodes Pruned | 1.93 ± 1.29 |
| Avg Edges Removed (true refine) | 0.33 ± 0.70 |
| Avg Edges Removed (false refine) | 1.00 ± 1.15 |
| Avg Edges Removed (prune) | 0.87 ± 1.36 |
| Avg SGS Size | 6.03 ± 2.76 |

Nhan xet: Do doan toan bo am tinh (TP=0, FP cao), precision/recall/F1 bang 0 du accuracy van trung binh nho ty le TN.

## fever (20260128_171734)

Source: `experiments/p02_evaluation_harness/results_comprehensive/20260128_171734_maveric_fever_summary.txt`

| Metric | Value |
| --- | --- |
| Samples (verified) | 30 (30) |
| Accuracy | 0.7667 (23/30) |
| Precision | 0.8333 |
| Recall | 0.8696 |
| F1 | 0.8511 |
| Confusion Matrix | TP 20, FP 4, FN 3, TN 3 |
| Avg Budget Used | 7.35 ± 1.04 |
| Min Budget | 4.00 |
| Max Budget | 8.00 |
| Avg Tool Calls | 6.93 ± 1.59 |
| Total Tool Calls | 208 |
| Avg Runtime | 98.05s per sample |
| Total Runtime | 2941.43s |
| Avg Nodes Pruned | 1.90 ± 1.16 |
| Avg Edges Removed (true refine) | 0.23 ± 0.67 |
| Avg Edges Removed (false refine) | 0.87 ± 1.06 |
| Avg Edges Removed (prune) | 1.13 ± 1.52 |
| Avg SGS Size | 6.77 ± 3.29 |

Nhan xet: Ket qua can bang, precision/recall cao, nhung TN thap cho thay mo hinh nghieng ve du doan True.

## truthfulqa (20260128_162830)

Source: `experiments/p02_evaluation_harness/results_comprehensive/20260128_162830_maveric_truthfulqa_summary.txt`

| Metric | Value |
| --- | --- |
| Samples (verified) | 30 (30) |
| Accuracy | 0.8333 (25/30) |
| Precision | 1.0000 |
| Recall | 0.8333 |
| F1 | 0.9091 |
| Confusion Matrix | TP 25, FP 0, FN 5, TN 0 |
| Avg Budget Used | 7.58 ± 0.70 |
| Min Budget | 5.00 |
| Max Budget | 8.00 |
| Avg Tool Calls | 7.57 ± 0.72 |
| Total Tool Calls | 227 |
| Avg Runtime | 130.07s per sample |
| Total Runtime | 3902.18s |
| Avg Nodes Pruned | 3.07 ± 1.36 |
| Avg Edges Removed (true refine) | 0.50 ± 0.67 |
| Avg Edges Removed (false refine) | 0.63 ± 0.91 |
| Avg Edges Removed (prune) | 1.70 ± 1.75 |
| Avg SGS Size | 3.63 ± 1.40 |

Nhan xet: Precision 1.0 nhung TN=0, cho thay tap mau co the thieu mau False hoac model nghieng du doan True.

## scifact (20260128_065114)

Source: `experiments/p02_evaluation_harness/results_comprehensive/20260128_065114_maveric_scifact_summary.txt`

| Metric | Value |
| --- | --- |
| Samples (verified) | 15 (15) |
| Accuracy | 0.8667 (13/15) |
| Precision | 0.8462 |
| Recall | 1.0000 |
| F1 | 0.9167 |
| Confusion Matrix | TP 11, FP 2, FN 0, TN 2 |
| Avg Budget Used | 4.34 ± 1.25 |
| Min Budget | 1.00 |
| Max Budget | 5.00 |
| Avg Tool Calls | 4.33 ± 1.25 |
| Total Tool Calls | 65 |
| Avg Runtime | 295.77s per sample |
| Total Runtime | 4436.51s |
| Avg Nodes Pruned | 1.07 ± 0.68 |
| Avg Edges Removed (true refine) | 0.00 ± 0.00 |
| Avg Edges Removed (false refine) | 0.53 ± 0.72 |
| Avg Edges Removed (prune) | 0.27 ± 0.57 |
| Avg SGS Size | 4.73 ± 2.11 |

Nhan xet: Hieu qua tot voi recall 1.0; runtime cao hon cac bo 15 mau khac.

## hover (20260128_053716)

Source: `experiments/p02_evaluation_harness/results_comprehensive/20260128_053716_maveric_hover_summary.txt`

| Metric | Value |
| --- | --- |
| Samples (verified) | 15 (15) |
| Accuracy | 0.4667 (7/15) |
| Precision | 0.6667 |
| Recall | 0.4000 |
| F1 | 0.5000 |
| Confusion Matrix | TP 4, FP 2, FN 6, TN 3 |
| Avg Budget Used | 4.75 ± 0.41 |
| Min Budget | 4.05 |
| Max Budget | 5.00 |
| Avg Tool Calls | 4.73 ± 0.44 |
| Total Tool Calls | 71 |
| Avg Runtime | 455.09s per sample |
| Total Runtime | 6826.34s |
| Avg Nodes Pruned | 1.33 ± 0.87 |
| Avg Edges Removed (true refine) | 0.40 ± 0.71 |
| Avg Edges Removed (false refine) | 0.20 ± 0.54 |
| Avg Edges Removed (prune) | 1.47 ± 3.12 |
| Avg SGS Size | 5.27 ± 1.88 |

Nhan xet: Hieu qua thap, recall 0.4 va FN cao; runtime cao nhat trong cac bo hien co.
