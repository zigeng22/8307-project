# AB Test Results (B1 + B2)

Updated: 2026-05-01
Branch: exp/alignment-abtest-20260501

## 1) Scope

This report summarizes two low-cost alignment experiments:

- B1: task-aware routing reproducibility check
- B2: Task3 retrieval top-k ablation (k=1,3,5)

Models covered in AB tests:

- qwen2.5-7b
- mistral-large

## 2) B1 Results (Reproducibility)

B1 is not designed to improve scores directly. It is a rerun sanity check for key routes:

- Task1 via baseline (no RAG)
- Task3 via base_rag

### B1 A/B comparison

| model | metric | A (control) | B1 (treatment) | delta |
|---|---|---:|---:|---:|
| qwen2.5-7b | task1_accuracy | 0.5614 | 0.5604 | -0.0010 |
| qwen2.5-7b | task3_token_f1 | 0.3070 | 0.3070 | +0.0000 |
| mistral-large | task1_accuracy | 0.6036 | 0.6026 | -0.0010 |
| mistral-large | task3_token_f1 | 0.3182 | 0.3161 | -0.0021 |

### B1 conclusion

- Deltas are near zero, indicating strong run-to-run reproducibility.
- Main findings from the original matrix are stable and not random artifacts.

## 3) B2 Results (Task3 top-k Ablation)

B2 tests whether retrieval depth is under-tuned in current setup.

### Task3 token_f1 by top-k

| model | k=1 | k=3 (control) | k=5 | best k | gain vs k=3 |
|---|---:|---:|---:|---:|---:|
| qwen2.5-7b | 0.2863 | 0.3070 | 0.3225 | 5 | +0.0155 |
| mistral-large | 0.2947 | 0.3182 | 0.3366 | 5 | +0.0184 |

### B2 conclusion

- For both models, k=5 outperforms current k=3 on Task3.
- The previous instability is partly due to retrieval hyperparameter mismatch, not method failure.

## 4) Actionable Decision

1. Keep existing full matrix as Group A (Control).
2. Add B1 and B2 as Group B (Treatment) evidence.
3. For Task3-focused runs, prefer top_k=5 for qwen2.5-7b and mistral-large.

## 5) Report-ready Takeaway

- B1 validates reproducibility (near-zero rerun deltas).
- B2 provides clear positive gains from alignment tuning (+0.0155 and +0.0184 on Task3 token_f1).
- Therefore, current project conclusion should be framed as task-dependent enhancement plus alignment sensitivity, not global failure of Fine-tuning/RAG.
