# AB Test Plan: Alignment Experiments

Updated: 2026-05-01
Branch: exp/alignment-abtest-20260501

## 1) Goal

Keep current results as Group A (Control), and run low-cost method-alignment experiments as Group B (Treatment) without overwriting existing outputs.

## 2) Control Snapshot (A Group)

Current summary is frozen by metrics below (from current experiment table):

| model | config | task1_accuracy | task2_rouge_l | task3_token_f1 |
|---|---|---:|---:|---:|
| deepseek-v3 | baseline | 0.5855 | 0.1662 | 0.2065 |
| deepseek-v3 | base_rag | 0.5533 | 0.1688 | 0.2274 |
| mistral-large | baseline | 0.6036 | 0.1952 | 0.2747 |
| mistral-large | base_rag | 0.5724 | 0.1943 | 0.3182 |
| llama-3.1-8b | baseline | 0.4487 | 0.1961 | 0.2472 |
| llama-3.1-8b | finetuned | 0.4215 | 0.2317 | 0.2267 |
| llama-3.1-8b | base_rag | 0.4547 | 0.1981 | 0.2846 |
| llama-3.1-8b | finetuned_rag | 0.4447 | 0.2246 | 0.2406 |
| qwen2.5-7b | baseline | 0.5614 | 0.1960 | 0.2608 |
| qwen2.5-7b | finetuned | 0.5402 | 0.2366 | 0.2595 |
| qwen2.5-7b | base_rag | 0.5151 | 0.1943 | 0.3070 |
| qwen2.5-7b | finetuned_rag | 0.5181 | 0.2248 | 0.2689 |
| gemma-2-9b | baseline | 0.5483 | 0.1891 | 0.2510 |

Note: Raw CSV files are ignored by git in this project, so this markdown table is the versioned control snapshot.

## 3) Treatment Output Isolation (B Group)

Use separate output roots to avoid any overwrite:

- /home/hiteam/results_abtest/b1_task_aware_rag
- /home/hiteam/results_abtest/b2_task3_topk
- /home/hiteam/results_abtest/b3_multitask_lora

Do not write B-group outputs into existing:

- /home/hiteam/results_gangda
- /home/hiteam/results_llama
- /home/hiteam/results_deepseek
- /home/hiteam/results_mistral
- /home/hiteam/results_gemma

## 4) Low-Cost Alignment Experiments

### B1: Task-aware RAG routing (no retraining)

- Disable RAG for Task1.
- Keep RAG for Task3.
- Task2 with/without RAG as a small comparison.
- Goal: verify that retrieval harms classification but helps QA.

### B2: Task3 retrieval ablation (inference only)

- Evaluate top_k in {1, 3, 5}.
- Optional chunk-size check: {300, 500}.
- Goal: find stable retrieval strength for QA gains.

### B3: Lightweight multitask LoRA continuation (one model first)

- Start from qwen2.5-7b only.
- Mix Task1 + Task2 + Task3 training samples with balanced sampling.
- Train short run (about 1 epoch).
- Goal: reduce cross-task negative transfer and improve finetuned+rag stability.

## 5) Acceptance Criteria

- Keep A-group metrics unchanged and reproducible.
- B-group experiments produce isolated outputs under results_abtest.
- At least one of these is observed:
  - smaller Task1 drop after alignment,
  - stronger Task3 gain under tuned retrieval,
  - improved finetuned+rag consistency on open-source models.

## 6) Reporting Rule

In report tables:

- Label existing matrix as Group A (Control).
- Label new aligned runs as Group B (Treatment).
- Compare deltas against this file's snapshot table.
