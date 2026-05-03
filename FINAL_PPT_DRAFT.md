# Final PPT Draft

更新日期：2026-05-03

说明：
- 这个文件是 final PPT 的工作底稿。
- 所有准备直接放进 PPT 的文字保持英文。
- 其余解释、排版建议、使用说明全部改成中文，方便你们直接讨论和二次修改。
- 我已经把大多数页内正文压成了更适合 PPT 的 bullets；只有标题、takeaway 和少量结论句保留完整句子。
- 当前涉及结果汇报的数字，统一以最终版 PROJECT_PLAN 为准。

## 已锁定的关键结果

- Task 1 全局最优：Mistral Large baseline，Accuracy = 0.6036
- Task 2 全局最优：Gemma fine-tuned，ROUGE-L = 0.2586
- Task 3 全局最优：Mistral Large base+RAG，Token-F1 = 0.3182
- Gemma Task 3 修正口径：base+RAG = 0.2908，fine-tuned = 0.2503，fine-tuned+RAG = 0.2676
- 主叙事口径：Task 1 看 baseline，Task 2 看 fine-tuning，Task 3 看 base+RAG

## 第 1 页：封面

### 页内标题（英文）
Enhancing LLM Performance on Mental Health Tasks via Fine-tuning and RAG

### 页内副标题（英文）
A Comparative Study across Classification, Counseling Dialogue, and Medical QA

### 页内文字（英文）
- STAT8307 Final Project
- Team Members: [Add names]
- May 2026

### 版式建议
- 大标题放在左侧或左上。
- 副标题紧贴主标题下方。
- 课程、组员和日期放在右下的小信息块里。

### 图表建议
- 封面尽量干净，不要堆信息。
- 如果要加背景，只建议用很淡的对话、医疗、分析类图标做氛围。

### 备注
- 这一页的目标只是把项目显得像一个完整研究，而不是课堂作业式展示。

## 第 2 页：问题与研究问题

### 页内标题（英文）
Why Mental Health LLM Evaluation Matters

### 页面块位
- 左侧模块：问题背景 + 三个任务卡片
- 右侧模块：Research Questions
- 页底：一句方法比较说明

### 左侧引导句（英文）
Mental health support requires multiple abilities.

### 左侧卡片文字（英文）
- Classification
- Dialogue generation
- Medical QA

### 右侧模块标题（英文）
Research Questions

### 右侧模块文字（英文）
1. Which enhancement method helps each task the most?
2. Are the gains consistent across different models?
3. Does Fine-tuned + RAG provide additive benefits?

### 页底短句（英文）
We compare fine-tuning and RAG instead of assuming one universal solution.

### 版式建议
- 左边放三个任务卡片：Classification、Dialogue、QA。
- 右边放三条研究问题。

### 图表建议
- 用三个紧凑小图标就够了，不要用大插图。

### 备注
- 这一页本质上是把你们现在的 motivation 和 research questions 合并成一页。

## 第 3 页：实验设计与工作量

### 页内标题（英文）
Experimental Design at a Glance

### 页面块位
- 左侧模块：主实验矩阵图
- 右侧模块：Workload Snapshot 信息框
- 页底：补充验证说明

### 左侧图标题（英文）
Main Experiment Matrix

### 左侧图例文字（英文）
- API models: Baseline, Base + RAG
- Open-source models: Baseline, Fine-tuned, Base + RAG, Fine-tuned + RAG

### 右侧信息框（英文）
Workload Snapshot

- 5 models x 3 tasks x 4 configurations
- 48 task-level results in the main matrix
- 3 datasets and 3 task types
- Training, inference, evaluation, and reporting

### 页底短句（英文）
Follow-up validation includes B1 reproducibility and B2 top-k ablation.

### 版式建议
- 左侧做实验矩阵图。
- 右侧放高亮的工作量信息框。

### 图表建议
- 用简洁矩阵图，不要上完整大表。
- 可直接使用：wtc/ppt_review_assets/generated/slide3_experiment_matrix.svg

### 备注
- 这一页是体现“我们做得系统、工作量充足”的关键页。

## 第 4 页：任务、数据与指标

### 页内标题（英文）
Tasks, Datasets, and Evaluation Metrics

### 页面块位
- 主体：四列表格
- 页底：一句任务-指标对齐说明

### 页内表格（英文）
| Task | Dataset | What the model does | Main metrics |
| --- | --- | --- | --- |
| Task 1 | Sentiment Analysis for Mental Health | Predict the mental health category from user text | Accuracy, Macro-F1 |
| Task 2 | MentalChat16K | Generate a professional and empathetic counseling response | ROUGE-L, BERTScore |
| Task 3 | MedQuAD mental-health subset | Answer factual mental-health medical questions | Token-F1, Exact Match, ROUGE-L |

### 页底短句（英文）
Metrics are matched to task type rather than reused uniformly.

### 版式建议
- 用一张干净的四列表格居中摆放。
- 如果版面单调，可以给三个 task 做轻微颜色区分。

### 图表建议
- 这一页不要再堆额外段落。
- 核心是让老师快速看懂“任务和指标是对应设计的”。

## 第 5 页：两种增强策略

### 页内标题（英文）
Two Enhancement Strategies

### 页面块位
- 左侧模块：LoRA 流程
- 右侧模块：RAG 流程
- 页底：一句总结短句

### 左侧模块标题（英文）
LoRA Fine-tuning

### 左侧模块文字（英文）
- Adapts open-source models to counseling-style language
- Learns domain-specific response patterns

### 右侧模块标题（英文）
RAG

### 右侧模块文字（英文）
- Injects external medical knowledge at inference time
- Strengthens factual grounding for knowledge-intensive QA

### 页脚短句（英文）
These methods solve different problems.

### 版式建议
- 左边放 LoRA 小流程图。
- 右边放 RAG 小流程图。
- 中间或页脚放一句总结短句。

### 图表建议
- LoRA 流程：Base model -> LoRA adapters -> adapted model
- RAG 流程：Query -> retrieval -> retrieved context -> answer
- 可直接使用：wtc/ppt_review_assets/generated/slide5_method_overview.svg

### 备注
- 这一页只讲方法逻辑，不讲参数细节。

## 第 6 页：主结果总览

### 页内标题（英文）
Main Results Overview

### 页面块位
- 左侧模块：winner table
- 右侧模块：三个高亮结果
- 页底：一句总 takeaway

### 左侧表格（英文）
| Task | Best model/configuration | Key result |
| --- | --- | --- |
| Task 1 | Mistral Large Baseline | Accuracy = 0.6036 |
| Task 2 | Gemma Fine-tuned | ROUGE-L = 0.2586 |
| Task 3 | Mistral Large Base + RAG | Token-F1 = 0.3182 |

### 右侧高亮结果（英文）
- Task 1 best: 0.6036
- Task 2 best: 0.2586
- Task 3 best: 0.3182

### 主结论短句（英文）
No single configuration is best across all three tasks.

### 版式建议
- 左边放三行 winner table。
- 右边放 task-wise winner chart，或者三个大数字高亮。

### 图表建议
- 每个 task 用一个固定强调色。
- 整页尽量留白，让观众记住三个 winner 就够了。
- 可直接使用：wtc/ppt_review_assets/generated/slide6_task_winners.svg

## 第 7 页：Task 1 与 Task 2 的分化

### 页内标题（英文）
Task 1 and Task 2 Favor Different Strategies

### 页面块位
- 左侧模块：Task 1 结论卡片
- 右侧模块：Task 2 结论卡片
- 页底：一句总 takeaway

### 左侧模块文字（英文）
Task 1: Classification favors strong baselines

- Best result: Mistral Large baseline = 0.6036
- Baseline remains the strongest classification setting
- Retrieval often adds noise to short-text inputs

### 右侧模块文字（英文）
Task 2: Dialogue favors fine-tuning

- Qwen, Llama, and Gemma all peak with fine-tuning
- Global best: Gemma fine-tuned = ROUGE-L 0.2586
- Domain adaptation helps more than retrieval

### 主结论短句（英文）
Dialogue quality benefits more from domain adaptation than from retrieval.

### 版式建议
- 左半页放 Task 1 小标题、3 条 bullets 和 1 张小图。
- 右半页放 Task 2 小标题、3 条 bullets 和 1 张小图。
- 页底横向放 takeaway，不要再放额外解释段落。

### 图表建议
- 左图可做 Task 1 configuration comparison。
- 右图可做 Task 2 configuration comparison。
- 如果只能补一张图，就做 Task1 vs Task2 的平均增益分组柱状图。
- 优先资产：wtc/ppt_review_assets/generated/slide7_task12_avg_delta.svg

### 备注
- 这一页要让听众第一次明确感受到：不同任务确实对应不同增强策略。

## 第 8 页：Task 3 明确偏向 Base + RAG

### 页内标题（英文）
Task 3 Favors Base + RAG

### 页面块位
- 左侧模块：Task 3 主图
- 右侧模块：结果解释 bullets
- 页底：一句总 takeaway

### 左侧图标题（英文）
Best Task 3 Result by Model

### 左侧图下注（英文）
All five Task 3 winners come from Base + RAG.

### 右侧模块文字（英文）
- All five models achieve their best Task 3 result with Base + RAG.
- Global best: Mistral Large base + RAG = 0.3182.
- Gemma base + RAG = 0.2908, above fine-tuned = 0.2503 and fine-tuned + RAG = 0.2676.
- Fine-tuned + RAG is not automatically additive.

### 主结论短句（英文）
For factual mental-health QA, external knowledge grounding matters most.

### 版式建议
- 左边放一张主图，优先展示不同模型在 Task 3 上的最佳结果。
- 右边放 4 条解释 bullets，并把 Gemma 修正口径单独高亮。
- 页底只放一句 takeaway，不要再补第二句总结。

### 图表建议
- 最值得做的是 Gemma 四配置柱状图，这样“Gemma 不再是反例”会很直观。
- 如果版面够大，也可以把左图改成“全部模型在 Task 3 的最佳 configuration 对比图”。
- 主图资产：wtc/ppt_review_assets/generated/slide8_task3_best_by_model.svg
- 补充资产：wtc/ppt_review_assets/generated/slide8_gemma_task3_configs.svg

### 备注
- 这一页是最终主结论最重要的证据页之一。

## 第 9 页：结论、局限与验证

### 页内标题（英文）
Final Conclusions

### 页面块位
- 上半页：4 条主结论
- 左下：2 条局限
- 右下：B1/B2 验证提示
- 页尾：task-method alignment 小图

### 上半页结论（英文）
- Task 1 depends most on strong baseline classification ability.
- Task 2 benefits most from domain fine-tuning.
- Task 3 benefits most from retrieval-based external knowledge.
- The best strategy is task-method alignment, not one universal pipeline.

### 左下局限（英文）
- We did not run full multi-seed repeats for the entire 48-result matrix.
- Task 2 evaluation still relies mainly on automatic metrics and would benefit from broader human evaluation.

### 右下验证提示（英文）
- B1 confirms that the main pattern is reproducible.
- B2 shows that Task 3 can improve further when retrieval depth increases.

### 页尾对齐图文字（英文）
- Task 1 -> Baseline
- Task 2 -> Fine-tuning
- Task 3 -> Base + RAG

### 版式建议
- 上半页放 4 条主结论。
- 下半页左边放 2 条局限，右边放 B1/B2 的一句验证提示。

### 图表建议
- 页尾可以放一张很小的 alignment graphic：
  - Task 1 -> Baseline
  - Task 2 -> Fine-tuning
  - Task 3 -> Base + RAG
- 可直接使用：wtc/ppt_review_assets/generated/slide9_task_method_alignment.svg

### 备注
- 这一页负责收口，语气要稳，不要再展开新实验细节。

## 第 10 页：Backup 页

### 页内标题（英文）
Additional Validation: B1 and B2

### B1 模块文字（英文）
- B1 was designed to test whether the main pattern is stable, not to chase a higher score.
- Example results:
  - Mistral Task 1 Accuracy: 0.6036 -> 0.6026, Delta = -0.0010
  - Mistral Task 3 Token-F1: 0.3182 -> 0.3161, Delta = -0.0021
- Interpretation: the core ranking pattern remains stable under rerun and route checking.

### B2 模块文字（英文）
- B2 tests whether stronger retrieval can further improve Task 3.
- Example result:
  - Mistral Task 3 Token-F1: top-k 3 = 0.3182, top-k 5 = 0.3366, Delta = +0.0184
- Interpretation: the Task 3 RAG advantage is not only stable, but still improvable through retrieval tuning.

### 版式建议
- 上半页放 B1 小表格。
- 下半页放 B2 小图或两行表格。

### 图表建议
- B1 可直接使用：wtc/ppt_review_assets/generated/backup_b1_reproducibility.svg
- B2 可直接使用：wtc/ppt_review_assets/generated/backup_b2_topk.svg

### 使用说明
- 只有在老师问到稳定性、复现性、参数敏感性时再翻这页。

## 制作备注

- 当前 PDF 初稿里的 agenda 和 section divider 建议删除。
- setup 内容尽量收在第 2 到第 5 页。
- 讲述重心放在第 6 到第 9 页。
- 不要再把 Gemma Task 3 说成 RAG 结论的反例。
- 主线里只简要提 B1/B2，详细数字全部留在 backup。