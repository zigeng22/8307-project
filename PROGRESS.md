# 项目进度追踪文档

> 项目名称：Enhancing LLM Performance on Mental Health Tasks via Fine-tuning and RAG
> 最后更新：2026-04-30
> 维护人：wtc

---

## 一、项目总览

- 模型：DeepSeek V3 / Mistral Large / Llama-3.1-8B / Qwen2.5-7B / Gemma-2-9B
- 任务：Task1 分类 / Task2 对话生成 / Task3 医疗问答
- 配置：Base / Fine-tuned / Base+RAG / Fine-tuned+RAG
- 总实验点：48（API 模型仅 Base 与 Base+RAG）

---

## 二、阶段进度

| 阶段 | 内容 | 状态 | 备注 |
|------|------|------|------|
| Phase 0 | 框架搭建 + 环境验证 | ✅ 已完成 | 代码结构与评估流水线稳定 |
| Phase 1 | Baseline | ✅ 已完成 | DeepSeek/Qwen/Llama/Gemma/Mistral 五模型 Base 三任务已补齐 |
| Phase 2 | LoRA 微调（开源模型） | ⏳ 进行中 | Qwen/Llama 微调与评估已完成；Gemma 微调由队友在 Colab A100 推进 |
| Phase 3 | RAG 实验 | ⏳ 进行中 | DeepSeek/Mistral/Qwen/Llama RAG 主线已补齐；Gemma RAG 待队友结果回传 |
| Phase 4 | 汇总分析与报告 | 🔲 未开始 | 待实验结果齐全后执行 |

---

## 三、当前已确认完成

1. 代码与数据
- 代码仓库可在远程服务器稳定运行。
- 服务器已配置 Python 环境与主要依赖。
- 数据集已放置并可被 loader 正常读取。

2. Baseline
- DeepSeek V3：Task1/2/3 已完成。
- Qwen2.5-7B：Task1/2/3 已完成。
- Llama-3.1-8B：Task1/2/3 已完成。
- Gemma-2-9B：Task1/2/3 已完成。
- Mistral Large：Task1/2/3 已完成。

3. 微调
- Qwen2.5-7B LoRA 训练已完成。
- Qwen2.5-7B Fine-tuned：Task1/2/3 已完成。
- Llama-3.1-8B LoRA 训练已完成。
- Llama-3.1-8B Fine-tuned：Task1/2/3 已完成。
- 训练统计：
  - train_runtime: 9619.138 s
  - train_steps: 2922
  - epoch: 3.0
  - train_loss: 0.8047
  - checkpoint: /home/hiteam/checkpoints/qwen2.5-7b

4. RAG
- DeepSeek V3 Base+RAG：Task1/2/3 已完成。
- Mistral Large Base+RAG：Task1/2/3 已完成。
- Qwen2.5-7B Base+RAG：Task1/2/3 已完成。
- Qwen2.5-7B Fine-tuned+RAG：Task1/2/3 已完成。
- Llama-3.1-8B Base+RAG：Task1/2/3 已完成。
- Llama-3.1-8B Fine-tuned+RAG：Task1/2/3 已完成。

---

## 四、当前进行中

1. Gemma 微调与 RAG 收尾（队友 Colab）
- Gemma LoRA / Fine-tuned / Base+RAG / Fine-tuned+RAG 由队友在 Colab A100 推进。
- 你本地这条线当前主要任务是接收结果、合并到 `results_server/results_gemma`、并重建总表。

2. 汇总与报告准备
- 已完成结果需持续回填到课程计划文档与对比表。
- 待 Gemma 收尾后统一进行 48 点结果复核与图表生成。

### 4.1 当前结果快照（来自 experiment_table.csv）

| 模型 | 配置 | Task1 | Task2 | Task3 |
|------|------|-------|-------|-------|
| DeepSeek V3 | Base | ✅ | ✅ | ✅ |
| DeepSeek V3 | Base+RAG | ✅ | ✅ | ✅ |
| Mistral Large | Base | ✅ | ✅ | ✅ |
| Mistral Large | Base+RAG | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Base | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Fine-tuned | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Base+RAG | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Fine-tuned+RAG | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Base | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Fine-tuned | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Base+RAG | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Fine-tuned+RAG | ✅ | ✅ | ✅ |
| Gemma-2-9B | Base | ✅ | ✅ | ✅ |

---

## 五、当前排班与收尾建议

### 5.1 你本地远程 8 卡（当前状态）

| 资源 | 当前任务 | 状态 |
|------|----------|------|
| 远程 GPU 任务线 | Qwen/Llama/DeepSeek/Mistral 剩余实验 | ✅ 已完成并入表 |
| 远程 GPU 任务线 | Gemma 微调/RAG | ⏸️ 不在本机继续跑（移交队友 Colab A100） |

### 5.2 当前建议动作

| 优先级 | 动作 | 目标 |
|------|------|------|
| P0 | 接收队友 Gemma 结果并合并到本地 | 补齐最后未完成配置 |
| P1 | 执行 `sync_and_summarize_results.py --skip-sync` 重建 CSV | 确保总表与源结果一致 |
| P2 | 回填课程文档与最终展示表 | 进入报告撰写阶段 |

---

## 六、近期关键修复记录（远程实战）

1. 离线模型路径
- 服务器无法外网下载 HuggingFace 模型时，将 model_id 改为本地路径并设置离线环境变量。

2. 训练数据空值
- 微调阶段出现 NoneType 导致 template 报错，已在训练脚本中加入空值清洗与安全字符串转换。

3. 多卡设备不一致
- 出现 cuda:0 与 cuda:7 混用，已通过固定 CUDA_VISIBLE_DEVICES 与单卡加载策略解决。

4. 显存不足
- 在 4090 24GB 上将 batch 与序列长度下调后，Qwen 微调可稳定运行。

5. BERTScore 离线依赖
- Task2 评估时 roberta-large 未命中缓存导致报错，已在评估脚本增加本地路径解析（支持 BERTSCORE_MODEL_TYPE）。

6. 数据集导入流程
- 已明确采用“先检查 `/home/hiteam/Datasets`，缺失则从本地 Dropbox 上传”的标准流程。
- loader 自检命令通过后再启动实验，避免运行中才报找不到文件。

7. 网络出口限制（2026-04-29 实测）
- 出口 IP：123.58.249.106（中国北京联通）
- huggingface.co：超时，不可达
- openrouter.ai：可达（HTTP 200）
- api.openai.com：超时，不可达
- api.anthropic.com：可达但 403
- 执行策略：开源模型离线本地权重；API 优先走 OpenRouter。

8. 结果同步脚本修复（2026-04-29）
- 已修复 `tools/sync_and_summarize_results.py` 的 scp 交互问题。
- 旧版本因 `capture_output` 导致 scp 交互密码失败并报 `Connection closed`。
- 新版本可在终端正常输入 SSH 密码完成三目录同步。

9. API 429 限流重试（2026-04-30）
- Mistral Base+RAG 过程中出现 OpenRouter 429（上游临时限流）。
- 已采用 tmux + 重试循环收尾，最终成功补齐 `mistral-large base_rag` 三任务结果。

---

## 七、下一步计划（按优先级）

1. 跑完 Qwen Fine-tuned+RAG 三任务并落盘。
2. 接收队友 Gemma（Colab A100）输出并合并到 `results_server/results_gemma`。
3. 重建 `results/experiment_table.csv`，核对 48 个实验点完整性。
4. 更新课程计划文档与最终报告用图表。
5. 进入报告撰写与误差分析阶段。

---

## 八、结果目录约定

- /home/hiteam/results_gangda
- /home/hiteam/results_llama
- /home/hiteam/results_gemma

建议：不同人、不同模型使用独立 output_dir，避免并行写入覆盖。