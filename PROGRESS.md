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
| Phase 2 | LoRA 微调（开源模型） | ⏳ 进行中 | Qwen 微调已完成并评估完毕；Llama LoRA 已完成；Gemma LoRA 受显存限制待切换方案 |
| Phase 3 | RAG 实验 | ⏳ 进行中 | Qwen Base+RAG 三任务已完成；Llama Base+RAG 三任务已完成；其余持续补齐 |
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
- Llama-3.1-8B LoRA 训练已完成（待跑 Fine-tuned / Fine-tuned+RAG）。
- 训练统计：
  - train_runtime: 9619.138 s
  - train_steps: 2922
  - epoch: 3.0
  - train_loss: 0.8047
  - checkpoint: /home/hiteam/checkpoints/qwen2.5-7b

4. RAG
- Qwen2.5-7B Base+RAG：Task1/2/3 已完成。
- Llama-3.1-8B Base+RAG：Task1/2/3 已完成。

---

## 四、当前进行中

1. Qwen 评估
- Qwen Fine-tuned：三任务已补齐。
- Qwen Base+RAG：三任务已补齐。

2. Baseline 阶段
- Llama：Baseline 三任务已补齐。
- Gemma：Baseline 三任务已补齐。

3. API 线 RAG 推进
- Mistral：下一步推进 Base+RAG。
- DeepSeek：下一步推进 Base+RAG。

4. Gemma 微调线
- Gemma LoRA：当前在 4090 24GB 上持续 OOM，正在采用更保守参数/替代方案（QLoRA 或降配）推进。

### 4.1 当前结果快照（来自 experiment_table.csv）

| 模型 | 配置 | Task1 | Task2 | Task3 |
|------|------|-------|-------|-------|
| DeepSeek V3 | Base | ✅ | ✅ | ✅ |
| Mistral Large | Base | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Base | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Fine-tuned | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Base+RAG | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Base | ✅ | ✅ | ✅ |
| Llama-3.1-8B | Base+RAG | ✅ | ✅ | ✅ |
| Gemma-2-9B | Base | ✅ | ✅ | ✅ |

---

## 五、下一阶段 GPU 排班建议（8 卡）

### 5.1 立刻可开跑（不等前置）

| GPU | 任务 | 目标输出目录 |
|-----|------|--------------|
| GPU0 | Qwen Fine-tuned+RAG task1/2/3 | /home/hiteam/results_gangda |
| GPU1 | Llama Fine-tuned 评估 | /home/hiteam/results_llama |
| GPU2 | Llama Fine-tuned+RAG | /home/hiteam/results_llama |

### 5.2 待前置完成后再启动

| GPU | 任务 | 前置条件 | 目标输出目录 |
|-----|------|----------|--------------|
| GPU3 | DeepSeek Base+RAG（API） | 无 | /home/hiteam/results_deepseek |
| GPU4 | Mistral Base+RAG（API） | 无 | /home/hiteam/results_mistral |
| GPU5 | Gemma LoRA 微调（降配/QLoRA） | Gemma Baseline 已完成（满足） | /home/hiteam/checkpoints |
| GPU6 | Gemma Fine-tuned 评估（后续接 Fine-tuned+RAG） | Gemma LoRA 完成后 | /home/hiteam/results_gemma |
| GPU7 | 机动卡（失败任务重试/补跑） | 无 | 对应任务目录 |

说明：DeepSeek/Mistral 的 API 实验不占本地 GPU，可与上述任务并行推进；当前 API 线重点是 Base+RAG。

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

---

## 七、下一步计划（按优先级）

1. 跑完 Qwen Fine-tuned+RAG 三任务并落盘。
2. 用已完成的 Llama LoRA checkpoint 跑 Llama Fine-tuned 与 Fine-tuned+RAG。
3. 启动 DeepSeek/Mistral Base+RAG 并同步结果。
4. Gemma 微调线改为保守参数或 QLoRA 方案后重启。
5. 同步远程结果并更新 experiment_table.csv 与报告表格。

---

## 八、结果目录约定

- /home/hiteam/results_gangda
- /home/hiteam/results_llama
- /home/hiteam/results_gemma

建议：不同人、不同模型使用独立 output_dir，避免并行写入覆盖。