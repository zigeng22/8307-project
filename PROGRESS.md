# 项目进度追踪文档

> 项目名称：Enhancing LLM Performance on Mental Health Tasks via Fine-tuning and RAG
> 最后更新：2026-04-29
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
| Phase 1 | Baseline | ⏳ 进行中 | DeepSeek/Qwen 已完成；Llama 缺 task2；Gemma 缺 task2/task3 |
| Phase 2 | LoRA 微调（开源模型） | ⏳ 进行中 | Qwen 微调已完成；Fine-tuned task1/task2 已完成 |
| Phase 3 | RAG 实验 | ⏳ 进行中 | Qwen Base+RAG task1 已完成，其他持续补齐 |
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
- Llama-3.1-8B：Task1/3 已完成（Task2 补跑中）。
- Gemma-2-9B：Task1 已完成（Task2/3 补跑中）。

3. 微调
- Qwen2.5-7B LoRA 训练已完成。
- Qwen2.5-7B Fine-tuned：Task1/2 已完成（Task3 待补）。
- 训练统计：
  - train_runtime: 9619.138 s
  - train_steps: 2922
  - epoch: 3.0
  - train_loss: 0.8047
  - checkpoint: /home/hiteam/checkpoints/qwen2.5-7b

4. RAG
- Qwen2.5-7B Base+RAG：Task1 已完成（Task2/3 待补）。

---

## 四、当前进行中

1. Qwen 评估
- Qwen Fine-tuned：正在补 task3。
- Qwen Base+RAG：正在补 task2/task3。

2. Llama 与 Gemma Baseline 补齐
- Llama：当前只缺 Baseline task2。
- Gemma：当前缺 Baseline task2 与 task3。

### 4.1 当前结果快照（来自 experiment_table.csv）

| 模型 | 配置 | Task1 | Task2 | Task3 |
|------|------|-------|-------|-------|
| DeepSeek V3 | Base | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Base | ✅ | ✅ | ✅ |
| Qwen2.5-7B | Fine-tuned | ✅ | ✅ | ⏳ |
| Qwen2.5-7B | Base+RAG | ✅ | ⏳ | ⏳ |
| Llama-3.1-8B | Base | ✅ | ⏳ | ✅ |
| Gemma-2-9B | Base | ✅ | ⏳ | ⏳ |

---

## 五、当前 GPU 使用与并行计划（8 卡）

### 5.1 当前已占用（约 3/8）

| GPU | 当前任务 | 状态 |
|-----|----------|------|
| GPU0 | Qwen Fine-tuned 评估 / 后续 Qwen FT+RAG | 进行中 |
| GPU1 | Llama Baseline 补跑 | 进行中 |
| GPU2 | Gemma Baseline 补跑 | 进行中 |

### 5.2 其余 5 张卡可并行任务建议

| GPU | 建议任务 | 前置条件 | 备注 |
|-----|----------|----------|------|
| GPU3 | Llama LoRA 微调 | Llama Baseline 三任务补齐 | 重任务，单独占卡 |
| GPU4 | Gemma LoRA 微调 | Gemma Baseline 三任务补齐 | 重任务，单独占卡 |
| GPU5 | Qwen Fine-tuned + RAG | Qwen Fine-tuned task3 完成 | 可尽快启动 |
| GPU6 | Llama Base + RAG | Llama Baseline 补齐 | 与微调线并行 |
| GPU7 | Gemma Base + RAG | Gemma Baseline 补齐 | 与微调线并行 |

说明：DeepSeek/Mistral 的 API 实验不占本地 GPU，可与上述任务并行推进。

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

---

## 七、下一步计划（按优先级）

1. 完成 Qwen Fine-tuned 评估结果落盘并核验三任务指标文件。
2. 补齐 Llama Baseline 三任务结果。
3. 补齐 Gemma Baseline 三任务结果。
4. 启动 Qwen Fine-tuned + RAG。
5. 按同样流程推进 Llama/Gemma 的 Fine-tuned 与 RAG。
6. 汇总 48 个实验点并进入 Phase 4 分析与报告。

---

## 八、结果目录约定

- /home/hiteam/results_gangda
- /home/hiteam/results_llama
- /home/hiteam/results_gemma

建议：不同人、不同模型使用独立 output_dir，避免并行写入覆盖。