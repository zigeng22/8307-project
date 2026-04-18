# 项目进度追踪文档

> 项目名称：Enhancing LLM Performance on Mental Health Tasks via Fine-tuning and RAG  
> 最后更新：2026-04-18  
> 维护人：wtc

---

## 一、项目总览

### 实验矩阵
- **5 个模型**：DeepSeek V3 / Mistral Large / Llama-3.1-8B / Qwen2.5-7B / Gemma-2-9B
- **3 个任务**：心理状态分类 (Task1) / 临床对话生成 (Task2) / 医疗知识问答 (Task3)
- **4 种配置**：Base / Fine-tuned / Base+RAG / Fine-tuned+RAG
- **总计**：48 个实验数据点（API模型不微调）

> 注：原计划用 GPT-4o + Claude-3.5-Sonnet，因 OpenRouter 账户被这两家提供商封禁，改用 DeepSeek V3 + Mistral Large 替代。

### 四个阶段
| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 框架搭建 + 环境验证 | ✅ 已完成 |
| Phase 1 | Baseline 零样本评估 | ⏳ 进行中（DeepSeek V3 + Qwen 已完成） |
| Phase 2 | LoRA 微调（3个开源模型） | 🔲 未开始 |
| Phase 3 | RAG 检索增强实验 | 🔲 未开始 |
| Phase 4 | 全面对比 + 消融分析 + 报告撰写 | 🔲 未开始 |

---

## 二、已完成的工作

### ✅ 2.1 项目提案 (Proposal)
- 已提交，基于 PROJECT_PLAN.md 完成
- 位置：`Proposal & Project Plan/Proposal_Draft_v1.tex`

### ✅ 2.2 数据集准备
三个数据集均已下载到 `Datasets/` 目录：

| 数据集 | 文件 | 规模 | 备注 |
|--------|------|------|------|
| Sentiment Analysis | `Combined Data.csv` (31MB) | 53,043 条，7个类别 | 类别不均衡：Normal 16K, Personality disorder 1.2K |
| MentalChat16K | `MentalChat16K-main/` (两个CSV) | Interview 6K + Synthetic 10K = 16K | 格式: instruction/input/output |
| MedQuAD | `medquad.csv` (23MB) | 16,412 条，5,127 个 focus_area | 心理健康相关仅133条，改为使用通用医疗QA |

### ✅ 2.3 代码框架搭建
所有代码位于 `wtc/` 目录下：

```
wtc/
├── config.py                  # 全局配置（路径、模型注册表、超参数）
├── requirements.txt           # Python 依赖列表
├── instructions.md            # AI 协作规则
├── PROGRESS.md                # 本文档
│
├── data/                      # 数据处理
│   ├── __init__.py
│   ├── loader.py              # 三个数据集的加载函数
│   └── splitter.py            # 测试集划分（固定 seed=42）
│
├── prompts/                   # Prompt 模板
│   ├── __init__.py
│   └── templates.py           # 三个任务的标准化 Prompt + RAG 增强版本
│
├── models/                    # 模型接口
│   ├── __init__.py
│   ├── base.py                # 抽象基类 BaseModel（统一 generate 接口）
│   ├── api_model.py           # OpenAI + Anthropic API 封装
│   └── hf_model.py            # HuggingFace 本地模型（支持 LoRA 加载）
│
├── evaluation/                # 评估指标
│   ├── __init__.py
│   └── metrics.py             # Accuracy/F1/ROUGE-L/BERTScore/Token-F1/EM
│
├── rag/                       # RAG 流水线
│   ├── __init__.py
│   ├── indexer.py             # 构建 FAISS 向量索引（MedQuAD 全量）
│   └── retriever.py           # 检索相关段落
│
├── finetune/                  # LoRA 微调
│   ├── __init__.py
│   └── lora_train.py          # LoRA 训练脚本（SFTTrainer）
│
├── experiments/               # 实验运行脚本
│   ├── __init__.py
│   ├── run_baseline.py        # Phase 1: 零样本实验
│   ├── run_rag.py             # Phase 3: RAG 实验（支持 base 和 finetuned）
│   └── run_finetuned.py       # Phase 2: 微调后模型评估
│
├── notebooks/                 # Colab 相关
│   └── colab_quickstart.ipynb # Colab 快速启动 notebook
│
└── results/                   # 实验结果（自动生成）
    ├── baseline/{model}/      # 各模型的 baseline 结果
    ├── finetuned/{model}/     # 微调模型的结果
    ├── base_rag/{model}/      # base + RAG 结果
    └── finetuned_rag/{model}/ # 微调 + RAG 结果
```

### 框架设计要点
- **统一接口**：所有模型（API + 本地）都实现同一个 `generate(messages)` 方法
- **公平比较**：所有模型共用同一套 Prompt 模板、同一份测试集（seed=42）
- **模块化**：每个模块独立，可单独测试和替换
- **命令行驱动**：每个实验脚本都支持 `--model` 和 `--task` 参数

### ✅ 2.4 API 和 Git 配置
- **OpenRouter 注册完成**，API Key 可用（VPN 下正常）
- **模型 ID 确认**：GPT-4o → `openai/gpt-4o`，Claude → `anthropic/claude-sonnet-4`
- **端到端测试通过**：GPT-4o 和 Claude 各跑 5 条 Task1，pipeline 完整运行无误
- **GitHub 仓库**：https://github.com/zigeng22/8307-project （已推送所有代码）
- **本地 Git 环境**：已初始化并关联远程仓库

### ✅ 2.5 环境配置
- [x] OpenRouter 充值完成
- [x] Google Colab Pro 订阅完成（A100 可用）
- [x] Colab notebook 已创建（GPT-4o + Qwen2.5-7B baseline）
- [ ] 将 Datasets/ 上传到 Google Drive
- [ ] Colab 上跑通 Section 0 环境搭建
- [ ] HuggingFace 账号登录 + Gemma 模型 license 同意

---

## 三、尚未完成的工作

### ✅ 3.1 环境配置（已完成）
- [x] OpenRouter 注册 + 充值
- [x] Google Colab Pro 订阅（A100）
- [x] GitHub 仓库推送完毕
- [ ] 上传 Datasets/ 到 Google Drive
- [ ] 在 Colab 上运行 Section 0 验证

### ⏳ 3.2 Phase 1: Baseline 实验

| 模型 | Task1 | Task2 | Task3 | 备注 |
|------|-------|-------|-------|------|
| DeepSeek V3 | ✅ | ✅ | ✅ | API，已完成 |
| Mistral Large | ⬜ | ⬜ | ⬜ | API |
| Llama-3.1-8B | ⬜ | ⬜ | ⬜ | Colab A100 |
| Qwen2.5-7B | ✅ | ✅ | ✅ | Colab A100，已完成 |
| Gemma-2-9B | ⬜ | ⬜ | ⬜ | Colab A100 |

**运行方式**：
```bash
# API 模型（本地或 Colab 均可，不需要 GPU）
python experiments/run_baseline.py --model gpt-4o --task all
python experiments/run_baseline.py --model claude-3.5-sonnet --task all

# 开源模型（需要 Colab GPU）
python experiments/run_baseline.py --model llama-3.1-8b --task all
python experiments/run_baseline.py --model qwen2.5-7b --task all
python experiments/run_baseline.py --model gemma-2-9b --task all
```

### 🔲 3.3 Phase 2: LoRA 微调
仅限 3 个开源模型：

| 模型 | 微调 | 微调后评估 Task1 | Task2 | Task3 |
|------|------|-----------------|-------|-------|
| Llama-3.1-8B | ⬜ | ⬜ | ⬜ | ⬜ |
| Qwen2.5-7B | ⬜ | ⬜ | ⬜ | ⬜ |
| Gemma-2-9B | ⬜ | ⬜ | ⬜ | ⬜ |

**运行方式**：
```bash
# 微调（需要 Colab A100，约 2-4 小时/模型）
python finetune/lora_train.py --model llama-3.1-8b

# 微调后评估
python experiments/run_finetuned.py --model llama-3.1-8b --task all \
    --lora_path ./finetune/checkpoints/llama-3.1-8b
```

### 🔲 3.4 Phase 3: RAG 实验
先构建 FAISS 索引（一次性），然后为所有模型跑 RAG 增强实验：

| 步骤 | 状态 |
|------|------|
| 构建 FAISS 索引 | ⬜ |
| GPT-4o + RAG (3 tasks) | ⬜ |
| Claude-3.5 + RAG (3 tasks) | ⬜ |
| Llama Base + RAG (3 tasks) | ⬜ |
| Qwen Base + RAG (3 tasks) | ⬜ |
| Gemma Base + RAG (3 tasks) | ⬜ |
| Llama FT + RAG (3 tasks) | ⬜ |
| Qwen FT + RAG (3 tasks) | ⬜ |
| Gemma FT + RAG (3 tasks) | ⬜ |

**运行方式**：
```bash
# 构建索引（需要一次，约 5 分钟）
python rag/indexer.py

# RAG 实验
python experiments/run_rag.py --model gpt-4o --task all
python experiments/run_rag.py --model llama-3.1-8b --task all --lora_path ./finetune/checkpoints/llama-3.1-8b
```

### 🔲 3.5 Phase 4: 分析与报告
- [ ] 汇总全部 60 个实验数据点到结果表
- [ ] 消融分析（Base vs FT vs RAG vs FT+RAG）
- [ ] 错误分析（各模型在哪些类别/场景犯错）
- [ ] Case Study（3-5 个典型案例对比）
- [ ] 撰写 Final Report（ACL 格式，≤8 页）
- [ ] 制作 Presentation

---

## 四、关键决策记录

| 日期 | 决策 | 原因 |
|------|------|------|
| 2026-04-14 | Task 3 改用通用医疗 QA（500条） | MedQuAD 中心理健康条目仅 133 条，不够 |
| 2026-04-14 | 云平台选择 Google Colab Pro/Pro+ | 用户偏好，对学生友好 |
| 2026-04-14 | API 使用 OpenRouter 统一网关 | 一个 Key 覆盖 GPT-4o + Claude |
| 2026-04-14 | VPN 下才能调用 OpenRouter 付费模型 | 香港地区有 region 限制 |
| 2026-04-14 | Claude 模型更新为 Sonnet 4 | 原 3.5-Sonnet 已下线，改用 claude-sonnet-4 |
| 2026-04-14 | RAG 知识库使用 MedQuAD 全量（16K 条） | 全量知识库覆盖面更广 |

---

## 五、风险与注意事项

1. **Colab 断连**：免费版运行超过一定时间会断开，建议使用 Pro 版并定期保存 checkpoint
2. **API 费用**：GPT-4o 和 Claude 的 API 调用有成本，预计总共 $20-40
3. **数据集类别不均衡**：Sentiment Analysis 数据集的 Normal/Depression 条目远多于 Personality disorder，需要在评估时关注 Macro-F1
4. **模型下载时间**：开源模型 7-9B 约 14-18GB，首次在 Colab 上加载需要几分钟
5. **Gemma license**：使用 google/gemma-2-9b-it 需要在 HuggingFace 上同意 license
