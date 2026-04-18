# 项目团队操作指南

> 项目：Enhancing LLM Performance on Mental Health Tasks via Fine-tuning and RAG  
> 仓库：https://github.com/zigeng22/8307-project  
> 最后更新：2026-04-18

---

## 一、项目概况

### 1.1 五个模型

| 简称 | 全称 | 类型 | 调用方式 | 能否LoRA微调 |
|------|------|------|---------|-------------|
| `deepseek-v3` | DeepSeek V3 (671B) | 开源权重但太大无法本地跑 | OpenRouter API | ❌ |
| `mistral-large` | Mistral Large | 闭源 | OpenRouter API | ❌ |
| `llama-3.1-8b` | Meta Llama 3.1 8B Instruct | 开源 | Colab 本地 GPU | ✅ |
| `qwen2.5-7b` | Qwen 2.5 7B Instruct | 开源 | Colab 本地 GPU | ✅ |
| `gemma-2-9b` | Google Gemma 2 9B IT | 开源 | Colab 本地 GPU | ✅ |

### 1.2 三个任务

| 任务 | 数据集 | 测试集大小 | 评估指标 |
|------|--------|-----------|---------|
| Task1: 心理状态分类 | Combined Data.csv (7类) | 1000条 | Accuracy, Macro-F1 |
| Task2: 对话生成 | MentalChat16K | 500条 | ROUGE-L, BERTScore |
| Task3: 医疗问答 | MedQuAD | 500条 | Token-F1, Exact Match, ROUGE-L |

### 1.3 四种实验配置

| 配置 | 说明 | 适用模型 |
|------|------|---------|
| Base | 零样本，直接给 prompt | 全部5个 |
| Fine-tuned | LoRA微调后推理 | 仅3个开源模型 |
| Base + RAG | 零样本 + 检索增强 | 全部5个 |
| Fine-tuned + RAG | 微调 + 检索增强 | 仅3个开源模型 |

### 1.4 总实验量

- API 模型 (2个) × 3任务 × 2配置 (Base, Base+RAG) = **12**
- 开源模型 (3个) × 3任务 × 4配置 = **36**
- **总计：48 个实验数据点**

---

## 二、代码结构

```
wtc/                              ← 项目根目录（GitHub 仓库）
│
├── config.py                     ← 【核心配置】模型注册表、路径、超参数
│                                    所有模型名称、LoRA参数、RAG参数都在这里定义
│
├── data/                         ← 数据处理
│   ├── loader.py                 ← 加载三个 CSV 数据集的函数
│   └── splitter.py               ← 划分训练/测试集（固定 seed=42 保证可复现）
│
├── prompts/
│   └── templates.py              ← 三个任务的 Prompt 模板（所有模型共用同一份）
│                                    + RAG版本（在原prompt前面加检索到的上下文）
│
├── models/                       ← 模型接口（统一的 generate() 方法）
│   ├── base.py                   ← 抽象基类
│   ├── api_model.py              ← OpenAI / Anthropic 官方 API
│   ├── openrouter_model.py       ← OpenRouter 统一网关（当前使用这个）
│   └── hf_model.py               ← HuggingFace 本地模型（支持加载 LoRA）
│
├── evaluation/
│   └── metrics.py                ← 评估指标计算
│                                    Task1: Accuracy + F1
│                                    Task2: ROUGE-L + BERTScore
│                                    Task3: Token-F1 + Exact Match + ROUGE-L
│
├── rag/                          ← RAG 检索增强
│   ├── indexer.py                ← 构建 FAISS 向量索引（从 MedQuAD 全量数据）
│   └── retriever.py              ← 查询检索：输入文本 → 返回 top-k 相关段落
│
├── finetune/
│   └── lora_train.py             ← LoRA 微调训练脚本（基于 MentalChat16K）
│
├── experiments/                  ← 实验运行脚本（每个对应一个阶段）
│   ├── run_baseline.py           ← Phase 1: 零样本 baseline
│   ├── run_finetuned.py          ← Phase 2: 微调后评估
│   └── run_rag.py                ← Phase 3: RAG 增强实验（支持 base 和 finetuned）
│
├── notebooks/
│   └── colab_quickstart.ipynb    ← Colab 快速启动 notebook
│
├── results/                      ← 实验结果（自动生成）
│   ├── baseline/{模型名}/         ← task1_metrics.json, task1_predictions.json ...
│   ├── finetuned/{模型名}/
│   ├── base_rag/{模型名}/
│   └── finetuned_rag/{模型名}/
│
└── PROGRESS.md                   ← 进度追踪
```

### 关键设计：统一接口

不管是 API 模型还是本地模型，调用方式完全一样：
```python
model = get_model("deepseek-v3")      # API 模型
model = get_model("qwen2.5-7b")       # 本地模型
response = model.generate(messages)    # 同样的接口
```

这样所有实验脚本的逻辑都一样，只是 `--model` 参数不同。

---

## 三、环境准备

### 3.1 所有人都需要的

- GitHub 账号（clone 仓库）
- Google 账号（访问 Colab）

### 3.2 跑 API 模型实验

只需要 OpenRouter API Key（整个组共用一个就行）：
- 在 Colab 的 notebook 中运行时会弹出输入框，粘贴 key 即可
- **不需要 GPU**，免费 Colab 就能跑 API 模型

### 3.3 跑开源模型实验

需要 **Colab Pro**（A100 40GB GPU）：
- 7B/9B 模型在 fp16 下需要 14-18GB VRAM
- LoRA 微调需要更多显存
- 免费 Colab 的 T4 (16GB) 勉强能跑推理但跑不了微调

---

## 四、三大核心概念：Baseline / LoRA微调 / RAG

在看具体命令之前，先理解每个实验阶段到底在做什么。

### 4.1 数据集是怎么划分的

| 数据集 | 总量 | 训练集 | 测试集 | 用途 |
|--------|------|--------|--------|------|
| Sentiment Analysis | ~30K 条 | ❌ 不训练 | 1000 条（按7类均衡抽样） | Task1 分类评估 |
| MentalChat16K | 16,084 条 | ~15,584 条 | 500 条 | 训练集 → LoRA微调；测试集 → Task2 评估 |
| MedQuAD | ~47K 条 | 全部用于 RAG 索引 | 500 条（心理健康相关） | 全量 → RAG检索库；测试集 → Task3 评估 |

**关键**：所有4种配置（Base / Fine-tuned / Base+RAG / Fine-tuned+RAG）的评估都用 **完全相同的测试集**，唯一变化的是模型本身或 prompt 内容。这样才能公平对比。

### 4.2 Baseline 是什么

**一句话**：不做任何增强，直接把测试问题丢给模型，看它"裸考"能答多好。

```
测试问题 → [构造 Prompt] → 发给模型 → 模型回答 → 和标准答案比较 → 得到指标
```

这是所有实验的基准参照——后面 Fine-tuned 和 RAG 的指标都是和 Baseline 对比。

### 4.3 LoRA 微调是什么

**一句话**：用 MentalChat16K 的 15,584 条训练数据教模型"如何做心理咨询师"。

**训练数据长什么样**（MentalChat16K 每一行）：

| instruction | input | output |
|---|---|---|
| "你是一名心理咨询师..." | "最近工作压力很大，睡不着" | "我能感受到你的压力。你能告诉我这种状态持续多久了吗？" |

**训练过程**：
1. 把 instruction + input + output 组合成一段完整对话文本
2. 让模型 **学习在看到 instruction+input 之后，生成和 output 尽可能相似的回答**
3. 重复 3 遍（3 epochs），训练 loss 持续下降说明模型在学习
4. LoRA 只微调模型注意力层旁边插入的小矩阵（占总参数 ~0.5%），原始模型冻结不动

**训练完成后**：保存 LoRA 权重（几十 MB 的小文件）到 `finetune/checkpoints/{模型名}/`

**评估方式**：用微调后的模型重新跑 **同一份测试集**，对比 Baseline 指标：
- Baseline BERTScore = 0.848 → Fine-tuned BERTScore = 0.880 → 提升 +0.032

**注意**：虽然模型只在 MentalChat16K（Task2 的训练集）上训练，但我们也测试它在 Task1（分类）和 Task3（问答）上的表现，看是否有"跨任务迁移"效果。API 模型不做微调。

### 4.4 RAG 是什么

**一句话**：给模型"开卷考试"——在提问之前，先从知识库检索相关参考信息，拼到 prompt 里。

**RAG 没有训练过程**，它是在推理时增强 prompt 的方法。

**工作流程**：

```
                     ┌──────────────────────────────────────┐
                     │ MedQuAD 全部 47K 条问答              │
                     │ → 切成 500 字的小块（chunk）           │
                     │ → 用 sentence-transformers 转成向量   │
                     │ → 存入 FAISS 向量数据库               │
                     │   （一次性操作，python rag/indexer.py）│
                     └────────────────┬─────────────────────┘
                                      │
测试问题 ──→ 转成向量 ──→ 在 FAISS 中找最相似的 3 条 ──→ 拼到 Prompt 前面
                                                              │
                                              ┌───────────────┘
                                              ▼
            "请参考以下背景信息回答问题：
             参考1：[检索到的相关知识]
             参考2：[...]
             参考3：[...]
             ---
             Question: 抑郁症的主要症状有哪些？"
                              │
                              ▼
                    发给模型 → 生成回答 → 评估指标
```

**查询（query）从哪来？就是测试集里的数据**：

| 任务 | 查询内容 | 举例 |
|------|---------|------|
| Task1 | 用户帖子文本 | "I've been feeling hopeless for weeks..." |
| Task2 | 患者的描述 | "最近工作压力很大，睡不着" |
| Task3 | 医疗问题 | "What are the symptoms of depression?" |

**对比 Base 和 Base+RAG 的 prompt 区别**：

| 配置 | Prompt 内容 |
|------|------------|
| Base | "Question: 抑郁症和双相的区别？请回答。" |
| Base+RAG | "参考以下信息：\n[检索到的3条参考]\n---\nQuestion: 抑郁症和双相的区别？请回答。" |

**预期效果**：
- Task3（医疗问答）：RAG 效果最显著，因为检索库里可能直接有相关答案
- Task1/Task2：RAG 效果可能不明显——这本身就是有价值的发现

### 4.5 四种配置的完整对比

```
                          ┌─ Base: 原模型 + 原prompt
同一份测试数据 ──→ 4种方式 ─┤─ Fine-tuned: 微调后模型 + 原prompt
                          ├─ Base+RAG: 原模型 + 增强prompt（加了检索信息）
                          └─ Fine-tuned+RAG: 微调后模型 + 增强prompt
                                     │
                                     ▼
                            同一套评估指标
                       （Accuracy/F1/ROUGE/BERTScore）
                                     │
                                     ▼
                         对比表格 + 分析 = 报告内容
```

---

## 五、实验操作命令

### Phase 1: Baseline（零样本）

**做什么**：直接给模型 prompt，不做任何训练，看模型原始能力。

**命令**：
```bash
# API 模型（不需要 GPU，普通 Colab 也行）
python experiments/run_baseline.py --model deepseek-v3 --task task1
python experiments/run_baseline.py --model deepseek-v3 --task task2
python experiments/run_baseline.py --model deepseek-v3 --task task3
python experiments/run_baseline.py --model mistral-large --task task1
python experiments/run_baseline.py --model mistral-large --task task2
python experiments/run_baseline.py --model mistral-large --task task3

# 开源模型（需要 A100 GPU）
python experiments/run_baseline.py --model llama-3.1-8b --task task1
python experiments/run_baseline.py --model llama-3.1-8b --task task2
python experiments/run_baseline.py --model llama-3.1-8b --task task3
python experiments/run_baseline.py --model qwen2.5-7b --task task1
python experiments/run_baseline.py --model qwen2.5-7b --task task2
python experiments/run_baseline.py --model qwen2.5-7b --task task3
python experiments/run_baseline.py --model gemma-2-9b --task task1
python experiments/run_baseline.py --model gemma-2-9b --task task2
python experiments/run_baseline.py --model gemma-2-9b --task task3
```

**结果保存位置**：`results/baseline/{模型名}/task{N}_metrics.json`

**当前进度**：DeepSeek V3 ✅ 已完成 3 个任务，Qwen2.5-7B ✅ 已完成 3 个任务

---

### Phase 2: LoRA 微调（仅3个开源模型）

**做什么**：用 MentalChat16K 训练集对模型做 LoRA 参数高效微调，然后用微调后的模型重新跑 3 个任务。

**Step 1 — 训练**（每个模型约 1-3 小时，需要 A100）：
```bash
python finetune/lora_train.py --model llama-3.1-8b
python finetune/lora_train.py --model qwen2.5-7b
python finetune/lora_train.py --model gemma-2-9b
```
训练完成后，LoRA 权重保存在 `finetune/checkpoints/{模型名}/`

**Step 2 — 评估微调后模型**：
```bash
python experiments/run_finetuned.py --model llama-3.1-8b --task task1 \
    --lora_path ./finetune/checkpoints/llama-3.1-8b
python experiments/run_finetuned.py --model llama-3.1-8b --task task2 \
    --lora_path ./finetune/checkpoints/llama-3.1-8b
python experiments/run_finetuned.py --model llama-3.1-8b --task task3 \
    --lora_path ./finetune/checkpoints/llama-3.1-8b
# qwen 和 gemma 同理，换模型名和对应路径
```

**结果保存位置**：`results/finetuned/{模型名}/task{N}_metrics.json`

**注意**：API 模型（DeepSeek V3, Mistral Large）不做微调，跳过这步。

---

### Phase 3: RAG 检索增强

**做什么**：先用 MedQuAD 全量数据建一个向量索引，查询时检索相关段落拼接到 prompt 前面，看检索增强是否提升性能。

**Step 1 — 构建 FAISS 索引**（一次性操作，约 5 分钟）：
```bash
python rag/indexer.py
```
索引保存在 `rag/faiss_index/`

**Step 2 — Base + RAG**（全部5个模型）：
```bash
python experiments/run_rag.py --model deepseek-v3 --task task1
python experiments/run_rag.py --model deepseek-v3 --task task2
python experiments/run_rag.py --model deepseek-v3 --task task3
# 其他4个模型同理
```

**Step 3 — Fine-tuned + RAG**（仅3个开源模型，需要先完成 Phase 2）：
```bash
python experiments/run_rag.py --model llama-3.1-8b --task task1 \
    --lora_path ./finetune/checkpoints/llama-3.1-8b
# 其他 task 和模型同理
```

**结果保存位置**：
- Base + RAG → `results/base_rag/{模型名}/`
- Fine-tuned + RAG → `results/finetuned_rag/{模型名}/`

---

### Phase 4: 汇总分析 + 写报告

48 组实验全部跑完后，每组实验的结果在 `results/{配置}/{模型名}/task{N}_metrics.json`。

**需要做的分析工作**：

1. **填写完整结果表**：把 48 个指标汇总成一张大表（5模型 × 3任务 × 4配置）
2. **对比分析**：
   - Base vs Fine-tuned → 微调提升了多少？
   - Base vs Base+RAG → RAG 提升了多少？
   - Fine-tuned vs Fine-tuned+RAG → 两者叠加有没有额外收益？
3. **错误分析**：
   - Task1：哪些心理状态类别最容易被搞混？（看 per_class F1）
   - Task2/3：挑几个模型回答特别好/特别差的样本做 case study
4. **可视化**：柱状图对比各模型各配置的指标
5. **撰写报告**（50% 成绩）+ **制作 Presentation**（40% 成绩）

---

## 六、组员分工建议

| 角色 | 负责内容 | 需要什么 |
|------|---------|---------|
| 同学A（有Colab Pro） | 3个开源模型的全部实验（Baseline + 微调 + RAG） | Colab Pro A100 |
| 同学B | API模型 Baseline + RAG（DeepSeek V3） | 免费Colab + OpenRouter Key |
| 同学C | API模型 Baseline + RAG（Mistral Large） | 免费Colab + OpenRouter Key |
| 同学D | RAG索引构建 + 结果汇总分析 | 免费Colab |
| 同学E | 报告撰写 + 可视化 | 本地即可 |

---

## 七、在 Colab 上跑实验的操作流程

### 7.1 首次设置（每次新开 Colab 都需要）

1. **Runtime → Change runtime type → A100 GPU**（跑开源模型时选，API模型不需要）
2. 运行以下 cell：

```python
# clone 代码
!git clone https://github.com/zigeng22/8307-project.git /content/8307-project

# 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 创建符号链接到数据集
import os
os.symlink('/content/drive/MyDrive/8307/Datasets', '/content/Datasets')

# 安装依赖
!pip install -q transformers>=4.43.0 peft>=0.11.0 trl>=0.9.0 datasets accelerate
!pip install -q langchain langchain-community faiss-cpu sentence-transformers
!pip install -q rouge-score bert-score scikit-learn openai tqdm pandas

# 设置路径
import sys
sys.path.insert(0, '/content/8307-project')
os.chdir('/content/8307-project')

# 设置 API Key（运行时粘贴）
import getpass
os.environ['OPENROUTER_API_KEY'] = getpass.getpass('Paste OpenRouter Key: ')
```

### 7.2 数据集在 Google Drive 上的位置

需要提前上传到 `MyDrive/8307/Datasets/`：
```
MyDrive/8307/Datasets/
├── Combined Data.csv
├── medquad.csv
└── MentalChat16K-main/
    ├── Interview_Data_6K.csv
    └── Synthetic_Data_10K.csv
```

### 7.3 运行完后备份结果

```python
!cp -r results/ /content/drive/MyDrive/8307/results_backup/
```

---

## 八、常见问题

**Q: API 调用返回 403 怎么办？**  
A: OpenAI 和 Anthropic 的模型在 OpenRouter 上被封了。用 DeepSeek V3 和 Mistral Large 替代。如果找到新的能用的 API 渠道，可以随时加回 GPT-4o 和 Claude。

**Q: Colab 断开了数据还在吗？**  
A: 代码可以重新 `git clone`，数据在 Google Drive 不会丢。只有 `results/` 目录在 Colab 虚拟机上，断开就没了——所以跑完实验要及时 `cp -r results/ /content/drive/...` 备份。

**Q: 微调的 checkpoint 很大吗？**  
A: LoRA 只微调少量参数，checkpoint 只有几十 MB（不是整个模型的几 GB），备份到 Drive 很快。

**Q: 所有人的结果能放在一起吗？**  
A: 可以。每个人跑完后把 `results/` 文件夹备份到 Drive，最后汇总。结果按 `results/{配置}/{模型名}/task{N}_metrics.json` 组织，不会冲突。
