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

## 四、实验分四个阶段

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

- 汇总全部 48 个实验数据点
- 对比分析：Base vs Fine-tuned vs RAG vs Fine-tuned+RAG
- 错误分析：各模型在哪些类别/场景表现差
- Case Study：挑 3-5 个代表性案例展示
- 撰写最终报告

---

## 五、组员分工建议

| 角色 | 负责内容 | 需要什么 |
|------|---------|---------|
| 同学A（有Colab Pro） | 3个开源模型的全部实验（Baseline + 微调 + RAG） | Colab Pro A100 |
| 同学B | API模型 Baseline + RAG（DeepSeek V3） | 免费Colab + OpenRouter Key |
| 同学C | API模型 Baseline + RAG（Mistral Large） | 免费Colab + OpenRouter Key |
| 同学D | RAG索引构建 + 结果汇总分析 | 免费Colab |
| 同学E | 报告撰写 + 可视化 | 本地即可 |

---

## 六、在 Colab 上跑实验的操作流程

### 6.1 首次设置（每次新开 Colab 都需要）

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

### 6.2 数据集在 Google Drive 上的位置

需要提前上传到 `MyDrive/8307/Datasets/`：
```
MyDrive/8307/Datasets/
├── Combined Data.csv
├── medquad.csv
└── MentalChat16K-main/
    ├── Interview_Data_6K.csv
    └── Synthetic_Data_10K.csv
```

### 6.3 运行完后备份结果

```python
!cp -r results/ /content/drive/MyDrive/8307/results_backup/
```

---

## 七、需要人工调参的实验（核心 workload）

> **重要**：以上 Phase 1–4 的命令只是 **"用默认参数跑一次"**。
> 一个合格的研究项目需要 **实验设计 + 参数搜索 + 消融分析**，这才是真正的 workload。
> 以下列出每个阶段中需要人工反复实验和判断的内容。

---

### 7.1 Prompt Engineering 实验（Phase 1 阶段，但影响所有后续实验）

**当前状态**：`prompts/templates.py` 中写了一版基础 prompt，但这不一定是最优的。

**需要实验对比的 prompt 设计维度**：

| 维度 | 当前默认 | 需要对比的选项 | 预计实验次数 |
|------|---------|--------------|-------------|
| **语言** | 英文 | 英文 vs 中文 vs 中英混合 | ×3 |
| **示例数量** | 0-shot | 0-shot vs 1-shot vs 3-shot | ×3 |
| **推理方式** | 直接回答 | 直接 vs Chain-of-Thought | ×2 |
| **系统提示详细度** | 简洁 | 简洁 vs 详细角色描述 | ×2 |

**操作步骤**：
1. 在 `prompts/templates.py` 中创建多个版本的 prompt 变体
2. 用 **1 个模型**（如 Qwen2.5-7B）在 **1 个任务**（如 Task1）上测试所有变体
3. 选出最优 prompt，再统一用于所有模型
4. 在报告中写一节 "Prompt Selection"，展示不同 prompt 的指标对比表

**需要人工编写的代码**：
```python
# experiments/prompt_ablation.py —— 需要组员自己写
# 遍历不同 prompt 变体，固定模型和任务，对比指标
# 产出：一张 prompt 变体对比表（如 Accuracy 差异）
```

**这项工作的贡献**：对应 PROJECT_DESCRIPTION 中"新的评估协议"

---

### 7.2 LoRA 超参数搜索（Phase 2）

**当前状态**：`config.py` 中的 LoRA 参数是通用推荐值，**不是针对心理健康任务调优的**。

**需要搜索的超参数**：

| 参数 | 当前默认值 | 建议搜索范围 | 为什么要调 |
|------|----------|-------------|-----------|
| **r** (LoRA rank) | 16 | {4, 8, 16, 32} | 太小→欠拟合，太大→过拟合，需要找最优平衡点 |
| **lora_alpha** | 32 | 跟随 r 变化 (alpha/r ∈ {1, 2, 4}) | 缩放系数影响学习强度 |
| **learning_rate** | 2e-4 | {5e-5, 1e-4, 2e-4, 5e-4} | 不同模型对学习率敏感度不同 |
| **num_epochs** | 3 | {1, 2, 3, 5} | 需观察 loss 曲线判断何时过拟合 |
| **target_modules** | q/k/v/o_proj | attention-only vs attention+MLP | 微调更多层是否有收益 |
| **lora_dropout** | 0.05 | {0.0, 0.05, 0.1} | 正则化强度 |

**操作步骤**：
1. **先跑一次默认参数**，得到 baseline 微调结果
2. **逐一改变一个参数**（固定其他参数），观察 Task2 BERTScore 变化
3. **记录每次实验的**：
   - `config.py` 中修改的参数值
   - 训练 loss 曲线（training log 自动保存）
   - 最终 Task2 上的指标
4. **选出最优组合**，用最优参数跑全部 3 个任务的正式评估
5. 在报告中写一节 "Hyperparameter Study"，展示消融表

**具体修改方式（每次实验改 `config.py`）**：
```python
# 例1：测试 r=8
LORA_CONFIG = {
    "r": 8,            # ← 改这里（默认16）
    "lora_alpha": 16,  # ← 保持 alpha = 2*r
    ...
}

# 例2：测试 lr=1e-4
TRAINING_ARGS = {
    "learning_rate": 1e-4,  # ← 改这里（默认2e-4）
    ...
}
```

**每个开源模型至少跑 6-8 次微调**（不同参数组合），每次约 1-2 小时（A100）。
3 个模型 × 6-8 次 = **约 18-24 次微调实验**。

**需要人工编写的代码**：
```python
# experiments/lora_ablation.py —— 需要组员自己写
# 读取不同参数组合的 results，生成消融对比表
# 可能还要画 loss 曲线图
```

**这项工作的贡献**：对应课程类别 3 "预训练语言模型的高效适配"

---

### 7.3 RAG 参数消融（Phase 3）

**当前状态**：`config.py` 中的 RAG 参数是常见默认值，**未针对 MedQuAD 数据集优化**。

**需要搜索的超参数**：

| 参数 | 当前默认值 | 建议搜索范围 | 影响 |
|------|----------|-------------|------|
| **chunk_size** | 500 | {250, 500, 1000} | 小块→精准但碎片化；大块→完整但可能含噪声 |
| **chunk_overlap** | 50 | {0, 50, 100} | 重叠度影响边界信息是否丢失 |
| **top_k** | 3 | {1, 3, 5, 10} | 检索几条？太多会让 prompt 太长 |
| **embedding_model** | all-MiniLM-L6-v2 | MiniLM vs mpnet-base vs bge-small | 不同 embedding 检索质量不同 |

**操作步骤**：
1. **改 `config.py` 中的 RAG_CONFIG**，重新运行 `python rag/indexer.py`
2. **用一个模型（如 DeepSeek V3）在 Task3 上测**不同参数的效果
3. **每改一次参数就需要重建索引**（约 5 分钟），然后重跑实验
4. 选出最优参数组合，用于正式的全模型实验
5. 在报告中写一节 "RAG Ablation Study"

**具体修改方式**：
```python
# 例：测试 chunk_size=250, top_k=5
RAG_CONFIG = {
    "chunk_size": 250,       # ← 改这里（默认500）
    "chunk_overlap": 25,     # ← 对应调整
    "top_k": 5,              # ← 改这里（默认3）
    ...
}
# 然后重新 python rag/indexer.py 再跑实验
```

**RAG 消融大约需要 10-15 次实验**（不同参数组合 × Task3 评测）。

**需要人工编写的代码**：
```python
# experiments/rag_ablation.py —— 需要组员自己写
# 固定模型，遍历不同 RAG 参数，对比 Task3 F1
```

**这项工作的贡献**：对应课程类别 4 "检索与长上下文系统"

---

### 7.4 Error Analysis（Phase 4，最体现研究深度的部分）

**当前状态**：**完全没有代码**，需要组员自己设计和实现。

**需要做的分析**：

| 分析内容 | 具体做什么 | 产出 |
|---------|----------|------|
| **Task1 混淆矩阵** | 用 sklearn 画 7×7 混淆矩阵热力图，分析哪些类别互相混淆 | 混淆矩阵图 ×5 模型 ×4 配置 |
| **Per-class F1 对比** | 哪些心理状态类别最难分类？模型间差异在哪？ | 柱状图/雷达图 |
| **高风险类别分析** | "Suicidal" 和 "Bipolar" 被误判的概率和后果 | 专门讨论段落 |
| **Task2 质量分布** | BERTScore 的分布直方图，找出极低分和极高分的样本 | 分布图 + 极端案例 |
| **Task3 检索质量** | RAG 检索到的文档真的相关吗？抽样检查 top-k 检索结果 | 检索准确率估算 |
| **Case Study** | 选 3-5 个代表性问题，展示所有模型+配置的回答差异 | 表格/并排对比 |
| **Ablation 总表** | Base vs FT vs RAG vs FT+RAG 的提升幅度统计 | ΔAccuracy/ΔF1 表 |

**需要人工编写的代码**：
```python
# analysis/confusion_matrix.py —— 画混淆矩阵
# analysis/per_class_analysis.py —— 各类别详细指标
# analysis/case_study.py —— 提取有代表性的 case
# analysis/ablation_table.py —— 汇总消融实验结果
# analysis/visualize.py —— 图表生成（柱状图、雷达图、热力图）
```

**这项工作的贡献**：对应 PROJECT_DESCRIPTION 明确要求的 "仔细的错误分析"

---

### 7.5 Human Evaluation（Task2 对话生成的人工评估）

**当前状态**：**完全没有代码**。Task2 的自动指标（ROUGE/BERTScore）不足以评估咨询对话质量。

**需要做的事**：

1. **设计评分标准**（rubric），例如：
   - 共情度（1-5分）：是否理解患者情绪？
   - 专业性（1-5分）：是否提供合理建议？
   - 安全性（1-5分）：是否有潜在有害建议？
   - 流畅性（1-5分）：语言是否自然？

2. **抽样**：从 500 条 Task2 测试集中抽 50 条

3. **打分**：每条 × 5 个模型 × 最优配置 = 250 次打分
   - 建议 2-3 人独立打分，计算 inter-annotator agreement（Cohen's Kappa）

4. **汇总**：各模型的人工评分均值 ± 标准差

**需要人工编写的代码**：
```python
# evaluation/human_eval.py —— 生成打分表（Excel/CSV），汇总打分结果
```

---

### 7.6 统计显著性检验

**当前状态**：**没有代码**。目前的指标只是单个数字，无置信区间。

**需要做的事**：

1. **Bootstrap 置信区间**：对每个指标做 1000 次 bootstrap 采样，报告 95% CI
2. **配对检验**：Fine-tuned vs Base 的提升是否统计显著？（paired bootstrap test）
3. **在报告的 Results 表中加上 ± CI**，例如 `Accuracy = 72.3 ± 1.8`

**需要人工编写的代码**：
```python
# evaluation/significance.py —— bootstrap CI + paired test
```

---

### 7.7 总结：默认参数实验 vs 完整实验的区别

| 维度 | "只跑默认参数" | "完整研究项目" |
|------|--------------|--------------|
| Phase 1 | 1 套 prompt × 5 模型 | **3-5 套 prompt 先对比选最优** |
| Phase 2 | 1 组 LoRA 参数 × 3 模型 | **6-8 组参数 × 3 模型 → 选最优** |
| Phase 3 | 1 组 RAG 参数 × 5 模型 | **10-15 组 RAG 参数先消融** |
| Phase 4 | 只有数字表格 | **混淆矩阵 + case study + 显著性检验 + 人工评估** |
| 报告 | 列表格 | **深度分析每个发现的原因和意义** |
| 代码贡献 | 0行新代码 | **analysis/ + ablation 脚本 + 可视化 + 人工评估框架** |

**总实验量（含调参）**：

- 默认参数只跑一轮：48 次实验
- 加上调参和消融：**约 100-150 次实验**
- 再加上 Error Analysis + Human Evaluation + 报告撰写

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
