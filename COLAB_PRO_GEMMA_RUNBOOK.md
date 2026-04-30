# Colab Pro 跑 Gemma 剩余实验操作手册

> 更新时间：2026-04-30
> 适用范围：仅针对本项目 `gemma-2-9b` 的剩余实验（Fine-tuned / Base+RAG / Fine-tuned+RAG）
> 运行约束：A100 + 默认参数，不做任何 OOM 降配改参

---

## 1. 目标与范围

根据当前实验表，Gemma 已完成 Baseline，剩余 9 个结果点：

1. `finetuned`：task1 / task2 / task3
2. `base_rag`：task1 / task2 / task3
3. `finetuned_rag`：task1 / task2 / task3

本手册固定使用项目初始默认训练参数：

1. `per_device_train_batch_size = 4`
2. `gradient_accumulation_steps = 4`
3. `max_seq_length = 2048`

说明：本手册不包含任何降配参数，不考虑 OOM 分支；若不是 A100，请先切换运行时后再执行。

建议执行顺序：

1. 环境与数据准备
2. LoRA 训练（拿到 Gemma checkpoint）
3. Fine-tuned 评估（all）
4. 构建 RAG 索引（若已存在可跳过）
5. Base+RAG（all）
6. Fine-tuned+RAG（all）
7. 结果校验与回传

---

## 2. 开始前准备

### 2.1 Colab 运行时

1. 打开 Colab Pro。
2. 进入 Runtime -> Change runtime type。
3. Hardware accelerator 选择 GPU。
4. 必须选择 A100（不是 A100 就不要继续执行）。

说明：本 runbook 按 A100 固定参数设计，队友执行时不要改训练参数。

### 2.2 HuggingFace 访问

1. 在 HuggingFace 页面接受 `google/gemma-2-9b-it` 使用条款。
2. 创建 HF Token（Read 权限即可）。
3. 后续在 Notebook 里 `login()` 时粘贴该 token。

### 2.3 Google Drive 数据目录

确保 Drive 中存在：

- `/content/drive/MyDrive/8307/Datasets/Combined Data.csv`
- `/content/drive/MyDrive/8307/Datasets/medquad.csv`
- `/content/drive/MyDrive/8307/Datasets/MentalChat16K-main/Interview_Data_6K.csv`
- `/content/drive/MyDrive/8307/Datasets/MentalChat16K-main/Synthetic_Data_10K.csv`

---

## 3. Colab Notebook 一次性初始化

以下代码块按顺序执行。

### Cell 1: clone 代码 + 挂载 Drive + 建数据软链接

```python
import os

REPO_DIR = "/content/8307-project"
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/zigeng22/8307-project {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

from google.colab import drive
drive.mount('/content/drive')

DATA_SRC = '/content/drive/MyDrive/8307/Datasets'
DATA_LINK = '/content/Datasets'
assert os.path.exists(DATA_SRC), f'未找到数据目录: {DATA_SRC}'

if os.path.islink(DATA_LINK) or os.path.exists(DATA_LINK):
    !rm -rf {DATA_LINK}
os.symlink(DATA_SRC, DATA_LINK)
print(f'linked: {DATA_LINK} -> {DATA_SRC}')
```

### Cell 2: 安装依赖（用项目已验证版本）

```python
!pip install -q -U pip setuptools wheel
!pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
!pip install -q transformers==4.46.3 peft==0.12.0 trl==0.11.4 datasets==2.21.0 accelerate==0.34.2
!pip install -q langchain==0.2.16 langchain-community==0.2.16 faiss-cpu==1.8.0.post1 sentence-transformers==3.0.1
!pip install -q rouge-score==0.1.2 bert-score==0.3.13 scikit-learn==1.5.1
!pip install -q openai==1.52.2 anthropic==0.34.2 pandas==2.2.2 matplotlib==3.9.2 seaborn==0.13.2 tqdm==4.66.5 "httpx<0.28"
```

### Cell 3: 路径与 GPU 检查

```python
import os, sys, torch
os.chdir('/content/8307-project')
sys.path.insert(0, '/content/8307-project')

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print('GPU:', props.name)
    print('VRAM(GB):', round(props.total_memory / 1e9, 2))

from data.loader import load_sentiment, load_mentalchat, load_medquad
print('Sentiment rows:', len(load_sentiment()))
print('MentalChat rows:', len(load_mentalchat()))
print('MedQuAD rows:', len(load_medquad()))
```

### Cell 3.1: A100 与默认参数硬校验（必须通过）

```python
import torch
from pathlib import Path

assert torch.cuda.is_available(), '未检测到 GPU，请切换到 A100 运行时'
gpu_name = torch.cuda.get_device_properties(0).name
assert 'A100' in gpu_name, f'当前 GPU 为 {gpu_name}，本手册要求 A100'

cfg_text = Path('/content/8307-project/config.py').read_text(encoding='utf-8')
lora_text = Path('/content/8307-project/finetune/lora_train.py').read_text(encoding='utf-8')

assert '"per_device_train_batch_size": 4' in cfg_text, 'batch_size 不是默认值 4'
assert '"gradient_accumulation_steps": 4' in cfg_text, 'grad_accum 不是默认值 4'
assert 'max_seq_length=2048,' in lora_text, 'max_seq_length 不是默认值 2048'

print('A100 + 默认参数校验通过 ✓')
```

### Cell 4: HuggingFace 登录（Gemma 必做）

```python
from huggingface_hub import login
login()
```

### Cell 5: 持久化输出目录（直接写 Drive，避免断连丢结果）

```python
import os
DRIVE_ROOT = '/content/drive/MyDrive/8307'

RESULT_DIR = f'{DRIVE_ROOT}/results_colab_gemma'
CKPT_DIR = f'{DRIVE_ROOT}/checkpoints/gemma-2-9b'
INDEX_DIR = f'{DRIVE_ROOT}/rag/faiss_index'

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

print('RESULT_DIR =', RESULT_DIR)
print('CKPT_DIR   =', CKPT_DIR)
print('INDEX_DIR  =', INDEX_DIR)
```

---

## 4. RAG 索引持久化（建议）

为了避免每次重开 Colab 都重建索引，把 `rag/faiss_index` 链接到 Drive。

### Cell 6: 建索引目录软链接

```python
import os, shutil
repo_index = '/content/8307-project/rag/faiss_index'
drive_index = INDEX_DIR
os.makedirs(drive_index, exist_ok=True)

if os.path.islink(repo_index):
    os.remove(repo_index)
elif os.path.isdir(repo_index):
    shutil.rmtree(repo_index)

os.symlink(drive_index, repo_index)
print(f'linked: {repo_index} -> {drive_index}')
```

### Cell 7: 若索引为空则构建

```python
from pathlib import Path
p = Path('/content/8307-project/rag/faiss_index')
if len(list(p.glob('*'))) == 0:
    !python rag/indexer.py
    print('FAISS index built')
else:
    print('FAISS index already exists, skip build')
```

---

## 5. Gemma LoRA 训练（关键步骤）

### 5.1 先尝试标准命令（A100 优先）

```python
!python finetune/lora_train.py --model gemma-2-9b --output_dir "{CKPT_DIR}"
```

### 5.2 训练前参数一致性复核（建议保留）

```python
from pathlib import Path

cfg_text = Path('/content/8307-project/config.py').read_text(encoding='utf-8')
lora_text = Path('/content/8307-project/finetune/lora_train.py').read_text(encoding='utf-8')

print('batch_size default   =', '"per_device_train_batch_size": 4' in cfg_text)
print('grad_accum default   =', '"gradient_accumulation_steps": 4' in cfg_text)
print('max_seq_len default  =', 'max_seq_length=2048,' in lora_text)
```

### 5.3 训练结果落盘检查（checkpoint）

```python
from pathlib import Path
ckpt_root = Path(CKPT_DIR)
print('checkpoint dir exists:', ckpt_root.exists())
if ckpt_root.exists():
    print('sample files:', [p.name for p in list(ckpt_root.glob('*'))[:8]])
```

---

## 6. 跑 Gemma 剩余实验

### Cell 8: Fine-tuned（3 任务）

```python
!python experiments/run_finetuned.py --model gemma-2-9b --task all \
    --lora_path "{CKPT_DIR}" \
    --output_dir "{RESULT_DIR}"
```

### Cell 9: Base + RAG（3 任务）

```python
!python experiments/run_rag.py --model gemma-2-9b --task all \
    --output_dir "{RESULT_DIR}"
```

### Cell 10: Fine-tuned + RAG（3 任务）

```python
!python experiments/run_rag.py --model gemma-2-9b --task all \
    --lora_path "{CKPT_DIR}" \
    --output_dir "{RESULT_DIR}"
```

说明：`run_rag.py` 已支持 partial 续跑。若中断，直接重跑同一命令即可从 partial 文件继续。

### Cell 10.1: 结果文件快速点验（每步跑完都执行）

```python
from pathlib import Path

base = Path(RESULT_DIR)
checks = [
    base / 'finetuned' / 'gemma-2-9b',
    base / 'base_rag' / 'gemma-2-9b',
    base / 'finetuned_rag' / 'gemma-2-9b',
]

for d in checks:
    print('\n==', d)
    if d.exists():
        for f in sorted(d.glob('*_metrics.json')):
            print('  ', f.name)
    else:
        print('  not found yet')
```

---

## 7. 结果校验（必须）

### Cell 11: 检查 9 个 metrics 文件是否齐全

```python
from pathlib import Path

root = Path(RESULT_DIR)
required = [
    ('finetuned', 'task1_metrics.json'),
    ('finetuned', 'task2_metrics.json'),
    ('finetuned', 'task3_metrics.json'),
    ('base_rag', 'task1_metrics.json'),
    ('base_rag', 'task2_metrics.json'),
    ('base_rag', 'task3_metrics.json'),
    ('finetuned_rag', 'task1_metrics.json'),
    ('finetuned_rag', 'task2_metrics.json'),
    ('finetuned_rag', 'task3_metrics.json'),
]

missing = []
for cfg, fn in required:
    p = root / cfg / 'gemma-2-9b' / fn
    if not p.exists():
        missing.append(str(p))

if missing:
    print('Missing files:')
    for x in missing:
        print(' -', x)
else:
    print('All 9 Gemma remaining metrics files are ready ✓')
```

---

## 8. 回传到本地并更新总表

### 8.0 推荐同步方式：最终落到 Dropbox（你们当前方案）

按你现在的要求，流程统一为：

1. 队友在 Colab 跑完后先执行 8.1 打包。
2. 队友把 zip 下载到自己电脑。
3. 队友把 zip 内容拷贝到你们共享 Dropbox 目录。
4. 你在本机从 Dropbox 合并到 `wtc/results_server/results_gemma/`。

这样做的好处是：Colab 负责计算，Dropbox 负责团队共享与最终归档。

### 8.1 在 Colab 打包（可选）

```python
!cd "{DRIVE_ROOT}" && zip -r gemma_colab_outputs.zip results_colab_gemma checkpoints/gemma-2-9b
```

如果上面的 zip 不需要，也可直接在 Drive 网页下载这两个目录：

1. `MyDrive/8307/results_colab_gemma/`
2. `MyDrive/8307/checkpoints/gemma-2-9b/`

### 8.1.1 队友本地拷贝到 Dropbox（Windows 示例）

将下载后的压缩包先解压，再复制到你们 Dropbox 项目目录。示例命令：

```powershell
Expand-Archive -Path "C:\Users\<队友用户名>\Downloads\gemma_colab_outputs.zip" -DestinationPath "C:\Users\<队友用户名>\Downloads\gemma_colab_outputs" -Force
Copy-Item -Recurse -Force "C:\Users\<队友用户名>\Downloads\gemma_colab_outputs\results_colab_gemma\*" "C:\Users\Lenovo\Dropbox\8307\wtc\results_server\results_gemma\"
```

如果队友不是你的同一台电脑，就让队友把上述 `results_colab_gemma` 目录放入你们共用的 Dropbox 文件夹，你本地同步完成后再执行 8.3。

### 8.2 在本机项目中合并

将 `results_colab_gemma` 下的这三个目录合并到本地仓库 `wtc/results_server/results_gemma/`：

1. `finetuned/gemma-2-9b/`
2. `base_rag/gemma-2-9b/`
3. `finetuned_rag/gemma-2-9b/`

### 8.3 本地重建汇总 CSV

在本机项目的 `wtc/` 目录执行：

```powershell
python tools/sync_and_summarize_results.py --skip-sync
```

执行后会刷新：

1. `results/metrics_long.csv`
2. `results/experiment_table.csv`

---

## 9. 常见故障与处理

### 9.1 不是 A100（最常见）

处理方式：

1. 立即停止执行，不要继续训练。
2. 进入 Runtime -> Change runtime type，切到 A100。
3. 重新运行 3.1 校验 cell，通过后再继续。

### 9.2 HuggingFace 401 / 权限错误

1. 确认已在网页接受 Gemma 模型条款。
2. 在 notebook 重新 `login()`。
3. 确认 token 未过期。

### 9.3 找不到数据集

1. 确认 Drive 目录是否为 `MyDrive/8307/Datasets`。
2. 重新运行 Cell 1 建软链接。

### 9.4 RAG 中断

`run_rag.py` 支持 partial 文件续跑，直接重复执行同一条命令即可。

---

## 10. 建议的最稳执行策略

1. 优先在 A100 上先跑完 LoRA 训练。
2. 再跑 `finetuned all`。
3. 再跑 `base_rag all`。
4. 最后跑 `finetuned_rag all`。
5. 每个阶段结束后先检查 metrics 文件是否落盘，再进行下一阶段。
6. 全程保持默认参数，不做任何降配改参。

这样做的好处是：即使中途断线，也能最大化保住已完成结果。
