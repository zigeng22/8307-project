# 远程设备一步到位操作手册（真实参数版）

> 更新时间：2026-04-29
> 目标：按本文从上到下执行，执行完即可使用远程服务器算力跑实验。

---

## 1. 固定信息（直接用）

### 1.1 EasyConnect 登录信息

- VPN 入口地址：`https://123.58.249.106:10443/`
- VPN 账号：`gangda`
- VPN 密码：`Ht_GangDa@zsh#sh`

### 1.2 Linux 服务器登录信息

- SSH 地址：`172.17.0.18`
- SSH 账号：`hiteam`
- SSH 密码：`GangDA_HTzl@Tsh`

### 1.3 实验固定路径

- 项目目录：`/home/hiteam/8307-project`
- Python 虚拟环境：`/opt/venvs/llm8307/bin/activate`
- Qwen checkpoint 目录：`/home/hiteam/checkpoints/qwen2.5-7b`
- Qwen 结果目录：`/home/hiteam/results_gangda`
- Llama 结果目录：`/home/hiteam/results_llama`
- Gemma 结果目录：`/home/hiteam/results_gemma`
- BERTScore 本地模型目录：`/mnt/sdc/roberta-large`

---

## 2. 第一次连接（Windows）

### Step 1：打开 EasyConnect 并登录

1. 打开 EasyConnect 客户端。
2. 地址输入：`https://123.58.249.106:10443/`
3. 输入账号：`gangda`
4. 输入密码：`Ht_GangDa@zsh#sh`
5. 点击登录，保持 EasyConnect 在线。

### Step 2：打开 PowerShell 并 SSH 登录

在 Windows PowerShell 执行：

```bash
ssh hiteam@172.17.0.18
```

出现 password 提示后输入：`GangDA_HTzl@Tsh`，然后回车。

说明：输入密码时不显示字符，属于正常现象。

---

## 3. 登录后立即执行（环境就绪）

```bash
source /opt/venvs/llm8307/bin/activate
cd /home/hiteam/8307-project
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BERTSCORE_MODEL_TYPE=/mnt/sdc/roberta-large
export BERTSCORE_NUM_LAYERS=17
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

以上命令全部成功，说明可以开始跑实验。

### 3.1 项目目录与数据集检查（必须）

先在服务器执行：

```bash
cd /home/hiteam/8307-project
ls -lah /home/hiteam/Datasets
ls -lah /home/hiteam/Datasets/MentalChat16K-main
python -c "from data.loader import load_sentiment, load_mentalchat, load_medquad; print(len(load_sentiment()), len(load_mentalchat()), len(load_medquad()))"
```

如果最后一条命令报错找不到数据，说明服务器还没导入数据集。此时在本机 Windows PowerShell 执行：

```powershell
ssh hiteam@172.17.0.18 "mkdir -p /home/hiteam/Datasets"
scp "C:\Users\Lenovo\Dropbox\8307\Datasets\Combined Data.csv" hiteam@172.17.0.18:/home/hiteam/Datasets/
scp "C:\Users\Lenovo\Dropbox\8307\Datasets\medquad.csv" hiteam@172.17.0.18:/home/hiteam/Datasets/
scp -r "C:\Users\Lenovo\Dropbox\8307\Datasets\MentalChat16K-main" hiteam@172.17.0.18:/home/hiteam/Datasets/
```

上传后回到服务器重复执行上一段检查命令，直到 loader 检查成功。

---

## 4. 每次开工固定动作（照抄）

```bash
ssh hiteam@172.17.0.18
source /opt/venvs/llm8307/bin/activate
cd /home/hiteam/8307-project
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BERTSCORE_MODEL_TYPE=/mnt/sdc/roberta-large
export BERTSCORE_NUM_LAYERS=17
nvidia-smi
```

---

## 5. 直接可跑命令（Qwen）

### 5.1 Qwen Baseline

```bash
/opt/venvs/llm8307/bin/python experiments/run_baseline.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda
```

### 5.2 Qwen LoRA 微调

```bash
export CUDA_VISIBLE_DEVICES=0
/opt/venvs/llm8307/bin/python finetune/lora_train.py --model qwen2.5-7b --output_dir /home/hiteam/checkpoints/qwen2.5-7b
```

### 5.3 Qwen Fine-tuned 评估

```bash
export CUDA_VISIBLE_DEVICES=0
/opt/venvs/llm8307/bin/python experiments/run_finetuned.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda
```

### 5.4 Qwen Base + RAG

```bash
export CUDA_VISIBLE_DEVICES=3
[ -d rag/faiss_index ] || /opt/venvs/llm8307/bin/python rag/indexer.py
/opt/venvs/llm8307/bin/python experiments/run_rag.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda
```

### 5.5 Qwen Fine-tuned + RAG

```bash
export CUDA_VISIBLE_DEVICES=3
/opt/venvs/llm8307/bin/python experiments/run_rag.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda
```

---

## 6. 直接可跑命令（Llama / Gemma）

### 6.1 Llama Baseline（GPU1）

```bash
export CUDA_VISIBLE_DEVICES=1
/opt/venvs/llm8307/bin/python experiments/run_baseline.py --model llama-3.1-8b --task all --output_dir /home/hiteam/results_llama
```

### 6.2 Gemma Baseline（GPU2）

```bash
export CUDA_VISIBLE_DEVICES=2
/opt/venvs/llm8307/bin/python experiments/run_baseline.py --model gemma-2-9b --task all --output_dir /home/hiteam/results_gemma
```

---

## 7. tmux 用法（必须）

长任务都在 tmux 里跑，避免断线中断。

```bash
tmux new -s qwen_ft
tmux new -s llama_base
tmux new -s gemma_base
tmux ls
tmux attach -t qwen_ft
```

离开会话但不中断任务：按 `Ctrl+b`，再按 `d`。

---

## 8. 网络连通性自检（直接执行）

```bash
python - <<'PY'
import json, urllib.request
ip = urllib.request.urlopen("https://api64.ipify.org", timeout=10).read().decode().strip()
print("Public IP:", ip)
info = json.loads(urllib.request.urlopen(f"https://ipapi.co/{ip}/json/", timeout=10).read().decode())
print("Country:", info.get("country_name"), f"({info.get('country_code')})")
print("Region:", info.get("region"))
print("City:", info.get("city"))
print("Org:", info.get("org"))
PY
```

```bash
curl -I --max-time 15 https://huggingface.co
curl -I --max-time 15 https://openrouter.ai/api/v1/models
curl -I --max-time 15 https://api.openai.com/v1/models
curl -I --max-time 15 https://api.anthropic.com/v1/messages
```

判定规则：

1. `200/301/302`：可达
2. `401/403`：可达（只是未授权）
3. 超时/解析失败/TLS失败：网络路径有问题

### 8.1 当前服务器实测结论（2026-04-29）

根据已执行结果：

1. 公网出口 IP：`123.58.249.106`（中国北京联通）
2. `huggingface.co`：超时，不可达
3. `openrouter.ai`：HTTP 200，可达
4. `api.openai.com`：超时，不可达
5. `api.anthropic.com`：HTTP 403（网络可达，策略拒绝）

执行策略：

1. 开源模型必须使用本地预下载 + 离线模式（`HF_HUB_OFFLINE=1`）
2. API 模型优先走 OpenRouter
3. 不要依赖在线下载 HuggingFace 模型

---

## 9. 结果检查（收工前执行）

```bash
find /home/hiteam -type f -path "*/results*/*/*/*_metrics.json" | sort
```

看到各模型 task1/task2/task3 的 metrics 文件后，说明结果已成功落盘。

### 9.1 把远程结果同步回本地（用于填实验表）

在本机 Windows PowerShell 执行：

```powershell
mkdir "C:\Users\Lenovo\Dropbox\8307\wtc\results_server" -ErrorAction SilentlyContinue
scp -r hiteam@172.17.0.18:/home/hiteam/results_gangda "C:\Users\Lenovo\Dropbox\8307\wtc\results_server\"
scp -r hiteam@172.17.0.18:/home/hiteam/results_llama "C:\Users\Lenovo\Dropbox\8307\wtc\results_server\"
scp -r hiteam@172.17.0.18:/home/hiteam/results_gemma "C:\Users\Lenovo\Dropbox\8307\wtc\results_server\"
```

同步后，本地实验表填数统一从以下目录读取：

1. `C:\Users\Lenovo\Dropbox\8307\wtc\results\`（本机已跑结果）
2. `C:\Users\Lenovo\Dropbox\8307\wtc\results_server\`（远程同步结果）