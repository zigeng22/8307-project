# 远程设备快速上手手册（EasyConnect + SSH + 实验执行）

> 目标：让组员看到这份文档后，可在 Windows 上快速连上远程 Linux 服务器并开始跑实验。
> 更新时间：2026-04-28

---

## 1. 你需要的三类信息

在开始之前，先向算力公司确认以下信息：

1. EasyConnect 入口
- 客户端下载地址（例如公司给的地址）
- VPN 登录账号
- VPN 登录密码

2. Linux 服务器登录信息
- 服务器 IP（示例：172.17.0.18）
- SSH 用户名（示例：hiteam）
- SSH 密码

3. 项目路径信息
- 项目代码目录（示例：/home/hiteam/8307-project）
- Python 虚拟环境目录（示例：/opt/venvs/llm8307）
- 数据集目录（示例：/home/hiteam/Datasets）

---

## 2. 安装并登录 EasyConnect（Windows）

### Step 1：下载并安装

1. 打开公司给的 EasyConnect 下载地址。
2. 下载并安装客户端。
3. 安装完成后打开 EasyConnect。

### Step 2：填写 VPN 地址并登录

1. 在地址框输入公司给的 VPN 地址。
2. 输入 VPN 账号和密码。
3. 登录成功后，保持 EasyConnect 窗口处于连接状态。

说明：
- EasyConnect 的作用是打通网络通道。
- 真正执行实验命令是在 SSH 登录后的 Linux 终端里完成。

---

## 3. 打开 cmd/PowerShell 并 SSH 登录 Linux

### Step 1：打开本机终端

可使用以下任一终端：

1. Windows cmd
2. Windows PowerShell
3. Windows Terminal

### Step 2：SSH 登录

~~~bash
ssh 用户名@服务器IP
~~~

示例：

~~~bash
ssh hiteam@172.17.0.18
~~~

说明：
- 输入密码时不会显示字符，这是正常现象。
- 输入后直接按回车。

登录成功后会出现类似：

~~~bash
hiteam@hiteam:~$
~~~

---

## 4. 首次登录后先做环境检查

按顺序执行：

~~~bash
# 1) 激活环境
source /opt/venvs/llm8307/bin/activate

# 2) 进入项目目录
cd /home/hiteam/8307-project

# 3) 设置Python导入路径
export PYTHONPATH=$(pwd):$PYTHONPATH

# 4) 检查GPU
nvidia-smi

# 5) 检查PyTorch与CUDA
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 6) 检查数据集可读
python -c "from data.loader import load_sentiment, load_mentalchat, load_medquad; print(len(load_sentiment()), len(load_mentalchat()), len(load_medquad()))"
~~~

如果以上命令都正常，说明可以开始跑实验。

---

## 5. 最常用实验命令（直接复制）

### 5.1 Baseline

~~~bash
python experiments/run_baseline.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda
~~~

### 5.2 LoRA 微调

~~~bash
python finetune/lora_train.py --model qwen2.5-7b --output_dir /home/hiteam/checkpoints/qwen2.5-7b
~~~

### 5.3 微调后评估

~~~bash
python experiments/run_finetuned.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda
~~~

### 5.4 RAG

~~~bash
# 若索引不存在，先建索引
python rag/indexer.py

# Base + RAG
python experiments/run_rag.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda

# Fine-tuned + RAG
python experiments/run_rag.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda
~~~

---

## 6. 强烈建议：用 tmux 跑长任务

### 为什么要用 tmux

- 本地断网或关闭终端时，任务不会中断。
- 可以随时离开，再回来继续看进度。

### 基本操作

~~~bash
# 新建会话
tmux new -s exp

# 查看会话
tmux ls

# 回到会话
tmux attach -t exp
~~~

退出但不终止任务：

1. 按 Ctrl+b
2. 再按 d

---

## 7. 多 GPU 并行模板

原则：

1. 不同任务绑定不同 GPU
2. 不同任务使用不同 output_dir
3. 不同任务使用不同 tmux 会话

示例：

~~~bash
# 任务A用GPU1
CUDA_VISIBLE_DEVICES=1 python experiments/run_baseline.py --model llama-3.1-8b --task all --output_dir /home/hiteam/results_llama

# 任务B用GPU2
CUDA_VISIBLE_DEVICES=2 python experiments/run_baseline.py --model gemma-2-9b --task all --output_dir /home/hiteam/results_gemma
~~~

监控 GPU：

~~~bash
watch -n 2 nvidia-smi
~~~

---

## 8. 远程网络环境自检（不调用具体模型）

目标：确认服务器出口 IP 是否为境外，以及关键域名是否可达。

### 8.1 查询公网出口 IP 与地理位置

~~~bash
python - <<'PY'
import json, urllib.request

ip = urllib.request.urlopen("https://api64.ipify.org", timeout=10).read().decode().strip()
print("Public IP:", ip)

url = f"https://ipapi.co/{ip}/json/"
info = json.loads(urllib.request.urlopen(url, timeout=10).read().decode())
print("Country:", info.get("country_name"), f"({info.get('country_code')})")
print("Region:", info.get("region"))
print("City:", info.get("city"))
print("Org:", info.get("org"))
PY
~~~

### 8.2 测试关键域名 HTTPS 连通（无需 API Key）

~~~bash
curl -I --max-time 15 https://huggingface.co
curl -I --max-time 15 https://openrouter.ai/api/v1/models
curl -I --max-time 15 https://api.openai.com/v1/models
curl -I --max-time 15 https://api.anthropic.com/v1/messages
~~~

判定：

1. 200/301/302：可达
2. 401/403：也可达（只是没授权）
3. 超时/域名解析失败/TLS失败：网络路径有问题

---

## 9. 每次开工的最短清单（30秒版）

~~~bash
ssh hiteam@172.17.0.18
source /opt/venvs/llm8307/bin/activate
cd /home/hiteam/8307-project
export PYTHONPATH=$(pwd):$PYTHONPATH
nvidia-smi
~~~

确认无误后再运行实验命令。

---

## 10. 结果检查与收工动作

### 结果检查

~~~bash
find /home/hiteam -type f -path "*/results*/*/*/*_metrics.json" | sort
~~~

### 收工前建议

1. 确认所有任务 metrics 文件落盘。
2. 记录当前 tmux 会话名称。
3. 备份 logs、checkpoints、results。