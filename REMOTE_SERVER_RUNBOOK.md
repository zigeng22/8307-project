# 远程服务器实验操作手册（EasyConnect + SSH + tmux）

> 适用场景：已获得算力公司提供的 Linux 服务器，需在远程 GPU 上稳定运行本项目。
> 更新日期：2026-04-28

---

## 1. 一次性认知

1. EasyConnect 负责连 VPN 网络，不负责自动运行训练。
2. 真正使用远程算力的入口是 SSH 会话。
3. 长任务必须放在 tmux 中跑，防止本地断网后任务中断。

---

## 2. 从连接到开跑的标准流程

### Step A：连接 VPN
1. 打开 EasyConnect 并登录。
2. 看到“已成功登录”即可。

### Step B：SSH 登录服务器
1. 在本机 PowerShell 或 cmd 执行：
ssh hiteam@172.17.0.18
2. 输入 SSH 密码（密码输入时终端不回显，正常）。

### Step C：激活环境与进入项目
1. source /opt/venvs/llm8307/bin/activate
2. cd /home/hiteam/8307-project
3. export PYTHONPATH=$(pwd):$PYTHONPATH

### Step D：确认 GPU 与数据
1. nvidia-smi
2. python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
3. python -c "from data.loader import load_sentiment, load_mentalchat, load_medquad; print(len(load_sentiment()), len(load_mentalchat()), len(load_medquad()))"

### Step E：进入 tmux 跑任务
1. tmux new -s exp
2. 在会话里执行实验命令。
3. 退出不杀任务：Ctrl+b 然后按 d。
4. 回到会话：tmux attach -t exp。

---

## 3. 常用实验命令模板

### 3.1 Baseline
python experiments/run_baseline.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda

### 3.2 LoRA 微调
python finetune/lora_train.py --model qwen2.5-7b --output_dir /home/hiteam/checkpoints/qwen2.5-7b

### 3.3 Fine-tuned 评估
python experiments/run_finetuned.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda

### 3.4 RAG 索引与评估
python rag/indexer.py
python experiments/run_rag.py --model qwen2.5-7b --task all --output_dir /home/hiteam/results_gangda
python experiments/run_rag.py --model qwen2.5-7b --task all --lora_path /home/hiteam/checkpoints/qwen2.5-7b --output_dir /home/hiteam/results_gangda

---

## 4. 多人并行与多 GPU 规范

1. 每个任务固定一张卡：
CUDA_VISIBLE_DEVICES=1 python ...
2. 每个任务独立 output_dir：
/home/hiteam/results_llama, /home/hiteam/results_gemma, /home/hiteam/results_gangda
3. 每个任务独立 tmux 会话：
llama_base, gemma_base, qwen_ft
4. 开跑前检查占用：
watch -n 2 nvidia-smi

---

## 5. 我们遇到过的典型问题与修复

### 5.1 SSH 登录问题
问题：Connection closed / Permission denied。
处理：确认 VPN 已连接；确认使用 SSH 密码而不是 VPN 密码；必要时用 ssh -v 排查。

### 5.2 python 找不到
问题：Command 'python' not found。
处理：未激活虚拟环境。执行 source /opt/venvs/llm8307/bin/activate，或直接用 /opt/venvs/llm8307/bin/python。

### 5.3 数据路径报错
问题：找不到 /home/hiteam/Datasets/...。
处理：按代码要求，Datasets 必须位于项目上一级；可用软链接对齐路径。

### 5.4 Git pull 被本地改动阻塞
问题：local changes would be overwritten by merge。
处理：先 git stash push -- 指定文件，再 git pull；成功后再决定是否恢复 stash。

### 5.5 无外网导致模型下载失败
问题：huggingface 无法连接。
处理：改用本地模型目录，设置：
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

### 5.6 Gemma 模板不支持 system role
问题：TemplateError: System role not supported。
处理：在模型封装中对 gemma 自动移除 system 消息。

### 5.7 微调模板中 NoneType 报错
问题：NoneType not subscriptable / apply_chat_template 崩溃。
处理：训练前清洗空值行，并把 instruction/input/output 做安全字符串转换。

### 5.8 多卡设备不一致
问题：Expected all tensors to be on same device。
处理：固定单卡运行并设置 CUDA_VISIBLE_DEVICES，避免 auto device_map 跨卡。

### 5.9 OOM 显存不足
问题：CUDA out of memory on 4090 24GB。
处理：降低 per_device_train_batch_size，提升 gradient_accumulation_steps，降低 max_seq_length。

### 5.10 Task2 评估阶段 BERTScore 报错
问题：离线环境找不到 roberta-large。
处理：设置本地模型路径：
export BERTSCORE_MODEL_TYPE=/mnt/sdc/roberta-large
并使用已修复的 evaluation/metrics.py。

### 5.11 终端出现 ^[[B
问题：滚轮或方向键转义字符被原样打印。
处理：在训练窗口避免滚轮；另开窗口看日志；执行 stty sane 或 reset 恢复终端。

---

## 6. 日常推荐操作节奏

1. 登录后第一件事：激活环境 + 进入项目 + 设置 PYTHONPATH。
2. 长任务一律 tmux 跑。
3. 每个任务开跑时显式绑定 CUDA_VISIBLE_DEVICES。
4. 输出目录按人/按模型隔离。
5. 任务结束立即检查结果目录与 metrics 文件是否落盘。
6. 每日收工前备份 logs、checkpoints、results。

---

## 7. 常用检查命令速查

1. 看会话：tmux ls
2. 回会话：tmux attach -t 会话名
3. 退会话不杀任务：Ctrl+b, d
4. 看 GPU：nvidia-smi
5. 持续监控 GPU：watch -n 2 nvidia-smi
6. 看日志尾部：tail -f 日志文件
7. 查任务进程：ps -ef | grep python
8. 查结果文件：ls -lah 结果目录

---

## 8. 高频执行命令（从本轮实战沉淀）

### 8.1 登录后固定四步

source /opt/venvs/llm8307/bin/activate
cd /home/hiteam/8307-project
export PYTHONPATH=$(pwd):$PYTHONPATH
nvidia-smi

### 8.2 安全拉取更新（避免本地热修复冲突）

git stash push -m "temp-before-pull" -- finetune/lora_train.py models/hf_model.py evaluation/metrics.py
git pull

### 8.3 离线评估固定环境变量

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BERTSCORE_MODEL_TYPE=/mnt/sdc/roberta-large
export BERTSCORE_NUM_LAYERS=17

### 8.4 三模型并行模板（示例）

CUDA_VISIBLE_DEVICES=0 python ... --output_dir /home/hiteam/results_gangda
CUDA_VISIBLE_DEVICES=1 python ... --output_dir /home/hiteam/results_llama
CUDA_VISIBLE_DEVICES=2 python ... --output_dir /home/hiteam/results_gemma

### 8.5 统一盘点完成情况

find /home/hiteam -type f -path "*/results*/*/*/*_metrics.json" | sort

---

## 9. 本轮关键问答与经验结论（持续更新）

1. EasyConnect 仅负责 VPN 连接；真正使用算力要通过 SSH 登录服务器。
2. 密码输入不回显是正常行为，不是键盘失灵。
3. 未开 tmux 跑长任务风险高，断线可能导致任务中断。
4. Baseline 的 task 文件是分任务写盘；task2 评估阶段崩溃时通常只会留下 task1 的 json。
5. 4090 24GB 可跑 7B/9B，但微调需降低 batch 与序列长度以规避 OOM。
6. Gemma 不支持 system role 时，需做模板兼容处理。
7. 离线 BERTScore 需要本地 roberta-large 路径，并显式设置 BERTSCORE_NUM_LAYERS=17。
8. 出现 Connection reset 多数是本地 SSH 断联，不等于远端 tmux 任务停止。
9. 结果目录被 gitignore 忽略是正常现象，不能依赖 git 同步 results。
10. 多人并行必须遵守三原则：不同 GPU、不同 tmux 会话、不同 output_dir。
