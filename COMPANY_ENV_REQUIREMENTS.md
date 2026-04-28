# 预装环境清单

## 1. 系统与运行时
- OS: Ubuntu 22.04 LTS
- NVIDIA Driver: 550.144+
- CUDA: 12.4+
- Python: 3.10.14
- PyTorch: 2.5.1 + cu124

## 2. Python 环境与依赖
```bash
python3.10 -m venv /opt/venvs/llm8307
source /opt/venvs/llm8307/bin/activate
python -m pip install -U pip setuptools wheel

# PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Core
pip install transformers==4.46.3 peft==0.12.0 trl==0.11.4 datasets==2.21.0 accelerate==0.34.2

# RAG
pip install langchain==0.2.16 langchain-community==0.2.16 faiss-cpu==1.8.0.post1 sentence-transformers==3.0.1

# Evaluation
pip install rouge-score==0.1.2 bert-score==0.3.13 scikit-learn==1.5.1

# API + Utils
pip install openai==1.52.2 anthropic==0.34.2 pandas==2.2.2 matplotlib==3.9.2 seaborn==0.13.2 tqdm==4.66.5
```

## 3. 预下载模型
- Qwen/Qwen2.5-7B-Instruct
- google/gemma-2-9b-it
- sentence-transformers/all-MiniLM-L6-v2
- roberta-large
- meta-llama/Llama-3.1-8B-Instruct