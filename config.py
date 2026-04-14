"""
Project configuration — paths, model names, hyperparams.
"""
import os
from pathlib import Path

# project root = wtc/
PROJECT_ROOT = Path(__file__).parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = REPO_ROOT / "Datasets"

# dataset paths
SENTIMENT_CSV = DATA_DIR / "Combined Data.csv"
MEDQUAD_CSV = DATA_DIR / "medquad.csv"
MENTALCHAT_DIR = DATA_DIR / "MentalChat16K-main"
MENTALCHAT_INTERVIEW = MENTALCHAT_DIR / "Interview_Data_6K.csv"
MENTALCHAT_SYNTHETIC = MENTALCHAT_DIR / "Synthetic_Data_10K.csv"

# results
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# test set sizes
TASK1_TEST_SIZE = 1000
TASK2_TEST_SIZE = 500
TASK3_TEST_SIZE = 500

# random seed
SEED = 42

# sentiment labels
SENTIMENT_LABELS = [
    "Normal", "Depression", "Suicidal", "Anxiety",
    "Stress", "Bipolar", "Personality disorder"
]

# mental-health keywords for filtering MedQuAD
MENTAL_HEALTH_KEYWORDS = [
    "depress", "anxiety", "mental", "bipolar", "ptsd", "stress",
    "schizo", "suicide", "panic", "phobia", "obsessive",
    "eating disorder", "adhd", "autism", "psycho", "mood", "trauma",
    "counseling", "therapy", "psychiatric",
]

# ---- model registry ----
MODELS = {
    "gpt-4o": {
        "type": "api",
        "provider": "openai",
        "model_id": "gpt-4o",
        "can_finetune": False,
    },
    "claude-3.5-sonnet": {
        "type": "api",
        "provider": "anthropic",
        "model_id": "anthropic/claude-sonnet-4",
        "can_finetune": False,
    },
    "deepseek-v3": {
        "type": "api",
        "provider": "deepseek",
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "can_finetune": False,
    },
    "llama-3.1-8b": {
        "type": "hf",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "can_finetune": True,
    },
    "qwen2.5-7b": {
        "type": "hf",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "can_finetune": True,
    },
    "gemma-2-9b": {
        "type": "hf",
        "model_id": "google/gemma-2-9b-it",
        "can_finetune": True,
    },
}

# ---- LoRA hyperparams ----
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "lora_dropout": 0.05,
}

TRAINING_ARGS = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 50,
    "save_steps": 200,
    "warmup_ratio": 0.03,
}

# ---- RAG config ----
RAG_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 3,
    "index_dir": str(PROJECT_ROOT / "rag" / "faiss_index"),
}

# ---- API keys (set via env vars, never hardcode) ----
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# if OpenRouter key is available, use it as the default for API models
USE_OPENROUTER = bool(OPENROUTER_API_KEY)
