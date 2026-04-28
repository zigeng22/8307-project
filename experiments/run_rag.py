"""
Phase 3: RAG-augmented evaluation for all models × all tasks.

Usage:
    python experiments/run_rag.py --model gpt-4o --task task1
    python experiments/run_rag.py --model llama-3.1-8b --task all
    python experiments/run_rag.py --model llama-3.1-8b --task all --lora_path ./finetune/checkpoints/llama
"""
import argparse
import json
import sys
import math
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODELS, RESULTS_DIR, SENTIMENT_LABELS,
    OPENAI_API_KEY, ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY, USE_OPENROUTER,
)
from data.loader import load_sentiment, load_mentalchat, load_medquad
from data.splitter import split_task1, split_task2, split_task3
from prompts.templates import (
    build_task1_messages, build_task2_messages, build_task3_messages,
)
from evaluation.metrics import eval_task
from rag.retriever import retrieve


def _safe_text(value) -> str:
    """Convert potentially-null values to clean strings."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _load_partial_predictions(partial_path: Path, max_len: int) -> list:
    """Load partial predictions from disk for resume support."""
    if not partial_path.exists():
        return []
    try:
        with open(partial_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        if not isinstance(preds, list):
            return []
        return preds[:max_len]
    except Exception:
        return []


def _save_partial_predictions(partial_path: Path, predictions: list):
    """Persist partial predictions for resumable execution."""
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)


def get_model(model_name: str, lora_path: str = None):
    cfg = MODELS[model_name]
    if cfg["type"] == "api":
        if USE_OPENROUTER:
            from models.openrouter_model import OpenRouterModel
            return OpenRouterModel(cfg["model_id"], api_key=OPENROUTER_API_KEY)
        elif cfg["provider"] == "openai":
            from models.api_model import OpenAIModel
            return OpenAIModel(cfg["model_id"], api_key=OPENAI_API_KEY)
        elif cfg["provider"] == "anthropic":
            from models.api_model import AnthropicModel
            return AnthropicModel(cfg["model_id"], api_key=ANTHROPIC_API_KEY)
    elif cfg["type"] == "hf":
        from models.hf_model import HFModel
        return HFModel(cfg["model_id"], lora_path=lora_path)
    raise ValueError(f"Unknown model config: {model_name}")


def run_task1_rag(model, test_df, partial_path: Path = None, save_every: int = 20):
    predictions = []
    start_idx = 0
    if partial_path is not None:
        predictions = _load_partial_predictions(partial_path, len(test_df))
        start_idx = len(predictions)
        if start_idx > 0:
            print(f"Resuming Task1+RAG from sample {start_idx}/{len(test_df)}")

    pending_df = test_df.iloc[start_idx:]
    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Task1+RAG")
    ):
        text = _safe_text(row["text"])
        context = retrieve(text)
        msgs = build_task1_messages(text, rag_context=context)
        pred = model.generate(msgs, max_tokens=50)
        predictions.append(pred)

        if partial_path is not None and ((i + 1) % save_every == 0 or i + 1 == len(pending_df)):
            _save_partial_predictions(partial_path, predictions)
    results = eval_task("task1", predictions, test_df["label"].tolist(),
                        labels=SENTIMENT_LABELS)
    results["predictions"] = predictions
    return results


def run_task2_rag(model, test_df, partial_path: Path = None, save_every: int = 20):
    predictions = []
    start_idx = 0
    if partial_path is not None:
        predictions = _load_partial_predictions(partial_path, len(test_df))
        start_idx = len(predictions)
        if start_idx > 0:
            print(f"Resuming Task2+RAG from sample {start_idx}/{len(test_df)}")

    pending_df = test_df.iloc[start_idx:]
    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Task2+RAG")
    ):
        patient_input = _safe_text(row["input"])
        context = retrieve(patient_input)
        msgs = build_task2_messages(patient_input, rag_context=context)
        pred = model.generate(msgs, max_tokens=512)
        predictions.append(pred)

        if partial_path is not None and ((i + 1) % save_every == 0 or i + 1 == len(pending_df)):
            _save_partial_predictions(partial_path, predictions)
    references = test_df["output"].tolist()
    results = eval_task("task2", predictions, references)
    results["predictions"] = predictions
    return results


def run_task3_rag(model, test_df, partial_path: Path = None, save_every: int = 20):
    predictions = []
    start_idx = 0
    if partial_path is not None:
        predictions = _load_partial_predictions(partial_path, len(test_df))
        start_idx = len(predictions)
        if start_idx > 0:
            print(f"Resuming Task3+RAG from sample {start_idx}/{len(test_df)}")

    pending_df = test_df.iloc[start_idx:]
    for i, (_, row) in enumerate(
        tqdm(pending_df.iterrows(), total=len(pending_df), desc="Task3+RAG")
    ):
        question = _safe_text(row["question"])
        context = retrieve(question)
        msgs = build_task3_messages(question, rag_context=context)
        pred = model.generate(msgs, max_tokens=512)
        predictions.append(pred)

        if partial_path is not None and ((i + 1) % save_every == 0 or i + 1 == len(pending_df)):
            _save_partial_predictions(partial_path, predictions)
    references = test_df["answer"].tolist()
    results = eval_task("task3", predictions, references)
    results["predictions"] = predictions
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--lora_path", default=None,
                        help="Path to LoRA checkpoint (for finetuned+RAG)")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    # determine config label
    config_label = "finetuned_rag" if args.lora_path else "base_rag"
    out_dir = Path(args.output_dir) / config_label / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model, lora_path=args.lora_path)
    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running {task} on {args.model} ({config_label})")
        print(f"{'='*50}")

        partial_path = out_dir / f"{task}_predictions.partial.json"

        if task == "task1":
            df = load_sentiment()
            test_df = split_task1(df)
            results = run_task1_rag(model, test_df, partial_path=partial_path)
        elif task == "task2":
            df = load_mentalchat()
            _, test_df = split_task2(df)
            results = run_task2_rag(model, test_df, partial_path=partial_path)
        elif task == "task3":
            df = load_medquad(mental_health_only=False)
            test_df = split_task3(df)
            results = run_task3_rag(model, test_df, partial_path=partial_path)

        preds = results.pop("predictions")
        summary_path = out_dir / f"{task}_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {summary_path}")
        print(json.dumps(results, indent=2))

        preds_path = out_dir / f"{task}_predictions.json"
        with open(preds_path, "w") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)

        if partial_path.exists():
            partial_path.unlink()


if __name__ == "__main__":
    main()
