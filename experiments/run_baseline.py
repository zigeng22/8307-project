"""
Phase 1: Baseline zero-shot evaluation for all models × all tasks.

Usage:
    python experiments/run_baseline.py --model gpt-4o --task task1
    python experiments/run_baseline.py --model qwen2.5-7b --task all
"""
import argparse
import json
import sys
import os
from pathlib import Path
from tqdm import tqdm

# allow imports from project root
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


def get_model(model_name: str):
    """Instantiate the right model wrapper."""
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
        return HFModel(cfg["model_id"])
    raise ValueError(f"Unknown model config: {model_name}")


def run_task1(model, test_df):
    """Evaluate classification task."""
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Task1"):
        msgs = build_task1_messages(row["text"])
        pred = model.generate(msgs, max_tokens=50)
        predictions.append(pred)

    results = eval_task("task1", predictions, test_df["label"].tolist(),
                        labels=SENTIMENT_LABELS)
    results["predictions"] = predictions
    return results


def run_task2(model, test_df):
    """Evaluate dialogue generation task."""
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Task2"):
        msgs = build_task2_messages(row["input"])
        pred = model.generate(msgs, max_tokens=512)
        predictions.append(pred)

    references = test_df["output"].tolist()
    results = eval_task("task2", predictions, references)
    results["predictions"] = predictions
    return results


def run_task3(model, test_df):
    """Evaluate medical QA task."""
    predictions = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Task3"):
        msgs = build_task3_messages(row["question"])
        pred = model.generate(msgs, max_tokens=512)
        predictions.append(pred)

    references = test_df["answer"].tolist()
    results = eval_task("task3", predictions, references)
    results["predictions"] = predictions
    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--output_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / "baseline" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(args.model)
    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running {task} on {args.model} (baseline)")
        print(f"{'='*50}")

        if task == "task1":
            df = load_sentiment()
            test_df = split_task1(df)
            results = run_task1(model, test_df)
        elif task == "task2":
            df = load_mentalchat()
            _, test_df = split_task2(df)
            results = run_task2(model, test_df)
        elif task == "task3":
            df = load_medquad(mental_health_only=False)
            test_df = split_task3(df)
            results = run_task3(model, test_df)

        # save results (without raw predictions for the summary)
        preds = results.pop("predictions")
        summary_path = out_dir / f"{task}_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {summary_path}")
        print(json.dumps(results, indent=2))

        # save full predictions
        preds_path = out_dir / f"{task}_predictions.json"
        with open(preds_path, "w") as f:
            json.dump(preds, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
