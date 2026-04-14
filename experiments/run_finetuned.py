"""
Phase 2 (eval part): Evaluate fine-tuned models on all tasks.

Usage:
    python experiments/run_finetuned.py --model llama-3.1-8b --task all \
        --lora_path ./finetune/checkpoints/llama
"""
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, RESULTS_DIR, SENTIMENT_LABELS
from data.loader import load_sentiment, load_mentalchat, load_medquad
from data.splitter import split_task1, split_task2, split_task3
from prompts.templates import (
    build_task1_messages, build_task2_messages, build_task3_messages,
)
from evaluation.metrics import eval_task


def main():
    parser = argparse.ArgumentParser(description="Run finetuned model experiments")
    parser.add_argument("--model", required=True,
                        choices=[k for k, v in MODELS.items() if v["can_finetune"]])
    parser.add_argument("--task", required=True,
                        choices=["task1", "task2", "task3", "all"])
    parser.add_argument("--lora_path", required=True,
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--output_dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    from models.hf_model import HFModel

    cfg = MODELS[args.model]
    out_dir = Path(args.output_dir) / "finetuned" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model = HFModel(cfg["model_id"], lora_path=args.lora_path)
    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Running {task} on {args.model} (finetuned)")
        print(f"{'='*50}")

        if task == "task1":
            df = load_sentiment()
            test_df = split_task1(df)
            predictions = []
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                msgs = build_task1_messages(row["text"])
                predictions.append(model.generate(msgs, max_tokens=50))
            results = eval_task("task1", predictions, test_df["label"].tolist(),
                                labels=SENTIMENT_LABELS)

        elif task == "task2":
            df = load_mentalchat()
            _, test_df = split_task2(df)
            predictions = []
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                msgs = build_task2_messages(row["input"])
                predictions.append(model.generate(msgs, max_tokens=512))
            results = eval_task("task2", predictions, test_df["output"].tolist())

        elif task == "task3":
            df = load_medquad(mental_health_only=False)
            test_df = split_task3(df)
            predictions = []
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                msgs = build_task3_messages(row["question"])
                predictions.append(model.generate(msgs, max_tokens=512))
            results = eval_task("task3", predictions, test_df["answer"].tolist())

        summary_path = out_dir / f"{task}_metrics.json"
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
