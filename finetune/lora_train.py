"""
Phase 2: LoRA fine-tuning for open-source models on MentalChat16K.

Usage:
    python finetune/lora_train.py --model llama-3.1-8b
    python finetune/lora_train.py --model qwen2.5-7b
    python finetune/lora_train.py --model gemma-2-9b
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODELS, LORA_CONFIG, TRAINING_ARGS, SEED


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--model", required=True,
                        choices=[k for k, v in MODELS.items() if v["can_finetune"]])
    parser.add_argument("--output_dir", default=None,
                        help="Override output dir for checkpoint")
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer

    from data.loader import load_mentalchat
    from data.splitter import split_task2

    cfg = MODELS[args.model]
    model_id = cfg["model_id"]
    output_dir = args.output_dir or str(
        Path(__file__).parent / "checkpoints" / args.model
    )

    print(f"Loading tokenizer and model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
    )

    # LoRA config
    lora_config = LoraConfig(
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        target_modules=LORA_CONFIG["target_modules"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # load and format training data
    print("Loading training data...")
    df = load_mentalchat()
    train_df, _ = split_task2(df)

    # Some rows may contain null/blank fields; drop them to avoid chat-template errors.
    required_cols = ["instruction", "input", "output"]
    before_rows = len(train_df)
    train_df = train_df.dropna(subset=required_cols).copy()
    for col in required_cols:
        train_df[col] = train_df[col].astype(str).str.strip()
    train_df = train_df[
        (train_df["instruction"] != "") &
        (train_df["output"] != "")
    ]
    after_rows = len(train_df)
    print(f"Training rows: {before_rows} -> {after_rows} after cleaning")

    def format_example(row):
        """Format into chat-style text for SFT."""
        system_text = str(row.get("instruction", "") or "")
        user_text = str(row.get("input", "") or "")
        assistant_text = str(row.get("output", "") or "")
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            text = (
                f"### System: {system_text}\n"
                f"### User: {user_text}\n"
                f"### Assistant: {assistant_text}"
            )
        return {"text": text}

    train_dataset = Dataset.from_pandas(train_df)
    train_dataset = train_dataset.map(format_example)

    # training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=TRAINING_ARGS["num_train_epochs"],
        per_device_train_batch_size=TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps=TRAINING_ARGS["gradient_accumulation_steps"],
        learning_rate=TRAINING_ARGS["learning_rate"],
        fp16=TRAINING_ARGS["fp16"],
        logging_steps=TRAINING_ARGS["logging_steps"],
        save_steps=TRAINING_ARGS["save_steps"],
        warmup_ratio=TRAINING_ARGS["warmup_ratio"],
        save_total_limit=2,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=2048,
    )

    print("Starting training...")
    trainer.train()

    # save final checkpoint
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
