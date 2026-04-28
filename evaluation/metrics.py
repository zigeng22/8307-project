"""
Evaluation metrics for all three tasks.
"""
import os
import re
from collections import Counter
from pathlib import Path
from typing import List

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)


# ---- Task 1: Classification ----

def normalize_label(pred: str, valid_labels: List[str]) -> str:
    """Try to match a model's free-text prediction to a valid label."""
    pred_clean = pred.strip().strip('"').strip("'").strip(".").lower()
    for label in valid_labels:
        if label.lower() == pred_clean:
            return label
        if label.lower() in pred_clean:
            return label
    return pred.strip()


def eval_classification(y_true: List[str], y_pred: List[str],
                        labels: List[str]) -> dict:
    """Compute accuracy, macro-F1, and per-class F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels,
                        average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            label: round(report.get(label, {}).get("f1-score", 0), 4)
            for label in labels
        },
    }


# ---- Task 2 & 3: Generation metrics ----

def eval_rouge(predictions: List[str], references: List[str]) -> dict:
    """Compute ROUGE-L score."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [scorer.score(ref, pred)["rougeL"].fmeasure
              for ref, pred in zip(references, predictions)]
    avg = sum(scores) / len(scores) if scores else 0
    return {"rouge_l": round(avg, 4)}


def eval_bertscore(predictions: List[str], references: List[str],
                   lang: str = "en") -> dict:
    """Compute BERTScore (F1)."""
    from bert_score import score as bert_score_fn

    def resolve_local_bertscore_model() -> str:
        """Resolve local roberta-large path for offline evaluation."""
        candidates = []
        env_model = os.environ.get("BERTSCORE_MODEL_TYPE")
        env_path = os.environ.get("BERTSCORE_MODEL_PATH")
        if env_model:
            candidates.append(env_model)
        if env_path:
            candidates.append(env_path)

        # Common predownload/cache locations on local servers
        candidates.extend([
            "/mnt/sdc/roberta-large",
            "/workspace/.cache/huggingface/hub/models--roberta-large",
            str(Path.home() / ".cache" / "huggingface" / "hub" / "models--roberta-large"),
        ])

        for raw in candidates:
            p = Path(raw)
            if not p.exists():
                continue

            # Plain model directory layout
            if (p / "config.json").exists():
                return str(p)

            # Hugging Face cache snapshots layout
            snapshot_cfgs = list(p.glob("snapshots/*/config.json"))
            if snapshot_cfgs:
                return str(snapshot_cfgs[0].parent)

        return ""

    model_type = resolve_local_bertscore_model()
    try:
        if model_type:
            # Local path is not in bert-score's built-in model2layers map.
            # roberta-large commonly uses 17 for BERTScore.
            num_layers = int(os.environ.get("BERTSCORE_NUM_LAYERS", "17"))
            P, R, F1 = bert_score_fn(
                predictions,
                references,
                model_type=model_type,
                num_layers=num_layers,
                verbose=False,
            )
        else:
            P, R, F1 = bert_score_fn(
                predictions, references, lang=lang, verbose=False
            )
    except Exception as e:
        raise RuntimeError(
            "BERTScore failed. If running offline, set BERTSCORE_MODEL_TYPE "
            "to a local roberta-large directory containing config.json."
        ) from e

    return {"bertscore_f1": round(F1.mean().item(), 4)}


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def eval_f1_token(prediction: str, reference: str) -> float:
    """Token-level F1 (for QA evaluation)."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def eval_qa(predictions: List[str], references: List[str]) -> dict:
    """Evaluate QA with token-level F1 and exact match."""
    em = sum(1 for p, r in zip(predictions, references)
             if p.strip().lower() == r.strip().lower()) / len(predictions)
    f1s = [eval_f1_token(p, r) for p, r in zip(predictions, references)]
    avg_f1 = sum(f1s) / len(f1s) if f1s else 0
    return {
        "exact_match": round(em, 4),
        "token_f1": round(avg_f1, 4),
    }


def eval_task(task: str, predictions: List[str], references: List[str],
              labels: List[str] = None) -> dict:
    """Unified evaluation entry point."""
    if task == "task1":
        if labels:
            y_pred = [normalize_label(p, labels) for p in predictions]
        else:
            y_pred = predictions
        return eval_classification(references, y_pred, labels or [])
    elif task == "task2":
        rouge = eval_rouge(predictions, references)
        bert = eval_bertscore(predictions, references)
        return {**rouge, **bert}
    elif task == "task3":
        qa = eval_qa(predictions, references)
        rouge = eval_rouge(predictions, references)
        return {**qa, **rouge}
    else:
        raise ValueError(f"Unknown task: {task}")
