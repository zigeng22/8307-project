import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

CONFIG_NAMES = {"baseline", "finetuned", "base_rag", "finetuned_rag"}


def parse_ab_metrics(ab_root: Path):
    rows = []
    if not ab_root.exists():
        return rows

    for f in ab_root.rglob("*_metrics.json"):
        parts = list(f.parts)

        cfg_idx = -1
        for i, p in enumerate(parts):
            if p in CONFIG_NAMES:
                cfg_idx = i
                break
        if cfg_idx == -1 or cfg_idx + 1 >= len(parts):
            continue

        config = parts[cfg_idx]
        model = parts[cfg_idx + 1]
        task = f.stem.replace("_metrics", "")

        # Group path is everything between ab_root and config dir.
        rel_parts = list(f.relative_to(ab_root).parts)
        rel_cfg_idx = rel_parts.index(config)
        group = "/".join(rel_parts[:rel_cfg_idx]) if rel_cfg_idx > 0 else "root"

        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        rows.append(
            {
                "group": group,
                "model": model,
                "config": config,
                "task": task,
                "accuracy": data.get("accuracy", ""),
                "macro_f1": data.get("macro_f1", ""),
                "rouge_l": data.get("rouge_l", ""),
                "bertscore_f1": data.get("bertscore_f1", ""),
                "token_f1": data.get("token_f1", ""),
                "exact_match": data.get("exact_match", ""),
                "source_file": str(f),
            }
        )

    return rows


def load_control_table(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    return {(r["model"], r["config"]): r for r in rows}


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def build_compare_rows(ab_rows, control_idx):
    compare = []
    for r in ab_rows:
        ctrl = control_idx.get((r["model"], r["config"]))
        if not ctrl:
            continue

        metric_name = None
        ab_value = None
        ctrl_value = None

        if r["task"] == "task1":
            metric_name = "task1_accuracy"
            ab_value = safe_float(r["accuracy"])
            ctrl_value = safe_float(ctrl.get("task1_accuracy", ""))
        elif r["task"] == "task2":
            metric_name = "task2_rouge_l"
            ab_value = safe_float(r["rouge_l"])
            ctrl_value = safe_float(ctrl.get("task2_rouge_l", ""))
        elif r["task"] == "task3":
            metric_name = "task3_token_f1"
            ab_value = safe_float(r["token_f1"])
            ctrl_value = safe_float(ctrl.get("task3_token_f1", ""))

        if metric_name is None or ab_value is None or ctrl_value is None:
            continue

        compare.append(
            {
                "group": r["group"],
                "model": r["model"],
                "config": r["config"],
                "task": r["task"],
                "metric": metric_name,
                "control": round(ctrl_value, 6),
                "treatment": round(ab_value, 6),
                "delta": round(ab_value - ctrl_value, 6),
            }
        )

    return compare


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_group_summary(compare_rows):
    agg = defaultdict(list)
    for r in compare_rows:
        agg[(r["group"], r["metric"])].append(r["delta"])

    if not agg:
        print("[WARN] no comparable rows found")
        return

    print("\n=== Mean Delta by Group ===")
    for (group, metric), vals in sorted(agg.items()):
        mean = sum(vals) / len(vals)
        print(f"{group:40s} {metric:16s} mean_delta={mean:+.4f} n={len(vals)}")


def main():
    parser = argparse.ArgumentParser(description="Compare AB alignment results against control table")
    parser.add_argument(
        "--ab-root",
        default="results_server/results_abtest",
        help="Root directory of synced AB results",
    )
    parser.add_argument(
        "--control-table",
        default="results/experiment_table.csv",
        help="Control experiment table CSV",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    ab_root = (repo_root / args.ab_root).resolve()
    control_table = (repo_root / args.control_table).resolve()

    ab_rows = parse_ab_metrics(ab_root)
    if not ab_rows:
        print(f"[WARN] no AB metrics found under: {ab_root}")
        return

    control_idx = load_control_table(control_table)
    compare_rows = build_compare_rows(ab_rows, control_idx)

    long_out = repo_root / "results" / "abtest_alignment_long.csv"
    cmp_out = repo_root / "results" / "abtest_alignment_compare.csv"

    write_csv(
        long_out,
        [
            "group",
            "model",
            "config",
            "task",
            "accuracy",
            "macro_f1",
            "rouge_l",
            "bertscore_f1",
            "token_f1",
            "exact_match",
            "source_file",
        ],
        ab_rows,
    )

    write_csv(
        cmp_out,
        ["group", "model", "config", "task", "metric", "control", "treatment", "delta"],
        compare_rows,
    )

    print(f"[OK] wrote: {long_out}")
    print(f"[OK] wrote: {cmp_out}")
    print_group_summary(compare_rows)


if __name__ == "__main__":
    main()
