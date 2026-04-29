import argparse
import csv
import json
import subprocess
from pathlib import Path

CONFIG_NAMES = {"baseline", "finetuned", "base_rag", "finetuned_rag"}


def run_cmd(cmd, interactive=False):
    if interactive:
        result = subprocess.run(cmd)
        return result.returncode, "", ""

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr


def sync_remote_results(project_root: Path, user: str, host: str):
    dst = project_root / "results_server"
    dst.mkdir(parents=True, exist_ok=True)

    remote_dirs = [
        "/home/hiteam/results_gangda",
        "/home/hiteam/results_llama",
        "/home/hiteam/results_gemma",
    ]

    for remote in remote_dirs:
        cmd = ["scp", "-r", f"{user}@{host}:{remote}", str(dst)]
        code, out, err = run_cmd(cmd, interactive=True)
        if code != 0:
            print(f"[WARN] sync failed: {remote}")
        else:
            print(f"[OK] synced: {remote}")


def parse_metrics_files(roots):
    rows = []

    for root in roots:
        if not root.exists():
            continue

        for f in root.rglob("*_metrics.json"):
            parts = f.parts
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

            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue

            row = {
                "config": config,
                "model": model,
                "task": task,
                "accuracy": data.get("accuracy", ""),
                "macro_f1": data.get("macro_f1", ""),
                "rouge_l": data.get("rouge_l", ""),
                "bertscore_f1": data.get("bertscore_f1", ""),
                "token_f1": data.get("token_f1", ""),
                "exact_match": data.get("exact_match", ""),
                "source_file": str(f),
            }
            rows.append(row)

    return rows


def build_experiment_table(rows):
    table = {}

    for r in rows:
        key = (r["model"], r["config"])
        if key not in table:
            table[key] = {
                "model": r["model"],
                "config": r["config"],
                "task1_accuracy": "",
                "task1_macro_f1": "",
                "task2_rouge_l": "",
                "task2_bertscore_f1": "",
                "task3_token_f1": "",
                "task3_exact_match": "",
                "task3_rouge_l": "",
            }

        if r["task"] == "task1":
            table[key]["task1_accuracy"] = r["accuracy"]
            table[key]["task1_macro_f1"] = r["macro_f1"]
        elif r["task"] == "task2":
            table[key]["task2_rouge_l"] = r["rouge_l"]
            table[key]["task2_bertscore_f1"] = r["bertscore_f1"]
        elif r["task"] == "task3":
            table[key]["task3_token_f1"] = r["token_f1"]
            table[key]["task3_exact_match"] = r["exact_match"]
            table[key]["task3_rouge_l"] = r["rouge_l"]

    return list(table.values())


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Sync remote results and build CSV summaries")
    parser.add_argument("--host", default="172.17.0.18")
    parser.add_argument("--user", default="hiteam")
    parser.add_argument("--skip-sync", action="store_true", help="Only summarize local files")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    if not args.skip_sync:
        sync_remote_results(project_root, args.user, args.host)

    roots = [
        project_root / "results",
        project_root / "results_server",
    ]

    rows = parse_metrics_files(roots)
    if not rows:
        print("[WARN] no metrics json found")
        return

    long_csv = project_root / "results" / "metrics_long.csv"
    exp_csv = project_root / "results" / "experiment_table.csv"

    write_csv(
        long_csv,
        [
            "config", "model", "task", "accuracy", "macro_f1", "rouge_l",
            "bertscore_f1", "token_f1", "exact_match", "source_file",
        ],
        rows,
    )

    exp_rows = build_experiment_table(rows)
    write_csv(
        exp_csv,
        [
            "model", "config",
            "task1_accuracy", "task1_macro_f1",
            "task2_rouge_l", "task2_bertscore_f1",
            "task3_token_f1", "task3_exact_match", "task3_rouge_l",
        ],
        exp_rows,
    )

    print(f"[OK] wrote: {long_csv}")
    print(f"[OK] wrote: {exp_csv}")


if __name__ == "__main__":
    main()
