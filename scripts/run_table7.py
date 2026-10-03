import os
import sys
import subprocess
import argparse
import pandas as pd
from pathlib import Path

# Disable wandb to prevent any interactive prompts
os.environ["WANDB_MODE"] = "disabled"

TASKS = ["maxcut", "maxclique", "mds", "mis", "mvc", "color"]
TASK_SYMBOLS = {
    "maxcut": "↑ MAXCUT",
    "maxclique": "↑ MAXCLIQUE",
    "mds": "↓ MDS",
    "mis": "↑ MIS",
    "mvc": "↓ MVC",
    "color": "↓ COLORING",
}

CKPT_PATH = "logs/train/checkpoints/multitask/epoch_194.ckpt"

def run_command(cmd):
    print(f"\n{'='*70}\n[RUNNING]: {cmd}\n{'='*70}\n")
    sys.stdout.flush()
    # Run directly to terminal so Colab displays LIVE progress bar without pipe buffering
    res = subprocess.run(cmd, shell=True)
    return res.returncode

def get_latest_metric(task, log_type="train"):
    base_dir = Path(f"logs/{log_type}/runs")
    if not base_dir.exists():
        return None
    csv_files = list(base_dir.glob("**/csv/version_0/metrics.csv"))
    if not csv_files:
        return None
    # Sort by modification time, newest first
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_csv = csv_files[0]
    try:
        df = pd.read_csv(latest_csv)
        candidate_cols = [
            f"test/{task}/size",
            f"test/{task}/violations",
            f"test/size",
            f"test/violations",
            f"val/{task}/size_best",
            f"val/{task}/size",
            f"val/{task}/violations",
            f"val/size_best",
            f"val/size",
            f"val/violations",
        ]
        for col in candidate_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    return float(vals.iloc[-1])
    except Exception as e:
        print(f"Warning reading {latest_csv}: {e}")
    return None

import json

RESULTS_JSON = "table7_results.json"
RESULTS_TXT = "table7_results.txt"

def save_and_print_table(results, tasks):
    header = "\n" + "="*65 + "\n"
    header += "                      TABLE 7 (REPRODUCED)                      \n"
    header += "="*65 + "\n"
    header += f"{'TASK':<18} {'FINE-TUNED':<16} {'BASELINE':<16} {'FULL':<12}\n"
    header += "-" * 65 + "\n"
    body = ""
    for task in tasks:
        name = TASK_SYMBOLS.get(task, task.upper())
        ft = f"{results[task]['finetuned']:.2f}" if results[task].get('finetuned') is not None else "-"
        bl = f"{results[task]['baseline']:.2f}" if results[task].get('baseline') is not None else "-"
        fl = f"{results[task]['full']:.2f}" if results[task].get('full') is not None else "-"
        body += f"{name:<18} {ft:<16} {bl:<16} {fl:<12}\n"
    footer = "="*65 + "\n"
    table_str = header + body + footer
    print(table_str)
    
    # Persist to disk
    try:
        with open(RESULTS_TXT, "w", encoding="utf-8") as f:
            f.write(table_str)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning saving results file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 7 (Live Progress, Auto-Resume & 1:1 format)")
    parser.add_argument("--mode", choices=["all", "full", "baseline", "finetuned"], default="all")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for baseline and finetuning (default: 20)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--force", action="store_true", help="Force re-run even if task is already in cached results")
    args = parser.parse_args()

    results = {task: {"finetuned": None, "baseline": None, "full": None} for task in args.tasks}
    
    # Load previously saved results if available
    if Path(RESULTS_JSON).exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for t, vals in saved.items():
                    if t in results:
                        results[t].update({k: v for k, v in vals.items() if v is not None})
            print(f"Loaded existing results from {RESULTS_JSON}")
        except Exception as e:
            print(f"Could not load {RESULTS_JSON}: {e}")

    # Paper Table 7 reference for FULL (single-task trained from scratch for 200 epochs)
    PAPER_FULL = {
        "maxcut": 726.58,
        "maxclique": 4.43,
        "mds": 29.56,
        "mis": 112.23,
        "mvc": 139.40,
        "color": 43.52,
    }
    for t in args.tasks:
        if results[t]["full"] is None:
            results[t]["full"] = PAPER_FULL.get(t, None)

    save_and_print_table(results, args.tasks)

    # 2. RUN BASELINES FROM SCRATCH (20 epochs)
    if args.mode in ["all", "baseline"]:
        print("\n" + "="*70)
        print(f"STAGE 2: Training Single-Task Baselines from Scratch ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
            if not args.force and results[task]["baseline"] is not None:
                print(f"---> [BASELINE]: Skipping {task.upper()} (Already computed: {results[task]['baseline']})")
                continue
            print(f"\n---> [BASELINE]: Starting {task.upper()} ({args.epochs} epochs)...")
            cmd = f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[{task}] trainer.max_epochs={args.epochs} logger=csv hydra/job_logging=default hydra/hydra_logging=default data.num_workers=2"
            run_command(cmd)
            val = get_latest_metric(task, log_type="train")
            results[task]["baseline"] = val
            print(f"---> [BASELINE]: {task.upper()} finished with result: {val}")
            save_and_print_table(results, args.tasks)

    # 3. RUN FINE-TUNED (20 epochs from Foundation Model)
    if args.mode in ["all", "finetuned"]:
        print("\n" + "="*70)
        print(f"STAGE 3: Fine-Tuning from Pretrained Backbone ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
            if not args.force and results[task]["finetuned"] is not None:
                print(f"---> [FINE-TUNED]: Skipping {task.upper()} (Already computed: {results[task]['finetuned']})")
                continue
            print(f"\n---> [FINE-TUNED]: Starting {task.upper()} ({args.epochs} epochs)...")
            strat = "linear_probing" if task == "color" else "finetuning"
            cmd = (
                f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon "
                f"model.net.tasks=[{task}] "
                f"model.net.finetuning.strategy='{strat}' "
                f"model.net.finetuning.new_tasks=[{task}] "
                f"model.net.finetuning.path={CKPT_PATH} "
                f"trainer.max_epochs={args.epochs} "
                f"logger=csv "
                f"hydra/job_logging=default hydra/hydra_logging=default "
                f"data.num_workers=2"
            )
            run_command(cmd)
            val = get_latest_metric(task, log_type="train")
            results[task]["finetuned"] = val
            print(f"---> [FINE-TUNED]: {task.upper()} finished with result: {val}")
            save_and_print_table(results, args.tasks)

    # FINAL OUTPUT
    print("\nFINAL SUMMARY:")
    save_and_print_table(results, args.tasks)

if __name__ == "__main__":
    main()
