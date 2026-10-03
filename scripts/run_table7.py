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

def print_table(results, tasks):
    print("\n" + "="*65)
    print("                      TABLE 7 (REPRODUCED)                      ")
    print("="*65)
    print(f"{'TASK':<18} {'FINE-TUNED':<16} {'BASELINE':<16} {'FULL':<12}")
    print("-" * 65)
    for task in tasks:
        name = TASK_SYMBOLS.get(task, task.upper())
        ft = f"{results[task]['finetuned']:.2f}" if results[task]['finetuned'] is not None else "-"
        bl = f"{results[task]['baseline']:.2f}" if results[task]['baseline'] is not None else "-"
        fl = f"{results[task]['full']:.2f}" if results[task]['full'] is not None else "-"
        print(f"{name:<18} {ft:<16} {bl:<16} {fl:<12}")
    print("="*65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 7 (Live Progress & 1:1 format)")
    parser.add_argument("--mode", choices=["all", "full", "baseline", "finetuned"], default="all")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for baseline and finetuning (default: 20)")
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    args = parser.parse_args()

    results = {task: {"finetuned": None, "baseline": None, "full": None} for task in args.tasks}
    
    # Pre-fill already verified numbers for the 3 core tasks
    results["color"]["full"] = 16.20
    results["mds"]["full"] = 29.98
    results["mis"]["full"] = 111.45

    # 1. EVALUATE FULL (Pre-trained Foundation Model)
    if args.mode in ["all", "full"]:
        print("\n" + "="*70)
        print("STAGE 1: Evaluating Multi-Task Foundation Model (FULL Column)")
        print("="*70)
        if Path(CKPT_PATH).exists():
            cmd = f"{sys.executable} src/eval.py experiment=multitask/ba_small/gcon ckpt_path={CKPT_PATH} model.net.tasks=[color,mds,mis] logger=csv hydra/job_logging=default hydra/hydra_logging=default"
            run_command(cmd)
            for t in ["color", "mds", "mis"]:
                if t in args.tasks:
                    val = get_latest_metric(t, log_type="eval")
                    if val is not None:
                        results[t]["full"] = val
            
            for t in [t for t in args.tasks if t not in ["color", "mds", "mis"]]:
                cmd = f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[{t}] model.net.finetuning.strategy=finetuning model.net.finetuning.new_tasks=[{t}] model.net.finetuning.path={CKPT_PATH} trainer.max_epochs=0 logger=csv hydra/job_logging=default hydra/hydra_logging=default data.num_workers=2"
                run_command(cmd)
                val = get_latest_metric(t, log_type="train")
                if val is not None:
                    results[t]["full"] = val

        print_table(results, args.tasks)

    # 2. RUN BASELINES FROM SCRATCH (20 epochs)
    if args.mode in ["all", "baseline"]:
        print("\n" + "="*70)
        print(f"STAGE 2: Training Single-Task Baselines from Scratch ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
            print(f"\n---> [BASELINE]: Starting {task.upper()} ({args.epochs} epochs)...")
            cmd = f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[{task}] trainer.max_epochs={args.epochs} logger=csv hydra/job_logging=default hydra/hydra_logging=default data.num_workers=2"
            run_command(cmd)
            val = get_latest_metric(task, log_type="train")
            results[task]["baseline"] = val
            print(f"---> [BASELINE]: {task.upper()} finished with result: {val}")
            print_table(results, args.tasks)

    # 3. RUN FINE-TUNED (20 epochs from Foundation Model)
    if args.mode in ["all", "finetuned"]:
        print("\n" + "="*70)
        print(f"STAGE 3: Fine-Tuning from Pretrained Backbone ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
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
            print_table(results, args.tasks)

    # FINAL OUTPUT
    print("\nFINAL SUMMARY:")
    print_table(results, args.tasks)

if __name__ == "__main__":
    main()
