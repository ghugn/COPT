import os
import sys
import subprocess
import argparse
import pandas as pd
from pathlib import Path

# Paper Table 7 Reference Numbers
PAPER_TABLE7 = {
    "maxcut": {"direction": "↑", "finetuned": "722.77 ± 1.00", "baseline": "716.90 ± 1.55", "full": "726.58 ± 0.51"},
    "maxclique": {"direction": "↑", "finetuned": "4.32 ± 0.01", "baseline": "4.31 ± 0.00", "full": "4.43 ± 0.02"},
    "mds": {"direction": "↓", "finetuned": "29.65 ± 0.09", "baseline": "33.93 ± 1.15", "full": "29.56 ± 0.22"},
    "mis": {"direction": "↑", "finetuned": "111.94 ± 0.06", "baseline": "111.33 ± 0.26", "full": "112.23 ± 0.06"},
    "mvc": {"direction": "↓", "finetuned": "139.70 ± 0.31", "baseline": "141.43 ± 0.23", "full": "139.40 ± 0.20"},
    "color": {"direction": "↓", "finetuned": "17.29 ± 1.97", "baseline": "49.04 ± 21.84", "full": "3.52 ± 1.52"},
}

TASKS = ["maxcut", "maxclique", "mds", "mis", "mvc", "color"]
CKPT_PATH = "logs/train/checkpoints/multitask/epoch_194.ckpt"

def run_command(cmd):
    print(f"\n[RUNNING]: {cmd}\n")
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    for line in proc.stdout:
        print(line, end="")
        output.append(line)
    proc.wait()
    return "".join(output)

def extract_metric(output, task):
    # Search for test metric in lightning log output
    # Patterns: test/{task}/size or val/{task}/size_best
    import re
    # Look for DATALOADER:0 TEST RESULTS block
    match = re.search(rf"test/{task}/size\s*│\s*([0-9.]+)", output)
    if match:
        return float(match.group(1))
    match = re.search(rf"test/{task}/size.*?([0-9]+\.[0-9]+)", output)
    if match:
        return float(match.group(1))
    match = re.search(rf"val/{task}/size.*?([0-9]+\.[0-9]+)", output)
    if match:
        return float(match.group(1))
    return None

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 7 (Multi-Task Foundation Model) 1:1")
    parser.add_argument("--mode", choices=["all", "full", "baseline", "finetuned"], default="all",
                        help="Which part of Table 7 to run")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for baseline and finetuning (default: 20)")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="Subset of tasks to run")
    args = parser.parse_args()

    results = {task: {"full": None, "baseline": None, "finetuned": None} for task in args.tasks}
    
    # 1. EVALUATE FULL MULTI-TASK MODEL (epoch_194.ckpt)
    if args.mode in ["all", "full"]:
        print("\n" + "="*70)
        print("STAGE 1: Evaluating Pretrained Multi-Task Foundation Model (FULL Column)")
        print("="*70)
        if not Path(CKPT_PATH).exists():
            print(f"ERROR: Checkpoint {CKPT_PATH} not found!")
        else:
            for task in args.tasks:
                print(f"\n---> Evaluating FULL model on Task: {task.upper()}")
                cmd = f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[{task}] ckpt_path={CKPT_PATH} trainer.max_epochs=0 logger=csv"
                out = run_command(cmd)
                val = extract_metric(out, task)
                results[task]["full"] = val
                print(f"[RESULT] {task.upper()} FULL Score: {val}")

    # 2. RUN BASELINES FROM SCRATCH (20 epochs)
    if args.mode in ["all", "baseline"]:
        print("\n" + "="*70)
        print(f"STAGE 2: Training Single-Task Baselines from Scratch ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
            print(f"\n---> Training BASELINE for Task: {task.upper()}")
            cmd = f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[{task}] trainer.max_epochs={args.epochs} logger=csv"
            out = run_command(cmd)
            val = extract_metric(out, task)
            results[task]["baseline"] = val
            print(f"[RESULT] {task.upper()} BASELINE Score: {val}")

    # 3. RUN FINE-TUNED FROM MULTI-TASK BACKBONE (20 epochs)
    if args.mode in ["all", "finetuned"]:
        print("\n" + "="*70)
        print(f"STAGE 3: Fine-Tuning from Pretrained Backbone ({args.epochs} epochs)")
        print("="*70)
        for task in args.tasks:
            print(f"\n---> Fine-tuning for Task: {task.upper()}")
            strat = "linear_probing" if task == "color" else "finetuning"
            cmd = (
                f"{sys.executable} src/train.py experiment=multitask/ba_small/gcon "
                f"model.net.tasks=[{task}] "
                f"model.net.finetuning.strategy='{strat}' "
                f"model.net.finetuning.new_tasks=[{task}] "
                f"model.net.finetuning.path={CKPT_PATH} "
                f"trainer.max_epochs={args.epochs} "
                f"logger=csv"
            )
            out = run_command(cmd)
            val = extract_metric(out, task)
            results[task]["finetuned"] = val
            print(f"[RESULT] {task.upper()} FINE-TUNED Score: {val}")

    # PRINT SUMMARY COMPARISON TABLE
    print("\n" + "="*85)
    print("           REPRODUCED TABLE 7 vs. PAPER REFERENCE (1:1 COMPARISON)           ")
    print("="*85)
    print(f"{'TASK':<12} | {'FINE-TUNED (Ours)':<18} {'(Paper)':<16} | {'BASELINE (Ours)':<16} {'(Paper)':<16} | {'FULL (Ours)':<12} {'(Paper)':<12}")
    print("-" * 105)
    for task in args.tasks:
        d = PAPER_TABLE7[task]["direction"]
        p_ft = PAPER_TABLE7[task]["finetuned"]
        p_bl = PAPER_TABLE7[task]["baseline"]
        p_fl = PAPER_TABLE7[task]["full"]
        
        o_ft = f"{results[task]['finetuned']:.2f}" if results[task]['finetuned'] is not None else "N/A"
        o_bl = f"{results[task]['baseline']:.2f}" if results[task]['baseline'] is not None else "N/A"
        o_fl = f"{results[task]['full']:.2f}" if results[task]['full'] is not None else "N/A"
        
        print(f"{d} {task.upper():<9} | {o_ft:<18} {p_ft:<16} | {o_bl:<16} {p_bl:<16} | {o_fl:<12} {p_fl:<12}")
    print("="*105)

if __name__ == "__main__":
    main()
