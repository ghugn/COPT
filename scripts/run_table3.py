import os
import sys
import time
import subprocess
import argparse
import shutil
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 stdout/stderr on all platforms (especially Windows PowerShell/cmd)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Disable wandb to prevent interactive prompts
os.environ["WANDB_MODE"] = "disabled"
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parent.parent)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

RESULTS_JSON = "table3_results.json"
RESULTS_TXT = "table3_results.txt"

CKPT_DIR = PROJECT_ROOT / "logs" / "train" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [12345, 42, 2024]

ROWS = [
    ("BASELINE", "—", "BASELINE"),
    ("FREEZE", "RESET + FT", "FREEZE_RESET"),
    ("FREEZE", "INVERT + FT", "FREEZE_INVERT"),
    ("FT", "INVERT + FT", "FT_INVERT"),
]

PAPER_TABLE3 = {
    "BASELINE": {"gnn": "BASELINE", "head": "—", "mis": "18.12 ± 0.11", "mvc": "211.69 ± 0.16"},
    "FREEZE_RESET": {"gnn": "FREEZE", "head": "RESET + FT", "mis": "17.68 ± 0.04", "mvc": "212.46 ± 0.25"},
    "FREEZE_INVERT": {"gnn": "FREEZE", "head": "INVERT + FT", "mis": "17.69 ± 0.05", "mvc": "212.39 ± 0.21"},
    "FT_INVERT": {"gnn": "FT", "head": "INVERT + FT", "mis": "18.00 ± 0.05", "mvc": "211.56 ± 0.24"},
}

def run_command(cmd):
    print(f"\n{'='*70}\n[RUNNING]: {cmd}\n{'='*70}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=True)
    return res.returncode

def get_latest_metric(task, min_mtime=None, log_type="train"):
    base_dir = PROJECT_ROOT / "logs" / log_type / "runs"
    if not base_dir.exists():
        return None
    csv_files = list(base_dir.glob("**/metrics.csv"))
    if not csv_files:
        return None
    if min_mtime is not None:
        csv_files = [p for p in csv_files if p.stat().st_mtime >= min_mtime - 1.0]
        if not csv_files:
            return None
    # Sort newest first
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_csv = csv_files[0]
    try:
        df = pd.read_csv(latest_csv)
        candidate_cols = [
            f"test/{task}/size",
            f"test/size",
            f"val/{task}/size_best",
            f"val/size_best",
            f"val/{task}/size",
            f"val/size",
        ]
        for col in candidate_cols:
            if col in df.columns:
                vals = df[col].dropna()
                if not vals.empty:
                    return float(vals.iloc[-1])
    except Exception as e:
        print(f"Warning reading {latest_csv}: {e}")
    return None

def find_latest_checkpoint(min_mtime=None):
    runs_dir = PROJECT_ROOT / "logs" / "train" / "runs"
    if not runs_dir.exists():
        return None
    ckpt_files = list(runs_dir.glob("**/checkpoints/*.ckpt"))
    if not ckpt_files:
        return None
    if min_mtime is not None:
        ckpt_files = [p for p in ckpt_files if p.stat().st_mtime >= min_mtime - 1.0]
    valid = [p for p in ckpt_files if p.stat().st_size > 1000]
    if not valid:
        return None
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]

def format_stat(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    return f"{mean:.2f} ± {std:.2f}"

def save_and_print_table(runs, active_seeds):
    header = "\n" + "="*70 + "\n"
    num_done_seeds = sum(
        1 for s in active_seeds 
        if str(s) in runs and any(runs[str(s)][k]["mis"] is not None for k in ["BASELINE", "FREEZE_RESET", "FREEZE_INVERT", "FT_INVERT"])
    )
    seed_str = ", ".join(str(s) for s in active_seeds)
    header += f"          TABLE 3 (REPRODUCED - {len(active_seeds)} SEEDS: [{seed_str}])          \n"
    header += "="*70 + "\n"
    header += f"{'GNN':<12} {'OUT-HEAD':<20} {'MIS ↑':<18} {'MVC ↓':<16}\n"
    header += "-" * 70 + "\n"

    body = ""
    summary = {}
    for gnn, head, key in ROWS:
        mis_vals = [
            runs[str(s)][key]["mis"] 
            for s in active_seeds 
            if str(s) in runs and runs[str(s)].get(key, {}).get("mis") is not None
        ]
        mvc_vals = [
            runs[str(s)][key]["mvc"] 
            for s in active_seeds 
            if str(s) in runs and runs[str(s)].get(key, {}).get("mvc") is not None
        ]
        mis_str = format_stat(mis_vals)
        mvc_str = format_stat(mvc_vals)
        body += f"{gnn:<12} {head:<20} {mis_str:<18} {mvc_str:<16}\n"
        summary[key] = {
            "mis_vals": mis_vals,
            "mvc_vals": mvc_vals,
            "mis_str": mis_str,
            "mvc_str": mvc_str,
        }

    mid_divider = "-" * 70 + "\n"
    mid_divider += "PAPER REFERENCE (Table 3, 3 runs average):\n"
    paper_body = ""
    for gnn, head, key in ROWS:
        ref = PAPER_TABLE3[key]
        paper_body += f"{gnn:<12} {head:<20} {ref['mis']:<18} {ref['mvc']:<16}\n"

    footer = "="*70 + "\n"

    # Seed-by-seed breakdown
    breakdown = "\n--- PER-SEED BREAKDOWN ---\n"
    for s in active_seeds:
        s_str = str(s)
        breakdown += f"[Seed {s}]:\n"
        if s_str in runs:
            for gnn, head, key in ROWS:
                r = runs[s_str].get(key, {})
                mis_v = f"{r.get('mis'):.2f}" if r.get('mis') is not None else "-"
                mvc_v = f"{r.get('mvc'):.2f}" if r.get('mvc') is not None else "-"
                breakdown += f"  {gnn:<10} {head:<16} MIS={mis_v:<8} MVC={mvc_v:<8}\n"
        else:
            breakdown += "  (Not yet run)\n"

    table_str = header + body + mid_divider + paper_body + footer + breakdown
    try:
        print(table_str)
    except UnicodeEncodeError:
        print(table_str.encode("ascii", errors="replace").decode("ascii"))

    try:
        with open(RESULTS_TXT, "w", encoding="utf-8") as f:
            f.write(table_str)
        save_data = {
            "active_seeds": active_seeds,
            "runs": runs,
            "summary": {k: {"mis": v["mis_str"], "mvc": v["mvc_str"]} for k, v in summary.items()}
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
    except Exception as e:
        print(f"Warning saving results file: {e}")

def init_seed_dict():
    return {
        "BASELINE": {"mis": None, "mvc": None},
        "FREEZE_RESET": {"mis": None, "mvc": None},
        "FREEZE_INVERT": {"mis": None, "mvc": None},
        "FT_INVERT": {"mis": None, "mvc": None},
    }

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 3: MIS <-> MVC Pairwise Transferability on RB-small across seeds")
    parser.add_argument("--epochs_baseline", type=int, default=100, help="Epochs for training baseline models (default: 100)")
    parser.add_argument("--epochs_ft", type=int, default=20, help="Epochs for fine-tuning / transfer (default: 20)")
    parser.add_argument("--val_every", type=int, default=5, help="Check validation every N epochs (default: 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="List of random seeds (default: 12345 42 2024)")
    parser.add_argument("--seed", type=int, default=None, help="Single seed override (e.g. --seed 12345)")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of seeds to run from defaults (e.g. --num_seeds 3)")
    parser.add_argument("--step", choices=["all", "baseline", "transfer"], default="all")
    parser.add_argument("--task", choices=["all", "mis", "mvc"], default="all")
    parser.add_argument("--force", action="store_true", help="Force re-run even if result already cached")
    parser.add_argument("--mis_ckpt", type=str, default=None, help="Optional manual MIS checkpoint")
    parser.add_argument("--mvc_ckpt", type=str, default=None, help="Optional manual MVC checkpoint")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers (default: 4)")
    args, extra_args = parser.parse_known_args()
    extra_cmd = " " + " ".join(extra_args) if extra_args else ""

    # Determine seeds to run
    if args.seed is not None:
        active_seeds = [args.seed]
    elif args.seeds is not None:
        active_seeds = args.seeds
    elif args.num_seeds is not None:
        base_pool = DEFAULT_SEEDS + [777, 999, 888, 111, 222]
        active_seeds = base_pool[:args.num_seeds]
    else:
        active_seeds = DEFAULT_SEEDS  # Default: [12345, 42, 2024] (3 runs)

    runs = {}

    # Load cached results if available and not forcing full re-run
    if Path(RESULTS_JSON).exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
                if "runs" in saved:
                    runs = saved["runs"]
                elif "BASELINE" in saved:
                    # Backward compatibility with single-run json
                    runs["12345"] = saved
            print(f"Loaded existing results from {RESULTS_JSON}")
        except Exception as e:
            print(f"Could not load {RESULTS_JSON}: {e}")

    # Ensure all active seeds are initialized in runs dict
    for s in active_seeds:
        s_str = str(s)
        if s_str not in runs:
            runs[s_str] = init_seed_dict()
        else:
            for k in ["BASELINE", "FREEZE_RESET", "FREEZE_INVERT", "FT_INVERT"]:
                if k not in runs[s_str]:
                    runs[s_str][k] = {"mis": None, "mvc": None}

    save_and_print_table(runs, active_seeds)

    print(f"\nTarget Seeds to evaluate: {active_seeds}")

    # Loop through each seed
    for seed_idx, seed in enumerate(active_seeds, 1):
        s_str = str(seed)
        seed_res = runs[s_str]
        print("\n" + "#"*70)
        print(f"### RUN {seed_idx}/{len(active_seeds)}: SEED = {seed} ###")
        print("#"*70)

        seed_mis_ckpt = Path(args.mis_ckpt) if args.mis_ckpt else CKPT_DIR / f"mis_rb_small_seed_{seed}.ckpt"
        seed_mvc_ckpt = Path(args.mvc_ckpt) if args.mvc_ckpt else CKPT_DIR / f"mvc_rb_small_seed_{seed}.ckpt"

        # Fallback to general ckpt if seed-specific does not exist yet
        fallback_mis_ckpt = CKPT_DIR / "mis_rb_small.ckpt"
        fallback_mvc_ckpt = CKPT_DIR / "mvc_rb_small.ckpt"

        # =========================================================================
        # STAGE 1: BASELINE (TRAIN FROM SCRATCH FOR THIS SEED)
        # =========================================================================
        if args.step in ["all", "baseline"]:
            # Baseline MIS
            if args.task in ["all", "mis"]:
                if args.force or seed_res["BASELINE"]["mis"] is None:
                    print(f"\n---> [SEED {seed} | BASELINE MIS]: Training GCON on RB-small ({args.epochs_baseline} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mis/rb_small/gcon "
                        f"seed={seed} "
                        f"trainer.max_epochs={args.epochs_baseline} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                    )
                    t_start = time.time()
                    ret = run_command(cmd)
                    if ret == 0:
                        val = get_latest_metric("mis", min_mtime=t_start, log_type="train")
                        seed_res["BASELINE"]["mis"] = val
                        print(f"---> [SEED {seed} | BASELINE MIS] Result: {val}")

                        latest_ckpt = find_latest_checkpoint(min_mtime=t_start)
                        if latest_ckpt and latest_ckpt.exists():
                            shutil.copyfile(latest_ckpt, seed_mis_ckpt)
                            if not fallback_mis_ckpt.exists():
                                shutil.copyfile(latest_ckpt, fallback_mis_ckpt)
                            print(f"---> Saved MIS checkpoint to {seed_mis_ckpt}")
                        save_and_print_table(runs, active_seeds)
                    else:
                        print(f"---> [ERROR]: Baseline MIS (seed={seed}) failed with exit code {ret}!")
                else:
                    print(f"---> [SEED {seed} | BASELINE MIS]: Cached ({seed_res['BASELINE']['mis']})")

            # Baseline MVC
            if args.task in ["all", "mvc"]:
                if args.force or seed_res["BASELINE"]["mvc"] is None:
                    print(f"\n---> [SEED {seed} | BASELINE MVC]: Training GCON on RB-small ({args.epochs_baseline} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon "
                        f"seed={seed} "
                        f"trainer.max_epochs={args.epochs_baseline} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                    )
                    t_start = time.time()
                    ret = run_command(cmd)
                    if ret == 0:
                        val = get_latest_metric("mvc", min_mtime=t_start, log_type="train")
                        seed_res["BASELINE"]["mvc"] = val
                        print(f"---> [SEED {seed} | BASELINE MVC] Result: {val}")

                        latest_ckpt = find_latest_checkpoint(min_mtime=t_start)
                        if latest_ckpt and latest_ckpt.exists():
                            shutil.copyfile(latest_ckpt, seed_mvc_ckpt)
                            if not fallback_mvc_ckpt.exists():
                                shutil.copyfile(latest_ckpt, fallback_mvc_ckpt)
                            print(f"---> Saved MVC checkpoint to {seed_mvc_ckpt}")
                        save_and_print_table(runs, active_seeds)
                    else:
                        print(f"---> [ERROR]: Baseline MVC (seed={seed}) failed with exit code {ret}!")
                else:
                    print(f"---> [SEED {seed} | BASELINE MVC]: Cached ({seed_res['BASELINE']['mvc']})")

        # =========================================================================
        # STAGE 2: TRANSFER LEARNING FOR THIS SEED
        # =========================================================================
        if args.step in ["all", "transfer"]:
            # Pick MVC checkpoint for MIS transfer
            mvc_ckpt_to_use = seed_mvc_ckpt if seed_mvc_ckpt.exists() else fallback_mvc_ckpt
            # Pick MIS checkpoint for MVC transfer
            mis_ckpt_to_use = seed_mis_ckpt if seed_mis_ckpt.exists() else fallback_mis_ckpt

            # ---------------------------------------------------------------------
            # DIRECTION 1: Pretrained MVC -> Evaluated on MIS (Column MIS ↑)
            # ---------------------------------------------------------------------
            if args.task in ["all", "mis"]:
                if not mvc_ckpt_to_use.exists():
                    print(f"ERROR: MVC pretrained checkpoint not found at {mvc_ckpt_to_use}!")
                else:
                    mvc_ckpt_str = str(mvc_ckpt_to_use.resolve()).replace("\\", "/")

                    # Setting 1: FREEZE (RESET + FT)
                    if args.force or seed_res["FREEZE_RESET"]["mis"] is None:
                        print(f"\n---> [SEED {seed} | MIS ↑ | FREEZE (RESET + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                            f"model.pretrain_path='{mvc_ckpt_str}' "
                            f"model.freeze=backbone "
                            f"model.reset_head=True "
                            f"model.invert_head=False "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mis", min_mtime=t_start, log_type="train")
                            seed_res["FREEZE_RESET"]["mis"] = val
                            save_and_print_table(runs, active_seeds)

                    # Setting 2: FREEZE (INVERT + FT)
                    if args.force or seed_res["FREEZE_INVERT"]["mis"] is None:
                        print(f"\n---> [SEED {seed} | MIS ↑ | FREEZE (INVERT + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                            f"model.pretrain_path='{mvc_ckpt_str}' "
                            f"model.freeze=backbone "
                            f"model.reset_head=False "
                            f"model.invert_head=True "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mis", min_mtime=t_start, log_type="train")
                            seed_res["FREEZE_INVERT"]["mis"] = val
                            save_and_print_table(runs, active_seeds)

                    # Setting 3: FT (INVERT + FT)
                    if args.force or seed_res["FT_INVERT"]["mis"] is None:
                        print(f"\n---> [SEED {seed} | MIS ↑ | FT (INVERT + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                            f"model.pretrain_path='{mvc_ckpt_str}' "
                            f"model.freeze=False "
                            f"model.reset_head=False "
                            f"model.invert_head=True "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mis", min_mtime=t_start, log_type="train")
                            seed_res["FT_INVERT"]["mis"] = val
                            save_and_print_table(runs, active_seeds)

            # ---------------------------------------------------------------------
            # DIRECTION 2: Pretrained MIS -> Evaluated on MVC (Column MVC ↓)
            # ---------------------------------------------------------------------
            if args.task in ["all", "mvc"]:
                if not mis_ckpt_to_use.exists():
                    print(f"ERROR: MIS pretrained checkpoint not found at {mis_ckpt_to_use}!")
                else:
                    mis_ckpt_str = str(mis_ckpt_to_use.resolve()).replace("\\", "/")

                    # Setting 1: FREEZE (RESET + FT)
                    if args.force or seed_res["FREEZE_RESET"]["mvc"] is None:
                        print(f"\n---> [SEED {seed} | MVC ↓ | FREEZE (RESET + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                            f"model.pretrain_path='{mis_ckpt_str}' "
                            f"model.freeze=backbone "
                            f"model.reset_head=True "
                            f"model.invert_head=False "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mvc", min_mtime=t_start, log_type="train")
                            seed_res["FREEZE_RESET"]["mvc"] = val
                            save_and_print_table(runs, active_seeds)

                    # Setting 2: FREEZE (INVERT + FT)
                    if args.force or seed_res["FREEZE_INVERT"]["mvc"] is None:
                        print(f"\n---> [SEED {seed} | MVC ↓ | FREEZE (INVERT + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                            f"model.pretrain_path='{mis_ckpt_str}' "
                            f"model.freeze=backbone "
                            f"model.reset_head=False "
                            f"model.invert_head=True "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mvc", min_mtime=t_start, log_type="train")
                            seed_res["FREEZE_INVERT"]["mvc"] = val
                            save_and_print_table(runs, active_seeds)

                    # Setting 3: FT (INVERT + FT)
                    if args.force or seed_res["FT_INVERT"]["mvc"] is None:
                        print(f"\n---> [SEED {seed} | MVC ↓ | FT (INVERT + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                        cmd = (
                            f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                            f"model.pretrain_path='{mis_ckpt_str}' "
                            f"model.freeze=False "
                            f"model.reset_head=False "
                            f"model.invert_head=True "
                            f"seed={seed} "
                            f"trainer.max_epochs={args.epochs_ft} "
                            f"trainer.check_val_every_n_epoch={args.val_every} "
                            f"trainer.accelerator=auto "
                            f"logger=csv "
                            f"hydra/job_logging=default hydra/hydra_logging=default "
                            f"paths.root_dir=. data.num_workers={args.num_workers}{extra_cmd}"
                        )
                        t_start = time.time()
                        ret = run_command(cmd)
                        if ret == 0:
                            val = get_latest_metric("mvc", min_mtime=t_start, log_type="train")
                            seed_res["FT_INVERT"]["mvc"] = val
                            save_and_print_table(runs, active_seeds)

    print("\nFINAL SUMMARY:")
    save_and_print_table(runs, active_seeds)

if __name__ == "__main__":
    main()
