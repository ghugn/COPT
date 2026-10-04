import os
import sys
import subprocess
import argparse
import shutil
import json
import pandas as pd
from pathlib import Path

# Disable wandb to prevent interactive prompts
os.environ["WANDB_MODE"] = "disabled"
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = str(Path(__file__).resolve().parent.parent)

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"])

RESULTS_JSON = "table3_results.json"
RESULTS_TXT = "table3_results.txt"

CKPT_DIR = PROJECT_ROOT / "logs" / "train" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MIS_CKPT = CKPT_DIR / "mis_rb_small.ckpt"
DEFAULT_MVC_CKPT = CKPT_DIR / "mvc_rb_small.ckpt"

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

def get_latest_metric(task, log_type="train"):
    base_dir = PROJECT_ROOT / "logs" / log_type / "runs"
    if not base_dir.exists():
        return None
    csv_files = list(base_dir.glob("**/metrics.csv"))
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

def find_latest_checkpoint():
    runs_dir = PROJECT_ROOT / "logs" / "train" / "runs"
    if not runs_dir.exists():
        return None
    ckpt_files = list(runs_dir.glob("**/checkpoints/*.ckpt"))
    if not ckpt_files:
        return None
    # Filter for non-empty checkpoints, sort newest first
    valid = [p for p in ckpt_files if p.stat().st_size > 1000]
    if not valid:
        return None
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]

def save_and_print_table(results):
    header = "\n" + "="*65 + "\n"
    header += "                      TABLE 3 (REPRODUCED)                      \n"
    header += "="*65 + "\n"
    header += f"{'GNN':<12} {'OUT-HEAD':<20} {'MIS ↑':<16} {'MVC ↓':<12}\n"
    header += "-" * 65 + "\n"
    body = ""
    for gnn, head, key in ROWS:
        mis_val = f"{results[key]['mis']:.2f}" if results[key].get('mis') is not None else "-"
        mvc_val = f"{results[key]['mvc']:.2f}" if results[key].get('mvc') is not None else "-"
        body += f"{gnn:<12} {head:<20} {mis_val:<16} {mvc_val:<12}\n"
    footer = "="*65 + "\n"
    table_str = header + body + footer
    print(table_str)
    
    try:
        with open(RESULTS_TXT, "w", encoding="utf-8") as f:
            f.write(table_str)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning saving results file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 3: MIS <-> MVC Pairwise Transferability on RB-small")
    parser.add_argument("--epochs_baseline", type=int, default=50, help="Epochs for training baseline models (default: 50)")
    parser.add_argument("--epochs_ft", type=int, default=20, help="Epochs for fine-tuning / transfer (default: 20)")
    parser.add_argument("--val_every", type=int, default=5, help="Check validation every N epochs (default: 5)")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed (default: 12345)")
    parser.add_argument("--step", choices=["all", "baseline", "transfer"], default="all")
    parser.add_argument("--task", choices=["all", "mis", "mvc"], default="all")
    parser.add_argument("--force", action="store_true", help="Force re-run even if result already cached")
    parser.add_argument("--mis_ckpt", type=str, default=None, help="Path to pretrained MIS checkpoint")
    parser.add_argument("--mvc_ckpt", type=str, default=None, help="Path to pretrained MVC checkpoint")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader & generator workers (default: 4)")
    args, extra_args = parser.parse_known_args()
    extra_cmd = " " + " ".join(extra_args) if extra_args else ""

    results = {
        "BASELINE": {"mis": None, "mvc": None},
        "FREEZE_RESET": {"mis": None, "mvc": None},
        "FREEZE_INVERT": {"mis": None, "mvc": None},
        "FT_INVERT": {"mis": None, "mvc": None},
    }

    if Path(RESULTS_JSON).exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for k, v in saved.items():
                    if k in results:
                        results[k].update({subk: subv for subk, subv in v.items() if subv is not None})
            print(f"Loaded existing results from {RESULTS_JSON}")
        except Exception as e:
            print(f"Could not load {RESULTS_JSON}: {e}")

    save_and_print_table(results)

    mis_ckpt = Path(args.mis_ckpt) if args.mis_ckpt else DEFAULT_MIS_CKPT
    mvc_ckpt = Path(args.mvc_ckpt) if args.mvc_ckpt else DEFAULT_MVC_CKPT

    # =========================================================================
    # STEP 1: BASELINE (TRAIN FROM SCRATCH)
    # =========================================================================
    if args.step in ["all", "baseline"]:
        print("\n" + "="*70)
        print("STAGE 1: Single-Task Baselines from Scratch on RB-small")
        print("="*70)

        # Baseline MIS
        if args.task in ["all", "mis"]:
            if args.force or results["BASELINE"]["mis"] is None:
                print(f"\n---> [BASELINE MIS]: Training GCON on RB-small ({args.epochs_baseline} epochs, seed={args.seed})...")
                cmd = (
                    f"{sys.executable} src/train.py experiment=mis/rb_small/gcon "
                    f"seed={args.seed} "
                    f"trainer.max_epochs={args.epochs_baseline} "
                    f"trainer.check_val_every_n_epoch={args.val_every} "
                    f"trainer.accelerator=auto "
                    f"logger=csv "
                    f"hydra/job_logging=default hydra/hydra_logging=default "
                    f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                )
                run_command(cmd)
                val = get_latest_metric("mis", log_type="train")
                results["BASELINE"]["mis"] = val
                print(f"---> [BASELINE MIS] Finished: {val}")

                # Save checkpoint as pretrained MIS model
                latest_ckpt = find_latest_checkpoint()
                if latest_ckpt and latest_ckpt.exists():
                    shutil.copyfile(latest_ckpt, mis_ckpt)
                    print(f"---> Saved MIS checkpoint to {mis_ckpt}")
                save_and_print_table(results)
            else:
                print(f"---> [BASELINE MIS]: Already computed ({results['BASELINE']['mis']})")

        # Baseline MVC
        if args.task in ["all", "mvc"]:
            if args.force or results["BASELINE"]["mvc"] is None:
                print(f"\n---> [BASELINE MVC]: Training GCON on RB-small ({args.epochs_baseline} epochs, seed={args.seed})...")
                cmd = (
                    f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon "
                    f"seed={args.seed} "
                    f"trainer.max_epochs={args.epochs_baseline} "
                    f"trainer.check_val_every_n_epoch={args.val_every} "
                    f"trainer.accelerator=auto "
                    f"logger=csv "
                    f"hydra/job_logging=default hydra/hydra_logging=default "
                    f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                )
                run_command(cmd)
                val = get_latest_metric("mvc", log_type="train")
                results["BASELINE"]["mvc"] = val
                print(f"---> [BASELINE MVC] Finished: {val}")

                # Save checkpoint as pretrained MVC model
                latest_ckpt = find_latest_checkpoint()
                if latest_ckpt and latest_ckpt.exists():
                    shutil.copyfile(latest_ckpt, mvc_ckpt)
                    print(f"---> Saved MVC checkpoint to {mvc_ckpt}")
                save_and_print_table(results)
            else:
                print(f"---> [BASELINE MVC]: Already computed ({results['BASELINE']['mvc']})")

    # =========================================================================
    # STEP 2: TRANSFER LEARNING EXPERIMENTS
    # =========================================================================
    if args.step in ["all", "transfer"]:
        print("\n" + "="*70)
        print("STAGE 2: MIS <-> MVC Transferability on RB-small")
        print("="*70)

        # ---------------------------------------------------------------------
        # DIRECTION 1: Pretrained MVC -> Evaluated on MIS (Column MIS ↑)
        # ---------------------------------------------------------------------
        if args.task in ["all", "mis"]:
            if not mvc_ckpt.exists():
                print(f"ERROR: MVC pretrained checkpoint not found at {mvc_ckpt}! Run baseline first or provide --mvc_ckpt.")
            else:
                mvc_ckpt_str = str(mvc_ckpt.resolve()).replace("\\", "/")

                # Setting 1: FREEZE (RESET + FT)
                if args.force or results["FREEZE_RESET"]["mis"] is None:
                    print(f"\n---> [MIS ↑ | FREEZE (RESET + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                        f"model.pretrain_path='{mvc_ckpt_str}' "
                        f"model.freeze=backbone "
                        f"model.reset_head=True "
                        f"model.invert_head=False "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mis", log_type="train")
                    results["FREEZE_RESET"]["mis"] = val
                    save_and_print_table(results)

                # Setting 2: FREEZE (INVERT + FT)
                if args.force or results["FREEZE_INVERT"]["mis"] is None:
                    print(f"\n---> [MIS ↑ | FREEZE (INVERT + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                        f"model.pretrain_path='{mvc_ckpt_str}' "
                        f"model.freeze=backbone "
                        f"model.reset_head=False "
                        f"model.invert_head=True "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mis", log_type="train")
                    results["FREEZE_INVERT"]["mis"] = val
                    save_and_print_table(results)

                # Setting 3: FT (INVERT + FT)
                if args.force or results["FT_INVERT"]["mis"] is None:
                    print(f"\n---> [MIS ↑ | FT (INVERT + FT)]: Pretrained MVC -> MIS ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mis/rb_small/gcon_pretrained_mvc "
                        f"model.pretrain_path='{mvc_ckpt_str}' "
                        f"model.freeze=False "
                        f"model.reset_head=False "
                        f"model.invert_head=True "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mis", log_type="train")
                    results["FT_INVERT"]["mis"] = val
                    save_and_print_table(results)

        # ---------------------------------------------------------------------
        # DIRECTION 2: Pretrained MIS -> Evaluated on MVC (Column MVC ↓)
        # ---------------------------------------------------------------------
        if args.task in ["all", "mvc"]:
            if not mis_ckpt.exists():
                print(f"ERROR: MIS pretrained checkpoint not found at {mis_ckpt}! Run baseline first or provide --mis_ckpt.")
            else:
                mis_ckpt_str = str(mis_ckpt.resolve()).replace("\\", "/")

                # Setting 1: FREEZE (RESET + FT)
                if args.force or results["FREEZE_RESET"]["mvc"] is None:
                    print(f"\n---> [MVC ↓ | FREEZE (RESET + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                        f"model.pretrain_path='{mis_ckpt_str}' "
                        f"model.freeze=backbone "
                        f"model.reset_head=True "
                        f"model.invert_head=False "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mvc", log_type="train")
                    results["FREEZE_RESET"]["mvc"] = val
                    save_and_print_table(results)

                # Setting 2: FREEZE (INVERT + FT)
                if args.force or results["FREEZE_INVERT"]["mvc"] is None:
                    print(f"\n---> [MVC ↓ | FREEZE (INVERT + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                        f"model.pretrain_path='{mis_ckpt_str}' "
                        f"model.freeze=backbone "
                        f"model.reset_head=False "
                        f"model.invert_head=True "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mvc", log_type="train")
                    results["FREEZE_INVERT"]["mvc"] = val
                    save_and_print_table(results)

                # Setting 3: FT (INVERT + FT)
                if args.force or results["FT_INVERT"]["mvc"] is None:
                    print(f"\n---> [MVC ↓ | FT (INVERT + FT)]: Pretrained MIS -> MVC ({args.epochs_ft} epochs)...")
                    cmd = (
                        f"{sys.executable} src/train.py experiment=mvc/rb_small/gcon_pretrained_mis "
                        f"model.pretrain_path='{mis_ckpt_str}' "
                        f"model.freeze=False "
                        f"model.reset_head=False "
                        f"model.invert_head=True "
                        f"seed={args.seed} "
                        f"trainer.max_epochs={args.epochs_ft} "
                        f"trainer.check_val_every_n_epoch={args.val_every} "
                        f"trainer.accelerator=auto "
                        f"logger=csv "
                        f"hydra/job_logging=default hydra/hydra_logging=default "
                        f"paths.root_dir=. data.num_workers={args.num_workers} data.multiprocessing=True{extra_cmd}"
                    )
                    run_command(cmd)
                    val = get_latest_metric("mvc", log_type="train")
                    results["FT_INVERT"]["mvc"] = val
                    save_and_print_table(results)

    print("\nFINAL SUMMARY:")
    save_and_print_table(results)

if __name__ == "__main__":
    main()
