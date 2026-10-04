import os
import sys
import time
import subprocess
import argparse
import shutil
import json
from pathlib import Path
import numpy as np
import pandas as pd

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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_JSON = PROJECT_ROOT / "table4_results.json"
RESULTS_TXT = PROJECT_ROOT / "table4_results.txt"

CKPT_DIR = PROJECT_ROOT / "logs" / "train" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [12345, 42, 2024]

# Paper reference table
PAPER_TABLE4 = {
    1: {"feats": "G", "gnn": "BASELINE", "gt": "—", "comp": "FALSE", "paper": "16.92 ± 0.13", "rank": ""},
    2: {"feats": "G || G_bar", "gnn": "BASELINE", "gt": "—", "comp": "FALSE", "paper": "16.63 ± 0.05", "rank": ""},
    3: {"feats": "G", "gnn": "RANDOM", "gt": "—", "comp": "FALSE", "paper": "10.71 ± 0.14", "rank": ""},
    4: {"feats": "G", "gnn": "FROZEN", "gt": "—", "comp": "FALSE", "paper": "16.12 ± 0.20", "rank": ""},
    5: {"feats": "G", "gnn": "FINE-TUNED", "gt": "—", "comp": "FALSE", "paper": "16.55 ± 0.03", "rank": "Bronze"},
    6: {"feats": "G || G_bar", "gnn": "FROZEN", "gt": "—", "comp": "FALSE", "paper": "15.98 ± 0.12", "rank": ""},
    7: {"feats": "G || G_bar", "gnn": "FINE-TUNED", "gt": "—", "comp": "FALSE", "paper": "16.63 ± 0.03", "rank": "Silver"},
    8: {"feats": "G", "gnn": "FROZEN", "gt": "3-MHA", "comp": "FALSE", "paper": "16.13 ± 0.17", "rank": ""},
    9: {"feats": "G || G_bar", "gnn": "FROZEN", "gt": "3-MHA", "comp": "FALSE", "paper": "16.11 ± 0.63", "rank": ""},
    10: {"feats": "G || G_bar", "gnn": "FROZEN", "gt": "—", "comp": "TRUE", "paper": "15.52 ± 0.08", "rank": ""},
    11: {"feats": "G || G_bar", "gnn": "FINE-TUNED", "gt": "—", "comp": "TRUE", "paper": "16.82 ± 0.04", "rank": "Gold"},
}


def run_command(cmd):
    print(f"\n{'='*70}\n[RUNNING]: {cmd}\n{'='*70}\n")
    sys.stdout.flush()
    res = subprocess.run(cmd, shell=True)
    return res.returncode


def get_latest_metric(task="maxclique", min_mtime=None, log_type="train"):
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
    csv_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest_csv = csv_files[0]
    try:
        df = pd.read_csv(latest_csv)
        candidate_cols = [
            f"test/{task}/size",
            f"test/size",
            f"test/mis/size",
            f"test/maxclique/size",
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


def ensure_data_c_pt():
    """Ensure data-c.pt exists for complement experiments so PyG doesn't regenerate."""
    processed_dir = PROJECT_ROOT / "data" / "rb" / "small" / "processed"
    if not processed_dir.exists():
        processed_dir.mkdir(parents=True, exist_ok=True)
    
    data_pt = processed_dir / "data.pt"
    data_c_pt = processed_dir / "data-c.pt"
    
    # Check if backup exists in /kaggle/input/copt-rb/
    kaggle_backup = Path("/kaggle/input/copt-rb")
    if not data_pt.exists():
        for candidate in [
            kaggle_backup / "data.pt",
            kaggle_backup / "small" / "processed" / "data.pt",
            kaggle_backup / "rb" / "small" / "processed" / "data.pt",
        ]:
            if candidate.exists():
                print(f"[DATA SETUP] Copying {candidate} -> {data_pt}")
                shutil.copy(candidate, data_pt)
                break
    
    if data_pt.exists() and not data_c_pt.exists():
        print(f"[DATA SETUP] Linking/Copying {data_pt} -> {data_c_pt} for complement support...")
        try:
            shutil.copy(data_pt, data_c_pt)
        except Exception as e:
            print(f"Warning copying data-c.pt: {e}")


def ensure_mis_checkpoint(seed, max_epochs=200, val_every=10, num_workers=2):
    """Ensure pretrained MIS checkpoint with G features exists."""
    candidates = [
        CKPT_DIR / f"mis_rb_small_seed_{seed}.ckpt",
        CKPT_DIR / "mis_rb_small.ckpt",
        Path(f"/kaggle/working/COPT/mis_rb_small_seed_{seed}.ckpt"),
        Path(f"/kaggle/working/COPT/mis_rb_small.ckpt"),
        Path(f"/kaggle/input/copt-rb/mis_rb_small_seed_{seed}.ckpt"),
        Path(f"/kaggle/input/copt-rb/mis_rb_small.ckpt"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1000:
            target = CKPT_DIR / f"mis_rb_small_seed_{seed}.ckpt"
            if not target.exists():
                shutil.copy(c, target)
            return target

    print(f"\n[SETUP] Pretraining MIS baseline for seed {seed}...")
    start_time = time.time()
    cmd = (
        f"python src/train.py experiment=mis/rb_small/gcon "
        f"seed={seed} "
        f"trainer.max_epochs={max_epochs} "
        f"callbacks.early_stopping.patience=40 "
        f"trainer.check_val_every_n_epoch={val_every} "
        f"paths.root_dir=. data.num_workers={num_workers}"
    )
    ret = run_command(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to pretrain MIS baseline for seed {seed}")

    latest_ckpt = find_latest_checkpoint(min_mtime=start_time)
    if not latest_ckpt:
        raise RuntimeError(f"Could not find trained MIS checkpoint for seed {seed}")

    target = CKPT_DIR / f"mis_rb_small_seed_{seed}.ckpt"
    shutil.copy(latest_ckpt, target)
    print(f"[SETUP] Saved MIS checkpoint -> {target}")
    return target


def ensure_mis_cdata_checkpoint(seed, max_epochs=200, val_every=10, num_workers=2):
    """Ensure pretrained MIS checkpoint with G || G_bar (cdata) features exists."""
    candidates = [
        CKPT_DIR / f"mis_rb_small_cdata_seed_{seed}.ckpt",
        CKPT_DIR / "mis_rb_small_cdata.ckpt",
        Path(f"/kaggle/working/COPT/mis_rb_small_cdata_seed_{seed}.ckpt"),
        Path(f"/kaggle/input/copt-rb/mis_rb_small_cdata_seed_{seed}.ckpt"),
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1000:
            target = CKPT_DIR / f"mis_rb_small_cdata_seed_{seed}.ckpt"
            if not target.exists():
                shutil.copy(c, target)
            return target

    ensure_data_c_pt()
    print(f"\n[SETUP] Pretraining MIS on G || G_bar (cdata) for seed {seed}...")
    start_time = time.time()
    cmd = (
        f"python src/train.py experiment=mis/rb_small/gcon_cdata "
        f"seed={seed} "
        f"trainer.max_epochs={max_epochs} "
        f"callbacks.early_stopping.patience=40 "
        f"trainer.check_val_every_n_epoch={val_every} "
        f"paths.root_dir=. data.num_workers={num_workers}"
    )
    ret = run_command(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to pretrain MIS cdata for seed {seed}")

    latest_ckpt = find_latest_checkpoint(min_mtime=start_time)
    if not latest_ckpt:
        raise RuntimeError(f"Could not find trained MIS cdata checkpoint for seed {seed}")

    target = CKPT_DIR / f"mis_rb_small_cdata_seed_{seed}.ckpt"
    shutil.copy(latest_ckpt, target)
    print(f"[SETUP] Saved MIS cdata checkpoint -> {target}")
    return target


def load_results():
    if RESULTS_JSON.exists():
        try:
            with open(RESULTS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"seeds": [], "runs": {}}


def save_results(data):
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def render_table(results_data, target_seeds):
    lines = []
    lines.append("=" * 108)
    lines.append("TABLE 4: MIS -> MaxClique Transferability on RB-small (Cantürk et al.)")
    lines.append("=" * 108)
    lines.append(f"{'#':<4}{'FEATS':<14}{'GNN STACK':<14}{'GT':<8}{'COMP?':<8}{'PAPER MC SIZE':<18}{'OUR MC SIZE':<22}{'STATUS':<12}")
    lines.append("-" * 108)

    for row_id, meta in PAPER_TABLE4.items():
        feats = meta["feats"]
        gnn = meta["gnn"]
        gt = meta["gt"]
        comp = meta["comp"]
        paper_val = meta["paper"]
        rank_tag = f" ({meta['rank']})" if meta["rank"] else ""

        # Collect results across active seeds
        vals = []
        for s in target_seeds:
            s_str = str(s)
            if s_str in results_data.get("runs", {}):
                row_val = results_data["runs"][s_str].get(str(row_id))
                if row_val is not None:
                    vals.append(row_val)

        if len(vals) == 0:
            our_str = "—"
            status = "PENDING"
        elif len(vals) == 1:
            our_str = f"{vals[0]:.2f}{rank_tag}"
            status = "COMPLETED"
        else:
            mean = np.mean(vals)
            std = np.std(vals)
            our_str = f"{mean:.2f} ± {std:.2f}{rank_tag}"
            status = "COMPLETED"

        lines.append(f"{row_id:<4}{feats:<14}{gnn:<14}{gt:<8}{comp:<8}{paper_val:<18}{our_str:<22}{status:<12}")

    lines.append("=" * 108)
    table_str = "\n".join(lines)
    return table_str


def print_and_save_table(results_data, target_seeds):
    tbl = render_table(results_data, target_seeds)
    print("\n" + tbl + "\n")
    sys.stdout.flush()
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(tbl + "\n")


def run_row(row_id, seed, args, results_data):
    s_str = str(seed)
    if s_str not in results_data["runs"]:
        results_data["runs"][s_str] = {}

    if str(row_id) in results_data["runs"][s_str] and not args.force:
        val = results_data["runs"][s_str][str(row_id)]
        print(f"[SKIP] Row #{row_id} for Seed {seed} already completed: {val:.4f}")
        return val

    print(f"\n{'#'*80}")
    print(f"STARTING ROW #{row_id} | Seed: {seed} | {PAPER_TABLE4[row_id]['gnn']} | FEATS: {PAPER_TABLE4[row_id]['feats']}")
    print(f"{'#'*80}")

    ensure_data_c_pt()
    start_time = time.time()
    val_every = args.val_every
    num_workers = args.num_workers
    max_epochs_baseline = args.max_epochs_baseline
    max_epochs_transfer = args.max_epochs_transfer

    # Build Hydra command per row
    if row_id == 1:
        # G BASELINE
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon "
            f"seed={seed} "
            f"trainer.max_epochs={max_epochs_baseline} "
            f"callbacks.early_stopping.patience=100 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 2:
        # G || G_bar BASELINE
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_cdata "
            f"seed={seed} "
            f"trainer.max_epochs={max_epochs_baseline} "
            f"callbacks.early_stopping.patience=100 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 3:
        # G RANDOM (Evaluate randomly initialized model)
        cmd = (
            f"python src/eval.py experiment=maxclique/rb_small/gcon "
            f"seed={seed} "
            f"ckpt_path=null "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "eval"

    elif row_id == 4:
        # G FROZEN
        mis_ckpt = ensure_mis_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis "
            f"seed={seed} "
            f"model.pretrain_path='{mis_ckpt.as_posix()}' "
            f"model.freeze='backbone' "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 5:
        # G FINE-TUNED
        mis_ckpt = ensure_mis_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis "
            f"seed={seed} "
            f"model.pretrain_path='{mis_ckpt.as_posix()}' "
            f"model.freeze=False "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 6:
        # G || G_bar FROZEN
        mis_cdata_ckpt = ensure_mis_cdata_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis_cdata "
            f"seed={seed} "
            f"model.pretrain_path='{mis_cdata_ckpt.as_posix()}' "
            f"model.freeze='backbone' "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 7:
        # G || G_bar FINE-TUNED
        mis_cdata_ckpt = ensure_mis_cdata_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis_cdata "
            f"seed={seed} "
            f"model.pretrain_path='{mis_cdata_ckpt.as_posix()}' "
            f"model.freeze=False "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 8:
        # G FROZEN 3-MHA
        mis_ckpt = ensure_mis_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_with_mha_mis "
            f"seed={seed} "
            f"model.pretrain_path='{mis_ckpt.as_posix()}' "
            f"model.freeze='backbone' "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 9:
        # G || G_bar FROZEN 3-MHA
        mis_cdata_ckpt = ensure_mis_cdata_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_with_mha_mis_cdata "
            f"seed={seed} "
            f"model.pretrain_path='{mis_cdata_ckpt.as_posix()}' "
            f"model.freeze='backbone' "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "maxclique"
        log_type = "train"

    elif row_id == 10:
        # G || G_bar FROZEN on COMP (G_bar)
        mis_cdata_ckpt = ensure_mis_cdata_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis_c "
            f"seed={seed} "
            f"model.pretrain_path='{mis_cdata_ckpt.as_posix()}' "
            f"model.freeze='gnn_stack' "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "mis"  # Equivalent reduction uses MIS metric over complement
        log_type = "train"

    elif row_id == 11:
        # G || G_bar FINE-TUNED on COMP (G_bar)
        mis_cdata_ckpt = ensure_mis_cdata_checkpoint(seed, val_every=val_every, num_workers=num_workers)
        cmd = (
            f"python src/train.py experiment=maxclique/rb_small/gcon_pretrained_mis_c "
            f"seed={seed} "
            f"model.pretrain_path='{mis_cdata_ckpt.as_posix()}' "
            f"model.freeze=False "
            f"trainer.max_epochs={max_epochs_transfer} "
            f"callbacks.early_stopping.patience=50 "
            f"trainer.check_val_every_n_epoch={val_every} "
            f"paths.root_dir=. data.num_workers={num_workers}"
        )
        task = "mis"  # Equivalent reduction uses MIS metric over complement
        log_type = "train"

    else:
        raise ValueError(f"Unknown Row ID: {row_id}")

    # Append any extra overrides passed via CLI
    if args.extra_args:
        cmd += f" {args.extra_args}"

    ret = run_command(cmd)
    if ret != 0:
        print(f"[ERROR] Row #{row_id} failed with exit code {ret}")
        return None

    val = get_latest_metric(task=task, min_mtime=start_time, log_type=log_type)
    if val is None:
        # Try generic size lookup across any log folder
        val = get_latest_metric(task="size", min_mtime=start_time, log_type=log_type)

    if val is not None:
        results_data["runs"][s_str][str(row_id)] = val
        save_results(results_data)
        print(f"[SUCCESS] Row #{row_id} for Seed {seed} => MC Size: {val:.4f} (Paper: {PAPER_TABLE4[row_id]['paper']})")
    else:
        print(f"[WARNING] Could not extract metric for Row #{row_id} for Seed {seed}")

    return val


def parse_rows(rows_arg):
    if not rows_arg:
        return list(range(1, 12))
    
    arg_lower = str(rows_arg[0]).lower() if len(rows_arg) == 1 else ""
    if arg_lower == "all":
        return list(range(1, 12))
    elif arg_lower == "core":
        # Key comparison rows: Baselines, Random, G transfer, and COMP transfer
        return [1, 3, 4, 5, 10, 11]
    elif arg_lower == "g_only":
        # Only rows using G features (no cdata pretraining required)
        return [1, 3, 4, 5, 8]
    elif arg_lower == "transfer":
        # All transfer rows (4 to 11)
        return list(range(4, 12))
    
    # Otherwise parse integers
    selected = []
    for r in rows_arg:
        try:
            val = int(r)
            if 1 <= val <= 11:
                selected.append(val)
        except ValueError:
            pass
    return selected if selected else list(range(1, 12))


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 4: MIS -> MaxClique Transferability on RB-small")
    parser.add_argument("--rows", nargs="+", default=["all"], help="Rows to run: 'all', 'core', 'g_only', 'transfer', or list like '1 3 4 5 10 11'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[12345], help="Random seeds (default: 12345; for 3 runs: 12345 42 2024)")
    parser.add_argument("--val_every", type=int, default=10, help="Check validation every N epochs to speed up training (default: 10)")
    parser.add_argument("--max_epochs_baseline", type=int, default=300, help="Max epochs for baselines trained from scratch (default: 300)")
    parser.add_argument("--max_epochs_transfer", type=int, default=150, help="Max epochs for transfer / fine-tuning (default: 150)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers (default: 2)")
    parser.add_argument("--force", action="store_true", help="Force rerun of rows even if already cached")
    parser.add_argument("--extra_args", type=str, default="", help="Extra Hydra arguments to pass through")
    args = parser.parse_args()

    target_rows = parse_rows(args.rows)
    target_seeds = args.seeds

    print("\n" + "=" * 80)
    print("REPRODUCING TABLE 4: MIS -> MaxClique Transferability on RB-small")
    print(f"Target Rows: {target_rows}")
    print(f"Seeds: {target_seeds}")
    print(f"Val Check Interval: every {args.val_every} epochs")
    print(f"Max Epochs: Baseline={args.max_epochs_baseline}, Transfer={args.max_epochs_transfer}")
    print("=" * 80 + "\n")

    results_data = load_results()
    for s in target_seeds:
        if s not in results_data.get("seeds", []):
            results_data.setdefault("seeds", []).append(s)
    save_results(results_data)

    print_and_save_table(results_data, target_seeds)

    for seed in target_seeds:
        print(f"\n>>> PROCESSING SEED: {seed} <<<\n")
        for row_id in target_rows:
            run_row(row_id, seed, args, results_data)
            print_and_save_table(results_data, target_seeds)

    print("\n" + "=" * 80)
    print("FINISHED TABLE 4 EXECUTION!")
    print(f"Results stored in: {RESULTS_JSON} and {RESULTS_TXT}")
    print("=" * 80 + "\n")
    print_and_save_table(results_data, target_seeds)


if __name__ == "__main__":
    main()
