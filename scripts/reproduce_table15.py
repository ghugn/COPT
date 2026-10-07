"""Reproduce COPT Table 15 without patch inference or external-paper methods."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.data.datasets.dimacs_dataset import DIMACSDataset
from src.evaluation.dimacs_reference import TABLE15, TABLE15_INSTANCES
from src.evaluation.dimacs_table15 import (
    complement_data,
    is_clique,
    score_complement,
    sequential_independent_set_decode,
)
from src.models.copt_module import COPTModule, MultiCOPTModule


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def model_stats(model) -> list[str]:
    try:
        return list(model.encoder.node_encoder.stat_list)
    except AttributeError as exc:
        raise ValueError("Checkpoint does not expose GraphStatsEncoder.stat_list") from exc


def reset_model(model) -> None:
    if not hasattr(model, "reset_parameters"):
        raise ValueError("Model has no reset_parameters(); cannot build random baseline")
    model.reset_parameters()


def load_module(checkpoint: Path, device: str):
    """Load either a single-task or multi-task COPT Lightning checkpoint."""
    metadata = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    net = metadata.get("hyper_parameters", {}).get("net")
    module_type = MultiCOPTModule if hasattr(net, "tasks") else COPTModule
    return module_type.load_from_checkpoint(
        str(checkpoint), map_location=device, weights_only=False
    )


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate(name, model, stats, args):
    dataset = DIMACSDataset(
        root=str(args.data_dir), name=f"table15_{name}", instance_names=[name]
    )
    if len(dataset) != 1:
        raise FileNotFoundError(f"DIMACS instance '{name}' was not found in {args.data_dir}")
    original = dataset[0]

    total_start = time.perf_counter()
    prep_start = time.perf_counter()
    complement = complement_data(original, stats)
    prep_seconds = time.perf_counter() - prep_start

    synchronize(args.device)
    inference_start = time.perf_counter()
    scores = score_complement(model, complement, task="mis", device=args.device)
    synchronize(args.device)
    inference_seconds = time.perf_counter() - inference_start

    decode_start = time.perf_counter()
    clique = sequential_independent_set_decode(
        scores,
        complement.edge_index,
        original.num_nodes,
        num_seeds=args.num_seeds,
        dec_length=args.dec_length,
    )
    decode_seconds = time.perf_counter() - decode_start
    if not is_clique(clique, original.edge_index, original.num_nodes):
        raise RuntimeError(f"Decoder returned an invalid clique for {name}")

    return {
        "clique_size": len(clique),
        "clique": clique,
        "valid": True,
        "preprocess_seconds": prep_seconds,
        "inference_seconds": inference_seconds,
        "decode_seconds": decode_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "paper_value": TABLE15[name][args.mode],
        "difference_from_paper": len(clique) - TABLE15[name][args.mode],
        "best_known": TABLE15[name]["best_known"],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data" / "dimacs" / "maxclique"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("pretrained", "random"), default="pretrained")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--dec-length",
        type=int,
        default=300,
        help="Maximum number of ranked vertices scanned by the COPT decoder (paper/config default: 300)",
    )
    parser.add_argument("--instances", nargs="*", default=list(TABLE15_INSTANCES))
    return parser.parse_args()


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    args.instances = [name.lower() for name in args.instances]
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(12345)
    module = load_module(args.checkpoint, args.device)
    model = copy.deepcopy(module.net)
    if args.mode == "random":
        reset_model(model)
    stats = model_stats(model)

    unknown = sorted(set(args.instances) - set(TABLE15))
    if unknown:
        raise ValueError(f"Instances not present in Table 15: {unknown}")

    results = {}
    for name in args.instances:
        row = evaluate(name, model, stats, args)
        results[name] = row
        print(
            f"{name:16s} found={row['clique_size']:4d} "
            f"paper={row['paper_value']:4d} delta={row['difference_from_paper']:+3d} "
            f"time={row['total_seconds']:.3f}s"
        )

    output = args.output or (
        REPO_ROOT / "reports" / "reproduction" / "table15" / f"{args.mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "COPT Table 15: MIS on explicit complement, full-graph inference, sequential decoder",
        "mode": args.mode,
        "git_commit": git_commit(),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": args.device,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "graph_stats": stats,
        "num_seeds": args.num_seeds,
        "dec_length": args.dec_length,
        "results": results,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
