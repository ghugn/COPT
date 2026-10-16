"""Paired replay pilot. Run as python -m experiments.guidance.pilot --help."""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from experiments.guidance.solver import er_graph, solve, color_bound, vertices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 60, 80])
    parser.add_argument("--densities", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--graphs-per-cell", type=int, default=3)
    parser.add_argument("--states-per-graph", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--node-limit", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.sizes) < 2 or any(not 0 < p < 1 for p in args.densities):
        parser.error("sizes must be >=2 and densities strictly between 0 and 1")
    if min(args.graphs_per_cell, args.repeats, args.seconds, args.node_limit) <= 0 or args.states_per_graph < 0:
        parser.error("invalid experiment budget")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "paired.jsonl"
    if raw_path.exists():
        raise FileExistsError("Use a new output directory; existing results will not be overwritten")
    metadata = {**vars(args), "python": platform.python_version(), "platform": platform.platform(),
                "design": "one intervention at replay root; structural continuation",
                "checkpoint": str(args.checkpoint), "output": str(args.output)}
    metadata["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    metadata["git_status"] = subprocess.run(["git", "status", "--short"], capture_output=True, text=True).stdout
    metadata["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path(__file__).parent.glob("*.py")}
    scorer = None
    if args.checkpoint:
        from experiments.guidance.copt import COPTScorer
        with args.checkpoint.open("rb") as handle:
            metadata["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        start = time.perf_counter()
        scorer = COPTScorer(args.checkpoint, args.device)
        metadata["model_load_s"] = time.perf_counter() - start
        import torch
        import torch_geometric
        metadata.update(torch=torch.__version__, pyg=torch_geometric.__version__,
                        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None)
        scorer.score(er_graph(24, 0.5, 17), (1 << 24) - 1)  # untimed warmup, not a test instance
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    rows = []
    graph_index = 0
    with raw_path.open("x", encoding="utf-8") as handle:
        for n in args.sizes:
            for density in args.densities:
                for sample in range(args.graphs_per_cell):
                    seed = args.seed + graph_index
                    graph_index += 1
                    adj = er_graph(n, density, seed)
                    graph_id = f"er_n{n}_p{density}_seed{seed}"
                    base = solve(adj, seconds=args.seconds, node_limit=args.node_limit,
                                 capture=args.states_per_graph, capture_seed=seed)
                    states = [{"prefix": [], "candidates": (1 << n) - 1,
                               "incumbent": [], "depth": 0}] + base["snapshots"]
                    for state_index, state in enumerate(states):
                        mask = state["candidates"]
                        ids = list(vertices(mask))
                        feature_start = time.perf_counter()
                        degrees = [(adj[v] & mask).bit_count() for v in ids]
                        features = {"candidate_n": len(ids), "density": sum(degrees) / max(1, len(ids)*(len(ids)-1)),
                                    "depth": state["depth"], "incumbent_size": len(state["incumbent"]),
                                    "color_bound": color_bound(adj, mask), "degree_mean": statistics.mean(degrees),
                                    "degree_max": max(degrees), "degree_min": min(degrees)}
                        feature_s = time.perf_counter() - feature_start
                        for repeat in range(args.repeats):
                            methods = ["structural", "random"] + (["copt"] if scorer else [])
                            rng = random.Random(seed * 10000 + state_index * 100 + repeat)
                            rng.shuffle(methods)
                            results = {}
                            for method in methods:
                                priority, reason = None, None
                                overhead = 0.0
                                if method == "random":
                                    tick = time.perf_counter()
                                    priority = {v: rng.random() for v in ids}
                                    overhead = time.perf_counter() - tick
                                elif method == "copt":
                                    priority, overhead, reason = scorer.score(adj, mask)
                                remaining = max(0.0, args.seconds - overhead)
                                result = solve(adj, mask, state["prefix"], state["incumbent"],
                                               priority, seconds=remaining, node_limit=args.node_limit)
                                result.pop("snapshots")
                                result["trace"] = [(0.0, len(state["incumbent"]))] + [
                                    (t + overhead, value) for t, value in result["trace"]]
                                result.update(overhead_s=overhead, total_s=overhead + result["search_s"],
                                              fallback_reason=reason)
                                results[method] = result
                            completed = [r["best_size"] for r in results.values() if r["exhausted"]]
                            if len(set(completed)) > 1:
                                raise AssertionError("Exact replay results disagree")
                            for result in results.values():
                                events = result["trace"] + [(args.seconds, result["best_size"])]
                                area = 0.0
                                for (t, value), (end, _) in zip(events, events[1:]):
                                    area += max(0.0, min(end, args.seconds)-min(t, args.seconds)) * (value-len(state["incumbent"]))
                                result["incumbent_gain_area"] = area
                            row = {"graph_id": graph_id, "n": n, "density": density, "seed": seed,
                                   "state_id": state_index, "state": state, "features": features,
                                   "feature_s": feature_s, "repeat": repeat, "order": methods, "results": results}
                            rows.append(row)
                            handle.write(json.dumps(row) + "\n")
                            handle.flush()
                    print(f"{graph_id}: {len(states)} states, {len(rows)} paired rows saved", flush=True)
    summary = summarize(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def summarize(rows):
    """Descriptive only; repeats are NOT independent statistical samples."""
    summary = {"paired_rows": len(rows), "graphs": len({r["graph_id"] for r in rows}),
               "warning": "Hindsight pairwise oracle, not an executable learned gate; no statistical claim."}
    eligible = [r for r in rows if "copt" in r["results"] and
                all(r["results"][m]["exhausted"] for m in ("structural", "copt"))]
    summary["both_exhausted_rows"] = len(eligible)
    if eligible:
        base = sum(r["results"]["structural"]["total_s"] for r in eligible)
        neural = sum(r["results"]["copt"]["total_s"] for r in eligible)
        oracle = sum(min(r["results"]["structural"]["total_s"], r["results"]["copt"]["total_s"])
                     for r in eligible)
        summary.update(structural_total_s=base, copt_total_s=neural,
                       hindsight_oracle_total_s=oracle,
                       hindsight_saving_fraction=1-oracle/base if base else 0,
                       copt_faster_rows=sum(r["results"]["copt"]["total_s"] < r["results"]["structural"]["total_s"] for r in eligible))
    return summary


if __name__ == "__main__":
    main()
