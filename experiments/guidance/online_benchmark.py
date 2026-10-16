"""
Milestone M3: Online Search Benchmark & Distribution Drift Analysis.
Compares End-to-End search performance on fresh graphs:
1. Pure Structural (Max-Degree)
2. Naive Neural (Always-Call at Root)
3. Online Cost-Aware Guided Solver (M3)
"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from experiments.guidance.solver import er_graph, solve_online
from experiments.guidance.gate import CostAwareGate


def run_online_benchmark(checkpoint, device="cuda", sizes=(120, 150), densities=(0.6, 0.7),
                         graphs_per_cell=3, seconds=5.0, top_k=3, seed=32345, output_dir="artifacts/m3_online"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scorer = None
    if checkpoint:
        from experiments.guidance.copt import COPTScorer
        print(f"Đang nạp mô hình COPT từ: {checkpoint}")
        scorer = COPTScorer(checkpoint, device)
        
    gate = CostAwareGate(mode="rule")
    
    results = []
    g_count = 0
    total_graphs = len(sizes) * len(densities) * graphs_per_cell
    
    print(f"\nBắt đầu Milestone M3 Online Benchmark trên {total_graphs} đồ thị...")
    print(f"Sizes: {sizes}, Densities: {densities}, Graphs/cell: {graphs_per_cell}, Cutoff: {seconds}s, Top-k: {top_k}\n")
    
    for n in sizes:
        for p in densities:
            for g_idx in range(graphs_per_cell):
                g_count += 1
                g_seed = seed + g_count * 17
                adj = er_graph(n, p, g_seed)
                graph_id = f"er_n{n}_p{p}_seed{g_seed}"
                
                # 1. Pure Structural (Cổ điển)
                res_struct = solve_online(adj, scorer=None, gate=None, seconds=seconds)
                
                # 2. Naive COPT (Luôn gọi tại Root)
                class AlwaysGate:
                    def extract_features(self, *args, **kwargs): return {}
                    def should_call(self, *args, **kwargs): return True
                    
                res_always = solve_online(adj, scorer=scorer, gate=AlwaysGate(), top_k=top_k, seconds=seconds) if scorer else res_struct
                
                # 3. Online Cost-Aware Gate (Đề xuất M3)
                res_gate = solve_online(adj, scorer=scorer, gate=gate, top_k=top_k, seconds=seconds) if scorer else res_struct
                
                results.append({
                    "graph_id": graph_id,
                    "n": n,
                    "density": p,
                    # Structural
                    "struct_time": res_struct["total_s"],
                    "struct_nodes": res_struct["nodes"],
                    "struct_exhausted": res_struct["exhausted"],
                    # Always-Call
                    "always_time": res_always["total_s"],
                    "always_nodes": res_always["nodes"],
                    "always_exhausted": res_always["exhausted"],
                    # Gate M3
                    "gate_time": res_gate["total_s"],
                    "gate_nodes": res_gate["nodes"],
                    "gate_exhausted": res_gate["exhausted"],
                    "gate_called": res_gate["neural_called"],
                    "gate_overhead": res_gate["overhead_s"]
                })
                print(f"[{g_count}/{total_graphs}] {graph_id}: Struct={res_struct['total_s']:.2f}s | Gate={res_gate['total_s']:.2f}s (Called: {res_gate['neural_called']})")

    df = pd.DataFrame(results)
    df.to_csv(output_path / "online_benchmark_results.csv", index=False)
    
    # Tổng kết
    summary = {
        "total_graphs": len(df),
        "struct_total_time": df["struct_time"].sum(),
        "always_total_time": df["always_time"].sum(),
        "gate_total_time": df["gate_time"].sum(),
        "gate_net_savings": df["struct_time"].sum() - df["gate_time"].sum(),
        "gate_call_count": int(df["gate_called"].sum()),
        "gate_call_rate_pct": float(df["gate_called"].mean() * 100.0)
    }
    
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*70)
    print("           TỔNG KẾT THỰC NGHIỆM MILESTONE M3 (ONLINE BENCHMARK)")
    print("="*70)
    print(f"Tổng thời gian Structural:   {summary['struct_total_time']:.2f}s")
    print(f"Tổng thời gian Always COPT:  {summary['always_total_time']:.2f}s")
    print(f"Tổng thời gian Gate M3:      {summary['gate_total_time']:.2f}s")
    print(f"Tiết kiệm ròng của Gate M3:  {summary['gate_net_savings']:+.2f}s")
    print(f"Tỷ lệ kích hoạt Gate:        {summary['gate_call_count']}/{len(df)} ({summary['gate_call_rate_pct']:.1f}%)")
    print("="*70)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=str, default="artifacts/m3_online")
    args = parser.parse_args()
    
    run_online_benchmark(
        checkpoint=args.checkpoint,
        device=args.device,
        seconds=args.seconds,
        top_k=args.top_k,
        output_dir=args.output
    )
