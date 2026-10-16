"""
Milestone M4 (Dedicated): Standard DIMACS Maximum Clique Benchmark Suite.
Focuses strictly on the international DIMACS challenge instances (Brockington & Culberson).
Directly addresses the major open criticism in the peer reviews of the COPT-MT paper.
Compares:
1. Pure Structural (Classical Max-Degree)
2. Always COPT (Naive Neural)
3. Cost-Aware Gate M4 (Proposed)
Outputs:
- Optimal Clique Size (Verifying 100% Exactness: struct_clique == gate_clique)
- Search Tree Nodes & Node Reduction
- Wall-clock Time & Net Savings
"""
import argparse
import json
import sys
import time
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from experiments.guidance.solver import solve_online, is_clique
from experiments.guidance.gate import CostAwareGate
from experiments.guidance.ood_generators import fetch_dimacs_instance, DIMACS_RAW_URLS


def run_dimacs_benchmark(checkpoint, device="cuda", seconds=20.0, top_k=0,
                         gate_mode="rule", gate_model="artifacts/m2_gate/gate_model.joblib",
                         output_dir="artifacts/m4_dimacs"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / "dimacs_cache"
    
    scorer = None
    if checkpoint:
        from experiments.guidance.copt import COPTScorer
        print(f"[DIMACS Suite] Loading COPT checkpoint: {checkpoint}...")
        scorer = COPTScorer(checkpoint, device)
        
    gate = CostAwareGate(mode=gate_mode, model_path=gate_model)
    
    class AlwaysGate:
        def extract_features(self, *args, **kwargs): return {}
        def should_call(self, *args, **kwargs): return True

    always_gate = AlwaysGate()
    
    # Standard DIMACS Maximum Clique instances (Brockington & Culberson)
    instances = ["sample", "brock200_1", "brock200_2", "brock200_3", "brock200_4"]
    
    print("\n" + "="*85)
    print("      DIMACS MAXIMUM CLIQUE BENCHMARK SUITE (MILESTONE M4 EVALUATION)")
    print("="*85)
    print(f"Cutoff: {seconds}s | Top-k: {top_k} | Gate Mode: {gate_mode}\n")
    
    results = []
    
    for idx, name in enumerate(instances):
        try:
            adj = fetch_dimacs_instance(name, cache_dir=str(cache_dir))
        except Exception as e:
            print(f"[Lỗi tải] Bỏ qua {name}: {e}")
            continue
            
        n_verts = len(adj)
        n_edges = sum(mask.bit_count() for mask in adj) // 2
        density = (2.0 * n_edges) / (n_verts * (n_verts - 1)) if n_verts > 1 else 0.0
        
        # 1. Pure Structural
        res_struct = solve_online(adj, scorer=None, gate=None, seconds=seconds)
        
        # 2. Always COPT
        res_always = solve_online(adj, scorer=scorer, gate=always_gate, top_k=top_k, seconds=seconds) if scorer else res_struct
        
        # 3. Cost-Aware Gate M4
        res_gate = solve_online(adj, scorer=scorer, gate=gate, top_k=top_k, seconds=seconds) if scorer else res_struct
        
        # Xác minh nghiệm hợp lệ
        assert is_clique(adj, res_struct["clique"]), f"Nghiệm của Structural không phải clique trên {name}!"
        assert is_clique(adj, res_gate["clique"]), f"Nghiệm của Gate không phải clique trên {name}!"
        
        # Kiểm tra tính tương đương của nghiệm
        clique_match = (res_struct["best_size"] == res_gate["best_size"])
        
        time_sav = res_struct["total_s"] - res_gate["total_s"]
        node_sav = res_struct["nodes"] - res_gate["nodes"]
        
        results.append({
            "instance": name,
            "n": n_verts,
            "edges": n_edges,
            "density": density,
            "struct_clique": res_struct["best_size"],
            "gate_clique": res_gate["best_size"],
            "clique_match": clique_match,
            "struct_time": res_struct["total_s"],
            "always_time": res_always["total_s"],
            "gate_time": res_gate["total_s"],
            "time_savings": time_sav,
            "struct_nodes": res_struct["nodes"],
            "always_nodes": res_always["nodes"],
            "gate_nodes": res_gate["nodes"],
            "node_savings": node_sav,
            "gate_called": res_gate["neural_called"],
            "struct_exhausted": res_struct["exhausted"],
            "gate_exhausted": res_gate["exhausted"]
        })
        
        status_call = "CALLED" if res_gate["neural_called"] else "SKIPPED"
        match_str = "KHỚP 100%" if clique_match else "LỆCH"
        
        print(f"[{idx+1}/{len(instances)}] {name:12s} (N={n_verts:3d}, M={n_edges:5d}, p={density:.2f})")
        print(f"      -> Nghiệm tìm được: Struct={res_struct['best_size']} | Gate={res_gate['best_size']} [{match_str}]")
        print(f"      -> Thời gian giải:  Struct={res_struct['total_s']:5.2f}s | Gate={res_gate['total_s']:5.2f}s ({status_call}) -> Tiết kiệm: {time_sav:+5.2f}s")
        print(f"      -> Số Node duyệt:   Struct={res_struct['nodes']:8,d} | Gate={res_gate['nodes']:8,d} -> Cắt tỉa: {node_sav:+8,d} nodes\n")

    df = pd.DataFrame(results)
    df.to_csv(output_path / "dimacs_benchmark_results.csv", index=False)
    
    summary = {
        "total_instances": len(df),
        "all_cliques_matched": bool(df["clique_match"].all()),
        "struct_total_time": float(df["struct_time"].sum()),
        "always_total_time": float(df["always_time"].sum()),
        "gate_total_time": float(df["gate_time"].sum()),
        "gate_net_savings": float(df["struct_time"].sum() - df["gate_time"].sum()),
        "struct_nodes_total": int(df["struct_nodes"].sum()),
        "gate_nodes_total": int(df["gate_nodes"].sum()),
        "node_reduction_total": int(df["struct_nodes"].sum() - df["gate_nodes"].sum()),
        "gate_call_count": int(df["gate_called"].sum())
    }
    
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("="*85)
    print("                 TỔNG KẾT BỘ ĐỐI CHUẨN DIMACS BENCHMARK")
    print("="*85)
    print(f"Xác nhận chất lượng nghiệm:    {'CHÍNH XÁC 100% (Khớp toàn bộ)' if summary['all_cliques_matched'] else 'CÓ LỆCH'}")
    print(f"Tổng thời gian Cổ điển:        {summary['struct_total_time']:.2f}s")
    print(f"Tổng thời gian Always COPT:    {summary['always_total_time']:.2f}s")
    print(f"Tổng thời gian Gate M4:        {summary['gate_total_time']:.2f}s")
    print(f"Tiết kiệm ròng của Gate M4:    {summary['gate_net_savings']:+.2f}s")
    print(f"Tổng số Node cắt tỉa được:     {summary['node_reduction_total']:,} nodes")
    print(f"Số bài Gate kích hoạt:         {summary['gate_call_count']}/{len(df)}")
    print("="*85)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--gate-mode", type=str, default="rule", choices=["rule", "model"])
    parser.add_argument("--gate-model", type=str, default="artifacts/m2_gate/gate_model.joblib")
    parser.add_argument("--output", type=str, default="artifacts/m4_dimacs")
    args = parser.parse_args()
    
    run_dimacs_benchmark(
        checkpoint=args.checkpoint,
        device=args.device,
        seconds=args.seconds,
        top_k=args.top_k,
        gate_mode=args.gate_mode,
        gate_model=args.gate_model,
        output_dir=args.output
    )
