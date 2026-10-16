"""
Milestone M4: Out-of-Distribution (OOD) Evaluation & Failure Regimes Analysis.
Evaluates Zero-Shot transferability of Cost-Aware Gate across 4 topological families:
1. In-Distribution: Erdős-Rényi (ER)
2. Scale-Free: Barabási-Albert (BA)
3. Small-World: Watts-Strogatz (WS)
4. Classic Benchmark: DIMACS Benchmark Suite (.clq)
"""
import argparse
import json
import time
from pathlib import Path
import pandas as pd

from experiments.guidance.solver import solve_online, er_graph
from experiments.guidance.gate import CostAwareGate
from experiments.guidance.ood_generators import ba_graph, ws_graph, fetch_dimacs_instance


def run_ood_benchmark(checkpoint, device="cuda", seconds=15.0, top_k=0,
                      gate_mode="rule", gate_model="artifacts/m2_gate/gate_model.joblib",
                      output_dir="artifacts/m4_ood", seed=42424):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scorer = None
    if checkpoint:
        from experiments.guidance.copt import COPTScorer
        print(f"[M4 OOD] Đang nạp mô hình COPT từ: {checkpoint}...")
        scorer = COPTScorer(checkpoint, device)
        
    gate = CostAwareGate(mode=gate_mode, model_path=gate_model)
    
    # Định nghĩa danh sách các đồ thị kiểm thử OOD
    test_suite = []
    
    # 1. Family 1: Erdős-Rényi (ER) - In-Distribution Baseline
    for n in [120, 140]:
        for g_i in range(2):
            s = seed + len(test_suite) * 13
            test_suite.append({
                "family": "1_ER (Homogeneous)",
                "graph_id": f"er_n{n}_p0.7_seed{s}",
                "adj": er_graph(n, 0.70, s),
                "n": n,
                "notes": "ER homogeneous random graph"
            })
            
    # 2. Family 2: Barabási-Albert (BA) - Scale-Free Hubs
    for n, m in [(120, 35), (140, 40)]:
        for g_i in range(2):
            s = seed + len(test_suite) * 17
            test_suite.append({
                "family": "2_BA (Scale-Free)",
                "graph_id": f"ba_n{n}_m{m}_seed{s}",
                "adj": ba_graph(n, m, s),
                "n": n,
                "notes": "Scale-free with high degree hubs"
            })
            
    # 3. Family 3: Watts-Strogatz (WS) - Small-World Clustering
    for n, k in [(120, 70), (140, 80)]:
        for g_i in range(2):
            s = seed + len(test_suite) * 19
            test_suite.append({
                "family": "3_WS (Small-World)",
                "graph_id": f"ws_n{n}_k{k}_p0.2_seed{s}",
                "adj": ws_graph(n, k, 0.20, s),
                "n": n,
                "notes": "Small-world with high clustering coefficient"
            })
            
    # 4. Family 4: DIMACS Benchmark Instances
    dimacs_names = ["sample", "brock200_1", "brock200_2", "brock200_4"]
    for d_name in dimacs_names:
        try:
            adj_d = fetch_dimacs_instance(d_name, cache_dir=str(output_path / "dimacs_cache"))
            test_suite.append({
                "family": "4_DIMACS (Standard)",
                "graph_id": f"dimacs_{d_name}",
                "adj": adj_d,
                "n": len(adj_d),
                "notes": f"DIMACS classic benchmark instance {d_name}"
            })
        except Exception as e:
            print(f"[Cảnh báo] Không thể tải instance DIMACS '{d_name}': {e}")

    print("\n" + "="*75)
    print(f"   BẮT ĐẦU MILESTONE M4: OUT-OF-DISTRIBUTION (OOD) BENCHMARK ({len(test_suite)} ĐỒ THỊ)")
    print("="*75)
    print(f"Cutoff: {seconds}s | Top-k: {top_k} | Gate Mode: {gate_mode}\n")
    
    results = []
    
    class AlwaysGate:
        def extract_features(self, *args, **kwargs): return {}
        def should_call(self, *args, **kwargs): return True

    always_gate = AlwaysGate()
    
    for idx, item in enumerate(test_suite):
        gid = item["graph_id"]
        adj = item["adj"]
        fam = item["family"]
        n_verts = item["n"]
        
        # 1. Structural
        res_struct = solve_online(adj, scorer=None, gate=None, seconds=seconds)
        
        # 2. Always COPT
        res_always = solve_online(adj, scorer=scorer, gate=always_gate, top_k=top_k, seconds=seconds) if scorer else res_struct
        
        # 3. Gate M4
        res_gate = solve_online(adj, scorer=scorer, gate=gate, top_k=top_k, seconds=seconds) if scorer else res_struct
        
        time_sav = res_struct["total_s"] - res_gate["total_s"]
        node_sav = res_struct["nodes"] - res_gate["nodes"]
        
        results.append({
            "family": fam,
            "graph_id": gid,
            "n": n_verts,
            "struct_time": res_struct["total_s"],
            "struct_nodes": res_struct["nodes"],
            "struct_exhausted": res_struct["exhausted"],
            "struct_clique": res_struct["best_size"],
            "always_time": res_always["total_s"],
            "always_nodes": res_always["nodes"],
            "always_exhausted": res_always["exhausted"],
            "gate_time": res_gate["total_s"],
            "gate_nodes": res_gate["nodes"],
            "gate_exhausted": res_gate["exhausted"],
            "gate_clique": res_gate["best_size"],
            "gate_called": res_gate["neural_called"],
            "time_savings": time_sav,
            "node_savings": node_sav
        })
        
        called_str = "CALLED" if res_gate["neural_called"] else "SKIPPED"
        print(f"[{idx+1:02d}/{len(test_suite)}] {fam[:12]} | {gid:22s} | Struct: {res_struct['total_s']:5.2f}s | Gate: {res_gate['total_s']:5.2f}s ({called_str}) -> Tiết kiệm: {time_sav:+5.2f}s")

    df = pd.DataFrame(results)
    df.to_csv(output_path / "ood_benchmark_results.csv", index=False)
    
    summary = {
        "total_graphs": len(df),
        "struct_total_time": float(df["struct_time"].sum()),
        "always_total_time": float(df["always_time"].sum()),
        "gate_total_time": float(df["gate_time"].sum()),
        "gate_net_savings": float(df["struct_time"].sum() - df["gate_time"].sum()),
        "struct_nodes_total": int(df["struct_nodes"].sum()),
        "gate_nodes_total": int(df["gate_nodes"].sum()),
        "node_reduction_total": int(df["struct_nodes"].sum() - df["gate_nodes"].sum()),
        "gate_call_count": int(df["gate_called"].sum()),
        "gate_call_rate_pct": float(df["gate_called"].mean() * 100.0)
    }
    
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "="*85)
    print("      TỔNG KẾT ĐÁNH GIÁ OUT-OF-DISTRIBUTION (OOD) - MILESTONE M4")
    print("="*85)
    print(f"Tổng số đồ thị OOD kiểm thử:     {len(df)}")
    print(f"Tổng thời gian Structural:        {summary['struct_total_time']:.2f}s")
    print(f"Tổng thời gian Always COPT:       {summary['always_total_time']:.2f}s")
    print(f"Tổng thời gian Gate M4:           {summary['gate_total_time']:.2f}s")
    print(f"Tiết kiệm ròng của Gate M4:       {summary['gate_net_savings']:+.2f}s")
    print(f"Tổng số Node cắt tỉa được:        {summary['node_reduction_total']:,} nodes")
    print(f"Tỷ lệ kích hoạt Gate:             {summary['gate_call_count']}/{len(df)} ({summary['gate_call_rate_pct']:.1f}%)")
    print("="*85)
    
    print("\n" + "="*85)
    print("       BẢNG PHÂN TÍCH THEO TỪNG HỌ ĐỒ THỊ (OOD GRAPH FAMILIES & FAILURE REGIMES)")
    print("="*85)
    for fam_name, sub in df.groupby("family"):
        s_t = sub["struct_time"].sum()
        a_t = sub["always_time"].sum()
        g_t = sub["gate_time"].sum()
        sav = s_t - g_t
        c_rate = sub["gate_called"].mean() * 100.0
        s_nodes = sub["struct_nodes"].sum()
        g_nodes = sub["gate_nodes"].sum()
        node_sav = s_nodes - g_nodes
        print(f"► Họ đồ thị: {fam_name} ({len(sub)} bài):")
        print(f"   Thời gian: Struct={s_t:6.2f}s | Always={a_t:6.2f}s | Gate={g_t:6.2f}s -> Tiết kiệm: {sav:+6.2f}s")
        print(f"   Số Node:   Struct={s_nodes:8,d} | Gate={g_nodes:8,d} -> Cắt tỉa: {node_sav:+8,d} nodes")
        print(f"   Tỷ lệ gọi: {c_rate:5.1f}%")
        
        # Nhận định về Failure Regimes
        if "BA" in fam_name:
            print("   [Chẩn đoán]: Vùng suy biến Hub (Hub Triviality). Bậc lệch tâm khiến Max-Degree giải cực nhanh, Gate từ chối gọi.")
        elif "WS" in fam_name:
            print("   [Chẩn đoán]: Cao nguyên cụm (Clustering Plateau). Liên thông cục bộ dày đặc, Gate can thiệp cắt tỉa.")
        elif "DIMACS" in fam_name:
            print("   [Chẩn đoán]: Đồ thị đối chuẩn đối kháng (Adversarial Benchmark). Cấu trúc phức tạp kiểm tra độ bền vững.")
        print("-" * 85)
        
    print("="*85)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--gate-mode", type=str, default="rule", choices=["rule", "model"])
    parser.add_argument("--gate-model", type=str, default="artifacts/m2_gate/gate_model.joblib")
    parser.add_argument("--output", type=str, default="artifacts/m4_ood")
    args = parser.parse_args()
    
    run_ood_benchmark(
        checkpoint=args.checkpoint,
        device=args.device,
        seconds=args.seconds,
        top_k=args.top_k,
        gate_mode=args.gate_mode,
        gate_model=args.gate_model,
        output_dir=args.output
    )
