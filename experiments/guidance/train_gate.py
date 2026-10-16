"""
Milestone M2: Cost-Sensitive Gate Training & Baseline Benchmarking.
Evaluates: Never-Call, Always-Call, Root-Only, Size-Threshold, Oracle, and Learned Cost-Aware Gate.
Uses Cost-Sensitive Regret Minimization (Elkan, 2001) with sample weights = |Delta(s)|.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import KFold


def load_paired_data(jsonl_paths):
    """Nạp và trích xuất đặc trưng phi(s), nhãn Delta(s) từ các file paired.jsonl."""
    records = []
    for path in jsonl_paths:
        path = Path(path)
        if not path.exists():
            print(f"Bỏ qua đường dẫn không tồn tại: {path}")
            continue
        print(f"Đang đọc dữ liệu từ: {path}")
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                res = row.get("results", {})
                st = res.get("structural", {})
                cp = res.get("copt", {})
                
                # Chỉ lấy các trạng thái hoàn tất chứng minh (both exhausted)
                if not (st.get("exhausted") and cp.get("exhausted")):
                    continue
                
                feats = row.get("features", {})
                t_struct = st.get("total_s", 0.0)
                t_copt = cp.get("total_s", 0.0)
                delta = t_struct - t_copt  # Delta > 0: COPT thắng, Delta <= 0: COPT thua
                
                candidate_n = feats.get("candidate_n", row.get("n", 0))
                density = feats.get("density", row.get("density", 0.0))
                depth = feats.get("depth", row.get("state", {}).get("depth", 0))
                color_bound = feats.get("color_bound", 0)
                incumbent_size = feats.get("incumbent_size", 0)
                color_gap = max(0, color_bound - incumbent_size)
                
                records.append({
                    "graph_id": row.get("graph_id", "unknown"),
                    "candidate_n": candidate_n,
                    "density": density,
                    "depth": depth,
                    "is_root": 1 if depth == 0 else 0,
                    "color_bound": color_bound,
                    "color_gap": color_gap,
                    "degree_mean": feats.get("degree_mean", 0.0),
                    "degree_max": feats.get("degree_max", 0),
                    "degree_min": feats.get("degree_min", 0),
                    "t_struct": t_struct,
                    "t_copt": t_copt,
                    "delta": delta,
                    "target_call": 1 if delta > 0 else 0,
                    "sample_weight": abs(delta)
                })
    df = pd.DataFrame(records)
    print(f"Tổng số mẫu nạp thành công: {len(df)} (Số ca COPT thắng: {df['target_call'].sum()})")
    return df


def evaluate_policy(df, decisions, name):
    """
    Đánh giá một chính sách (decisions: mảng boolean True nếu gọi COPT, False nếu dùng Structural).
    Tính tổng thời gian thực tế, thời gian tiết kiệm, và regret so với Hindsight Oracle.
    """
    # Nếu decision == 1 -> t_copt, nếu decision == 0 -> t_struct
    actual_times = np.where(decisions, df["t_copt"], df["t_struct"])
    total_time = np.sum(actual_times)
    
    struct_time = np.sum(df["t_struct"])
    oracle_time = np.sum(np.minimum(df["t_struct"], df["t_copt"]))
    
    net_savings = struct_time - total_time
    pct_savings = (net_savings / struct_time) * 100.0 if struct_time > 0 else 0.0
    
    regret = total_time - oracle_time
    pct_regret = (regret / struct_time) * 100.0 if struct_time > 0 else 0.0
    
    call_rate = np.mean(decisions) * 100.0
    
    return {
        "Policy": name,
        "Total_Time_s": total_time,
        "Net_Savings_s": net_savings,
        "Pct_Savings": pct_savings,
        "Regret_s": regret,
        "Pct_Regret": pct_regret,
        "Call_Rate_Pct": call_rate
    }


def train_and_evaluate():
    data_paths = [
        "artifacts/guidance_pilot_m1/paired.jsonl",
        "artifacts/guidance_pilot_m1b/paired.jsonl"
    ]
    df = load_paired_data(data_paths)
    if len(df) == 0:
        raise ValueError("Không tìm thấy dữ liệu hợp lệ trong các file paired.jsonl!")
    
    feature_cols = [
        "candidate_n", "density", "depth", "is_root",
        "color_gap", "degree_mean", "degree_max", "degree_min"
    ]
    X = df[feature_cols].values
    y = df["target_call"].values
    weights = df["sample_weight"].values
    
    # 1. Định nghĩa các baselines tĩnh
    decisions_never = np.zeros(len(df), dtype=bool)
    decisions_always = np.ones(len(df), dtype=bool)
    decisions_root = (df["is_root"] == 1).values
    
    # Tìm threshold tối ưu cho size-threshold baseline
    best_tau = 80
    best_tau_savings = -float("inf")
    for tau in [40, 60, 80, 100, 120, 140]:
        dec_tau = (df["candidate_n"] >= tau).values
        eval_tau = evaluate_policy(df, dec_tau, f"Size-Threshold (tau={tau})")
        if eval_tau["Net_Savings_s"] > best_tau_savings:
            best_tau_savings = eval_tau["Net_Savings_s"]
            best_tau = tau
    decisions_size = (df["candidate_n"] >= best_tau).values
    
    # 2. Huấn luyện Cost-Aware Gate qua 5-Fold Cross Validation (Out-of-Fold Predictions)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_decisions = np.zeros(len(df), dtype=bool)
    
    for train_idx, val_idx in kf.split(X):
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
        clf.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        oof_decisions[val_idx] = clf.predict(X[val_idx]).astype(bool)
        
    # 3. Huấn luyện mô hình cuối cùng trên toàn bộ dữ liệu để xuất luật Decision Tree
    final_gate = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
    final_gate.fit(X, y, sample_weight=weights)
    
    # 4. Hindsight Oracle (Cận trên lý tưởng)
    decisions_oracle = (df["delta"] > 0).values
    
    # 5. Đánh giá tổng hợp
    results = [
        evaluate_policy(df, decisions_never, "1. Never-Call (Structural)"),
        evaluate_policy(df, decisions_always, "2. Always-Call (Naive COPT)"),
        evaluate_policy(df, decisions_root, "3. Depth-Threshold (Root Only)"),
        evaluate_policy(df, decisions_size, f"4. Size-Threshold (|P| >= {best_tau})"),
        evaluate_policy(df, oof_decisions, "5. Cost-Aware Gate (Đề xuất - CV OOF)"),
        evaluate_policy(df, decisions_oracle, "6. Hindsight Oracle (Lý tưởng)")
    ]
    
    res_df = pd.DataFrame(results)
    print("\n" + "="*85)
    print("           BẢNG ĐỐI SOÁNH HIỆU NĂNG MILESTONE M2 (TIÊU CHÍ CHUYỂN BƯỚC)")
    print("="*85)
    print(res_df.to_string(index=False, formatters={
        "Total_Time_s": "{:10.2f}".format,
        "Net_Savings_s": "{:10.2f}".format,
        "Pct_Savings": "{:8.2f}%".format,
        "Regret_s": "{:10.2f}".format,
        "Pct_Regret": "{:8.2f}%".format,
        "Call_Rate_Pct": "{:8.1f}%".format
    }))
    print("="*85)
    
    # Xuất luật cây quyết định (Rule Extraction)
    tree_rules = export_text(final_gate, feature_names=feature_cols)
    print("\n[LUẬT ĐIỀU PHỐI CỦA CÂY QUYẾT ĐỊNH (COST-AWARE GATE)]:\n")
    print(tree_rules)
    
    # Lưu kết quả
    out_dir = Path("artifacts/m2_gate")
    out_dir.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_dir / "m2_evaluation_table.csv", index=False)
    with open(out_dir / "gate_rules.txt", "w", encoding="utf-8") as f:
        f.write(tree_rules)
    import joblib
    joblib.dump(final_gate, out_dir / "gate_model.joblib")
    print(f"\nĐã lưu kết quả đánh giá vào: {out_dir}/m2_evaluation_table.csv")
    print(f"Đã lưu luật điều phối vào: {out_dir}/gate_rules.txt")
    print(f"Đã lưu checkpoint mô hình Gate vào: {out_dir}/gate_model.joblib")



if __name__ == "__main__":
    train_and_evaluate()
