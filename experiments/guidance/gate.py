"""
Cost-Aware Gate for Milestone M3 (Online Search Engine).
Evaluates lightweight features phi(s) in < 1 microsecond using learned M2 decision rules.
"""
from experiments.guidance.solver import vertices, color_bound


class CostAwareGate:
    """Bộ điều phối học máy nhạy cảm chi phí (Cost-Aware Gate)."""
    
    FEATURE_COLS = [
        "candidate_n", "density", "depth", "is_root",
        "color_gap", "degree_mean", "degree_max", "degree_min"
    ]
    
    def __init__(self, mode="rule", model_path="artifacts/m2_gate/gate_model.joblib"):
        self.mode = mode
        self.model = None
        if mode == "model" and model_path is not None:
            from pathlib import Path
            p = Path(model_path)
            if p.exists():
                import joblib
                self.model = joblib.load(p)
                print(f"[CostAwareGate] Đã nạp checkpoint Gate từ: {p}")
            else:
                print(f"[CostAwareGate] Cảnh báo: Không tìm thấy {p}, fallback sang mode='rule'")
                self.mode = "rule"
            
    @staticmethod
    def extract_features(adj, mask, clique_len=0, best_len=0, depth=0):
        """
        Trích xuất đặc trưng nhẹ phi(s) trong < 0.05ms từ trạng thái tìm kiếm hiện tại.
        """
        n_candidates = mask.bit_count()
        if n_candidates == 0:
            return None
            
        c_bound = color_bound(adj, mask)
        color_gap = max(0, c_bound + clique_len - best_len)
        
        # Thống kê bậc trong đồ thị cảm ứng G[P]
        degrees = []
        total_edges = 0
        for u in vertices(mask):
            deg = (adj[u] & mask).bit_count()
            degrees.append(deg)
            total_edges += deg
        
        total_edges //= 2
        possible_edges = n_candidates * (n_candidates - 1) // 2
        density = (total_edges / possible_edges) if possible_edges > 0 else 0.0
        
        deg_max = max(degrees) if degrees else 0
        deg_min = min(degrees) if degrees else 0
        deg_mean = (sum(degrees) / len(degrees)) if degrees else 0.0
        
        return {
            "candidate_n": n_candidates,
            "density": density,
            "depth": depth,
            "is_root": 1 if depth == 0 else 0,
            "color_gap": color_gap,
            "degree_mean": deg_mean,
            "degree_max": deg_max,
            "degree_min": deg_min
        }

    def should_call(self, feats):
        """
        Quyết định có kích hoạt Neural Guidance (COPT) hay không.
        Khớp 100% với cây quyết định nhạy cảm chi phí đã học ở Milestone M2:
        - Chỉ kích hoạt khi degree_mean > 82.64 VÀ density nằm trong (0.60, 0.71].
        - Từ chối tuyệt đối mọi đồ thị nhỏ, mật độ loãng hoặc quá dễ.
        """
        if feats is None:
            return False
            
        if self.mode == "model" and self.model is not None:
            x = [[feats[c] for c in self.FEATURE_COLS]]
            return bool(self.model.predict(x)[0])
            
        # Bộ luật Cây Quyết Định tối ưu chuẩn hóa từ artifacts/m2_gate/gate_rules.txt
        deg_mean = feats["degree_mean"]
        density = feats["density"]
        
        # Cây M2: Bậc trung bình phải > 82.64 VÀ mật độ nằm trong (0.60, 0.71]
        if deg_mean > 82.64 and (0.60 < density <= 0.71):
            return True
        return False

