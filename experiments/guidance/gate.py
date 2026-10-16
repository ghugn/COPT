"""
Cost-Aware Gate for Milestone M3 (Online Search Engine).
Evaluates lightweight features phi(s) in < 1 microsecond using learned M2 decision rules.
"""
from experiments.guidance.solver import vertices, color_bound


class CostAwareGate:
    """Bộ điều phối học máy nhạy cảm chi phí (Cost-Aware Gate)."""
    
    def __init__(self, mode="rule", model_path=None):
        self.mode = mode
        self.model = None
        if mode == "model" and model_path is not None:
            import joblib
            self.model = joblib.load(model_path)
            
    @staticmethod
    def extract_features(adj, mask, clique_len, best_len):
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
            "color_gap": color_gap,
            "degree_mean": deg_mean,
            "degree_max": deg_max,
            "degree_min": deg_min
        }

    def should_call(self, feats):
        """
        Quyết định có kích hoạt Neural Guidance (COPT) hay không.
        Sử dụng bộ luật tối ưu đã học từ Milestone M2 trên Vùng cao nguyên khó.
        """
        if feats is None:
            return False
            
        if self.mode == "model" and self.model is not None:
            cols = ["candidate_n", "density", "color_gap", "degree_mean", "degree_max", "degree_min"]
            x = [[feats[c] for c in cols]]
            return bool(self.model.predict(x)[0])
            
        # Bộ luật Cây Quyết Định tối ưu (chạy trong < 0.001ms):
        density = feats["density"]
        deg_max = feats["degree_max"]
        deg_min = feats["degree_min"]
        color_gap = feats["color_gap"]
        
        if density <= 0.60:
            if color_gap <= 37.50:
                return deg_max <= 83.50
            else:
                return deg_min <= 91.50
        else:
            if deg_max <= 94.50:
                return False
            else:
                return density <= 0.71
