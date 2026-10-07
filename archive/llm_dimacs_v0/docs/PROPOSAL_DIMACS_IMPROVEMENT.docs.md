# ĐỀ XUẤT NGHIÊN CỨU (BẢN CHÍNH THỨC - ĐÃ QUA PHẢN BIỆN)
# Patch-based Transfer Learning for Maximum Clique on Real-World DIMACS Graphs

> **Tên dự kiến bài báo:** *"Can Small Graphs Teach Large Ones? Patch-based Zero-shot Transfer for Graph Combinatorial Optimization on DIMACS Benchmarks"*
>
> **Kế thừa từ:** Cantürk et al. — *"Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?"*
>
> **Repo gốc:** `COPT-MT-main/` (đã tái hiện thành công 100% benchmark gốc: Table 3, 5, 7)

---

## MỤC LỤC
1. [Câu hỏi nghiên cứu trung tâm](#1-câu-hỏi-nghiên-cứu-trung-tâm)
2. [Tóm tắt 3 đóng góp chính](#2-tóm-tắt-3-đóng-góp-chính)
3. [Đóng góp 1: Complement-Free MaxClique Loss](#3-đóng-góp-1-complement-free-maxclique-loss)
4. [Đóng góp 2: Patch-based Inference](#4-đóng-góp-2-patch-based-inference)
5. [Đóng góp 3: Ablation Transferability 3 chế độ](#5-đóng-góp-3-ablation-transferability-3-chế-độ)
6. [Thiết kế thực nghiệm chi tiết](#6-thiết-kế-thực-nghiệm-chi-tiết)
7. [Kế hoạch triển khai code](#7-kế-hoạch-triển-khai-code)
8. [Phản biện đã lường trước & Rebuttal](#8-phản-biện-đã-lường-trước--rebuttal)

---

## 1. Câu hỏi nghiên cứu trung tâm

> **"Một mô hình GNN pretrain trên đồ thị tổng hợp nhỏ (N = 200–300 đỉnh) có thể chuyển giao zero-shot lên đồ thị thế giới thực lớn (N = 1.000–4.000 đỉnh, DIMACS benchmark) và cho nghiệm Maximum Clique tốt hơn đáng kể so với mô hình khởi tạo ngẫu nhiên hay không?"**

### Tại sao câu hỏi này có giá trị:
- Paper gốc (Cantürk et al.) chỉ chứng minh transferability giữa các bài toán CO khác nhau (MIS → MVC, multi-task) trên CÙNG phân phối đồ thị nhỏ.
- **Chưa ai chứng minh** transferability từ đồ thị tổng hợp nhỏ sang đồ thị thế giới thực lớn với cấu trúc hoàn toàn khác.
- Đây là bước tiến tự nhiên và cần thiết để biến Graph Foundation Model từ khái niệm lý thuyết thành công cụ thực tiễn.

---

## 2. Tóm tắt 3 đóng góp chính

| # | Đóng góp | Vai trò | Tính mới |
|---|----------|---------|----------|
| 1 | **Complement-Free MaxClique Loss** | Điều kiện tiên quyết kỹ thuật (Engineering prerequisite) — loại bỏ OOM | Thấp (đã biết về mặt toán, nhưng paper gốc chưa áp dụng) |
| 2 | **Patch-based Inference** | Đóng góp phương pháp chính (Main methodological contribution) — cho phép GNN pretrain trên đồ thị nhỏ suy luận trên đồ thị lớn | **Cao** (chưa ai làm cho Graph CO) |
| 3 | **Ablation Transferability 3 chế độ + Anytime Plot** | Đóng góp thực nghiệm (Empirical contribution) — chứng minh pretrain có giá trị thực sự | **Cao** (câu hỏi chưa được trả lời) |

---

## 3. Đóng góp 1: Complement-Free MaxClique Loss

### 3.1. Vấn đề của paper gốc
Paper gốc giải MaxClique(G) bằng cách quy về MIS trên đồ thị bù G_bar:
- Sinh toàn bộ edge_index_c = negate_edge_index(edge_index) — tạo ra đồ thị bù tường minh.
- Trên DIMACS (N=1000, density=0.9): |E(G_bar)| chỉ ~50.000 cạnh nhưng |E(G)| ~450.000 cạnh.
  Tuy nhiên trên đồ thị thưa (density < 0.5): |E(G_bar)| bùng nổ lên hàng triệu cạnh → OOM.
- GNN phải message-passing trên đồ thị bù dày đặc → chậm và tốn RAM.

### 3.2. Giải pháp: Tính violation trực tiếp trên đồ thị gốc G

**Công thức toán học:**
Một tập S là Clique ⟺ mọi cặp đỉnh trong S đều kề nhau trong G.
Số cặp vi phạm (cặp đỉnh trong S KHÔNG kề nhau):

    Violations(x) = (1/2) * [ (Σ x_i)² - Σ_{(i,j)∈E(G)} x_i·x_j - Σ x_i² ]

Giải thích từng thành phần:
- (Σ x_i)²  = tổng số cặp đỉnh được chọn (bao gồm cả kề và không kề)
- Σ_{(i,j)∈E(G)} x_i·x_j = số cặp đỉnh được chọn MÀ có cạnh nối trong G
- Σ x_i² = số cặp tự ghép (loại bỏ)
- Hiệu = số cặp đỉnh được chọn MÀ KHÔNG có cạnh nối = vi phạm

**Hàm loss hoàn chỉnh:**

    L_MaxClique(x) = -α·Σ x_i + β · Violations(x)

Với β > α > 0 (mặc định: α=1.0, β=1.1–2.0).

### 3.3. Code triển khai

```python
def direct_maxclique_loss(batch, alpha=1.0, beta=1.1, reduction='sum'):
    """
    MaxClique loss trực tiếp trên đồ thị gốc G.
    KHÔNG sinh edge_index_c. Độ phức tạp O(|E(G)|). Không tràn RAM.
    
    Tương thích hoàn toàn với pipeline COPT hiện tại.
    """
    data_list = batch.to_data_list()
    loss = 0.0

    for data in data_list:
        x = data.x.squeeze(-1)  # [num_nodes], x_i ∈ [0, 1]
        src, dst = data.edge_index

        # Phần thưởng kích thước (Size reward): muốn chọn nhiều đỉnh
        size_term = -alpha * torch.sum(x)

        # Vi phạm: số cặp đỉnh được chọn mà KHÔNG kề nhau trong G
        sum_x = torch.sum(x)
        sum_x_sq = torch.sum(x ** 2)
        edge_sum = torch.sum(x[src] * x[dst])  # CHỈ tính trên cạnh gốc E(G)
        
        violations = 0.5 * (sum_x ** 2 - edge_sum - sum_x_sq)
        penalty_term = beta * violations

        loss += (size_term + penalty_term) / data.num_nodes

    if reduction == 'mean':
        return loss / len(data_list)
    return loss
```

### 3.4. Vị trí tích hợp trong repo
- **File:** `src/models/loss/copt_loss.py`
- **Thêm hàm** `direct_maxclique_loss` bên cạnh các hàm loss hiện có.
- **Cập nhật** `src/models/copt_module.py` để cho phép chọn loss mới qua config:
  ```yaml
  model:
    net:
      maxclique_loss: "direct"  # thay vì "complement"
  ```

---

## 4. Đóng góp 2: Patch-based Inference

### 4.1. Vấn đề cần giải quyết
GNN pretrain trên đồ thị kích thước N=200–300. Đồ thị DIMACS có N=1.000–4.000.
- Nạp trực tiếp đồ thị lớn vào GNN: kích thước không khớp, embedding bị méo, có thể OOM.
- Train lại trên mỗi DIMACS instance: mất thời gian, không còn ưu thế "nhanh hơn Gurobi".

### 4.2. Ý tưởng cốt lõi
Chia đồ thị lớn thành nhiều mảnh con (patches) có kích thước ĐÚNG BẰNG kích thước pretrain.
Chạy GNN inference trên từng mảnh. Tổng hợp xác suất từ các mảnh thành heatmap toàn cục.

### 4.3. Thuật toán chi tiết

```
THUẬT TOÁN: Patch-based GNN Inference cho đồ thị lớn
═══════════════════════════════════════════════════════

ĐẦU VÀO:
  - Đồ thị lớn G = (V, E) với |V| = N (ví dụ: N = 2.000)
  - Mô hình GNN θ đã pretrain trên đồ thị kích thước n (ví dụ: n = 300)
  - Số mảnh K (mặc định: K = max(20, N / n * 2))
  - Bán kính trích xuất r (mặc định: r = 2 hop)

ĐẦU RA:
  - Vector xác suất p ∈ [0, 1]^N cho mỗi đỉnh

CÁC BƯỚC:

1. CHỌN SEEDS (Đỉnh hạt giống):
   - Tính bậc d(v) cho mọi v ∈ V.
   - Sắp xếp giảm dần theo bậc.
   - Chọn K đỉnh hạt giống s_1, s_2, ..., s_K sao cho:
     + Ưu tiên đỉnh bậc cao (có khả năng thuộc Clique lớn).
     + Các seeds cách nhau ≥ 2 hop (để phủ nhiều vùng khác nhau).
     (Có thể dùng Farthest Point Sampling trên đồ thị)

2. TRÍCH XUẤT MẢNH CON (Patch extraction):
   Với mỗi seed s_k:
     a. Lấy tập đỉnh lân cận r-hop: N_r(s_k) = {v : dist(v, s_k) ≤ r}
     b. Nếu |N_r(s_k)| > n_max (ví dụ 500): lấy top-n_max đỉnh theo bậc.
     c. Nếu |N_r(s_k)| < n_min (ví dụ 50): mở rộng r ← r + 1.
     d. Trích xuất đồ thị con cảm ứng (induced subgraph): G_k = G[N_r(s_k)].
     e. Ghi nhận ánh xạ đỉnh cục bộ → toàn cục: mapping_k.

3. GNN INFERENCE TRÊN TỪNG MẢNH:
   Với mỗi mảnh G_k:
     a. Chuẩn bị input cho GNN (thêm đặc trưng nút nếu cần).
     b. Chạy forward pass: p_k = GNN_θ(G_k)  — vector xác suất cục bộ.
     c. Ánh xạ ngược về chỉ số đỉnh toàn cục qua mapping_k.

4. TỔNG HỢP XÁC SUẤT TOÀN CỤC:
   Với mỗi đỉnh v ∈ V:
     - Gom tất cả xác suất p_k(v) từ các mảnh chứa v.
     - Tính xác suất cuối: p(v) = mean({p_k(v) : v ∈ G_k}) 
       (hoặc max, hoặc weighted average theo khoảng cách đến seed).
     - Nếu v không thuộc mảnh nào: p(v) = 0.

5. GREEDY DECODING TỪ HEATMAP:
   a. Khởi tạo: Clique C = ∅, tập ứng viên K = V.
   b. Lặp:
      - Chọn u* = argmax_{u ∈ K} p(u).
      - Thêm u* vào C.
      - Cập nhật K ← K ∩ N(u*) (chỉ giữ đỉnh kề u* trong G).
      - Dừng khi K = ∅.
   c. (Tùy chọn) Local Search: thử swap 1 đỉnh trong C bằng 2 đỉnh ngoài C.
   d. Trả về Clique C (đảm bảo 100% hợp lệ).
```

### 4.4. Code triển khai

```python
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph, to_networkx

class PatchBasedInference:
    """
    Chia đồ thị lớn thành các mảnh con, chạy GNN pretrained trên từng mảnh,
    tổng hợp xác suất toàn cục, rồi greedy decode ra Clique.
    """
    
    def __init__(self, model, patch_size=300, num_patches=30, num_hops=2, device='cuda'):
        self.model = model
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_hops = num_hops
        self.device = device
        self.model.eval()
    
    def select_seeds(self, edge_index, num_nodes):
        """Chọn seeds bằng cách ưu tiên đỉnh bậc cao, cách nhau ≥ 2 hop."""
        # Tính bậc
        degree = torch.zeros(num_nodes, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))
        
        # Sắp xếp theo bậc giảm dần
        sorted_nodes = torch.argsort(degree, descending=True)
        
        seeds = []
        used = set()
        for node in sorted_nodes.tolist():
            if node in used:
                continue
            seeds.append(node)
            if len(seeds) >= self.num_patches:
                break
            # Đánh dấu lân cận 1-hop để tránh chọn seed quá gần
            neighbors, _, _, _ = k_hop_subgraph(
                node, 1, edge_index, num_nodes=num_nodes
            )
            used.update(neighbors.tolist())
        
        return seeds
    
    def extract_patch(self, seed, edge_index, num_nodes):
        """Trích xuất đồ thị con cảm ứng quanh seed (r-hop neighborhood)."""
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            seed, self.num_hops, edge_index,
            num_nodes=num_nodes, relabel_nodes=True
        )
        
        # Giới hạn kích thước patch nếu quá lớn
        if len(subset) > self.patch_size:
            # Giữ lại top-patch_size đỉnh gần seed nhất (theo bậc)
            degree = torch.zeros(len(subset), dtype=torch.long)
            degree.scatter_add_(0, sub_edge_index[0], 
                              torch.ones(sub_edge_index.size(1), dtype=torch.long))
            top_local = torch.argsort(degree, descending=True)[:self.patch_size]
            
            # Re-extract subgraph cho top đỉnh
            mask = torch.zeros(len(subset), dtype=torch.bool)
            mask[top_local] = True
            sub_edge_index, _ = subgraph(mask, sub_edge_index, relabel_nodes=True)
            subset = subset[top_local]
        
        patch_data = Data(
            x=torch.ones(len(subset), 1),
            edge_index=sub_edge_index,
            num_nodes=len(subset)
        )
        return patch_data, subset  # subset = ánh xạ local → global
    
    @torch.no_grad()
    def inference(self, edge_index, num_nodes):
        """Chạy patch-based inference, trả về heatmap xác suất toàn cục."""
        seeds = self.select_seeds(edge_index, num_nodes)
        
        # Tích lũy xác suất
        prob_sum = torch.zeros(num_nodes)
        prob_count = torch.zeros(num_nodes)
        
        for seed in seeds:
            patch_data, global_indices = self.extract_patch(seed, edge_index, num_nodes)
            patch_data = patch_data.to(self.device)
            
            # GNN forward pass
            output = self.model(patch_data)
            probs = output.x.squeeze(-1).cpu()  # [patch_size]
            
            # Tích lũy về toàn cục
            prob_sum[global_indices] += probs
            prob_count[global_indices] += 1
        
        # Trung bình xác suất
        mask = prob_count > 0
        heatmap = torch.zeros(num_nodes)
        heatmap[mask] = prob_sum[mask] / prob_count[mask]
        
        return heatmap
    
    def greedy_decode(self, heatmap, edge_index, num_nodes):
        """Greedy decoding từ heatmap xác suất → Clique hợp lệ 100%."""
        G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
        
        # Sắp xếp đỉnh theo xác suất giảm dần
        sorted_nodes = torch.argsort(heatmap, descending=True).tolist()
        
        clique = []
        candidates = set(range(num_nodes))
        
        for node in sorted_nodes:
            if node not in candidates:
                continue
            clique.append(node)
            # Chỉ giữ lại đỉnh kề với node vừa thêm
            neighbors = set(G.neighbors(node))
            candidates = candidates & neighbors
            if not candidates:
                break
        
        return clique
    
    def solve(self, edge_index, num_nodes):
        """Pipeline hoàn chỉnh: Patch inference → Greedy decode → Clique."""
        heatmap = self.inference(edge_index, num_nodes)
        clique = self.greedy_decode(heatmap, edge_index, num_nodes)
        return clique, heatmap
```

### 4.5. Vị trí tích hợp trong repo
- **File mới:** `src/inference/patch_inference.py`
- **Script chạy:** `scripts/run_dimacs_benchmark.py`

---

## 5. Đóng góp 3: Ablation Transferability 3 chế độ

### 5.1. Thiết kế thực nghiệm

Trên MỖI đồ thị DIMACS, chạy 3 chế độ rồi so sánh:

```
Chế độ A: RANDOM INIT + GREEDY
────────────────────────────────
- GNN với trọng số khởi tạo ngẫu nhiên (KHÔNG pretrain, KHÔNG train).
- Chạy forward pass → heatmap ngẫu nhiên → greedy decode.
- Mục đích: Mốc sàn (lower bound). Chứng minh greedy thuần túy không đủ.

Chế độ B: PRETRAINED + ZERO-SHOT + GREEDY  ← ĐÂY LÀ ĐÓNG GÓP CHÍNH
─────────────────────────────────────────────────────────────────────
- GNN pretrain trên đồ thị nhỏ (RB-small hoặc BA-small, seed 12345).
- KHÔNG fine-tune trên DIMACS. Chỉ chạy forward pass (zero-shot).
- Dùng Patch-based Inference → heatmap → greedy decode.
- Mục đích: Chứng minh pretrained knowledge có chuyển giao thực sự.

Chế độ C: PRETRAINED + FEW-SHOT FINE-TUNE + GREEDY
───────────────────────────────────────────────────
- GNN pretrain + fine-tune UNSUPERVISED 5–10 epochs trên chính instance DIMACS đó.
- Dùng complement-free MaxClique loss (không cần label, không cần G_bar).
- Sau fine-tune → Patch-based Inference → heatmap → greedy decode.
- Mục đích: Trần trên (upper bound). Cho thấy fine-tune thêm vài epochs đã đủ.
```

### 5.2. Kết quả mong đợi & Câu chuyện khoa học

```
Kịch bản lý tưởng (và rất có khả năng xảy ra):

  Clique Size:  A < B << C ≈ hoặc > Gurobi(10s)

  → B >> A: Pretrain có giá trị chuyển giao thực sự (NOVELTY chính).
  → C > B:  Fine-tune thêm vài epochs thu hẹp gap.
  → C ≥ Gurobi(10s): Trong cùng ngân sách thời gian, phương pháp của ta thắng.
```

### 5.3. Anytime Performance Plot

Vẽ biểu đồ **Solution Quality vs Wall-clock Time** trên mỗi instance:

```
  Clique
  Size ▲
       │
   k*  │· · · · · · · · · · · · · · · · · ·─────── Known Optimum
       │                         ╭──────────── Gurobi (chậm nhưng tìm ra optimum)
       │                    ╭────╯
       │               ╭────╯
       │          ╭────╯
       │     ╭────╯
       │████████████████████████████████████ ← Chế độ C (Pretrained + FT)
       │
       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ← Chế độ B (Pretrained zero-shot)
       │
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ← Chế độ A (Random init)
       │
       └──────────────────────────────────────► Time (seconds)
          0.1s   1s    10s    60s    300s
```

---

## 6. Thiết kế thực nghiệm chi tiết

### 6.1. Tập dữ liệu DIMACS (≥25 instances)

| Nhóm | Instances | N (đỉnh) | Density | Known Max Clique |
|-------|-----------|-----------|---------|------------------|
| c (random) | c125.9, c250.9, c500.9, c1000.9, c2000.5 | 125–2000 | 0.5–0.9 | 34, 44, 57, 68, 16 |
| brock | brock200_2, brock200_4, brock400_2, brock400_4, brock800_2 | 200–800 | 0.5–0.75 | 12, 17, 29, 33, 24 |
| p_hat | p_hat300-1, p_hat300-2, p_hat300-3, p_hat500-1, p_hat500-2 | 300–500 | 0.25–0.75 | 8, 25, 36, 9, 36 |
| MANN | MANN_a9, MANN_a27, MANN_a45 | 45–1035 | 0.93–0.99 | 16, 126, 345 |
| keller | keller4, keller5 | 171, 776 | 0.65, 0.75 | 11, 27 |
| san | san200_0.7_1, san200_0.9_1, san400_0.7_1 | 200–400 | 0.7–0.9 | 30, 70, 40 |
| hamming | hamming6-2, hamming8-4 | 64, 256 | 0.90, 0.96 | 32, 16 |

Nguồn tải: https://iridia.ulb.ac.be/~fmascia/maximum_clique/DIMACS-benchmark

### 6.2. Baselines so sánh

| Baseline | Loại | Thời gian | Ghi chú |
|----------|------|-----------|---------|
| **Gurobi** (TimeLimit=10s) | Exact solver | 10s | Nghiệm tốt nhất tìm được trong 10s |
| **Gurobi** (TimeLimit=60s) | Exact solver | 60s | Nghiệm tốt nhất tìm được trong 60s |
| **Gurobi** (Unlimited) | Exact solver | Không giới hạn | Known optimum (tham chiếu) |
| **NetworkX greedy** | Heuristic cổ điển | <1s | `nx.find_cliques` hoặc greedy thuần túy |
| **COPT gốc (với G_bar)** | GNN + complement | — | Ghi nhận OOM / thời gian / kích thước |
| **Chế độ A** (Random init) | GNN random + greedy | <1s | Mốc sàn cho ablation |
| **Chế độ B** (Pretrained zero-shot) | GNN pretrained + greedy | <1s | **Đóng góp chính** |
| **Chế độ C** (Pretrained + FT) | GNN pretrained + FT + greedy | <30s | Trần trên |

### 6.3. Metrics báo cáo

Với mỗi instance, báo cáo bảng sau:

| Instance | N | |E| | Density | Optimum | Gurobi(10s) | Gurobi(60s) | Mode A | Mode B | Mode C | Time(B) | Time(C) |
|----------|---|-----|---------|---------|-------------|-------------|--------|--------|--------|---------|---------|

Kèm theo:
- Approximation Ratio = Clique_found / Known_Optimum (cho mỗi phương pháp).
- GPU Peak Memory (MB) cho mỗi phương pháp.
- Anytime plot cho ≥3 instances đại diện.

---

## 7. Kế hoạch triển khai code

### 7.1. Các file cần tạo mới

| File | Mô tả |
|------|--------|
| `src/models/loss/direct_maxclique.py` | Hàm complement-free MaxClique loss |
| `src/inference/__init__.py` | Package mới cho inference pipeline |
| `src/inference/patch_inference.py` | Class PatchBasedInference (code ở mục 4.4) |
| `src/inference/greedy_decoder.py` | Greedy Clique decoder + Local Search |
| `scripts/run_dimacs_benchmark.py` | Script chạy toàn bộ benchmark 3 chế độ |
| `scripts/download_dimacs.py` | Script tải tự động dữ liệu DIMACS |
| `configs/data/dimacs_full.yaml` | Config mở rộng cho toàn bộ DIMACS instances |
| `configs/experiment/dimacs_transfer.yaml` | Config thực nghiệm transferability |

### 7.2. Các file cần sửa

| File | Thay đổi |
|------|----------|
| `src/models/copt_module.py` | Thêm option chọn `direct_maxclique_loss` |
| `src/models/discretizer.py` | Thêm `GreedyCliqueDiscretizer` |
| `src/data/datasets/dimacs_dataset.py` | Mở rộng hỗ trợ thêm định dạng `.clq`, `.col` |
| `configs/data/dimacs.yaml` | Mở rộng `instance_names` |

### 7.3. Thứ tự triển khai (Ưu tiên từ trên xuống)

```
Tuần 1: Nền tảng
  ✦ [1.1] Implement direct_maxclique_loss (30 phút)
  ✦ [1.2] Download toàn bộ DIMACS dataset (1 giờ)
  ✦ [1.3] Verify: chạy thử loss mới trên c125.9 (1 giờ)
  ✦ [1.4] Implement Greedy Clique Decoder (1 giờ)

Tuần 2: Patch-based Inference
  ✦ [2.1] Implement PatchBasedInference class (2 giờ)
  ✦ [2.2] Test trên c125.9, c250.9 với pretrained checkpoint (2 giờ)
  ✦ [2.3] Chạy Mode A (Random Init) trên toàn bộ DIMACS (2 giờ)
  ✦ [2.4] Chạy Mode B (Pretrained Zero-shot) trên toàn bộ DIMACS (2 giờ)

Tuần 3: Fine-tuning & So sánh
  ✦ [3.1] Implement unsupervised fine-tuning loop trên DIMACS instance (2 giờ)
  ✦ [3.2] Chạy Mode C (Pretrained + FT) trên toàn bộ DIMACS (4 giờ)
  ✦ [3.3] Chạy Gurobi baseline (cài gurobipy, TimeLimit=10s, 60s) (3 giờ)
  ✦ [3.4] Tổng hợp bảng kết quả + vẽ Anytime Plot (2 giờ)
```

---

## 8. Phản biện đã lường trước & Rebuttal

### Q1: "Complement-free loss không mới"
**A:** Đóng góp không nằm ở công thức mà ở việc chỉ ra paper gốc đã không tận dụng — họ sinh tường minh G_bar gây OOM. Đây là engineering contribution cho phép mở rộng lên DIMACS.

### Q2: "So sánh với Gurobi không công bằng"
**A:** Chúng tôi dùng Anytime Plot (tiêu chuẩn vàng trong OR+ML). Không claim "tốt hơn Gurobi mọi lúc" mà claim "trong regime real-time (t < 10s), GNN pretrained cho nghiệm tốt hơn".

### Q3: "Structural Gating có thể gây hại"
**A:** Đã loại bỏ hard-gating khỏi proposal. Thay bằng Patch-based Inference (không loại bỏ đỉnh nào, chỉ chia mảnh). Ablation study sẽ chứng minh.

### Q4: "Thiếu phân tích Transferability"
**A:** Ablation 3 chế độ (A/B/C) chính là trung tâm bài báo. Nếu B >> A: chứng minh pretrain có giá trị chuyển giao thực sự.

### Q5: "Greedy decoding không mới"
**A:** Greedy là thành phần phụ trợ, không phải đóng góp chính. Đóng góp chính là Patch-based Inference + Transferability ablation.

### Q6: "Thiếu Approximation Ratio"
**A:** Báo cáo Clique_found / Known_Optimum cho mỗi instance. Không claim worst-case guarantee (vì MaxClique là inapproximable). Đóng góp là thực nghiệm trên distribution cụ thể.

### Q7: "Thiếu đồ thị DIMACS"
**A:** Mở rộng lên ≥25 instances phủ tất cả nhóm cấu trúc (c, brock, p_hat, MANN, keller, san, hamming).

---

## GHI CHÚ CUỐI

File này được thiết kế để đưa trực tiếp vào AI/Agent lập trình.
Mọi code snippet đều tương thích với repo COPT-MT hiện tại (PyTorch + PyG + Hydra + Lightning).
Checkpoint pretrain có sẵn tại: `logs/train/checkpoints/` (MIS trên RB-small, Multi-task trên BA-small).
