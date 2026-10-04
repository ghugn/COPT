# COPT-MT: Foundation Model cho Bài toán Tối ưu Tổ hợp (Combinatorial Optimization) Dựa trên 21 Bài toán Karp

---

## 1. Giới thiệu & Ý tưởng Cốt lõi của Bài báo

### 1.1 Bối cảnh và Thách thức
Các bài toán **Tối ưu hóa Tổ hợp (Combinatorial Optimization - CO)** như Maximum Clique, Maximum Independent Set (MIS), Minimum Vertex Cover (MVC), Max-Cut... đều là các bài toán **NP-hard/NP-complete**. 
- Trong học máy truyền thống cho CO, mỗi bài toán thường yêu cầu thiết kế một mô hình riêng biệt và huấn luyện từ đầu (train from scratch) trên tập dữ liệu tương ứng.
- Khi gặp một bài toán tối ưu mới hoặc một phân phối đồ thị mới, toàn bộ quá trình huấn luyện phải lặp lại từ đầu, gây lãng phí tài nguyên tính toán và không tận dụng được tri thức cấu trúc chung.

### 1.2 Ý tưởng Chuyển giao Tri thức Dựa trên 21 Bài toán của Karp
Năm 1972, Richard M. Karp chứng minh rằng 21 bài toán kinh điển (Karp's 21 NP-complete problems) đều có thể **quy dẫn đa thức (polynomial-time reduction)** lẫn nhau. Ví dụ:
- **MIS & Maximum Clique**: Tập độc lập cực đại trên đồ thị $G$ chính là clique cực đại trên đồ thị bù $\overline{G}$.
- **MIS & MVC**: Tập đỉnh $S$ là Vertex Cover của $G$ khi và chỉ khi $V \setminus S$ là Independent Set của $G$ ($|MVC| = |V| - |MIS|$).
- **Max-Cut & MIS / QUBO**: Đều có thể biểu diễn qua ma trận Hamiltonian hoặc bài toán Quadratic Unconstrained Binary Optimization (QUBO).

**Ý tưởng trung tâm của dự án COPT-MT**:
> Thay vì huấn luyện các mô hình đơn lẻ, ta huấn luyện một **Foundation Graph Neural Network (GNN)** đa nhiệm (Multi-Task Learning) trên một tập các bài toán tối ưu tổ hợp cơ sở. Nhờ khả năng biểu diễn cấu trúc đồ thị mạnh mẽ, backbone của mô hình học được các đặc trưng tô-pô sâu sắc. Từ đó, ta có thể **chuyển giao tri thức (Transfer Learning)** sang các bài toán tối ưu khác hoặc đồ thị mới thông qua:
> 1. **Zero-shot / Freeze Backbone**: Đóng băng toàn bộ thân mô hình, chỉ huấn luyện một projection head mỏng (Linear Probing).
> 2. **Fine-tuning**: Tinh chỉnh toàn bộ mô hình với learning rate nhỏ trên bài toán mục tiêu.

---

## 2. Kiến trúc Tổng thể Hệ thống

Hệ thống được xây dựng trên nền tảng **PyTorch Lightning** kết hợp bộ quản lý cấu hình phân tầng **Hydra** (`lightning-hydra-template`).

```
                    +------------------------------------+
                    |        Input Graph Data (PyG)      |
                    |  (BA, RB, ER, DIMACS, GNNBenchmark)|
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |         Pre-transforms &           |
                    |    Graph Structural Statistics     |
                    | (Degree, Clustering, Triangles...) |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |       Backbone Network (GNN)       |
                    |  (HybridGNN / MultiHybridGNN)      |
                    |  - Feature Encoder                 |
                    |  - Pre-MP MLP                      |
                    |  - Message Passing (GCON/GCN/GIN)  |
                    |  - Concat/Sum Stage Aggregation    |
                    +-----------------+------------------+
                                      |
         +----------------------------+----------------------------+
         |                                                         |
         v                                                         v
[Single-Task Head]                                       [Multi-Task Heads]
- Node/Inductive Head                                    - Task Head: MaxCut
- Output: Soft assignments $x_i \in [0, 1]$               - Task Head: MaxClique
         |                                               - Task Head: MIS
         v                                               - Task Head: MVC, MDS...
[Unsupervised Hamiltonian Loss]                                    |
(QUBO / Penalty for constraint violations)                         v
         |                                       [MTL Strategy: Sum / Alt / PCGrad]
         v                                                         |
[Gradient Optimization (AdamW)] <----------------------------------+
```

---

## 3. Các Thành phần Chi tiết trong Pipeline

### 3.1 Dữ liệu & Tiền xử lý (`src/data/`)

1. **Bộ sinh đồ thị tổng hợp (`SyntheticDataModule`)**:
   - **BA (Barabási–Albert)**: Đồ thị mạng tỷ lệ tự do (scale-free network), đặc trưng bởi sự xuất hiện của các nút hub.
   - **RB (Xu-Li Model)**: Đồ thị chuẩn benchmark cho bài toán Maximum Clique / MIS với độ khó thuật toán được kiểm soát.
   - **ER (Erdős–Rényi)**: Đồ thị ngẫu nhiên đồng nhất.
   - **DIMACS**: Tập dữ liệu benchmark thực tế cho Maximum Clique.
2. **Trích xuất đặc trưng cấu trúc tô-pô (`GraphStatsEncoder`)**:
   - `degree`: Bậc của từng nút.
   - `cluster_coefficient`: Hệ số phân cụm cục bộ.
   - `triangle_count`: Số tam giác chứa nút.
   - `eccentricity`: Độ lệch tâm (khoảng cách lớn nhất tới nút khác).
   - Các chỉ số này được chiếu qua MLP/Embedding để cung cấp thông tin vị trí & cấu trúc ban đầu cho GNN (Structural Encodings).

### 3.2 Mô hình Mạng Nơ-ron (`src/models/network/`)

1. **`FeatureEncoder`**: Ánh xạ đặc trưng thô của nút và các thống kê đồ thị thành vector nhúng ẩn $d_{inner}$.
2. **`GCONConv` (Graph Convolutional Operator with Scattering)**:
   - Tận dụng toán tử khuếch tán đa bước $\mathbf{X}' = (\mathbf{\hat{D}}^{-1/2}\mathbf{\hat{A}}\mathbf{\hat{D}}^{-1/2})^K \mathbf{X}\mathbf{\Theta}$ với nhiều kênh (channels) $[0, 1, 2, 4]$.
   - Sử dụng cơ chế Attention (`att_bias`) để kết hợp thông tin đa bước nhảy, giúp mô hình bắt được tương tác tầm xa mà không bị over-smoothing.
3. **`GNNConcatStage` / `HybridGNN`**:
   - Nối (concatenate) biểu diễn của các nút qua tất cả các lớp message passing (`batch.x_list`), giữ lại cả thông tin cục bộ lẫn toàn cục trước khi đưa vào Prediction Head.
4. **`MultiHybridGNN`**:
   - Chia sẻ chung toàn bộ Backbone (Encoder + Message Passing Stack).
   - Tách ra các **Task Heads** riêng biệt (`self.post_mp = nn.ModuleDict({...})`) cho từng bài toán tối ưu.

### 3.3 Hàm Mất mát Không Giám sát (Unsupervised Losses) (`src/models/loss/copt_loss.py`)

Mô hình không cần nhãn ground-truth (NP-hard nên rất đắt để gán nhãn). Thay vào đó, mô hình tối ưu trực tiếp **hàm mục tiêu liên tục hóa (Relaxed Hamiltonian / QUBO)**:

1. **Max-Cut**:
   $$\mathcal{L}_{\text{MaxCut}} = \frac{1}{|B|} \sum_{(u, v) \in E} (2x_u - 1)(2x_v - 1)$$
   (Tối thiểu hóa tích này đồng nghĩa với việc tối đa hóa số cạnh có hai đầu mút khác dấu).
2. **Maximum Clique**:
   $$\mathcal{L}_{\text{MaxClique}} = -\alpha \sum_{u \in V} x_u + \beta \sum_{(u, v) \notin E} x_u x_v$$
   (Tối đa hóa số đỉnh được chọn, phạt nặng nếu giữa hai đỉnh được chọn không có cạnh).
3. **Maximum Independent Set (MIS)**:
   $$\mathcal{L}_{\text{MIS}} = -\alpha \sum_{u \in V} x_u + \beta \sum_{(u, v) \in E} x_u x_v \quad (\text{với } \beta > \alpha)$$
   (Tối đa hóa kích thước tập hợp, phạt nặng nếu hai đỉnh kề nhau cùng được chọn).
4. **Minimum Vertex Cover (MVC)**:
   $$\mathcal{L}_{\text{MVC}} = \alpha \sum_{u \in V} x_u + \beta \sum_{(u, v) \in E} (1 - x_u)(1 - x_v)$$
   (Tối thiểu hóa số đỉnh, phạt nếu cạnh $(u, v)$ có cả 2 đầu mút đều không được chọn).
5. **Minimum Dominating Set (MDS)** & **Graph Coloring (Color)** & **Hamiltonian Cycle (HCP)**:
   Cũng được thiết lập dưới dạng hàm phạt vi phạm ràng buộc liên tục.

### 3.4 Chiến lược Huấn luyện Đa nhiệm (Multi-Task Learning) (`MultiCOPTModule`)

Khi huấn luyện trên nhiều bài toán cùng lúc:
- **`strategy: sum`**: Cộng có trọng số các loss của từng bài toán: $\mathcal{L} = \sum_t w_t \mathcal{L}_t$.
- **`strategy: alternate`**: Luân phiên huấn luyện từng bài toán qua mỗi batch/epoch.
- **`strategy: pcgrad` (Projected Conflicting Gradients)**: Khi gradient giữa hai nhiệm vụ xung đột (tích vô hướng $< 0$), gradient của nhiệm vụ này sẽ được chiếu vuông góc lên mặt phẳng trực giao với gradient nhiệm vụ kia để loại bỏ thành phần triệt tiêu lẫn nhau.

### 3.5 Cơ chế Chuyển giao & Fine-tuning (`COPTTransferModule` & `MultiHybridGNN`)

Khi đã có checkpoint mô hình pre-trained (ví dụ trên MaxCut hoặc MaxClique):
1. **`linear_probing`**:
   - Đóng băng hoàn toàn Backbone: `encoder.requires_grad = False`, `mp.requires_grad = False`.
   - Chỉ khởi tạo và huấn luyện Head mới cho bài toán đích (ví dụ MIS hoặc MVC).
2. **`finetuning`**:
   - Nạp trọng số backbone từ checkpoint.
   - Cho phép toàn bộ mô hình tiếp tục cập nhật với bài toán mới.
3. **`pre_post`**:
   - Đóng băng các lớp message passing (`mp`), chỉ huấn luyện lại `pre_mp` và `post_mp`.

---

## 4. Hướng dẫn Vận hành Thực tế

Môi trường đã được cài đặt hoàn chỉnh với Python 3.11 trong thư mục `.venv`.

### 4.1 Kích hoạt Môi trường
Trong PowerShell:
```powershell
$env:Path = "C:\Users\ADMIN\.local\bin;$env:Path"
.\.venv\Scripts\Activate.ps1
```

### 4.2 Huấn luyện Đơn nhiệm (Single-Task Training)
Huấn luyện GCON trên bài toán Maximum Clique với dữ liệu `rb_small`:
```bash
python src/train.py data=rb_small task=maxclique model=gcon trainer.max_epochs=20
```

### 4.3 Huấn luyện Đa nhiệm Foundation Model (Multi-Task Pretraining)
Huấn luyện đồng thời trên MaxCut và MIS:
```bash
python src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[maxcut,mis] trainer.max_epochs=50
```

Với trọng số tùy chỉnh cho từng loss:
```bash
python src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[maxcut,mis] model.weights.maxcut=0.7 model.weights.mis=0.3
```

Với giải thuật giải xung đột gradient PCGrad:
```bash
python src/train.py experiment=multitask/ba_small/gcon model.net.tasks=[maxcut,mis] model.strategy=pcgrad
```

### 4.4 Chuyển giao Tri thức & Fine-tuning (Transfer Learning)
Sau khi đã có checkpoint pre-train tại `logs/train/checkpoints/maxcut/last.ckpt`:

1. **Linear Probing** (Freeze Backbone, chỉ train Head cho bài toán MIS mới):
```bash
python src/train.py experiment=multitask/ba_small/gcon model.net.finetuning.strategy=linear_probing model.net.finetuning.new_tasks=[mis] model.net.finetuning.path=logs/train/checkpoints/maxcut/last.ckpt
```

2. **Full Fine-tuning**:
```bash
python src/train.py experiment=multitask/ba_small/gcon model.net.finetuning.strategy=finetuning model.net.finetuning.new_tasks=[mis] model.net.finetuning.path=logs/train/checkpoints/maxcut/last.ckpt
```

### 4.5 Đánh giá Checkpoint (Evaluation)
```bash
python src/eval.py ckpt_path=logs/train/checkpoints/maxcut/last.ckpt
```

### 4.6 Bộ giải Đối chuẩn Ground-Truth (`solver.py`)
Tính toán nghiệm tối ưu chính xác (exact solution) thông qua quy dẫn NetworkX complement graph:
```bash
python solver.py
```

---

## 5. Tóm tắt các Thay đổi Đã Thực hiện để Dự án Hoạt động

1. **Môi trường**: Khởi tạo virtual environment Python 3.11 chuẩn hóa qua công cụ `uv`, cài đặt thành công 172 thư viện gồm `torch 2.14.1`, `torch-geometric 2.8.0`, `lightning 2.6.6`, `dimod`, `dwave-networkx`, `scikit-learn`, `numba`, v.v.
2. **Khắc phục phụ thuộc C++ `torch_scatter` trên Windows**:
   - Thêm cơ chế fallback an toàn sang `torch_geometric.utils.scatter` trong các file `src/models/loss/copt_loss.py`, `src/utils/metrics.py`, `src/utils/utils_graphgym.py`.
3. **Đồng bộ hóa Cấu hình Đặc trưng Tô-pô**:
   - Điều chỉnh `configs/model/base.yaml` để ràng buộc động trường `stat_list` theo `${data.graph_stats}`, tránh xung đột thiếu thuộc tính (`eccentricity`) giữa dataset và model encoder.
4. **Kiểm thử Xác thực (Smoke Test)**:
   - Đã chạy thành công cả single-task pipeline (`debug=fdr`) lẫn multi-task pipeline (`multitask/ba_small/gcon`) qua đầy đủ các vòng lặp: Data loading $\to$ Pre-transform $\to$ Forward pass $\to$ Loss $\to$ Backward pass $\to$ Validation $\to$ Testing.
