# BẢN THIẾT KẾ CHIẾN LƯỢC: HỢP NHẤT 3 Ý TƯỞNG ĐỘT PHÁ ĐỂ GIẢI BÀI TOÁN MAXCLIQUE/MIS TRÊN DIMACS

> **Tài liệu chiến lược học thuật:** Đúc kết và hợp nhất 3 phát kiến từ các công trình mới nhất tại **ICLR 2026** và **ICML 2026** nhằm giải quyết dứt điểm các giới hạn của mô hình nền tảng đồ thị (*Cantürk et al. - COPT*), mở rộng thành công lên bộ benchmark thế giới thực **DIMACS** và thiết lập lợi thế cạnh tranh trước **Gurobi**.

---

## I. TỔNG HỢP 3 "VŨ KHÍ TƯ DUY" TỪ CÁC HỘI NGHỊ HÀNG ĐẦU (ICLR / ICML 2026)

| Công trình / Ý tưởng | Bản chất cốt lõi | Giá trị ứng dụng vào bài toán DIMACS |
|---|---|---|
| **1. NExCO (ICLR 2026)**<br>*Native Solution Expansion* | Không sinh nghiệm một lần (*Global Prediction*) mà **mở rộng dần nghiệm bộ phận** kết hợp **toán tử chiếu khả thi $\Gamma(\cdot)$**. Mọi trạng thái trung gian đều 100% hợp lệ. | Cung cấp thuật toán giải mã từng bước, triệt tiêu hoàn toàn vi phạm ràng buộc trên các đồ thị dày đặc của DIMACS. |
| **2. ASAP (ICML 2026)**<br>*Satisficing Generalization Edge* | Dự đoán **một tập hợp ứng viên tiềm năng (*Candidate Set*)** có tính tổng quát hóa bền vững qua các miền dữ liệu hơn nhiều so với việc cố đoán đúng duy nhất 1 nghiệm tối ưu. | Giải phóng áp lực cho GNN: GNN chỉ cần đóng vai trò **bộ lọc khoanh vùng (*Proposal*)**, không cần cố dự đoán chính xác nhãn nhị phân 0-1. |
| **3. Certified Correctness (ICML 2026)**<br>*Symbolic Integration* | Khi tồn tại ràng buộc cứng, **bắt buộc phải tích hợp phương pháp biểu tượng (*Symbolic Verification*)**, không thể phó mặc cho mô hình nơ-ron thuần túy (*Pure Neural Network*). | Cung cấp luận điểm lý luận vững chắc trước Reviewer: GNN định hướng heuristic toàn cục + Symbolic chốt nghiệm khả thi 100%. |

---

## II. ĐIỂM NGHẼN CỦA ĐỀ TÀI & CÁCH 3 Ý TƯỞNG TRÊN GIẢI QUYẾT TRIỆT ĐỂ

| Điểm nghẽn của COPT gốc | Hậu quả thực tế trên DIMACS | Cơ chế khắc phục từ 3 ý tưởng |
|---|---|---|
| **1. Ma trận bù $\bar{G}$ cồng kềnh** | Đồ thị bù làm số cạnh bùng nổ lên hàng triệu $\to$ **GPU tràn RAM (OOM)** ngay lập tức. | **Loại bỏ 100% ma trận bù**, tính toán trực tiếp trên đồ thị gốc $G$ thưa thớt thông qua công thức đại số đóng (*Complement-Free*). |
| **2. Sinh Heatmap 1 lần (*Global Prediction*)** | Xác suất bị mờ (~0.5), làm tròn nhị phân gây **xung đột ràng buộc dữ dội**. | **Áp dụng NExCO & ASAP:** Coi GNN là bộ lọc đề xuất ứng viên (*Proposal*), sau đó dùng toán tử chiếu khả thi (*Feasibility Projector*) mở rộng nghiệm dần dần. |
| **3. Reviewer hoài nghi tính chính xác** | GNN chỉ là thuật toán xấp xỉ, bị đặt dấu hỏi khi so sánh với Gurobi. | **Áp dụng Certified Correctness & Anytime Plot:** Chứng minh trong khung thời gian hẹp ($t < 10s$), GNN + Symbolic cho nghiệm vượt trội Gurobi bị timeout. |

---

## III. KIẾN TRÚC HỆ THỐNG ĐỀ XUẤT: P-SAFE
*(**P**atch-based **S**atisficing **A**daptive **F**easible **E**xpansion)*

Toàn bộ quy trình giải bài toán **MaxClique / MIS trên DIMACS** được hợp nhất qua **3 tầng logic liên hoàn**:

```text
               ĐỒ THỊ DIMACS LỚN G = (V, E) (Hàng nghìn đỉnh)
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 1: PATCH-BASED PROPOSAL (Kế thừa tư tưởng ASAP)                       │
│  - Chia đồ thị lớn thành các mảnh con (Patches) ~200-300 đỉnh               │
│    (Đúng bằng kích thước mô hình nền tảng COPT đã pretrain).                │
│  - GNN suy luận song song trên từng mảnh (KHÔNG dùng ma trận bù).           │
│  - Tổng hợp lại thành một Heatmap độ tự tin toàn cục p_v ∈ [0, 1].          │
│  ==> Vai trò: "Màng lọc thô" bền vững, khoanh vùng top 20-30% đỉnh tiềm năng│
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 2: NATIVE FEASIBILITY PROJECTION (Kế thừa tư tưởng NExCO)              │
│  - Bắt đầu từ tập Clique rỗng C = ∅.                                        │
│  - Lấy các ứng viên có độ tự tin cao nhất từ Tầng 1.                        │
│  - Chiếu khả thi: Duyệt tuần tự theo độ tự tin giảm dần.                    │
│    Chỉ kết nạp đỉnh u vào C nếu u có cạnh nối với TẤT CẢ các đỉnh đã có.    │
│  - Mở rộng dần C cho đến khi không thể thêm đỉnh nào nữa.                   │
│  ==> Vai trò: Nghiệm sinh ra đảm bảo 100% là một Clique hợp lệ (Zero vi phạm)│
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TẦNG 3: FAST SYMBOLIC REFINEMENT (Kế thừa tư tưởng Certified Correctness)  │
│  - Chạy thuật toán Local Search siêu nhanh (< 0.05 giây):                   │
│    Thực hiện các phép hoán đổi (1-swap / 2-swap) để tối đa hóa kích thước.  │
│  ==> Vai trò: Chốt nghiệm cuối cùng, tiệm cận nghiệm tối ưu lý thuyết       │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                      KẾT QUẢ: CLIQUE LỚN & FEASIBLE 100%
```

---

## IV. BẢN THIẾT KẾ THỰC NGHIỆM ĐỐI ĐẦU VỚI GUROBI & THUYẾT PHỤC REVIEWER

### 1. Trục 1: So sánh Anytime Performance với Gurobi
Thay vì chỉ so sánh tại một mốc thời gian tĩnh, đồ thị **Anytime Performance Plot** (*Chất lượng nghiệm vs Thời gian chạy Wall-clock*) sẽ chứng minh ưu thế vượt trội:

```text
  Clique Size
       ▲
       │                                     ╭─────── Gurobi (Exact Optimum)
       │                                ╭────╯
       │                           ╭────╯
       │                      ╭────╯
       │                 ╭────╯
    ★  │████████████████████████████████████ ← P-SAFE (Đạt đỉnh ngay từ t = 0.5s)
       │
       │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ← Random GNN + Projector
       │
       │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ← Pure Greedy Heuristic
       │
       └────────────────────────────────────────► Thời gian (Log-scale)
          0.1s         1s        10s        60s       300s
```

* **Vùng thời gian thực ($t \le 10s$):** Gurobi bị bế tắc ở thuật toán nhánh-cận (*Branch-and-Bound*) trên các đồ thị DIMACS khó (`c1000.9`, `brock800`, `MANN_a45`), trả về nghiệm rất nhỏ hoặc bị timeout. Mô hình P-SAFE hoàn tất suy luận trong $< 1s$, cho nghiệm áp đảo.
* **Vùng hội tụ dài hạn ($t > 300s$):** Gurobi dần tiệm cận nghiệm tối ưu tuyệt đối. Điều này hoàn toàn tự nhiên vì Gurobi là *Exact Solver*, nhưng trong ứng dụng công nghiệp thời gian thực, nghiệm nhanh trong 1 giây của P-SAFE có giá trị thực tiễn vượt trội.

### 2. Trục 2: Ablation Study chứng minh tính chuyển giao (Transferability)
Để chứng minh việc pretrain mô hình nền tảng trên đồ thị nhỏ mang lại giá trị thực tế chứ không phải chỉ là thuật toán heuristic ngẫu nhiên, thực nghiệm chia làm 3 chế độ trên $\ge 25$ đồ thị DIMACS:

1. **Mode A (Random GNN + Projector):** Trọng số nơ-ron ngẫu nhiên $\to$ Đóng vai trò mốc sàn (*Lower Bound*).
2. **Mode B (Pretrained COPT Foundation Model + Zero-shot Transfer + Projector):** **Trọng tâm đóng góp.** Mô hình pretrain từ đồ thị nhỏ giải trực tiếp DIMACS mà không qua huấn luyện lại.
3. **Mode C (Pretrained COPT + Fine-tune 5 epochs + Projector):** Trần trên (*Upper Bound*) khi cho phép thích ứng nhẹ.

* **Luận điểm khoa học cốt lõi:** Khi **Mode B $\gg$ Mode A**, ta có bằng chứng đanh thép khẳng định:
  > *"Biểu diễn không gian đồ thị học được từ mô hình nền tảng trên đồ thị tổng hợp nhỏ thực sự có tính chuyển giao (Transferability) mạnh mẽ sang các đồ thị thế giới thực khổng lồ!"*
