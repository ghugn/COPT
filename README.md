# COPT-MT: Transferable Models for Graph Combinatorial Optimization

Implementation and benchmark reproduction of the paper:
> **"Can Computational Reducibility Lead to Transferable Models for Graph Combinatorial Optimization?"**  
> *Semih Cantürk, Frederik Wenkel, Michael Perlmutter, Guy Wolf*

> **Research status:** the active codebase uses the COPT-MT method only.
> Earlier LLM-generated patch-inference experiments are preserved under
> `archive/llm_dimacs_v0/` and are not part of the active scientific pipeline.
> See `docs/RESEARCH_SCOPE.md` for the boundary between reproduction and future
> extensions.

---

## 📌 Repository Structure

```text
COPT-MT/
├── configs/               # Hydra configuration files
│   ├── data/              # Dataset configs (ba_small, rb_small, dimacs, etc.)
│   ├── experiment/        # Experiment recipes (multitask, transfer, baselines)
│   ├── model/             # GCON, GNN backbones, discretizer, losses
│   └── trainer/           # PyTorch Lightning trainer configs
├── data/                  # Graph datasets (BA-small, RB-small, processed cache)
├── docs/                  # Documentation and reports
│   ├── papers/            # Original paper PDF & reference section texts
│   └── reports/           # Word docx reports & generation scripts
├── logs/                  # Pretrained model checkpoints (.ckpt) & metric logs
├── scripts/               # 1-click benchmark reproduction scripts
│   ├── run_table3.py      # Table 3: MIS ↔ MVC Pairwise Transfer on RB-small
│   ├── run_table4.py      # Table 4: MIS → MaxClique on RB-small
│   ├── run_table5.py      # Table 5: Leave-One-Out Fine-Tuning (20 epochs)
│   ├── run_table7.py      # Table 7: Multi-Task Foundation Model on BA-small
│   ├── generate_dataset.py# Graph generation utility
│   └── solver.py          # Exact solver baseline utility
├── src/                   # Core implementation
│   ├── data/              # PyG Datamodules and synthetic generation
│   ├── models/            # GCON module, Hamiltonian QUBO losses, discretizers
│   └── transforms/        # Graph statistics and feature transforms
├── pyproject.toml         # Python dependencies & build config
└── README.md              # Project guide
```

---

## 🚀 Quickstart & Installation

```bash
# Install dependencies with uv (or pip)
uv sync
uv pip install yacs einops dwave-networkx wandb ogb performer_pytorch python-sat
uv pip install torch_scatter torch_sparse --no-build-isolation
```

---

## 📊 1-Click Benchmark Reproduction

Each core table from the paper can be reproduced with a single command from the project root:

### 1. Table 7: Multi-Task Foundation Model on BA-small
Huấn luyện mô hình nền tảng đa nhiệm trên cả 6 bài toán NP-hard cốt lõi (MaxCut, MaxClique, MDS, MIS, MVC, Coloring):
```bash
python -u scripts/run_table7.py
```

### 2. Table 3: Pairwise Transferability (MIS ↔ MVC) on RB-small
Khảo sát chuyển giao bảo toàn cấu trúc và kỹ thuật **Invert Head** ($p_{\text{MIS}} = 1 - p_{\text{MVC}}$) trên 6.000 đồ thị RB-small:
```bash
python -u scripts/run_table3.py
```

### 3. Table 5: Leave-One-Out Low-Resource Fine-Tuning (20 epochs)
Đánh giá khả năng chuyển giao siêu tốc trong điều kiện ít tài nguyên (20 epochs) so với train từ đầu:
```bash
python -u scripts/run_table5.py
```

### 4. Table 4: MIS → MaxClique Transfer via Complement Graphs
Khảo sát chuyển giao qua đồ thị bù $\bar{G}$ (MaxClique(G) = MIS($\bar{G}$)):
```bash
python -u scripts/run_table4.py
```

---

## 🏆 Tóm Tắt Kết Quả Tái Hiện Thực Tế

### Table 5: Leave-One-Out Fine-Tuning (BA-small, 20 epochs)
```text
================================================================================
TABLE 5: Leave-One-Out Fine-Tuning in Low-Resource Regime on BA-small (20 epochs)
================================================================================
TASK                 FROM SCRATCH            FINE-TUNED
--------------------------------------------------------------------------------
↑ MAXCUT             718.92                  720.12
↑ MAXCLIQUE          4.33                    4.36
↓ MDS                34.54                   29.59
↑ MIS                111.14                  111.67
↓ MVC                140.83                  139.65
↓ COLOR              57.47                   18.91
================================================================================
```

### Table 3: MIS ↔ MVC Pairwise Transfer (RB-small, 6.000 graphs)
* **Baseline train từ đầu:** MIS = 18.31 (Paper: 18.12), MVC = 213.14 (Paper: 211.69)
* **FT Invert + FT:** MIS = 18.01, MVC = 212.06 (Đánh bại Baseline từ đầu 213.14)

---

## 📑 Báo Cáo & Tài Liệu Nghiên Cứu
* **Báo cáo tóm tắt:** [docs/reports/Bao_Cao_Y_Nghia_Benchmark_COPT.docx](docs/reports/Bao_Cao_Y_Nghia_Benchmark_COPT.docx)
* **Tài liệu phản biện & phân tích:** [docs/reports/Giai_thich_Bench.docx](docs/reports/Giai_thich_Bench.docx)
* **Script cập nhật báo cáo:** [docs/reports/generate_report_docx.py](docs/reports/generate_report_docx.py)
