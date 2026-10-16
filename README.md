# COPT-MT research workspace

Không gian nghiên cứu sạch dựa trên mã nguồn chính thức của
[COPT-MT](https://github.com/semihcanturk/COPT-MT), upstream commit
`ffa19d01984e178252fd874c8a956664f93a186d`.

`src/`, `configs/`, `run/`, `solver.py` và `pyproject.toml` được giữ nguyên từ
upstream. Những thử nghiệm mới nên được phát triển tách biệt và ghi rõ thay đổi
so với baseline.

## Cấu trúc thư mục

```text
.
|-- artifacts/
|   `-- checkpoints/mis_rb_small/epoch_189.ckpt
|-- configs/                    # Hydra configs của COPT-MT
|-- data/
|   `-- rb/small/processed/data.pt
|-- docs/
|   |-- papers/                 # Tài liệu PDF
|   `-- proposal/               # Proposal LaTeX và bản ZIP cho Overleaf
|-- experiments/guidance/       # Reference solver và paired replay benchmark mới
|-- run/                        # Job wrappers nguyên bản
|-- src/                        # Mã nguồn COPT-MT nguyên bản
|-- README_UPSTREAM.md          # README của tác giả
|-- pyproject.toml
`-- solver.py
```

`data/`, `artifacts/`, log huấn luyện và cache được bỏ khỏi Git để tránh commit
nhầm file lớn. Hiện tại cần giữ hai artifact cục bộ:

- `data/rb/small/processed/data.pt`: RB-small đã transform.
- `artifacts/checkpoints/mis_rb_small/epoch_189.ckpt`: checkpoint tốt nhất đã
  huấn luyện.

## Chạy baseline trên Kaggle

### 1. Kiểm tra GPU

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
```

Chỉ tiếp tục khi `CUDA available: True`. Không cài lại PyTorch nếu Kaggle đã
cung cấp CUDA build phù hợp.

### 2. Cài dependency

```python
%pip install -q \
    "hydra-core>=1.3,<1.4" \
    "hydra-colorlog>=1.2,<1.3" \
    "lightning>=2.5,<2.7" \
    "torch-geometric>=2.6,<2.8" \
    hydra-submitit-launcher rootutils networkx numba POT rich wandb \
    yacs einops dwave-networkx dimod ogb performer-pytorch python-sat
```

Cài PyG extensions theo đúng Torch/CUDA của session:

```python
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "Hãy bật GPU trước khi cài PyG extensions"

torch_version = torch.__version__.split("+")[0]
cuda_tag = "cu" + torch.version.cuda.replace(".", "")
wheel_index = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_tag}.html"

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "torch-scatter",
    "torch-sparse",
    "-f",
    wheel_index,
])
```

### 3. Dùng lại RB-small đã transform

Khi upload archive/dataset lên Kaggle, bảo đảm file cuối cùng nằm tại:

```text
/kaggle/working/COPT-MT/data/rb/small/processed/data.pt
```

### 4. Train lại khi cần

```python
%cd /kaggle/working/COPT-MT

!WANDB_MODE=disabled python -u src/train.py \
    experiment=mis/rb_small/gcon \
    logger=csv \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    data.num_workers=2
```

### 5. Checkpoint đã giữ

Checkpoint hiện tại:

```text
artifacts/checkpoints/mis_rb_small/epoch_189.ckpt
```

Trên Kaggle có thể đặt đường dẫn tuyệt đối tương ứng, ví dụ:

```python
CHECKPOINT = "/kaggle/working/COPT-MT/artifacts/checkpoints/mis_rb_small/epoch_189.ckpt"
```

## Proposal nghiên cứu

Proposal hiện hành nằm tại [`docs/proposal/main.tex`](docs/proposal/main.tex).
File [`docs/proposal/overleaf_proposal.zip`](docs/proposal/overleaf_proposal.zip)
có thể upload trực tiếp lên Overleaf. Hai proposal DIMACS cũ do LLM tạo đã được
loại bỏ để tránh nhầm với hướng nghiên cứu hiện tại.

Hướng hiện tại: học khi nào nên gọi COPT trong exact clique search, có tính chi phí
preprocessing và inference. Xem [hướng dẫn Kaggle](experiments/guidance/README.md).
Bước đầu là paired replay để đo dư địa cải thiện; chưa cần train lại checkpoint.
Ý tưởng optimal-continuation trước đó được lưu tại
`docs/proposal/archive/optimal_continuation_v1.tex` để đối chiếu.

## Nguyên tắc thực nghiệm

1. Giữ baseline upstream bất biến.
2. Ghi lại commit, seed, config, checkpoint và môi trường cho mỗi thí nghiệm.
3. So sánh learned ranking với random, degree, core và coloring trên cùng trạng
   thái tìm kiếm.
4. Tách rõ greedy/multi-start evaluation khỏi exact-search evaluation.
5. Luôn tính neural inference overhead khi báo cáo thời gian.
