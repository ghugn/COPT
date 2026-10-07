# COPT-MT — clean upstream baseline

Repository này đã được làm sạch và khôi phục từ source chính thức:

- Upstream: <https://github.com/semihcanturk/COPT-MT>
- Upstream commit: `ffa19d01984e178252fd874c8a956664f93a186d`
- `src/`, `configs/`, `run/`, `solver.py` và `pyproject.toml` được lấy nguyên trạng từ commit trên.
- README nguyên bản của tác giả được giữ tại [`README_UPSTREAM.md`](README_UPSTREAM.md).
- Các PDF nghiên cứu được giữ ở thư mục gốc.
- Hai đề xuất DIMACS cũ được giữ trong [`research_notes/`](research_notes/). Chúng là ghi chú ý tưởng, không phải implementation đã được xác minh.

Mục tiêu trước mắt là tái hiện baseline chính thức trên code sạch. Không thêm DIMACS, patch inference hoặc local search trước khi baseline RB-small khớp hợp lý với paper.

## Chạy từ đầu trên Kaggle

### 1. Tạo notebook và bật GPU

Trong **Notebook settings**, chọn **Accelerator → GPU**. Sau đó kiểm tra:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
```

Chỉ tiếp tục khi `CUDA available: True`. Không cài lại `torch` bằng pip vì có thể làm mất CUDA build của Kaggle.

### 2. Clone repository

```python
%cd /kaggle/working
!git clone https://github.com/ghugn/COPT.git COPT-MT
%cd /kaggle/working/COPT-MT
!git rev-parse HEAD
```

Không copy `data/*/processed`, `logs/` hoặc checkpoint từ lần chạy cũ vào clean run.

### 3. Cài dependency Python

```python
%pip install -q \
    "hydra-core>=1.3,<1.4" \
    "hydra-colorlog>=1.2,<1.3" \
    "lightning>=2.5,<2.7" \
    "torch-geometric>=2.6,<2.8" \
    hydra-submitit-launcher rootutils networkx numba POT rich wandb \
    yacs einops dwave-networkx dimod ogb performer-pytorch python-sat
```

### 4. Cài PyG extensions đúng Torch/CUDA

Không dùng một URL wheel viết cứng. Cell sau tự tạo index phù hợp với Torch/CUDA hiện tại:

```python
import subprocess
import sys
import torch

assert torch.cuda.is_available(), "Hãy bật GPU trước khi cài PyG extensions"

torch_version = torch.__version__.split("+")[0]
cuda_tag = "cu" + torch.version.cuda.replace(".", "")
wheel_index = (
    f"https://data.pyg.org/whl/"
    f"torch-{torch_version}+{cuda_tag}.html"
)
print("Wheel index:", wheel_index)

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

Nếu báo `No matching distribution found`, dừng lại và ghi lại ba giá trị `torch.__version__`, `torch.version.cuda` và phiên bản Python. Không nên tự compile hoặc đổi Torch trước khi xác định wheel tương thích.

Kiểm tra môi trường:

```python
import hydra
import lightning
import torch
import torch_geometric
import torch_scatter

print("Hydra:", hydra.__version__)
print("Lightning:", lightning.__version__)
print("Torch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

### 5. Train baseline MIS trên RB-small

Lệnh dưới đây dùng nguyên experiment config chính thức: seed `12345`, 6.000 RB-small graphs và tối đa 200 epochs.

```python
%cd /kaggle/working/COPT-MT

!WANDB_MODE=disabled python -u src/train.py \
    experiment=mis/rb_small/gcon \
    logger=csv \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    data.num_workers=2
```

Không override loss, decoder, batch size, số layer, learning rate hoặc số epoch trong clean reproduction đầu tiên.

### 6. Xác định best checkpoint

```python
from pathlib import Path

root = Path("/kaggle/working/COPT-MT")
checkpoints = sorted(
    root.glob("logs/train/runs/*/checkpoints/*.ckpt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

for checkpoint in checkpoints[:10]:
    size_mb = checkpoint.stat().st_size / 1024**2
    print(checkpoint, f"{size_mb:.1f} MB")
```

`epoch_XXX.ckpt` là checkpoint tốt nhất theo `val/size`; `last.ckpt` chỉ là trạng thái epoch cuối. Khi đánh giá paper, ưu tiên `epoch_XXX.ckpt`.

### 7. Lưu checkpoint khỏi Kaggle

```python
from IPython.display import FileLink

BEST_CHECKPOINT = next(p for p in checkpoints if p.name != "last.ckpt")
print("Selected:", BEST_CHECKPOINT)
FileLink(str(BEST_CHECKPOINT))
```

Bấm link được tạo để tải file về máy và **Save Version** notebook để kết quả trở thành Kaggle Output. `/kaggle/working` có thể mất khi session kết thúc.

## Quy trình nghiên cứu đề nghị

1. Chạy clean upstream với seed `12345` và lưu đầy đủ commit, config, metrics, checkpoint.
2. Kiểm tra baseline MIS RB-small trước khi đánh giá DIMACS.
3. Lặp lại các seed của paper nếu baseline một seed hợp lý.
4. Chỉ sau đó mới tạo nhánh riêng để thêm loader/evaluator DIMACS.
5. Mọi thay đổi so với upstream phải được ghi thành ablation riêng; không sửa trực tiếp baseline.

## Những gì không còn trong repository

Clean reset đã loại bỏ source thử nghiệm LLM, script reproduction tự viết, cache dữ liệu, logs, reports, checkpoint cũ, môi trường `.venv` và các file Word. Nếu cần khôi phục một nội dung cũ, lấy từ lịch sử Git thay vì trộn lại vào baseline sạch.
