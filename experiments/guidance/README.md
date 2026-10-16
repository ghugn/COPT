# Paired neural-guidance pilot (M1)

Mục đích: đo **cùng state, cùng incumbent**, neural có giúp ích sau chi phí không.
Chưa train gate, chưa claim novelty hoặc speedup. Không cần RB transformed cho ER pilot.

## Cách hoạt động

- Solver Python exact dùng bitsets + greedy coloring bound tính lại trên remaining set.
- Snapshot root và reservoir internal states từ structural trace.
- Tại replay root: degree / random / COPT priority. Mọi descendant quay về degree.
- COPT chạy trên complement của induced candidate graph, score cao thử trước.
- Mỗi phương án cùng total time budget; neural preprocessing/inference trừ vào budget.
- Mỗi repetition tạo lại neural scores, synchronize CUDA, thứ tự phương án đảo theo seed.
- Model load báo riêng; inference đã warmup. Summary chỉ mô tả, không đủ kết luận thống kê.
- `exhausted` ở internal state chỉ chứng nhận subtree, không phải optimum toàn graph.
- Kết quả từng pair được flush vào JSONL. Nếu session mất, file chưa lưu ra ngoài vẫn có thể mất.
- Không resume vào cùng thư mục: dùng output khác để tránh trộn protocol.

## Chạy trên Kaggle

Upload `artifacts/guidance_kaggle_bundle.zip` dưới dạng Kaggle Dataset. Gói được
tạo cục bộ gồm code benchmark, proposal và checkpoint epoch 189; không gồm RB data
vì ER pilot không cần dữ liệu đã transform. Giải nén gói vào repo trên Kaggle.

```python
from pathlib import Path
import zipfile

bundles = list(Path('/kaggle/input').rglob('guidance_kaggle_bundle.zip'))
assert len(bundles) == 1, bundles
with zipfile.ZipFile(bundles[0]) as z:
    z.extractall('/kaggle/working/COPT-MT')  # gói do dự án tạo, chỉ experiments/ và docs/
```

Kiểm tra checkpoint của bạn (chỉ load checkpoint tin cậy):

```python
from pathlib import Path
import torch

assert torch.cuda.is_available(), (torch.__version__, torch.version.cuda)
candidates = sorted(set(Path('/kaggle/input').rglob('*189*.ckpt')) |
                    set(Path('/kaggle/working/COPT-MT/artifacts').rglob('*189*.ckpt')))
for p in candidates:
    print(p)
assert len(candidates) == 1, 'Chọn đúng một đường dẫn checkpoint epoch 189 từ danh sách'
CHECKPOINT = str(candidates[0])
```

Giữ môi trường đã chạy COPT thành công. Dependency của adapter cần torch, PyG,
torch-scatter, torch-sparse và các module upstream mà checkpoint tham chiếu.
Không thay torch/CUDA chỉ để chạy pilot. Nếu lỗi import, gửi traceback.

### Bước 1: correctness và smoke COPT

```python
%cd /kaggle/working/COPT-MT
!python -m unittest experiments.guidance.test_solver -v
!python -u -m experiments.guidance.pilot --checkpoint "{CHECKPOINT}" --device cuda --sizes 30 40 --densities 0.5 0.7 --graphs-per-cell 1 --states-per-graph 2 --repeats 2 --seconds 1 --output /kaggle/working/guidance_smoke_copt
```

### Bước 2: pilot M1, chỉ sau khi smoke thành công

```python
!python -u -m experiments.guidance.pilot --checkpoint "{CHECKPOINT}" --device cuda --sizes 40 60 80 120 --densities 0.3 0.5 0.7 --graphs-per-cell 5 --states-per-graph 4 --repeats 3 --seconds 2 --node-limit 200000 --output /kaggle/working/guidance_pilot_m1
```

Tối đa 60 graph, 300 state, 900 pair, 2700 lượt replay; số internal state có thể ít hơn.
Với cutoff 2 giây, riêng replay search khoảng tối đa 90 phút nếu mọi lượt timeout;
collector thêm tối đa khoảng 2 phút. Preprocessing/khởi tạo và I/O có thể tăng thời gian,
đặc biệt inference không bị ngắt giữa chừng. Thực tế graph dễ sẽ nhanh hơn nhiều.
Chưa chạy DIMACS lớn hoặc train thêm.

### Bước 3: tải kết quả

```python
import shutil
from pathlib import Path
from IPython.display import FileLink, display

output = Path('/kaggle/working/guidance_pilot_m1')
print((output / 'summary.json').read_text())
archive = shutil.make_archive('/kaggle/working/guidance_pilot_m1', 'zip', output)
%cd /kaggle/working
display(FileLink(Path(archive).name))
```

Nếu link tải không hoạt động, tải ZIP từ panel Output của Kaggle và lưu notebook output.
Gửi nguyên ZIP gồm `paired.jsonl`, `metadata.json`, `summary.json`, kể cả kết quả âm.

## Chạy trên Google Colab

Import `experiments/guidance/colab_pilot_m1.ipynb` vào Colab, bật T4 GPU và
upload `artifacts/guidance_kaggle_bundle.zip` khi notebook yêu cầu. Bundle dùng
chung cho Kaggle/Colab và đã chứa checkpoint epoch 189. Chạy đến smoke trước.
Pilot chính mount Google Drive và ghi từng paired row vào
`MyDrive/COPT-MT-results/guidance_pilot_m1_<timestamp>` để kết quả đã hoàn thành
không mất khi runtime Colab ngắt. Sau khi chạy xong, cell cuối nén và tải ZIP.

### M1b: kiểm tra tín hiệu ở root của đồ thị khó

M1 đã chạy xong; không chạy lại. Dùng
`experiments/guidance/colab_pilot_m1b.ipynb` để kiểm tra có điều kiện tín hiệu yếu
quan sát được ở root của hai graph `n=120, p=0.7`. M1b chỉ lấy root, dùng nhiều
graph độc lập hơn, tăng kích thước/mật độ và cutoff lên 5 giây. Output được ghi
trực tiếp vào `MyDrive/COPT-MT-results/guidance_pilot_m1b_<timestamp>`.

Tiêu chí diễn giải được chốt trước:

- Đơn vị độc lập là graph, không phải repeat.
- Không kết luận từ một vài lượt timing thắng sát nút.
- Tín hiệu đáng theo tiếp phải lặp lại trên nhiều graph độc lập và có lợi ích sau
  toàn bộ overhead; nếu không, kết luận COPT root guidance chưa đủ headroom trên
  solver/protocol hiện tại và chưa train gate.

## Giới hạn diễn giải

- Reference solver chưa phải solver hiệu năng cao; thắng nó không chứng minh SOTA.
- Oracle chọn min sau khi nhìn timing là hindsight; không triển khai được và dễ lạc quan vì noise.
- Summary only-both-exhausted có selection bias; phân tích đầy đủ phải gồm timeout và anytime area.
- Median theo state, bootstrap theo graph; không coi 3 repetitions là 3 graph.
- Gate chưa tồn tại: feature extraction time được lưu nhưng chưa cộng vào solver baseline.
- Score adapter chưa xác nhận bằng golden output của harness cũ. Cần parity trước claim chính thức.
- Log-degree zero của complement được fallback, ghi rõ reason; không đổi preprocessing ngầm.
- Trace chỉ lấy từ structural policy. Gate online sau này cần kiểm tra distribution shift do policy.

## Local checks không cần Torch

```text
python -m unittest experiments.guidance.test_solver -v
python -m experiments.guidance.pilot --sizes 20 30 --graphs-per-cell 1 --repeats 1 --output artifacts/guidance_local
```

Không có `--checkpoint` thì chỉ chạy structural/random để kiểm tra pipeline,
không tạo bằng chứng về hiệu quả COPT.
