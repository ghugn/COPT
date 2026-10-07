import os
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch_geometric.data import Data
from src.data.datasets.dimacs_dataset import DIMACSDataset, parse_dimacs


def verify_checkpoint(ckpt_path: str):
    print(f"\n[1] Verifying Checkpoint: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint NOT found at: {ckpt_path}")
        return False

    size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"  ✓ Checkpoint file found ({size_mb:.2f} MB)")

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', {})
        print(f"  ✓ Successfully loaded state_dict with {len(state_dict)} tensor parameters")
        
        # Check model prefix layers
        net_keys = [k for k in state_dict.keys() if k.startswith('net.')]
        print(f"  ✓ Found {len(net_keys)} neural network parameters under 'net.' prefix")
        sample_keys = net_keys[:5]
        for k in sample_keys:
            print(f"     - {k}: {state_dict[k].shape}")
        return True
    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {e}")
        return False


def verify_dataset(instances: list):
    print(f"\n[2] Verifying DIMACSDataset Ingestion for {len(instances)} instances")
    data_dir = os.path.join("data", "dimacs", "maxclique")
    
    passed = 0
    for name in instances:
        try:
            ds = DIMACSDataset(root=data_dir, name=f"test_{name}", instance_names=[name])
            if len(ds) == 0:
                print(f"  ❌ [{name}]: Dataset is empty!")
                continue
            data = ds[0]
            n = data.num_nodes
            m = data.edge_index.size(1) // 2  # undirected edges
            x_nodes = data.x.size(0)
            
            # Check 0-based indexing validity
            min_idx = data.edge_index.min().item()
            max_idx = data.edge_index.max().item()

            if min_idx < 0 or max_idx >= n:
                print(f"  ❌ [{name}]: Indexing out of bounds! min={min_idx}, max={max_idx}, num_nodes={n}")
                continue

            if x_nodes != n:
                print(f"  ❌ [{name}]: Feature dimension mismatch! x={x_nodes}, num_nodes={n}")
                continue

            density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
            print(f"  ✓ [{name:<14}] N={n:<5} | |E|={m:<8} | Density={density:.4f} | 0-based index [0..{n-1}]: VALID")
            passed += 1
        except Exception as e:
            print(f"  ❌ [{name}]: Error: {e}")

    print(f"\n  Summary: {passed}/{len(instances)} instances verified successfully.")
    return passed == len(instances)


def main():
    print("=" * 70)
    print("        GIAI ĐOẠN 0: VERIFICATION & SMOKE TEST PIPELINE")
    print("=" * 70)

    # 1. Verify Checkpoints
    ckpt_path = os.path.join("logs", "train", "checkpoints", "multitask", "epoch_194.ckpt")
    ckpt_ok = verify_checkpoint(ckpt_path)

    # 2. Verify Sample Instances Across Groups
    sample_instances = [
        "c125.9",
        "c250.9",
        "c500.9",
        "brock200_2",
        "p_hat300-1",
        "p_hat300-2",
        "MANN_a27"
    ]
    data_ok = verify_dataset(sample_instances)

    print("\n" + "=" * 70)
    if ckpt_ok and data_ok:
        print("🎉 GIAI ĐOẠN 0 HOÀN TẤT THÀNH CÔNG: MÔI TRƯỜNG & DỮ LIỆU SẴN SÀNG!")
    else:
        print("⚠️ Có cảnh báo trong quá trình kiểm thử Giai đoạn 0. Vui lòng kiểm tra log.")
    print("=" * 70)


if __name__ == "__main__":
    main()
