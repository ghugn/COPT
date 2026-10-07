import os
import sys
import time
import argparse

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.data.datasets.dimacs_dataset import DIMACSDataset
from src.models.loss.direct_maxclique import direct_maxclique_loss
from src.inference.greedy_decoder import greedy_clique_decode, is_valid_clique
from scripts.download_dimacs import KNOWN_OPTIMA


def run_single_instance(
    name: str,
    known_opt: int,
    epochs: int = 60,
    lr: float = 0.15,
    alpha: float = 1.0,
    beta: float = 1.1,
    num_seeds: int = 10,
    verbose: bool = True
):
    data_dir = os.path.join("data", "dimacs", "maxclique")
    ds = DIMACSDataset(root=data_dir, name=f"run_{name}", instance_names=[name])
    if len(ds) == 0:
        print(f"❌ Instance {name} not found in {data_dir}!")
        return None

    data = ds[0]
    n = data.num_nodes
    m = data.edge_index.size(1) // 2
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Logits parameter
    logits = torch.nn.Parameter(torch.zeros(n, 1, device=device, requires_grad=True))
    optimizer = torch.optim.Adam([logits], lr=lr)

    if verbose:
        print(f"\n{'='*75}")
        print(f"🚀 Chạy Tối ưu Hóa Instance: {name} (N={n}, |E|={m}, Density={density:.4f}, Optimum={known_opt})")
        print(f"{'='*75}")

    t0 = time.time()
    best_clique = []
    initial_loss = None

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        probs = torch.sigmoid(logits)
        data.x = probs
        loss = direct_maxclique_loss(
            data,
            alpha=alpha,
            beta=beta,
            gamma=0.3,
            epoch=epoch,
            max_epochs=epochs
        )
        loss.backward()
        optimizer.step()

        if epoch == 1:
            initial_loss = loss.item()

        # Greedy decoding
        clique = greedy_clique_decode(
            scores=probs.detach().cpu(),
            edge_index=data.edge_index.cpu(),
            num_nodes=n,
            num_seeds=num_seeds,
            use_local_search=True
        )

        if len(clique) > len(best_clique):
            best_clique = clique

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == epochs):
            valid, _ = is_valid_clique(clique, data.edge_index.cpu(), n)
            ratio = (len(best_clique) / known_opt) * 100.0 if known_opt else 0.0
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.item():.4f} | "
                  f"Clique hiện tại: {len(clique):<2} | Kỷ lục: {len(best_clique):<2}/{known_opt} ({ratio:.1f}%) | Hợp lệ: {valid}")

    elapsed = time.time() - t0
    final_valid, _ = is_valid_clique(best_clique, data.edge_index.cpu(), n)
    approx_ratio = len(best_clique) / known_opt if known_opt else 0.0

    return {
        "instance": name,
        "nodes": n,
        "edges": m,
        "density": density,
        "optimum": known_opt,
        "clique_found": len(best_clique),
        "approx_ratio": approx_ratio,
        "time_sec": elapsed,
        "is_valid": final_valid,
        "initial_loss": initial_loss,
        "final_loss": loss.item()
    }


def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm Giai đoạn 1 trên DIMACS")
    parser.add_argument("--instances", nargs="+", default=["c125.9", "c250.9", "p_hat300-1", "keller4", "brock200_2"],
                        help="Danh sách instances cần chạy")
    parser.add_argument("--epochs", type=int, default=50, help="Số epochs tối ưu hóa")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    print("=" * 85)
    print("      KẾT QUẢ THỰC NGHIỆM GIAI ĐOẠN 1: COMPLEMENT-FREE LOSS + GREEDY DECODER")
    print("=" * 85)

    results = []
    for inst in args.instances:
        opt = KNOWN_OPTIMA.get(inst, 0)
        res = run_single_instance(inst, known_opt=opt, epochs=args.epochs, lr=args.lr, verbose=True)
        if res:
            results.append(res)

    # Print summary table
    print("\n" + "=" * 85)
    print("                     BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM")
    print("=" * 85)
    header = f"{'Instance':<15} | {'N':<5} | {'|E|':<8} | {'Density':<7} | {'Optimum':<7} | {'Tìm được':<8} | {'Tỷ lệ (Ratio)':<14} | {'Thời gian':<9} | {'Hợp lệ':<7}"
    print(header)
    print("-" * 85)

    for r in results:
        ratio_str = f"{r['approx_ratio']*100:.1f}%"
        time_str = f"{r['time_sec']:.2f}s"
        valid_str = "100% OK" if r['is_valid'] else "FAILED"
        print(f"{r['instance']:<15} | {r['nodes']:<5} | {r['edges']:<8} | {r['density']:<7.4f} | {r['optimum']:<7} | {r['clique_found']:<8} | {ratio_str:<14} | {time_str:<9} | {valid_str:<7}")

    print("=" * 85)
    
    # Save CSV
    out_dir = os.path.join("logs", "eval")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "phase1_benchmark_results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("Instance,N,Edges,Density,Optimum,CliqueFound,ApproxRatio,TimeSec,IsValid\n")
        for r in results:
            f.write(f"{r['instance']},{r['nodes']},{r['edges']},{r['density']:.4f},{r['optimum']},{r['clique_found']},{r['approx_ratio']:.4f},{r['time_sec']:.4f},{r['is_valid']}\n")

    print(f"\n✓ Đã lưu bảng kết quả chi tiết vào: {csv_path}")


if __name__ == "__main__":
    main()
