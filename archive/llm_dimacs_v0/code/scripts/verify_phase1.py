import os
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from src.models.loss.direct_maxclique import direct_maxclique_loss, DirectMaxCliqueLoss
from src.inference.greedy_decoder import (
    greedy_clique_decode,
    is_valid_clique,
    local_search_clique,
    GreedyCliqueDecoder,
    build_adj_sets,
)
from src.data.datasets.dimacs_dataset import DIMACSDataset


def test_mathematical_properties():
    print("\n[Test 1] Mathematical Exactness of direct_maxclique_loss")

    # Case A: Complete graph K_4 (all 4 nodes connected to all other nodes)
    # Pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) -> 6 undirected edges = 12 directed edges
    edges_k4 = []
    for i in range(4):
        for j in range(4):
            if i != j:
                edges_k4.append([i, j])
    edge_index_k4 = torch.tensor(edges_k4, dtype=torch.long).t()

    # If all nodes are selected x = [1, 1, 1, 1], violations must be 0!
    data_k4 = Data(x=torch.ones((4, 1), dtype=torch.float32), edge_index=edge_index_k4, num_nodes=4)
    loss_k4 = direct_maxclique_loss(data_k4, alpha=1.0, beta=1.1)

    expected_k4 = -1.0  # size_term = -4, violations = 0, penalty = 0 -> loss = -4 / 4 = -1.0
    assert abs(loss_k4.item() - expected_k4) < 1e-5, f"Expected {expected_k4}, got {loss_k4.item()}"
    print(f"  ✓ Complete Graph K_4 with full clique: Loss = {loss_k4.item():.4f} (Violations = 0, Exact: -1.0)")

    # Case B: Empty graph E_4 (4 nodes, 0 edges)
    # If 2 nodes are selected x = [1, 1, 0, 0], violations must be 1.0 (they lack an edge!)
    edge_index_empty = torch.empty((2, 0), dtype=torch.long)
    x_empty = torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32)
    data_empty = Data(x=x_empty, edge_index=edge_index_empty, num_nodes=4)
    loss_empty = direct_maxclique_loss(data_empty, alpha=1.0, beta=1.1)

    # size_term = -2.0, violations = 0.5 * (4 - 0 - 2) = 1.0, penalty = 1.1 * 1 = 1.1 -> loss = (-2 + 1.1)/4 = -0.225
    expected_empty = (-2.0 + 1.1) / 4.0
    assert abs(loss_empty.item() - expected_empty) < 1e-5, f"Expected {expected_empty}, got {loss_empty.item()}"
    print(f"  ✓ Empty Graph E_4 with non-clique pair: Loss = {loss_empty.item():.4f} (Violations = 1, Exact: -0.225)")


def test_autograd_and_optimization():
    print("\n[Test 2] Autograd Flow & Loss Gradient Descent")

    # Create a small graph: 5-cycle C_5 + chord (0, 2)
    # Max clique should be triangle {0, 1, 2}
    edges = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 0), (0, 2),  # triangle {0, 1, 2}
        (2, 3), (3, 2),
        (3, 4), (4, 3),
        (4, 0), (0, 4)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    num_nodes = 5

    # Parameter logits
    logits = nn.Parameter(torch.randn(num_nodes, 1, requires_grad=True))
    optimizer = torch.optim.Adam([logits], lr=0.05)

    initial_loss = None
    for step in range(30):
        optimizer.zero_grad()
        probs = torch.sigmoid(logits)
        data = Data(x=probs, edge_index=edge_index, num_nodes=num_nodes)
        loss = direct_maxclique_loss(data, alpha=1.0, beta=1.5)

        if step == 0:
            initial_loss = loss.item()
            loss.backward()
            assert logits.grad is not None and not torch.isnan(logits.grad).any()
            print(f"  ✓ Backward gradient flow verified: grad norm = {logits.grad.norm().item():.4f}")
        else:
            loss.backward()

        optimizer.step()

    final_loss = loss.item()
    print(f"  ✓ 30 Adam optimization steps: initial_loss={initial_loss:.4f} -> final_loss={final_loss:.4f}")
    assert final_loss < initial_loss, "Optimization failed to reduce loss!"


def test_greedy_decoder():
    print("\n[Test 3] GreedyCliqueDecoder & Validity Guarantee")

    # Test graph: 5-node graph with max clique {0, 1, 2}
    edges = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 0), (0, 2),
        (2, 3), (3, 2),
        (3, 4), (4, 3)
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    num_nodes = 5

    # Assign high scores to nodes in the clique
    scores = torch.tensor([0.9, 0.85, 0.8, 0.2, 0.1])
    decoder = GreedyCliqueDecoder(num_seeds=2, use_local_search=True)
    clique = decoder.decode(scores, edge_index, num_nodes)

    valid, msg = is_valid_clique(clique, edge_index, num_nodes)
    print(f"  ✓ Decoded clique: {clique}, verification: {msg}")
    assert valid, f"Decoded clique is invalid: {msg}"
    assert set(clique) == {0, 1, 2}, f"Expected clique {{0, 1, 2}}, got {clique}"

    # Test random scores on 10 random graphs
    torch.manual_seed(42)
    for trial in range(10):
        # Generate random Erdos-Renyi graph
        n = 30
        adj_mat = (torch.rand((n, n)) < 0.4).float()
        adj_mat = torch.triu(adj_mat, diagonal=1)
        adj_mat = adj_mat + adj_mat.t()
        src, dst = torch.where(adj_mat > 0)
        e_idx = torch.stack([src, dst], dim=0)

        rand_scores = torch.rand(n)
        clique_res = decoder.decode(rand_scores, e_idx, n)
        val, _ = is_valid_clique(clique_res, e_idx, n)
        assert val, f"Trial {trial}: Greedy decoder returned non-clique!"

    print("  ✓ 10/10 random graphs produced 100% valid maximal cliques.")


def test_dimacs_benchmark():
    print("\n[Test 4] Benchmark Smoke Test on Real DIMACS Instance (c125.9)")

    data_dir = os.path.join("data", "dimacs", "maxclique")
    ds = DIMACSDataset(root=data_dir, name="test_c125", instance_names=["c125.9"])
    data = ds[0]

    n = data.num_nodes
    m = data.edge_index.size(1) // 2

    # Forward pass simulation with direct loss
    torch.manual_seed(12345)
    simulated_probs = torch.rand((n, 1))
    data.x = simulated_probs

    t0 = time.perf_counter()
    loss_val = direct_maxclique_loss(data, alpha=1.0, beta=1.1)
    loss_time_ms = (time.perf_counter() - t0) * 1000

    print(f"  ✓ direct_maxclique_loss on c125.9 (N={n}, |E|={m}): Loss = {loss_val.item():.4f} in {loss_time_ms:.2f} ms")

    # Greedy decode
    t0 = time.perf_counter()
    decoder = GreedyCliqueDecoder(num_seeds=5, use_local_search=True)
    clique = decoder.decode(simulated_probs, data.edge_index, n)
    decode_time_ms = (time.perf_counter() - t0) * 1000

    is_val, msg = is_valid_clique(clique, data.edge_index, n)
    print(f"  ✓ GreedyCliqueDecoder on c125.9: Found Clique size = {len(clique)} in {decode_time_ms:.2f} ms")
    print(f"  ✓ Clique validation: {is_val} ({msg})")
    assert is_val, "Failed clique validation on c125.9!"


def main():
    print("=" * 70)
    print("        GIAI ĐOẠN 1: VERIFICATION & SMOKE TEST SUITE")
    print("=" * 70)

    test_mathematical_properties()
    test_autograd_and_optimization()
    test_greedy_decoder()
    test_dimacs_benchmark()

    print("\n" + "=" * 70)
    print("🎉 GIAI ĐOẠN 1 HOÀN TẤT THÀNH CÔNG: LOSS & DECODER ĐẠT CHUẨN 100%!")
    print("=" * 70)


if __name__ == "__main__":
    main()
