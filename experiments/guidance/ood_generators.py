"""
Milestone M4: Out-of-Distribution (OOD) Graph Generators and DIMACS Parsers.
Supports:
1. Barabasi-Albert (BA) scale-free graphs (Power-law degree distribution).
2. Watts-Strogatz (WS) small-world graphs (High clustering coefficient).
3. DIMACS Benchmark instances (.clq ASCII format).
"""
import os
import random
import urllib.request
from pathlib import Path
import networkx as nx


def nx_to_bitmask_adj(g):
    """Chuyển đổi đồ thị NetworkX sang danh sách bitmask kề tối ưu của Solver."""
    n = g.number_of_nodes()
    mapping = {node: i for i, node in enumerate(g.nodes())}
    adj = [0] * n
    for u, v in g.edges():
        i, j = mapping[u], mapping[v]
        if i != j:
            adj[i] |= 1 << j
            adj[j] |= 1 << i
    return adj


def ba_graph(n, m, seed):
    """
    Sinh đồ thị vô hướng Barabasi-Albert (Scale-free network).
    - n: Số đỉnh
    - m: Số cạnh nối từ đỉnh mới đến các đỉnh hiện hữu (m < n)
    """
    g = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return nx_to_bitmask_adj(g)


def ws_graph(n, k, p, seed):
    """
    Sinh đồ thị vô hướng Watts-Strogatz (Small-world network).
    - n: Số đỉnh
    - k: Bậc ban đầu của mỗi đỉnh (k chẵn, k < n)
    - p: Xác suất nối lại cạnh ngẫu nhiên (rewiring probability)
    """
    g = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    return nx_to_bitmask_adj(g)


def parse_dimacs_file(filepath):
    """
    Đọc file DIMACS (.clq hoặc .col) và chuyển sang danh sách bitmask kề.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file DIMACS tại: {filepath}")
        
    n_nodes = 0
    edges = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if parts[0] == "p":
                # p edge/col <nodes> <edges>
                n_nodes = int(parts[2])
            elif parts[0] == "e":
                u, v = int(parts[1]) - 1, int(parts[2]) - 1  # 1-based -> 0-based
                if u != v:
                    edges.append((u, v))
                    
    adj = [0] * n_nodes
    for u, v in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
    return adj


# Các instance DIMACS kinh điển từ kho lưu trữ của Pablo San Segundo (BBMC)
DIMACS_RAW_URLS = {
    "brock200_1": "https://raw.githubusercontent.com/psanse/graph/master/data/brock200_1.clq",
    "brock200_2": "https://raw.githubusercontent.com/psanse/graph/master/data/brock200_2.clq",
    "brock200_3": "https://raw.githubusercontent.com/psanse/graph/master/data/brock200_3.clq",
    "brock200_4": "https://raw.githubusercontent.com/psanse/graph/master/data/brock200_4.clq",
    "sample": "https://raw.githubusercontent.com/psanse/graph/master/data/sample.clq",
}


def fetch_dimacs_instance(name, cache_dir="data/dimacs_cache"):
    """
    Tải hoặc nạp từ cache một instance DIMACS chuẩn.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    target_file = cache_path / f"{name}.clq"
    
    if not target_file.exists():
        if name in DIMACS_RAW_URLS:
            url = DIMACS_RAW_URLS[name]
            print(f"[DIMACS] Downloading instance '{name}' from {url}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp, open(target_file, "wb") as f_out:
                f_out.write(resp.read())
        else:
            raise ValueError(f"Instance '{name}' khong co trong danh sach: {list(DIMACS_RAW_URLS.keys())}")


            
    return parse_dimacs_file(target_file)
