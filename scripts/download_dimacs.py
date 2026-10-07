import os
import sys
import urllib.request
import re
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.dimacs_reference import KNOWN_OPTIMA

DIMACS_DIR = os.path.join("data", "dimacs", "maxclique")

# Primary mirror: Carnegie Mellon University (CMU) COLOR02 archive
CMU_BASE = "https://mat.tepper.cmu.edu/COLOR02/INSTANCES"

# Secondary mirror: GitHub SanTanBan DIMACS repository
GITHUB_BASE = "https://raw.githubusercontent.com/SanTanBan/DIMACS-MCP-Cases/main"

# Mapping: instance key -> list of candidate remote filenames
DIMACS_CANDIDATES = {
    # Group: c (random)
    "c125.9": [f"{CMU_BASE}/C125.9.clq", f"{GITHUB_BASE}/C125.9.clq"],
    "c250.9": [f"{CMU_BASE}/C250.9.clq", f"{GITHUB_BASE}/C250.9.clq"],
    "c4000.5": [f"{CMU_BASE}/C4000.5.clq"],
    "c500.9": [f"{CMU_BASE}/C500.9.clq", f"{GITHUB_BASE}/C500.9.clq"],
    "c1000.9": [f"{CMU_BASE}/C1000.9.clq", f"{GITHUB_BASE}/C1000.9.clq"],
    "c2000.5": [f"{CMU_BASE}/C2000.5.clq", f"{GITHUB_BASE}/C2000.5.clq"],
    "c2000.9": [f"{CMU_BASE}/C2000.9.clq"],

    # Group: DSJC
    "dsjc500.5": [f"{CMU_BASE}/DSJC500.5.col"],
    "dsjc1000.5": [f"{CMU_BASE}/DSJC1000.5.col"],
    
    # Group: brock (hidden clique)
    "brock200_2": [f"{CMU_BASE}/brock200_2.clq", f"{GITHUB_BASE}/brock200_2.b"],
    "brock200_4": [f"{CMU_BASE}/brock200_4.clq", f"{GITHUB_BASE}/brock200_4.b"],
    "brock400_2": [f"{CMU_BASE}/brock400_2.clq", f"{GITHUB_BASE}/brock400_2.b"],
    "brock400_4": [f"{CMU_BASE}/brock400_4.clq", f"{GITHUB_BASE}/brock400_4.b"],
    "brock800_2": [f"{CMU_BASE}/brock800_2.clq", f"{GITHUB_BASE}/brock800_2.b"],
    "brock800_4": [f"{CMU_BASE}/brock800_4.clq", f"{GITHUB_BASE}/brock800_4.b"],
    
    # Group: p_hat (random with varying density)
    "p_hat300-1": [f"{CMU_BASE}/p_hat300-1.clq", f"{GITHUB_BASE}/p_hat300_1.clq"],
    "p_hat300-2": [f"{CMU_BASE}/p_hat300-2.clq", f"{GITHUB_BASE}/p_hat300-2.clq"],
    "p_hat300-3": [f"{CMU_BASE}/p_hat300-3.clq", f"{GITHUB_BASE}/p_hat300-3.clq"],
    "p_hat700-1": [f"{CMU_BASE}/p_hat700-1.clq", f"{GITHUB_BASE}/p_hat700-1.clq"],
    "p_hat700-2": [f"{CMU_BASE}/p_hat700-2.clq", f"{GITHUB_BASE}/p_hat700-2.clq"],
    "p_hat700-3": [f"{CMU_BASE}/p_hat700-3.clq", f"{GITHUB_BASE}/p_hat700-3.clq"],
    "p_hat1500-1": [f"{CMU_BASE}/p_hat1500-1.clq", f"{GITHUB_BASE}/p_hat1500-1.clq"],
    "p_hat1500-2": [f"{CMU_BASE}/p_hat1500-2.clq", f"{GITHUB_BASE}/p_hat1500-2.clq"],
    "p_hat1500-3": [f"{CMU_BASE}/p_hat1500-3.clq", f"{GITHUB_BASE}/p_hat1500-3.clq"],
    
    # Group: MANN
    "MANN_a27": [f"{CMU_BASE}/MANN_a27.clq", f"{GITHUB_BASE}/MANN_a27.clq.b"],
    "MANN_a45": [f"{CMU_BASE}/MANN_a45.clq", f"{GITHUB_BASE}/MANN_a45.clq.b"],
    "MANN_a81": [f"{CMU_BASE}/MANN_a81.clq"],
    
    # Group: keller
    "keller4": [f"{CMU_BASE}/keller4.clq", f"{GITHUB_BASE}/keller4.clq"],
    "keller5": [f"{CMU_BASE}/keller5.clq", f"{GITHUB_BASE}/keller5.clq"],
    "keller6": [f"{CMU_BASE}/keller6.clq", f"{GITHUB_BASE}/keller6.clq"],
    
    # Group: hamming
    "hamming8-4": [f"{CMU_BASE}/hamming8-4.clq", f"{GITHUB_BASE}/hamming8-4.clq"],
    "hamming10-4": [f"{CMU_BASE}/hamming10-4.clq", f"{GITHUB_BASE}/hamming10-4.clq"],

    # Group: gen (dense generators)
    "gen200_p0.9_44": [f"{CMU_BASE}/gen200_p0.9_44.clq", f"{GITHUB_BASE}/gen200_p0.9_44.b"],
    "gen200_p0.9_55": [f"{CMU_BASE}/gen200_p0.9_55.clq", f"{GITHUB_BASE}/gen200_p0.9_55.b"],
    "gen400_p0.9_55": [f"{CMU_BASE}/gen400_p0.9_55.clq", f"{GITHUB_BASE}/gen400_p0.9_55.b"],
    "gen400_p0.9_65": [f"{CMU_BASE}/gen400_p0.9_65.clq", f"{GITHUB_BASE}/gen400_p0.9_65.b"],
    "gen400_p0.9_75": [f"{CMU_BASE}/gen400_p0.9_75.clq", f"{GITHUB_BASE}/gen400_p0.9_75.b"],
}

def download_file(urls: List[str], dest_path: str) -> bool:
    """Download file from list of URLs, trying each until one succeeds."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                content = response.read()
                if len(content) > 0:
                    with open(dest_path, 'wb') as f:
                        f.write(content)
                    return True
        except Exception:
            continue
    return False


def parse_and_validate(filepath: str) -> Tuple[int, int, float]:
    """
    Parse a downloaded DIMACS file to validate vertex and edge counts.
    Returns: (num_nodes, num_edges, density)
    """
    max_node = 0
    edges = set()
    num_nodes_header = None
    num_edges_header = None

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                # e.g.: p col 125 6963 or p edge 125 6963
                try:
                    num_nodes_header = int(parts[2])
                    num_edges_header = int(parts[3])
                except (IndexError, ValueError):
                    pass
            elif parts[0] == 'e':
                try:
                    u, v = int(parts[1]), int(parts[2])
                    if u != v:
                        edge = (min(u, v), max(u, v))
                        edges.add(edge)
                        if u > max_node:
                            max_node = u
                        if v > max_node:
                            max_node = v
                except (IndexError, ValueError):
                    pass

    n = num_nodes_header if num_nodes_header is not None else max_node
    m = len(edges)
    density = (2.0 * m) / (n * (n - 1)) if n > 1 else 0.0
    return n, m, density


def main():
    os.makedirs(DIMACS_DIR, exist_ok=True)
    print(f"Target directory: {os.path.abspath(DIMACS_DIR)}")
    print(f"Total instances to fetch: {len(DIMACS_CANDIDATES)}")
    print("-" * 70)

    downloaded = 0
    skipped = 0
    failed = []

    for name, urls in DIMACS_CANDIDATES.items():
        # Destination filenames: save as both name.clq and name.txt for flexibility
        clq_dest = os.path.join(DIMACS_DIR, f"{name}.clq")
        txt_dest = os.path.join(DIMACS_DIR, f"{name}.txt")

        if os.path.exists(clq_dest) and os.path.getsize(clq_dest) > 100:
            dest = clq_dest
            skipped += 1
            action = "EXISTS"
        else:
            success = download_file(urls, clq_dest)
            if success:
                dest = clq_dest
                downloaded += 1
                action = "DOWNLOADED"
            else:
                failed.append(name)
                print(f"[{'FAIL':<10}] {name}: all download sources failed")
                continue

        # Also create a .txt symlink or copy so any parser expecting .txt works immediately
        if not os.path.exists(txt_dest):
            with open(clq_dest, 'rb') as f_in, open(txt_dest, 'wb') as f_out:
                f_out.write(f_in.read())

        n, m, density = parse_and_validate(dest)
        opt = KNOWN_OPTIMA.get(name.lower(), "N/A")
        print(f"[{action:<10}] {name:<16} | N={n:<5} | |E|={m:<8} | Density={density:.4f} | Opt={opt}")

    print("-" * 70)
    print(f"Summary: {downloaded} downloaded, {skipped} already present, {len(failed)} failed.")
    if failed:
        print(f"Failed instances: {failed}")
    else:
        print("All benchmark instances downloaded and validated successfully!")


if __name__ == "__main__":
    main()
