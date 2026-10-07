import os
import re
import hashlib
from typing import List, Optional
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx


def parse_dimacs(filepath: str) -> nx.Graph:
    """
    Parse a DIMACS graph file into a 0-indexed NetworkX undirected Graph.
    Properly handles 1-based indexing, stripping comments ('c'),
    and parsing 'p edge/col/clq NODES EDGES' and 'e U V' lines.
    """
    g = nx.Graph()
    declared_nodes = None

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            parts = line.split()
            if parts[0] == 'p':
                try:
                    declared_nodes = int(parts[2])
                    g.add_nodes_from(range(declared_nodes))
                except (IndexError, ValueError):
                    pass
            elif parts[0] == 'e':
                try:
                    u, v = int(parts[1]), int(parts[2])
                    # DIMACS format is 1-based, convert to 0-based
                    u0 = u - 1 if u > 0 else u
                    v0 = v - 1 if v > 0 else v
                    if u0 != v0:
                        g.add_edge(u0, v0)
                except (IndexError, ValueError):
                    pass

    # Fallback if no 'p' header was present: ensure contiguous 0-based nodes
    if declared_nodes is None:
        nodes = sorted(list(g.nodes()))
        if nodes:
            max_node = max(nodes)
            g.add_nodes_from(range(max_node + 1))

    return g


class DIMACSDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for DIMACS Maximum Clique benchmarks.
    Loads .clq, .col, .b, or .txt instances into in-memory PyG graphs.
    """
    def __init__(
        self,
        root: str,
        name: str = 'maxclique',
        instance_names: Optional[List[str]] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None
    ):
        self.name = name
        # Normalize instance names for case-insensitive matching without extension
        self.requested_instances = (
            [self._clean_name(n) for n in instance_names]
            if instance_names is not None else None
        )
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @staticmethod
    def _clean_name(name: str) -> str:
        """Strip file extension and convert to lowercase for uniform matching."""
        base = os.path.basename(name).lower()
        for ext in ['.txt', '.clq', '.col', '.b']:
            if base.endswith(ext):
                base = base[:-len(ext)]
        return base

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        if self.requested_instances and len(self.requested_instances) == 1:
            folder = self.requested_instances[0]
        elif self.requested_instances:
            suffix = hashlib.md5(''.join(sorted(self.requested_instances)).encode()).hexdigest()[:8]
            folder = f'{self.name}_{suffix}'
        else:
            folder = self.name
        return os.path.join(self.root, 'processed', folder)

    @property
    def raw_file_names(self) -> List[str]:
        supported_exts = ('.clq', '.txt', '.col', '.b')
        if not os.path.exists(self.raw_dir):
            return []

        all_files = [
            f for f in os.listdir(self.raw_dir)
            if any(f.lower().endswith(ext) for ext in supported_exts)
        ]

        # Group by clean name to avoid duplicate files (e.g. c125.9.clq and c125.9.txt)
        unique_instances = {}
        for fname in sorted(all_files):
            clean = self._clean_name(fname)
            # Prefer .clq over .txt / .b
            if clean not in unique_instances or fname.lower().endswith('.clq'):
                unique_instances[clean] = fname

        if self.requested_instances is not None:
            selected_files = [
                unique_instances[req]
                for req in self.requested_instances
                if req in unique_instances
            ]
            return selected_files

        return sorted(list(unique_instances.values()))

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        data_list = []
        for fname in self.raw_file_names:
            fpath = os.path.join(self.raw_dir, fname)
            g = parse_dimacs(fpath)

            if isinstance(g, nx.DiGraph):
                g = g.to_undirected()

            data = from_networkx(g)
            clean_name = self._clean_name(fname)
            data.instance_name = clean_name
            data.file_name = fname
            data.num_nodes = g.number_of_nodes()

            # Set node features x to ones [num_nodes, 1] if not already present
            if not hasattr(data, 'x') or data.x is None:
                data.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        os.makedirs(self.processed_dir, exist_ok=True)
        if len(data_list) > 0:
            self.save(data_list, self.processed_paths[0])
        else:
            torch.save(([], None), self.processed_paths[0])

    @property
    def num_node_features(self) -> int:
        return 1