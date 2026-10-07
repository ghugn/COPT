"""Paper-faithful utilities for COPT Table 15 reproduction.

Maximum Clique on G is evaluated as MIS on the complement graph. The decoder
matches the paper's sequential multi-seed construction and deliberately does not
perform local search or patch inference.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import from_networkx, to_networkx

from src.transforms.graph_stats import ComputeGraphStats


def complement_data(data: Data, stats: Sequence[str]) -> Data:
    """Construct the explicit complement and compute the model's node features."""
    graph = to_networkx(data, to_undirected=True)
    complement = nx.complement(graph)
    result = from_networkx(complement)
    result.num_nodes = data.num_nodes
    result.x = torch.ones((data.num_nodes, 1), dtype=torch.float32)
    result = ComputeGraphStats(stats_list=list(stats))(result)
    result.instance_name = getattr(data, "instance_name", "unknown")
    return result


def extract_node_scores(output, task: str = "mis") -> torch.Tensor:
    """Extract a flat node-score vector from single- or multi-task COPT output."""
    if isinstance(output, dict):
        if task not in output:
            raise ValueError(
                f"Checkpoint has task heads {sorted(output)}, but '{task}' is required."
            )
        output = output[task]
    scores = output.x if hasattr(output, "x") else output
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    return scores.detach().view(-1).cpu()


def sequential_independent_set_decode(
    scores: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    num_seeds: int = 10,
    dec_length: int | None = 300,
) -> List[int]:
    """COPT-style sequential multi-seed MIS decoder.

    Each seed starts at a different position in the shared score ordering. A
    vertex is accepted only when it is non-adjacent to every selected vertex.
    """
    if num_nodes == 0:
        return []
    order = torch.argsort(scores.view(-1), descending=True).tolist()
    limit = min(num_nodes, dec_length if dec_length is not None else num_nodes)
    seeds = min(num_seeds, limit)

    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().cpu().tolist():
        if u != v:
            adjacency[u].add(v)

    best: List[int] = []
    for seed_position in range(seeds):
        selected = [order[seed_position]]
        selected_set = {order[seed_position]}
        for position in range(seed_position, limit):
            vertex = order[position]
            if vertex in selected_set:
                continue
            if all(vertex not in adjacency[chosen] for chosen in selected):
                selected.append(vertex)
                selected_set.add(vertex)
        if len(selected) > len(best):
            best = selected
    return best


def is_clique(vertices: Iterable[int], edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Validate a clique against the original graph."""
    vertices = list(vertices)
    adjacency = [set() for _ in range(num_nodes)]
    for u, v in edge_index.t().cpu().tolist():
        if u != v:
            adjacency[u].add(v)
    return all(v in adjacency[u] for i, u in enumerate(vertices) for v in vertices[i + 1 :])


@torch.no_grad()
def score_complement(model, data: Data, task: str = "mis", device: str = "cpu") -> torch.Tensor:
    """Run full-graph COPT inference on a prepared complement graph."""
    model = model.to(device)
    model.eval()
    batch = Batch.from_data_list([data]).to(device)
    return extract_node_scores(model(batch), task=task)
