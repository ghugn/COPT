from typing import List, Set, Tuple, Union, Optional
import torch
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected


def build_adj_sets(edge_index: torch.Tensor, num_nodes: int) -> List[Set[int]]:
    """
    Build adjacency list of sets for O(1) neighbor membership lookup.
    """
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes)
    
    adj: List[Set[int]] = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    
    for u, v in zip(src, dst):
        adj[u].add(v)
        
    return adj


def is_valid_clique(
    clique: Union[List[int], Set[int], torch.Tensor],
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Verify whether the provided set of nodes forms a 100% valid clique in G.
    A clique is valid if and only if every distinct pair of nodes has an edge.

    Returns:
        (is_valid: bool, reason: str)
    """
    if isinstance(clique, torch.Tensor):
        clique_list = clique.view(-1).tolist()
    else:
        clique_list = list(clique)

    k = len(clique_list)
    if k <= 1:
        return True, f"Valid clique (size={k})"

    if num_nodes is None:
        num_nodes = max(max(clique_list) + 1, int(edge_index.max().item()) + 1)

    adj = build_adj_sets(edge_index, num_nodes)

    for i in range(k):
        u = clique_list[i]
        for j in range(i + 1, k):
            v = clique_list[j]
            if v not in adj[u]:
                return False, f"Invalid clique: missing edge between vertex {u} and vertex {v}"

    return True, f"Valid clique (size={k})"


def local_search_clique(
    clique: List[int],
    adj: List[Set[int]],
    max_steps: int = 50
) -> List[int]:
    """
    Local search improvement for a maximal clique.
    - Step 1: Greedy expansion if any non-clique node is adjacent to all clique members.
    - Step 2: (1, 2)-swap: If swapping 1 node u in C allows adding 2 nodes v1, v2 not in C,
              the clique size increases by +1.
    """
    current_clique = set(clique)
    num_nodes = len(adj)

    for _ in range(max_steps):
        improved = False

        # 1. Greedy expansion check
        candidates = set(range(num_nodes)) - current_clique
        for w in list(candidates):
            if current_clique.issubset(adj[w]):
                current_clique.add(w)
                candidates.remove(w)
                improved = True

        if improved:
            continue

        # 2. (1, 2)-swap check
        for u in list(current_clique):
            c_without_u = current_clique - {u}
            # Find nodes outside C that are connected to all nodes in c_without_u
            eligible = [w for w in (set(range(num_nodes)) - current_clique) if c_without_u.issubset(adj[w])]
            if len(eligible) > 60:
                eligible.sort(key=lambda w: len(adj[w]), reverse=True)
                eligible = eligible[:60]
            
            # Check if there is any pair (w1, w2) in eligible that are connected to each other
            found_swap = False
            for i in range(len(eligible)):
                w1 = eligible[i]
                for j in range(i + 1, len(eligible)):
                    w2 = eligible[j]
                    if w2 in adj[w1]:
                        # Swap u with {w1, w2} -> size increases by 1!
                        current_clique.remove(u)
                        current_clique.add(w1)
                        current_clique.add(w2)
                        found_swap = True
                        improved = True
                        break
                if found_swap:
                    break
            if found_swap:
                break

        if not improved:
            break

    return sorted(list(current_clique))


def greedy_clique_decode(
    scores: Union[torch.Tensor, List[float]],
    edge_index: torch.Tensor,
    num_nodes: Optional[int] = None,
    num_seeds: int = 1,
    use_local_search: bool = False
) -> List[int]:
    """
    Greedy Clique Decoding from node scores/heatmap.
    Guarantees a 100% valid maximal clique.

    Args:
        scores: Continuous node probabilities or scores [num_nodes].
        edge_index: PyG edge_index [2, num_edges].
        num_nodes: Total number of nodes (inferred if None).
        num_seeds: Number of top seeds to explore. Returns the largest clique found.
        use_local_search: Whether to apply (1,2)-swap local search to improve the clique.

    Returns:
        List[int]: List of vertex indices forming the clique.
    """
    if isinstance(scores, torch.Tensor):
        scores_t = scores.view(-1).detach().cpu()
    else:
        scores_t = torch.tensor(scores, dtype=torch.float32)

    if num_nodes is None:
        num_nodes = scores_t.size(0)

    adj = build_adj_sets(edge_index, num_nodes)
    sorted_nodes = torch.argsort(scores_t, descending=True).tolist()

    best_clique: List[int] = []
    num_seeds = min(num_seeds, num_nodes)

    for seed_idx in range(num_seeds):
        seed_node = sorted_nodes[seed_idx]
        clique = [seed_node]
        candidates = set(adj[seed_node])

        # Sequentially pick remaining candidates ordered by score
        for node in sorted_nodes:
            if not candidates:
                break
            if node in candidates:
                clique.append(node)
                candidates.intersection_update(adj[node])

        if len(clique) > len(best_clique):
            best_clique = clique

    if use_local_search and best_clique:
        best_clique = local_search_clique(best_clique, adj, max_steps=20)

    return best_clique


class GreedyCliqueDecoder:
    """
    Reusable Decoder class for GNN-guided Maximum Clique inference.
    """
    def __init__(self, num_seeds: int = 1, use_local_search: bool = False):
        self.num_seeds = num_seeds
        self.use_local_search = use_local_search

    def decode(
        self,
        scores: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None
    ) -> List[int]:
        return greedy_clique_decode(
            scores=scores,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_seeds=self.num_seeds,
            use_local_search=self.use_local_search
        )

    def decode_to_mask(
        self,
        scores: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        """
        Decodes to a binary indicator tensor of shape [num_nodes, 1].
        """
        if num_nodes is None:
            num_nodes = scores.view(-1).size(0)
        clique = self.decode(scores, edge_index, num_nodes)
        mask = torch.zeros((num_nodes, 1), dtype=torch.float32, device=scores.device)
        if clique:
            mask[clique] = 1.0
        return mask
