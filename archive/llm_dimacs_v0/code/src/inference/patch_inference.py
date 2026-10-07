import math
from typing import List, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph, subgraph, remove_self_loops, to_undirected

from src.inference.greedy_decoder import greedy_clique_decode, is_valid_clique


class PatchBasedInference:
    """
    Patch-based Inference for Graph Combinatorial Optimization (Maximum Clique).
    Divides large real-world graphs (N=1,000-4,000) into pretrain-sized induced patches (n~300),
    runs GNN inference on each patch, aggregates probabilities into a global heatmap,
    and decodes a 100% valid maximal clique.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        patch_size: int = 300,
        num_patches: int = 30,
        num_hops: int = 2,
        device: str = "cpu",
        task_head: str = "mis",
        aggregation: str = "mean"
    ):
        self.model = model
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.num_hops = num_hops
        self.device = torch.device(device)
        self.task_head = task_head
        self.aggregation = aggregation

        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()
            # Disable saturation-prone activation in task head to preserve raw logits
            if hasattr(self.model, 'post_mp') and self.task_head in self.model.post_mp:
                head = self.model.post_mp[self.task_head]
                if hasattr(head, 'last_act') and head.last_act is not None:
                    head.last_act = None

    def select_seeds(self, edge_index: torch.Tensor, num_nodes: int) -> List[int]:
        """
        Select K seed vertices prioritized by high degree with >= 2-hop spacing.
        """
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        # Compute degree per node
        degree = torch.zeros(num_nodes, dtype=torch.long)
        degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long))

        # Sort nodes descending by degree
        sorted_nodes = torch.argsort(degree, descending=True).tolist()

        seeds = []
        used = set()
        
        # Build 1-hop adjacency for fast spacing check
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        adj_1hop = [set() for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            adj_1hop[u].add(v)

        for node in sorted_nodes:
            if node in used:
                continue
            seeds.append(node)
            if len(seeds) >= self.num_patches:
                break
            # Mark 1-hop neighbors as used to enforce seed separation
            used.add(node)
            used.update(adj_1hop[node])

        # If not enough seeds found due to strict spacing, fill from remaining top-degree nodes
        if len(seeds) < self.num_patches:
            for node in sorted_nodes:
                if node not in seeds:
                    seeds.append(node)
                    if len(seeds) >= self.num_patches:
                        break

        return seeds

    def extract_patch(
        self,
        seed: int,
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> Tuple[Data, torch.Tensor]:
        """
        Extract an induced r-hop subgraph patch around seed, bounded to patch_size.
        Returns: (patch_data, global_indices)
        """
        subset, sub_edge_index, mapping, _ = k_hop_subgraph(
            seed,
            self.num_hops,
            edge_index,
            num_nodes=num_nodes,
            relabel_nodes=True
        )

        # If patch is larger than patch_size, retain top-patch_size nodes by internal degree
        if len(subset) > self.patch_size:
            deg = torch.zeros(len(subset), dtype=torch.long)
            deg.scatter_add_(0, sub_edge_index[0], torch.ones(sub_edge_index.size(1), dtype=torch.long))
            
            # Ensure seed node is always retained
            seed_local = mapping.item()
            deg[seed_local] += 999999
            
            top_local = torch.argsort(deg, descending=True)[:self.patch_size]
            mask = torch.zeros(len(subset), dtype=torch.bool)
            mask[top_local] = True
            
            sub_edge_index, _ = subgraph(mask, sub_edge_index, relabel_nodes=True)
            subset = subset[top_local]

        # Prepare patch data
        patch_n = len(subset)
        patch_data = Data(
            x=torch.ones((patch_n, 1), dtype=torch.float32),
            edge_index=sub_edge_index,
            num_nodes=patch_n
        )

        # Compute graph stats if model expects them
        try:
            from src.transforms.graph_stats import ComputeGraphStats
            stats_tf = ComputeGraphStats(stats_list=['degree', 'cluster_coefficient', 'triangle_count'])
            patch_data = stats_tf(patch_data)
            patch_data.degree = torch.nan_to_num(patch_data.degree)
            patch_data.cluster_coefficient = torch.nan_to_num(patch_data.cluster_coefficient)
            patch_data.triangle_count = torch.nan_to_num(patch_data.triangle_count)
        except Exception:
            pass

        return patch_data, subset

    @torch.no_grad()
    def inference(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Run patch-based inference:
        1. Select seeds
        2. Extract patches
        3. Forward pass on each patch
        4. Aggregate local predictions into global heatmap [num_nodes]
        """
        seeds = self.select_seeds(edge_index, num_nodes)

        prob_sum = torch.zeros(num_nodes, dtype=torch.float32)
        prob_count = torch.zeros(num_nodes, dtype=torch.float32)
        prob_max = torch.zeros(num_nodes, dtype=torch.float32)

        for seed in seeds:
            patch_data, global_indices = self.extract_patch(seed, edge_index, num_nodes)
            global_indices = global_indices.cpu()

            if self.model is not None:
                patch_batch = Batch.from_data_list([patch_data]).to(self.device)
                model_out = self.model(patch_batch)

                # Extract probabilities
                if isinstance(model_out, dict):
                    if self.task_head in model_out:
                        task_data = model_out[self.task_head]
                        probs = getattr(task_data, 'x', task_data).squeeze().cpu()
                    else:
                        first_key = list(model_out.keys())[0]
                        task_data = model_out[first_key]
                        probs = getattr(task_data, 'x', task_data).squeeze().cpu()
                elif hasattr(model_out, 'x'):
                    probs = model_out.x.squeeze().cpu()
                elif isinstance(model_out, torch.Tensor):
                    probs = model_out.squeeze().cpu()
                else:
                    probs = torch.ones(len(global_indices))
            else:
                # Fallback: degree-based heuristic if model is None
                deg = patch_data.edge_index.size(1) // 2
                probs = torch.ones(len(global_indices)) * (deg / max(1, patch_data.num_nodes))

            probs = torch.nan_to_num(probs, nan=0.0).view(-1)
            if probs.numel() == 1 and len(global_indices) > 1:
                probs = probs.repeat(len(global_indices))

            # Standardize patch probabilities to align scales across patches
            if probs.std() > 1e-6:
                probs = torch.sigmoid((probs - probs.mean()) / (probs.std() + 1e-6))
            elif probs.max() > probs.min():
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)

            prob_sum[global_indices] += probs
            prob_count[global_indices] += 1.0
            prob_max[global_indices] = torch.maximum(prob_max[global_indices], probs)

        mask = prob_count > 0
        heatmap = torch.zeros(num_nodes, dtype=torch.float32)

        if self.aggregation == "max":
            heatmap[mask] = prob_max[mask]
        else:
            heatmap[mask] = prob_sum[mask] / prob_count[mask]

        # For uncovered nodes, assign baseline degree-based score
        uncovered = ~mask
        if uncovered.any():
            deg_all = torch.zeros(num_nodes, dtype=torch.float32)
            deg_all.scatter_add_(0, edge_index[0].cpu(), torch.ones(edge_index.size(1)))
            max_deg = deg_all.max().item()
            heatmap[uncovered] = (deg_all[uncovered] / max(1.0, max_deg)) * 0.1

        return heatmap

    def greedy_decode(
        self,
        heatmap: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_seeds: int = 10,
        use_local_search: bool = True
    ) -> List[int]:
        """
        Decode global heatmap into a 100% valid maximal clique.
        """
        return greedy_clique_decode(
            scores=heatmap,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_seeds=num_seeds,
            use_local_search=use_local_search
        )

    def solve(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_seeds: int = 10,
        use_local_search: bool = True
    ) -> Tuple[List[int], torch.Tensor]:
        """
        End-to-End Pipeline:
        Patch Extraction -> GNN Inference -> Global Heatmap -> Greedy Decode
        """
        heatmap = self.inference(edge_index, num_nodes)
        clique = self.greedy_decode(
            heatmap=heatmap,
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_seeds=num_seeds,
            use_local_search=use_local_search
        )
        return clique, heatmap
