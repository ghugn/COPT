import torch
import torch.nn as nn
from typing import Optional
from torch_geometric.data import Batch, Data
from torch_geometric.utils import remove_self_loops


def direct_maxclique_loss(
    batch,
    alpha: float = 1.0,
    beta: float = 1.1,
    gamma: float = 0.3,
    epoch: Optional[int] = None,
    max_epochs: Optional[int] = None,
    scale_balance: bool = False,
    reduction: str = 'sum'
):
    """
    Enhanced Complement-Free MaxClique Loss calculated directly on graph G.
    Eliminates explicit generation of the complement graph G_bar.

    Enhancements:
    1. Complement-Free Violation Formulation: O(|E(G)|) time & memory complexity.
    2. Polarization Regularizer (gamma): Penalizes fractional probabilities x*(1-x),
       forcing continuous logits to decisively converge to 0 or 1.
    3. Dynamic Penalty Annealing (epoch/max_epochs): Starts with gentle penalty for
       global space exploration, then ramps up to strictly eliminate non-edges.
    4. Scale-Balanced Penalty (scale_balance): Balances O(k^2) penalty with O(k) reward
       to prevent gradient collapse on large graphs.

    Args:
        batch: PyG Batch or Data object. batch.x contains node probabilities/scores in [0, 1].
        alpha: Reward weight for selecting more nodes (default: 1.0).
        beta: Penalty weight for non-adjacent node pairs (default: 1.1).
        gamma: Polarization weight forcing binary {0, 1} decisions (default: 0.3).
        epoch: Current training epoch (1-indexed) for penalty annealing.
        max_epochs: Maximum epochs for annealing schedule.
        scale_balance: Whether to balance penalty by sum(x) to prevent O(N^2) blowout.
        reduction: 'sum' (total loss) or 'mean' (loss averaged across graphs in batch).

    Returns:
        torch.Tensor: Computed scalar loss value.
    """
    if isinstance(batch, Batch):
        data_list = batch.to_data_list()
    elif isinstance(batch, Data):
        data_list = [batch]
    elif hasattr(batch, 'to_data_list'):
        data_list = batch.to_data_list()
    else:
        raise TypeError(f"Unsupported graph batch type: {type(batch)}")

    total_loss = 0.0

    # Calculate effective beta with annealing if epoch and max_epochs provided
    if epoch is not None and max_epochs is not None and max_epochs > 0:
        progress = min(1.0, max(0.0, epoch / max_epochs))
        # Warmup from 0.5 * beta to 1.2 * beta
        effective_beta = beta * (0.5 + 0.7 * progress)
    else:
        effective_beta = beta

    for data in data_list:
        x = data.x.view(-1)  # [num_nodes]
        num_nodes = data.num_nodes if data.num_nodes is not None else x.size(0)

        # 1. Size reward: encourage picking more vertices
        sum_x = torch.sum(x)
        size_term = -alpha * sum_x

        # 2. Violation term: pairs of selected vertices that lack an edge in G
        sum_x_sq = torch.sum(x ** 2)

        if data.edge_index is not None and data.edge_index.numel() > 0:
            edge_index, _ = remove_self_loops(data.edge_index)
            src, dst = edge_index[0], edge_index[1]
            edge_sum = torch.sum(x[src] * x[dst])
        else:
            edge_sum = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        violations = 0.5 * (sum_x ** 2 - edge_sum - sum_x_sq)

        if scale_balance:
            penalty_term = effective_beta * (violations / torch.clamp(sum_x, min=1.0))
        else:
            penalty_term = effective_beta * violations

        # 3. Polarization term: penalize indecision x * (1 - x)
        polarization_term = gamma * torch.sum(x * (1.0 - x)) if gamma > 0.0 else 0.0

        # Normalize per-node to stabilize gradient variance across varying graph sizes
        graph_loss = (size_term + penalty_term + polarization_term) / max(num_nodes, 1)
        total_loss = total_loss + graph_loss

    if reduction == 'mean':
        return total_loss / max(len(data_list), 1)
    return total_loss


class DirectMaxCliqueLoss(nn.Module):
    """
    PyTorch nn.Module wrapper for direct_maxclique_loss to support Hydra / Lightning.
    """
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.1,
        gamma: float = 0.3,
        scale_balance: bool = False,
        reduction: str = 'sum'
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scale_balance = scale_balance
        self.reduction = reduction
        self.current_epoch: Optional[int] = None
        self.max_epochs: Optional[int] = None

    def set_epoch(self, epoch: int, max_epochs: int):
        self.current_epoch = epoch
        self.max_epochs = max_epochs

    def forward(self, batch):
        return direct_maxclique_loss(
            batch,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            epoch=self.current_epoch,
            max_epochs=self.max_epochs,
            scale_balance=self.scale_balance,
            reduction=self.reduction
        )
