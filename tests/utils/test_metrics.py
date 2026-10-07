import torch
from torch_geometric.data import Batch, Data

from src.utils.metrics import mis_decoder_pyg_parallel


def test_parallel_mis_decoder_orders_nodes_by_score():
    graph = Data(
        x=torch.tensor([[0.9], [0.8], [0.7]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        num_nodes=3,
    )

    decoded = mis_decoder_pyg_parallel(
        Batch.from_data_list([graph]), dec_length=3, num_seeds=2
    )

    assert decoded.to_data_list()[0].is_size == 2
