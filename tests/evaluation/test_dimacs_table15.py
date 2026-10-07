import torch

from src.data.datasets.dimacs_dataset import parse_dimacs
from src.evaluation.dimacs_table15 import is_clique, sequential_independent_set_decode


def test_parse_dimacs_converts_one_based_vertices(tmp_path):
    path = tmp_path / "tiny.clq"
    path.write_text("p edge 4 2\ne 1 2\ne 3 4\n", encoding="utf-8")
    graph = parse_dimacs(str(path))

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert set(graph.edges) == {(0, 1), (2, 3)}


def test_sequential_decoder_returns_independent_set():
    # Complement graph has edges (0,3), (1,3), and (2,3), so {0,1,2} is MIS.
    edge_index = torch.tensor(
        [[0, 3, 1, 3, 2, 3], [3, 0, 3, 1, 3, 2]], dtype=torch.long
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    result = sequential_independent_set_decode(scores, edge_index, 4, num_seeds=2)

    assert set(result) == {0, 1, 2}


def test_is_clique_validates_original_graph():
    edge_index = torch.tensor(
        [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long
    )
    assert is_clique([0, 1, 2], edge_index, 3)
    assert not is_clique([0, 1, 2], edge_index[:, :4], 3)
