"""Inference from a trusted local MIS checkpoint; includes all feature costs."""
import time


class COPTScorer:
    def __init__(self, checkpoint, device):
        import torch
        self.torch = torch
        self.device = device
        # Only load the user's trusted checkpoint; pickle is required upstream.
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = ckpt["hyper_parameters"]
        if hparams.get("task") != "mis":
            raise ValueError("This adapter requires a single-task MIS checkpoint")
        self.net = hparams["net"]
        weights = {k[4:]: v for k, v in ckpt["state_dict"].items() if k.startswith("net.")}
        self.net.load_state_dict(weights, strict=True)
        self.net.to(device).eval()
        torch.set_num_threads(1)

    def sync(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()

    def score(self, adj, candidates):
        from torch_geometric.data import Data, Batch
        from experiments.guidance.solver import vertices
        torch = self.torch
        self.sync()
        start = time.perf_counter()
        ids = list(vertices(candidates))
        n = len(ids)

        edges_sym = []
        for i in range(n):
            u_adj = adj[ids[i]]
            for j in range(i):
                if not (u_adj & (1 << ids[j])):
                    edges_sym.append((i, j))
                    edges_sym.append((j, i))

        if edges_sym:
            edge_index = torch.tensor(edges_sym, dtype=torch.long).t().contiguous()
            adj_comp = torch.zeros((n, n), dtype=torch.float32)
            adj_comp[edge_index[0], edge_index[1]] = 1.0
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            adj_comp = torch.zeros((n, n), dtype=torch.float32)

        deg_pt = adj_comp.sum(dim=1)
        # Original upstream uses log(degree), not log1p. Reject nonfinite
        # inputs rather than silently changing the pretrained feature contract.
        if len(deg_pt) == 0 or deg_pt.min() == 0:
            return None, time.perf_counter() - start, "zero_complement_degree"

        A2 = torch.matmul(adj_comp, adj_comp)
        tri_pt = 0.5 * (adj_comp * A2).sum(dim=1)
        possible_tri = deg_pt * (deg_pt - 1.0)
        clu_pt = torch.zeros_like(deg_pt)
        mask = possible_tri > 0
        clu_pt[mask] = (2.0 * tri_pt[mask]) / possible_tri[mask]

        data = Data(x=torch.ones(n, 1), y=torch.ones(n, 1), num_nodes=n,
                    edge_index=edge_index,
                    degree=deg_pt.log().view(-1, 1),
                    cluster_coefficient=clu_pt.view(-1, 1),
                    triangle_count=tri_pt.view(-1, 1))
        batch = Batch.from_data_list([data]).to(self.device)
        with torch.inference_mode():
            output = self.net(batch)
            values = output.x.flatten().detach().cpu()
        self.sync()
        elapsed = time.perf_counter() - start
        if len(values) != n or not torch.isfinite(values).all():
            raise ValueError("COPT returned invalid scores; check checkpoint and preprocessing")
        return dict(zip(ids, values.tolist())), elapsed, None
