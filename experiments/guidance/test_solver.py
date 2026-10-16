import itertools
import random
import unittest

from experiments.guidance.solver import er_graph, is_clique, solve


def brute(adj):
    return max(len(c) for k in range(len(adj)+1) for c in itertools.combinations(range(len(adj)), k)
               if is_clique(adj, c))


class ExactnessTests(unittest.TestCase):
    def test_all_graphs_through_five_vertices(self):
        for n in range(1, 6):
            edges = list(itertools.combinations(range(n), 2))
            for mask in range(1 << len(edges)):
                adj = [0]*n
                for i, (u, v) in enumerate(edges):
                    if mask & (1 << i):
                        adj[u] |= 1 << v
                        adj[v] |= 1 << u
                expected = brute(adj)
                for priority in (None, {v: v for v in range(n)}, {v: -v for v in range(n)}):
                    result = solve(adj, priority=priority)
                    self.assertTrue(result["exhausted"])
                    self.assertEqual(result["best_size"], expected)

    def test_replay_with_global_incumbent(self):
        for seed in range(20):
            adj = er_graph(10, 0.65, seed)
            base = solve(adj, capture=10, capture_min=1)
            for state in base["snapshots"]:
                p = [v for v in range(10) if state["candidates"] & (1 << v)]
                local = [sum(1 << j for j, w in enumerate(p) if adj[v] & (1 << w)) for v in p]
                expected = max(len(state["incumbent"]), len(state["prefix"])+brute(local))
                rng = random.Random(seed)
                result = solve(adj, state["candidates"], state["prefix"], state["incumbent"],
                               priority={v: rng.random() for v in p})
                self.assertEqual(expected, result["best_size"])

    def test_timeout_is_not_proof(self):
        result = solve(er_graph(20, 0.5, 1), seconds=0)
        self.assertFalse(result["exhausted"])

    def test_random_larger_graphs(self):
        for seed in range(40):
            adj = er_graph(12, 0.15 + 0.2 * (seed % 4), seed)
            rng = random.Random(seed)
            result = solve(adj, priority={v: rng.random() for v in range(12)})
            self.assertTrue(result["exhausted"])
            self.assertEqual(brute(adj), result["best_size"])

    def test_solve_online_exactness(self):
        from experiments.guidance.solver import solve_online
        from experiments.guidance.gate import CostAwareGate
        gate = CostAwareGate(mode="rule")
        for seed in range(25):
            adj = er_graph(11, 0.45, seed)
            expected = brute(adj)
            res = solve_online(adj, gate=gate, top_k=3)
            self.assertTrue(res["exhausted"])
            self.assertEqual(expected, res["best_size"])


if __name__ == "__main__":
    unittest.main()
