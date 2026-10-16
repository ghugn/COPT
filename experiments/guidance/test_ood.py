"""
Unit tests for Milestone M4: Out-of-Distribution (OOD) generators and benchmark integration.
"""
import unittest
from experiments.guidance.ood_generators import ba_graph, ws_graph, fetch_dimacs_instance
from experiments.guidance.solver import solve_online, is_clique
from experiments.guidance.gate import CostAwareGate


class TestOODGenerators(unittest.TestCase):
    
    def test_ba_generator(self):
        adj = ba_graph(n=40, m=10, seed=123)
        self.assertEqual(len(adj), 40)
        res = solve_online(adj, seconds=2.0)
        self.assertTrue(res["exhausted"])
        self.assertTrue(is_clique(adj, res["clique"]))
        
    def test_ws_generator(self):
        adj = ws_graph(n=40, k=20, p=0.2, seed=123)
        self.assertEqual(len(adj), 40)
        res = solve_online(adj, seconds=2.0)
        self.assertTrue(res["exhausted"])
        self.assertTrue(is_clique(adj, res["clique"]))
        
    def test_dimacs_sample(self):
        adj = fetch_dimacs_instance("sample")
        self.assertEqual(len(adj), 7)
        res = solve_online(adj, seconds=2.0)
        self.assertTrue(res["exhausted"])
        self.assertEqual(res["best_size"], 4)
        self.assertTrue(is_clique(adj, res["clique"]))
        
    def test_gate_on_ood_graphs(self):
        gate = CostAwareGate(mode="rule")
        # BA: low density & low average degree -> gate should skip
        adj_ba = ba_graph(n=60, m=10, seed=42)
        feats_ba = gate.extract_features(adj_ba, (1 << 60) - 1)
        self.assertFalse(gate.should_call(feats_ba))
        
        # WS with high k: high density & high degree -> gate can evaluate cleanly
        adj_ws = ws_graph(n=120, k=80, p=0.2, seed=42)
        feats_ws = gate.extract_features(adj_ws, (1 << 120) - 1)
        self.assertIsNotNone(feats_ws)
        self.assertIn("degree_mean", feats_ws)


if __name__ == "__main__":
    unittest.main()
