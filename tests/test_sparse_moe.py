import unittest
import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / 'model' / 'sparse_moe.py'
SPEC = importlib.util.spec_from_file_location('sthgcn_sparse_moe', MODULE_PATH)
SPARSE_MOE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPARSE_MOE)
HypergraphConditionedSharedSparseMoE = (
    SPARSE_MOE.HypergraphConditionedSharedSparseMoE
)


class HypergraphConditionedSharedSparseMoETest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            shared_alpha_init=0.15,
            router_context='hypergraph',
        )

    def test_zero_initialized_experts_preserve_backbone_output(self):
        final_state = torch.randn(5, 8)
        local_state = torch.randn(5, 8)
        output, balance_loss = self.moe(final_state, local_state)

        torch.testing.assert_close(output, final_state)
        self.assertEqual(output.shape, final_state.shape)
        self.assertTrue(torch.isfinite(balance_loss))

    def test_eval_diagnostics_record_top2_sparse_routing(self):
        self.moe.eval()
        final_state = torch.randn(7, 8)
        local_state = torch.randn(7, 8)
        self.moe(final_state, local_state)
        diagnostics = self.moe.diagnostics(reset=True)

        self.assertEqual(diagnostics['samples'], 7)
        self.assertAlmostEqual(sum(diagnostics['top1_fraction']), 1.0, places=6)
        self.assertAlmostEqual(sum(diagnostics['topk_fraction']), 1.0, places=6)
        self.assertEqual(diagnostics['expert_evaluations_per_sample'], 3)
        self.assertAlmostEqual(diagnostics['shared_alpha'], 0.15, places=6)
        self.assertEqual(self.moe.diagnostics()['samples'], 0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.moe(torch.randn(5, 8), torch.randn(4, 8))


if __name__ == '__main__':
    unittest.main()
