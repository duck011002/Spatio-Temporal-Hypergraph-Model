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

    def test_single_adapter_mode_has_no_router_parameters(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=1,
            top_k=1,
            router_hidden_size=6,
            dropout=0.0,
            use_shared_expert=False,
            residual_scale=0.5,
        )
        final_state = torch.randn(5, 8)
        local_state = torch.randn(5, 8)
        output, balance_loss = moe(final_state, local_state)

        torch.testing.assert_close(output, final_state)
        self.assertIsNone(moe.router)
        self.assertFalse(any(name.startswith('router.') for name, _ in moe.named_parameters()))
        self.assertEqual(float(balance_loss.item()), 1.0)

        moe.eval()
        moe(final_state, local_state)
        diagnostics = moe.diagnostics()
        self.assertTrue(diagnostics['single_adapter_mode'])
        self.assertEqual(diagnostics['expert_evaluations_per_sample'], 1)
        self.assertEqual(diagnostics['active_experts'], 1)

    def test_grouped_top2_selects_experts_from_different_groups(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=1,
        )
        moe._expert_group_ids.copy_(torch.tensor([0, 0, 1, 1]))
        moe._grouping_finalized.fill_(True)
        gate_weights = torch.tensor([[0.50, 0.40, 0.06, 0.04]])

        _, topk_indices = moe._select_topk(gate_weights)

        self.assertEqual(topk_indices.tolist(), [[0, 2]])

    def test_adaptive_grouping_finalizes_after_warmup(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=1,
            group_similarity_threshold=0.5,
        )
        moe.train()
        moe(torch.randn(9, 8), torch.randn(9, 8))

        self.assertTrue(bool(moe._grouping_finalized.item()))
        self.assertEqual(int(moe._grouping_batches.item()), 1)
        self.assertGreaterEqual(
            len(torch.unique(moe._expert_group_ids)),
            moe.top_k,
        )

    def test_grouping_uses_uncentered_routing_coactivation(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=100,
            group_similarity_threshold=0.8,
        )
        moe._group_routing_gram.copy_(
            torch.tensor(
                [
                    [10.0, 9.0, 0.0, 0.0],
                    [9.0, 10.0, 0.0, 0.0],
                    [0.0, 0.0, 10.0, 0.0],
                    [0.0, 0.0, 0.0, 10.0],
                ]
            )
        )

        moe._finalize_adaptive_groups()

        self.assertEqual(moe._expert_group_ids.tolist(), [0, 0, 1, 2])

    def test_target_groups_use_globally_optimal_expert_pairs(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=100,
            target_num_groups=2,
        )
        moe._group_routing_gram.copy_(
            torch.tensor(
                [
                    [10.0, 1.0, 1.0, 9.0],
                    [1.0, 10.0, 8.0, 1.0],
                    [1.0, 8.0, 10.0, 1.0],
                    [9.0, 1.0, 1.0, 10.0],
                ]
            )
        )

        moe._finalize_adaptive_groups()

        self.assertEqual(moe._expert_group_ids.tolist(), [0, 1, 1, 0])
        diagnostics = moe.diagnostics()
        self.assertEqual(diagnostics['target_num_groups'], 2)
        self.assertEqual(diagnostics['num_groups'], 2)
        self.assertEqual(diagnostics['group_sizes'], [2, 2])

    def test_grouping_state_is_checkpointed_and_legacy_state_still_loads(self):
        source = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=1,
        )
        source._expert_group_ids.copy_(torch.tensor([0, 0, 1, 1]))
        source._grouping_finalized.fill_(True)
        state_dict = source.state_dict()

        restored = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            adaptive_grouping=True,
            group_warmup_steps=1,
        )
        restored.load_state_dict(state_dict)
        self.assertEqual(restored._expert_group_ids.tolist(), [0, 0, 1, 1])
        self.assertTrue(bool(restored._grouping_finalized.item()))

        legacy_state = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith('_group')
            and key != '_expert_group_ids'
        }
        legacy_target = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
        )
        legacy_target.load_state_dict(legacy_state)

    def test_aux_loss_free_bias_balances_selection_without_changing_weights(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            aux_loss_free=True,
            router_bias_update_rate=0.001,
        )
        gate_weights = torch.tensor([[0.40, 0.30, 0.20, 0.10]])
        moe._router_selection_bias.copy_(
            torch.tensor([-0.25, 0.00, 0.15, 0.00])
        )

        _, selected = moe._select_topk(moe._selection_scores(gate_weights))
        mixture_weights = gate_weights.gather(1, selected)
        mixture_weights = mixture_weights / mixture_weights.sum(dim=-1, keepdim=True)

        self.assertEqual(selected.tolist(), [[2, 1]])
        torch.testing.assert_close(
            mixture_weights,
            torch.tensor([[0.40, 0.60]]),
        )

        moe.train()
        moe._router_selection_bias.zero_()
        moe._update_router_selection_bias(torch.tensor([[0, 1], [0, 1]]))
        torch.testing.assert_close(
            moe._router_selection_bias,
            torch.tensor([-0.001, -0.001, 0.001, 0.001]),
        )

        _, balance_loss = moe(torch.randn(5, 8), torch.randn(5, 8))
        self.assertEqual(float(balance_loss.item()), 0.0)

    def test_adaptive_dual_gates_start_at_a3_fusion_values(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            shared_alpha_init=0.15,
            residual_scale=0.5,
            adaptive_shared_gate=True,
            adaptive_residual_gate=True,
        )
        self.assertEqual(int(torch.count_nonzero(moe.shared_gate.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(moe.residual_gate.weight)), 0)

        final_state = torch.randn(7, 8)
        local_state = torch.randn(7, 8)
        moe.eval()
        output, _ = moe(final_state, local_state)
        diagnostics = moe.diagnostics()

        torch.testing.assert_close(output, final_state)
        self.assertAlmostEqual(diagnostics['shared_gate_mean'], 0.15, places=6)
        self.assertAlmostEqual(diagnostics['shared_gate_std'], 0.0, places=6)
        self.assertAlmostEqual(diagnostics['residual_gate_mean'], 0.5, places=6)
        self.assertAlmostEqual(diagnostics['residual_gate_std'], 0.0, places=6)
        self.assertTrue(diagnostics['adaptive_shared_gate'])
        self.assertTrue(diagnostics['adaptive_residual_gate'])

    def test_a3_checkpoint_can_initialize_adaptive_dual_gates(self):
        source = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            shared_alpha_init=0.15,
            residual_scale=0.5,
        )
        target = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            shared_alpha_init=0.15,
            residual_scale=0.5,
            aux_loss_free=True,
            router_bias_update_rate=0.001,
            adaptive_shared_gate=True,
            adaptive_residual_gate=True,
        )

        target.load_state_dict(source.state_dict())

        self.assertEqual(int(torch.count_nonzero(target.shared_gate.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(target.residual_gate.weight)), 0)
        self.assertEqual(
            int(torch.count_nonzero(target._router_selection_bias)),
            0,
        )

    def test_adaptive_dual_gates_receive_gradients_after_experts_activate(self):
        moe = HypergraphConditionedSharedSparseMoE(
            hidden_size=8,
            rank=3,
            num_experts=4,
            top_k=2,
            router_hidden_size=6,
            dropout=0.0,
            shared_alpha_init=0.15,
            residual_scale=0.5,
            adaptive_shared_gate=True,
            adaptive_residual_gate=True,
        )
        for expert in list(moe.experts) + [moe.shared_expert]:
            torch.nn.init.normal_(expert.up.weight, std=0.1)

        output, _ = moe(torch.randn(7, 8), torch.randn(7, 8))
        output.square().mean().backward()

        self.assertGreater(
            float(torch.linalg.vector_norm(moe.shared_gate.weight.grad)),
            0.0,
        )
        self.assertGreater(
            float(torch.linalg.vector_norm(moe.residual_gate.weight.grad)),
            0.0,
        )


if __name__ == '__main__':
    unittest.main()
