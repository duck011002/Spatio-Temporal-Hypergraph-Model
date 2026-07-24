import unittest

import numpy as np

try:
    import torch
    from model.llm_semantic_expert import LLMSemanticLogitExpert
except ImportError:
    torch = None
    LLMSemanticLogitExpert = None


@unittest.skipIf(torch is None, "PyTorch is unavailable in the local CPU runtime.")
class LlmSemanticExpertTest(unittest.TestCase):
    def make_expert(self):
        profiles = np.asarray(
            [
                [0.0, 0.2, 0.4],
                [1.0, 0.8, 0.6],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        return LLMSemanticLogitExpert(
            hidden_size=4,
            poi_profiles=profiles,
            profile_available=[True, True, False],
            rank=2,
            dropout=0.0,
            max_scale=0.5,
        )

    def test_zero_initialization_preserves_backbone_logits_exactly(self):
        expert = self.make_expert()
        hidden = torch.randn(5, 4)
        delta, profile_loss = expert(hidden)
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        self.assertEqual(float(profile_loss), 0.0)

    def test_scale_receives_gradient_then_unblocks_semantic_parameters(self):
        expert = self.make_expert()
        hidden = torch.randn(5, 4)
        loss = expert(hidden)[0].sum()
        loss.backward()
        self.assertIsNotNone(expert.raw_scale.grad)
        self.assertGreater(abs(float(expert.raw_scale.grad)), 0.0)

        expert.zero_grad(set_to_none=True)
        with torch.no_grad():
            expert.raw_scale.fill_(0.1)
        expert(hidden)[0].sum().backward()
        self.assertGreater(
            float(expert.profile_projector.weight.grad.abs().sum()),
            0.0,
        )

    def test_profile_distillation_warmup_preserves_logits_and_trains_predictor(self):
        profiles = np.asarray(
            [
                [0.0, 0.2, 0.4],
                [1.0, 0.8, 0.6],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        expert = LLMSemanticLogitExpert(
            hidden_size=4,
            poi_profiles=profiles,
            profile_available=[True, True, False],
            rank=2,
            dropout=0.0,
            max_scale=0.5,
            mode="profile_distillation",
            distill_warmup_steps=1,
        )
        hidden = torch.randn(2, 4)
        delta, profile_loss = expert(hidden, target_ids=torch.tensor([0, 1]))
        torch.testing.assert_close(delta, torch.zeros_like(delta), rtol=0, atol=0)
        self.assertGreater(float(profile_loss), 0.0)
        profile_loss.backward()
        self.assertGreater(
            float(expert.profile_decoder.weight.grad.abs().sum()),
            0.0,
        )

        with torch.no_grad():
            expert.raw_scale.fill_(0.1)
        delta, _ = expert(hidden, target_ids=torch.tensor([0, 1]))
        self.assertGreater(float(delta.abs().sum()), 0.0)
        torch.testing.assert_close(delta[:, 2], torch.zeros(2), rtol=0, atol=0)

    def test_profile_distillation_scale_cannot_be_negative(self):
        profiles = np.asarray([[0.0, 0.2], [1.0, 0.8]], dtype=np.float32)
        expert = LLMSemanticLogitExpert(
            hidden_size=4,
            poi_profiles=profiles,
            rank=2,
            dropout=0.0,
            mode="profile_distillation",
        )
        with torch.no_grad():
            expert.raw_scale.fill_(-1.0)
        self.assertEqual(float(expert.effective_scale), 0.0)


if __name__ == "__main__":
    unittest.main()
