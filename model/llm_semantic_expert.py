"""Lightweight logit experts distilled from offline LLM category profiles."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class LLMSemanticLogitExpert(nn.Module):
    """Match trajectory states to frozen LLM semantic profiles.

    ``joint_projection`` preserves the first pilot for ablation. The preferred
    ``profile_distillation`` mode first learns to predict the target POI's
    frozen semantic vector while emitting exactly zero logit residual. After
    warmup, only a non-negative calibrated residual can affect ranking.
    """

    MODES = ("joint_projection", "profile_distillation")

    def __init__(
        self,
        hidden_size,
        poi_profiles,
        profile_available=None,
        rank=32,
        dropout=0.1,
        max_scale=0.5,
        mode="joint_projection",
        distill_warmup_steps=0,
    ):
        super().__init__()
        profiles = torch.as_tensor(poi_profiles, dtype=torch.float32)
        if profiles.ndim != 2 or profiles.size(0) < 2:
            raise ValueError("poi_profiles must be a two-dimensional POI matrix.")
        if profile_available is None:
            available = torch.ones(profiles.size(0), dtype=torch.bool)
        else:
            available = torch.as_tensor(profile_available, dtype=torch.bool)
        if available.shape != (profiles.size(0),):
            raise ValueError("profile_available must align with POI rows.")
        if not bool(available.any()):
            raise ValueError("At least one semantic profile must be available.")
        if rank <= 0 or max_scale <= 0:
            raise ValueError("rank and max_scale must be positive.")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}.")
        if distill_warmup_steps < 0:
            raise ValueError("distill_warmup_steps must be non-negative.")

        normalized = profiles.clone()
        observed = profiles[available]
        mean = observed.mean(dim=0)
        std = observed.std(dim=0, unbiased=False).clamp_min(1e-6)
        normalized[available] = (observed - mean) / std
        normalized[~available] = 0.0

        self.hidden_size = int(hidden_size)
        self.profile_dim = int(profiles.size(1))
        self.rank = int(rank)
        self.max_scale = float(max_scale)
        self.mode = str(mode)
        self.distill_warmup_steps = int(distill_warmup_steps)
        self.register_buffer("poi_profiles", normalized)
        self.register_buffer("profile_available", available)
        self.register_buffer("profile_mean", mean)
        self.register_buffer("profile_std", std)
        self.register_buffer(
            "training_batches_seen",
            torch.zeros((), dtype=torch.long),
        )

        self.query_norm = nn.LayerNorm(self.hidden_size)
        self.query_projector = nn.Linear(
            self.hidden_size,
            self.rank,
            bias=False,
        )
        self.dropout = nn.Dropout(float(dropout))
        if self.mode == "joint_projection":
            self.profile_projector = nn.Linear(
                self.profile_dim,
                self.rank,
                bias=False,
            )
            nn.init.xavier_uniform_(self.profile_projector.weight)
        else:
            self.profile_decoder = nn.Linear(self.rank, self.profile_dim)
            nn.init.xavier_uniform_(self.profile_decoder.weight)
            nn.init.zeros_(self.profile_decoder.bias)
        self.gate = nn.Linear(self.hidden_size, 1)
        self.raw_scale = nn.Parameter(torch.zeros(()))

        nn.init.xavier_uniform_(self.query_projector.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.last_diagnostics = None

    @property
    def warmup_active(self):
        return (
            self.mode == "profile_distillation"
            and int(self.training_batches_seen.item())
            < self.distill_warmup_steps
        )

    @property
    def effective_scale(self):
        if self.mode == "joint_projection":
            return self.max_scale * torch.tanh(self.raw_scale)
        zero_point = F.softplus(torch.zeros_like(self.raw_scale))
        positive = F.softplus(self.raw_scale) - zero_point
        return self.max_scale * torch.clamp(positive, min=0.0, max=1.0)

    def _joint_projection(self, hidden):
        query = self.query_projector(self.query_norm(hidden))
        query = self.dropout(query)
        keys = self.profile_projector(self.poi_profiles)
        logits = torch.matmul(query, keys.transpose(0, 1))
        return logits / math.sqrt(self.rank), logits.sum() * 0.0

    def _profile_distillation(self, hidden, target_ids):
        encoded = self.query_projector(self.query_norm(hidden))
        encoded = self.dropout(F.gelu(encoded))
        predicted = self.profile_decoder(encoded)

        predicted_sq = predicted.square().mean(dim=1, keepdim=True)
        profile_sq = self.poi_profiles.square().mean(dim=1).unsqueeze(0)
        cross = torch.matmul(predicted, self.poi_profiles.transpose(0, 1))
        cross = 2.0 * cross / float(self.profile_dim)
        semantic_logits = -(predicted_sq + profile_sq - cross)
        semantic_logits = semantic_logits.masked_fill(
            ~self.profile_available.unsqueeze(0),
            0.0,
        )

        if target_ids is None:
            profile_loss = predicted.sum() * 0.0
        else:
            target_ids = target_ids.long()
            valid_target = (
                (target_ids >= 0)
                & (target_ids < self.poi_profiles.size(0))
            )
            safe_target = target_ids.clamp(
                min=0,
                max=self.poi_profiles.size(0) - 1,
            )
            valid_target = valid_target & self.profile_available[safe_target]
            if bool(valid_target.any()):
                target_profiles = self.poi_profiles[safe_target[valid_target]]
                profile_loss = F.smooth_l1_loss(
                    predicted[valid_target],
                    target_profiles,
                )
            else:
                profile_loss = predicted.sum() * 0.0
        return semantic_logits, profile_loss

    def forward(self, hidden, target_ids=None):
        if hidden.ndim != 2 or hidden.size(1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden shape [batch, {self.hidden_size}]."
            )
        if self.mode == "joint_projection":
            semantic_logits, profile_loss = self._joint_projection(hidden)
        else:
            semantic_logits, profile_loss = self._profile_distillation(
                hidden,
                target_ids,
            )

        gate = torch.sigmoid(self.gate(hidden))
        scale = self.effective_scale
        was_warmup_active = self.warmup_active
        if was_warmup_active:
            delta = semantic_logits * 0.0
        else:
            delta = scale * gate * semantic_logits

        if self.training and self.mode == "profile_distillation":
            self.training_batches_seen.add_(1)
        self.last_diagnostics = {
            "scale": scale.detach(),
            "gate_mean": gate.detach().mean(),
            "semantic_abs_mean": semantic_logits.detach().abs().mean(),
            "profile_loss": profile_loss.detach(),
            "warmup_active": torch.tensor(
                float(was_warmup_active),
                device=hidden.device,
            ),
            "training_batches_seen": self.training_batches_seen.detach().float(),
        }
        return delta, profile_loss

    def diagnostics(self):
        if self.last_diagnostics is None:
            return {
                "scale": self.effective_scale.detach(),
                "gate_mean": torch.tensor(float("nan")),
                "semantic_abs_mean": torch.tensor(float("nan")),
                "profile_loss": torch.tensor(float("nan")),
                "warmup_active": torch.tensor(float(self.warmup_active)),
                "training_batches_seen": self.training_batches_seen.detach().float(),
            }
        return dict(self.last_diagnostics)
