import math

import torch
from torch import nn
import torch.nn.functional as F


class LowRankAdapter(nn.Module):
    """A parameter-efficient residual expert with a zero-initialized output."""

    def __init__(self, hidden_size, rank, dropout):
        super(LowRankAdapter, self).__init__()
        self.down = nn.Linear(hidden_size, rank)
        self.up = nn.Linear(rank, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.dropout(self.activation(self.down(x))))


class HypergraphConditionedSharedSparseMoE(nn.Module):
    """Top-2 low-rank MoE conditioned on local and hypergraph representations."""

    def __init__(
        self,
        hidden_size,
        rank=32,
        num_experts=4,
        top_k=2,
        router_hidden_size=128,
        dropout=0.1,
        use_shared_expert=True,
        shared_alpha_init=0.15,
        residual_scale=1.0,
        router_context='hypergraph',
    ):
        super(HypergraphConditionedSharedSparseMoE, self).__init__()
        if num_experts < 2:
            raise ValueError('num_experts must be at least 2')
        if top_k < 1 or top_k > num_experts:
            raise ValueError('top_k must be between 1 and num_experts')
        if rank < 1:
            raise ValueError('rank must be positive')
        if router_hidden_size < 1:
            raise ValueError('router_hidden_size must be positive')
        if use_shared_expert and not 0.0 < shared_alpha_init < 1.0:
            raise ValueError('shared_alpha_init must be strictly between 0 and 1')
        if router_context not in ('local', 'final', 'hypergraph'):
            raise ValueError("router_context must be 'local', 'final', or 'hypergraph'")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = bool(use_shared_expert)
        self.residual_scale = float(residual_scale)
        self.router_context = router_context

        router_input_size = hidden_size * 3 if router_context == 'hypergraph' else hidden_size
        self.router_norm = nn.LayerNorm(router_input_size)
        self.router = nn.Sequential(
            nn.Linear(router_input_size, router_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_size, num_experts),
        )
        self.expert_norm = nn.LayerNorm(hidden_size)
        self.experts = nn.ModuleList(
            LowRankAdapter(hidden_size, rank, dropout) for _ in range(num_experts)
        )
        if self.use_shared_expert:
            self.shared_expert = LowRankAdapter(hidden_size, rank, dropout)
            alpha_logit = math.log(shared_alpha_init / (1.0 - shared_alpha_init))
            self.shared_alpha = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        self.reset_diagnostics()

    def _router_input(self, local_state, final_state):
        if self.router_context == 'local':
            return local_state
        if self.router_context == 'final':
            return final_state
        collaboration_delta = final_state - local_state
        return torch.cat([local_state, final_state, collaboration_delta], dim=-1)

    def reset_diagnostics(self):
        self._diag_samples = 0
        self._diag_top1_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self._diag_topk_counts = torch.zeros(self.num_experts, dtype=torch.long)
        self._diag_entropy_sum = 0.0
        self._diag_aux_sum = 0.0
        self._diag_shared_norm_sum = 0.0
        self._diag_routed_norm_sum = 0.0

    def _record_diagnostics(self, gate_weights, topk_indices, aux_loss, shared_out, routed_out):
        if self.training:
            return
        sample_count = int(gate_weights.size(0))
        self._diag_samples += sample_count
        self._diag_top1_counts += torch.bincount(
            topk_indices[:, 0].detach().cpu(), minlength=self.num_experts
        )
        self._diag_topk_counts += torch.bincount(
            topk_indices.detach().reshape(-1).cpu(), minlength=self.num_experts
        )
        entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-9))).sum(dim=-1)
        self._diag_entropy_sum += float(entropy.detach().sum().cpu().item())
        self._diag_aux_sum += float(aux_loss.detach().cpu().item()) * sample_count
        self._diag_shared_norm_sum += float(
            torch.linalg.vector_norm(shared_out.detach().float(), dim=-1).sum().cpu().item()
        )
        self._diag_routed_norm_sum += float(
            torch.linalg.vector_norm(routed_out.detach().float(), dim=-1).sum().cpu().item()
        )

    def diagnostics(self, reset=False):
        samples = max(self._diag_samples, 1)
        result = {
            'samples': self._diag_samples,
            'top1_fraction': (self._diag_top1_counts.float() / samples).tolist(),
            'topk_fraction': (
                self._diag_topk_counts.float() / max(self._diag_samples * self.top_k, 1)
            ).tolist(),
            'router_entropy': self._diag_entropy_sum / samples,
            'balance_loss': self._diag_aux_sum / samples,
            'shared_output_norm': self._diag_shared_norm_sum / samples,
            'routed_output_norm': self._diag_routed_norm_sum / samples,
            'shared_alpha': (
                float(torch.sigmoid(self.shared_alpha.detach()).cpu().item())
                if self.use_shared_expert else 0.0
            ),
            'expert_evaluations_per_sample': self.top_k + int(self.use_shared_expert),
        }
        if reset:
            self.reset_diagnostics()
        return result

    def forward(self, final_state, local_state):
        if final_state.shape != local_state.shape:
            raise ValueError(
                'final_state and local_state must have identical shapes, got '
                f'{tuple(final_state.shape)} and {tuple(local_state.shape)}'
            )

        router_input = self.router_norm(self._router_input(local_state, final_state))
        gate_weights = F.softmax(self.router(router_input), dim=-1)
        topk_weights, topk_indices = torch.topk(gate_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        expert_input = self.expert_norm(final_state)
        routed_out = torch.zeros_like(final_state)
        for expert_index, expert in enumerate(self.experts):
            sample_indices, topk_slots = torch.where(topk_indices == expert_index)
            if sample_indices.numel() == 0:
                continue
            expert_output = expert(expert_input.index_select(0, sample_indices))
            selected_weights = topk_weights[sample_indices, topk_slots].to(expert_output.dtype)
            routed_out.index_add_(
                0,
                sample_indices,
                expert_output * selected_weights.unsqueeze(-1),
            )

        if self.use_shared_expert:
            shared_out = self.shared_expert(expert_input)
            shared_alpha = torch.sigmoid(self.shared_alpha).to(final_state.dtype)
            moe_delta = shared_alpha * shared_out + (1.0 - shared_alpha) * routed_out
        else:
            shared_out = torch.zeros_like(routed_out)
            moe_delta = routed_out

        top1_fraction = F.one_hot(
            topk_indices[:, 0], num_classes=self.num_experts
        ).float().mean(dim=0)
        mean_probability = gate_weights.mean(dim=0)
        balance_loss = self.num_experts * torch.sum(top1_fraction * mean_probability)
        self._record_diagnostics(
            gate_weights, topk_indices, balance_loss, shared_out, routed_out
        )
        return final_state + self.residual_scale * moe_delta, balance_loss
