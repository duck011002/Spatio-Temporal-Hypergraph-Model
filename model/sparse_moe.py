import math
import logging

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
        adaptive_grouping=False,
        group_warmup_steps=3000,
        group_similarity_threshold=0.5,
    ):
        super(HypergraphConditionedSharedSparseMoE, self).__init__()
        if num_experts < 1:
            raise ValueError('num_experts must be at least 1')
        if top_k < 1 or top_k > num_experts:
            raise ValueError('top_k must be between 1 and num_experts')
        if num_experts == 1 and top_k != 1:
            raise ValueError('single-adapter mode requires top_k=1')
        if num_experts == 1 and use_shared_expert:
            raise ValueError('single-adapter mode must disable the shared expert')
        if rank < 1:
            raise ValueError('rank must be positive')
        if router_hidden_size < 1:
            raise ValueError('router_hidden_size must be positive')
        if use_shared_expert and not 0.0 < shared_alpha_init < 1.0:
            raise ValueError('shared_alpha_init must be strictly between 0 and 1')
        if router_context not in ('local', 'final', 'hypergraph'):
            raise ValueError("router_context must be 'local', 'final', or 'hypergraph'")
        if adaptive_grouping and num_experts <= top_k:
            raise ValueError('adaptive grouping requires num_experts > top_k')
        if group_warmup_steps < 1:
            raise ValueError('group_warmup_steps must be positive')
        if not -1.0 <= group_similarity_threshold <= 1.0:
            raise ValueError('group_similarity_threshold must be between -1 and 1')

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_shared_expert = bool(use_shared_expert)
        self.residual_scale = float(residual_scale)
        self.router_context = router_context
        self.single_adapter_mode = num_experts == 1
        self.adaptive_grouping = bool(adaptive_grouping)
        self.group_warmup_steps = int(group_warmup_steps)
        self.group_similarity_threshold = float(group_similarity_threshold)

        if self.single_adapter_mode:
            # The control contains exactly one normalized low-rank adapter and
            # no unused router parameters.
            self.router_norm = nn.Identity()
            self.router = None
        else:
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

        # Adaptive grouping keeps all experts. During a short warm-up it stores
        # only O(E^2) routing statistics, then clusters experts once and freezes
        # the assignment for the rest of training and inference.
        self.register_buffer(
            '_group_routing_gram',
            torch.zeros(num_experts, num_experts, dtype=torch.float32),
        )
        self.register_buffer(
            '_group_routing_sum',
            torch.zeros(num_experts, dtype=torch.float32),
        )
        self.register_buffer(
            '_grouping_samples',
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            '_grouping_batches',
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            '_expert_group_ids',
            torch.arange(num_experts, dtype=torch.long),
        )
        self.register_buffer(
            '_grouping_finalized',
            torch.tensor(not self.adaptive_grouping, dtype=torch.bool),
        )
        self.reset_diagnostics()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # R1 checkpoints created before adaptive grouping do not contain these
        # buffers. Supplying current defaults preserves strict legacy loading,
        # while new checkpoints retain finalized expert assignments.
        grouping_buffers = (
            '_group_routing_gram',
            '_group_routing_sum',
            '_grouping_samples',
            '_grouping_batches',
            '_expert_group_ids',
            '_grouping_finalized',
        )
        for buffer_name in grouping_buffers:
            state_key = prefix + buffer_name
            if state_key not in state_dict:
                state_dict[state_key] = getattr(self, buffer_name)
        super(HypergraphConditionedSharedSparseMoE, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

    def _update_adaptive_grouping(self, gate_weights):
        if (
            not self.training
            or not self.adaptive_grouping
            or bool(self._grouping_finalized.item())
        ):
            return

        with torch.no_grad():
            routing = gate_weights.detach().float()
            self._group_routing_gram.add_(routing.transpose(0, 1).matmul(routing))
            self._group_routing_sum.add_(routing.sum(dim=0))
            self._grouping_samples.add_(routing.size(0))
            self._grouping_batches.add_(1)
            if int(self._grouping_batches.item()) >= self.group_warmup_steps:
                self._finalize_adaptive_groups()

    def _finalize_adaptive_groups(self):
        if bool(self._grouping_finalized.item()):
            return

        gram = self._group_routing_gram.detach().cpu()
        routing_norms = gram.diag().clamp_min(0.0).sqrt()
        denominator = torch.outer(routing_norms, routing_norms).clamp_min(1e-12)
        similarity = (gram / denominator).clamp(0.0, 1.0)
        similarity.fill_diagonal_(1.0)

        groups = [[expert_index] for expert_index in range(self.num_experts)]
        while len(groups) > self.top_k:
            best_pair = None
            best_similarity = -float('inf')
            for left_index in range(len(groups)):
                for right_index in range(left_index + 1, len(groups)):
                    pair_values = similarity[groups[left_index]][:, groups[right_index]]
                    pair_similarity = float(pair_values.mean().item())
                    if pair_similarity > best_similarity:
                        best_similarity = pair_similarity
                        best_pair = (left_index, right_index)

            if (
                best_pair is None
                or best_similarity < self.group_similarity_threshold
            ):
                break
            left_index, right_index = best_pair
            groups[left_index] = groups[left_index] + groups[right_index]
            del groups[right_index]

        group_ids = torch.empty(self.num_experts, dtype=torch.long)
        for group_index, expert_indices in enumerate(groups):
            group_ids[expert_indices] = group_index
        self._expert_group_ids.copy_(group_ids.to(self._expert_group_ids.device))
        self._grouping_finalized.fill_(True)
        logging.info(
            '[MoE] Adaptive expert grouping finalized after %s batches: %s',
            int(self._grouping_batches.item()),
            groups,
        )

    def _select_topk(self, gate_weights):
        if (
            not self.adaptive_grouping
            or not bool(self._grouping_finalized.item())
        ):
            return torch.topk(gate_weights, self.top_k, dim=-1)

        group_ids = self._expert_group_ids.to(gate_weights.device)
        num_groups = int(group_ids.max().item()) + 1
        if num_groups < self.top_k:
            return torch.topk(gate_weights, self.top_k, dim=-1)

        group_scores = []
        for group_index in range(num_groups):
            group_scores.append(
                gate_weights[:, group_ids == group_index].max(dim=-1).values
            )
        group_scores = torch.stack(group_scores, dim=-1)
        _, selected_groups = torch.topk(group_scores, self.top_k, dim=-1)

        selected_experts = []
        for slot_index in range(self.top_k):
            membership = (
                group_ids.unsqueeze(0)
                == selected_groups[:, slot_index].unsqueeze(1)
            )
            candidate_scores = gate_weights.masked_fill(
                ~membership,
                torch.finfo(gate_weights.dtype).min,
            )
            selected_experts.append(candidate_scores.argmax(dim=-1))
        topk_indices = torch.stack(selected_experts, dim=-1)
        topk_weights = gate_weights.gather(1, topk_indices)
        return topk_weights, topk_indices

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
        top1_fraction = self._diag_top1_counts.float() / samples
        topk_fraction = (
            self._diag_topk_counts.float()
            / max(self._diag_samples * self.top_k, 1)
        )
        router_entropy = self._diag_entropy_sum / samples
        max_entropy = math.log(self.num_experts) if self.num_experts > 1 else 0.0
        group_ids = self._expert_group_ids.detach().cpu()
        num_groups = int(group_ids.max().item()) + 1
        group_topk_fraction = []
        for group_index in range(num_groups):
            group_topk_fraction.append(
                float(topk_fraction[group_ids == group_index].sum().item())
            )
        result = {
            'samples': self._diag_samples,
            'top1_fraction': top1_fraction.tolist(),
            'topk_fraction': topk_fraction.tolist(),
            'router_entropy': router_entropy,
            'normalized_router_entropy': (
                router_entropy / max_entropy if max_entropy > 0.0 else 0.0
            ),
            'active_experts': int((self._diag_topk_counts > 0).sum().item()),
            'max_top1_fraction': float(top1_fraction.max().item()),
            'balance_loss': self._diag_aux_sum / samples,
            'shared_output_norm': self._diag_shared_norm_sum / samples,
            'routed_output_norm': self._diag_routed_norm_sum / samples,
            'shared_alpha': (
                float(torch.sigmoid(self.shared_alpha.detach()).cpu().item())
                if self.use_shared_expert else 0.0
            ),
            'expert_evaluations_per_sample': self.top_k + int(self.use_shared_expert),
            'single_adapter_mode': self.single_adapter_mode,
            'adaptive_grouping': self.adaptive_grouping,
            'grouping_finalized': bool(self._grouping_finalized.item()),
            'grouping_batches': int(self._grouping_batches.item()),
            'expert_group_ids': group_ids.tolist(),
            'group_topk_fraction': group_topk_fraction,
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

        if self.single_adapter_mode:
            gate_weights = final_state.new_ones((final_state.size(0), 1))
        else:
            router_input = self.router_norm(self._router_input(local_state, final_state))
            gate_weights = F.softmax(self.router(router_input), dim=-1)
        self._update_adaptive_grouping(gate_weights)
        topk_weights, topk_indices = self._select_topk(gate_weights)
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
