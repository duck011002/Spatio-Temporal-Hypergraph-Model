import math
import logging
from functools import lru_cache

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
        target_num_groups=0,
        aux_loss_free=False,
        router_bias_update_rate=0.0,
        adaptive_shared_gate=False,
        adaptive_residual_gate=False,
        residual_gate_max_delta=0.0,
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
        if target_num_groups < 0 or target_num_groups > num_experts:
            raise ValueError('target_num_groups must be between 0 and num_experts')
        if target_num_groups and not adaptive_grouping:
            raise ValueError('target_num_groups requires adaptive grouping')
        if target_num_groups and target_num_groups < top_k:
            raise ValueError('target_num_groups must be at least top_k')
        if router_bias_update_rate < 0.0:
            raise ValueError('router_bias_update_rate must be non-negative')
        if adaptive_shared_gate and not use_shared_expert:
            raise ValueError('adaptive_shared_gate requires the shared expert')
        if adaptive_residual_gate and not 0.0 < residual_scale < 1.0:
            raise ValueError(
                'adaptive_residual_gate requires residual_scale strictly between 0 and 1'
            )
        if residual_gate_max_delta < 0.0:
            raise ValueError('residual_gate_max_delta must be non-negative')
        if residual_gate_max_delta and not adaptive_residual_gate:
            raise ValueError(
                'residual_gate_max_delta requires adaptive_residual_gate'
            )
        if residual_gate_max_delta > min(residual_scale, 1.0 - residual_scale):
            raise ValueError(
                'residual_gate_max_delta keeps the bounded residual gate outside [0, 1]'
            )

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
        self.target_num_groups = int(target_num_groups)
        self.aux_loss_free = bool(aux_loss_free)
        self.router_bias_update_rate = float(router_bias_update_rate)
        self.adaptive_shared_gate = bool(adaptive_shared_gate)
        self.adaptive_residual_gate = bool(adaptive_residual_gate)
        self.residual_gate_max_delta = float(residual_gate_max_delta)
        router_input_size = (
            hidden_size * 3 if router_context == 'hypergraph' else hidden_size
        )

        if self.single_adapter_mode:
            # The control contains exactly one normalized low-rank adapter and
            # no unused router parameters.
            self.router_norm = nn.Identity()
            self.router = None
        else:
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
            if self.adaptive_shared_gate:
                self.shared_gate = nn.Linear(router_input_size, 1, bias=False)
                nn.init.zeros_(self.shared_gate.weight)
        if self.adaptive_residual_gate:
            residual_logit = math.log(residual_scale / (1.0 - residual_scale))
            self.register_buffer(
                '_residual_scale_logit',
                torch.tensor(residual_logit, dtype=torch.float32),
            )
            self.residual_gate = nn.Linear(router_input_size, 1, bias=False)
            nn.init.zeros_(self.residual_gate.weight)

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
        # Following auxiliary-loss-free MoE routing, this expert-wise bias is
        # used only for Top-k selection. The unbiased router probabilities
        # still determine mixture weights and adaptive-grouping statistics.
        self.register_buffer(
            '_router_selection_bias',
            torch.zeros(num_experts, dtype=torch.float32),
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
            '_router_selection_bias',
        )
        for buffer_name in grouping_buffers:
            state_key = prefix + buffer_name
            if state_key not in state_dict:
                state_dict[state_key] = getattr(self, buffer_name)
        optional_module_state = []
        if self.adaptive_shared_gate:
            optional_module_state.append('shared_gate.weight')
        if self.adaptive_residual_gate:
            optional_module_state.extend(
                ('_residual_scale_logit', 'residual_gate.weight')
            )
        current_state = self.state_dict()
        for local_key in optional_module_state:
            state_key = prefix + local_key
            if state_key not in state_dict:
                state_dict[state_key] = current_state[local_key]
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
        self._diag_balance_proxy_sum = 0.0
        self._diag_shared_norm_sum = 0.0
        self._diag_routed_norm_sum = 0.0
        self._diag_shared_gate_sum = 0.0
        self._diag_shared_gate_sq_sum = 0.0
        self._diag_shared_gate_min = float('inf')
        self._diag_shared_gate_max = -float('inf')
        self._diag_residual_gate_sum = 0.0
        self._diag_residual_gate_sq_sum = 0.0
        self._diag_residual_gate_min = float('inf')
        self._diag_residual_gate_max = -float('inf')

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

        if self.target_num_groups * 2 == self.num_experts:
            groups = self._optimal_pair_groups(similarity)
        else:
            groups = [[expert_index] for expert_index in range(self.num_experts)]
            minimum_group_count = (
                self.target_num_groups
                if self.target_num_groups
                else self.top_k
            )
            maximum_group_size = (
                math.ceil(self.num_experts / self.target_num_groups)
                if self.target_num_groups
                else self.num_experts
            )
            while len(groups) > minimum_group_count:
                best_pair = None
                best_similarity = -float('inf')
                for left_index in range(len(groups)):
                    for right_index in range(left_index + 1, len(groups)):
                        if (
                            len(groups[left_index]) + len(groups[right_index])
                            > maximum_group_size
                        ):
                            continue
                        pair_values = similarity[groups[left_index]][:, groups[right_index]]
                        pair_similarity = float(pair_values.mean().item())
                        if pair_similarity > best_similarity:
                            best_similarity = pair_similarity
                            best_pair = (left_index, right_index)

                if best_pair is None:
                    raise RuntimeError(
                        'unable to satisfy target_num_groups with balanced group sizes'
                    )
                if (
                    not self.target_num_groups
                    and best_similarity < self.group_similarity_threshold
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
            '[MoE] Adaptive expert grouping finalized after %s batches '
            '(target_groups=%s): %s',
            int(self._grouping_batches.item()),
            self.target_num_groups or 'threshold',
            groups,
        )

    def _optimal_pair_groups(self, similarity):
        if self.num_experts % 2 != 0:
            raise ValueError('optimal pair grouping requires an even number of experts')

        @lru_cache(maxsize=None)
        def solve(mask):
            if mask == 0:
                return 0.0, ()

            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            remaining = mask ^ first_bit
            best_score = -float('inf')
            best_pairs = None

            candidate_mask = remaining
            while candidate_mask:
                partner_bit = candidate_mask & -candidate_mask
                partner = partner_bit.bit_length() - 1
                tail_score, tail_pairs = solve(remaining ^ partner_bit)
                score = float(similarity[first, partner].item()) + tail_score
                pairs = ((first, partner),) + tail_pairs
                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and (best_pairs is None or pairs < best_pairs)
                    )
                ):
                    best_score = score
                    best_pairs = pairs
                candidate_mask ^= partner_bit

            return best_score, best_pairs

        _, pairs = solve((1 << self.num_experts) - 1)
        return [list(pair) for pair in pairs]

    def _selection_scores(self, gate_weights):
        if self.router_bias_update_rate == 0.0:
            return gate_weights
        return gate_weights + self._router_selection_bias.to(
            device=gate_weights.device,
            dtype=gate_weights.dtype,
        )

    def _select_topk(self, selection_scores):
        if (
            not self.adaptive_grouping
            or not bool(self._grouping_finalized.item())
        ):
            return torch.topk(selection_scores, self.top_k, dim=-1)

        group_ids = self._expert_group_ids.to(selection_scores.device)
        num_groups = int(group_ids.max().item()) + 1
        if num_groups < self.top_k:
            return torch.topk(selection_scores, self.top_k, dim=-1)

        group_scores = []
        for group_index in range(num_groups):
            group_scores.append(
                selection_scores[:, group_ids == group_index].max(dim=-1).values
            )
        group_scores = torch.stack(group_scores, dim=-1)
        _, selected_groups = torch.topk(group_scores, self.top_k, dim=-1)

        selected_experts = []
        for slot_index in range(self.top_k):
            membership = (
                group_ids.unsqueeze(0)
                == selected_groups[:, slot_index].unsqueeze(1)
            )
            candidate_scores = selection_scores.masked_fill(
                ~membership,
                torch.finfo(selection_scores.dtype).min,
            )
            selected_experts.append(candidate_scores.argmax(dim=-1))
        topk_indices = torch.stack(selected_experts, dim=-1)
        topk_scores = selection_scores.gather(1, topk_indices)
        return topk_scores, topk_indices

    def _update_router_selection_bias(self, topk_indices):
        if not self.training or self.router_bias_update_rate == 0.0:
            return
        with torch.no_grad():
            expert_load = torch.bincount(
                topk_indices.detach().reshape(-1),
                minlength=self.num_experts,
            ).to(self._router_selection_bias)
            mean_load = expert_load.mean()
            self._router_selection_bias.add_(
                self.router_bias_update_rate * torch.sign(mean_load - expert_load)
            )

    @staticmethod
    def _gate_summary(gate_values):
        detached = gate_values.detach().double().reshape(-1)
        return (
            float(detached.sum().cpu().item()),
            float(detached.square().sum().cpu().item()),
            float(detached.min().cpu().item()),
            float(detached.max().cpu().item()),
        )

    def _record_diagnostics(
        self,
        gate_weights,
        topk_indices,
        aux_loss,
        balance_proxy,
        shared_out,
        routed_out,
        shared_gate,
        residual_gate,
    ):
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
        self._diag_balance_proxy_sum += (
            float(balance_proxy.detach().cpu().item()) * sample_count
        )
        self._diag_shared_norm_sum += float(
            torch.linalg.vector_norm(shared_out.detach().float(), dim=-1).sum().cpu().item()
        )
        self._diag_routed_norm_sum += float(
            torch.linalg.vector_norm(routed_out.detach().float(), dim=-1).sum().cpu().item()
        )
        if shared_gate is not None:
            gate_sum, gate_sq_sum, gate_min, gate_max = self._gate_summary(shared_gate)
            self._diag_shared_gate_sum += gate_sum
            self._diag_shared_gate_sq_sum += gate_sq_sum
            self._diag_shared_gate_min = min(self._diag_shared_gate_min, gate_min)
            self._diag_shared_gate_max = max(self._diag_shared_gate_max, gate_max)
        gate_sum, gate_sq_sum, gate_min, gate_max = self._gate_summary(residual_gate)
        self._diag_residual_gate_sum += gate_sum
        self._diag_residual_gate_sq_sum += gate_sq_sum
        self._diag_residual_gate_min = min(self._diag_residual_gate_min, gate_min)
        self._diag_residual_gate_max = max(self._diag_residual_gate_max, gate_max)

    @staticmethod
    def _mean_std(sum_value, square_sum, sample_count):
        mean = sum_value / sample_count
        variance = max(square_sum / sample_count - mean * mean, 0.0)
        return mean, math.sqrt(variance)

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
        base_shared_alpha = (
            float(torch.sigmoid(self.shared_alpha.detach()).cpu().item())
            if self.use_shared_expert else 0.0
        )
        if self._diag_samples and self.use_shared_expert:
            shared_gate_mean, shared_gate_std = self._mean_std(
                self._diag_shared_gate_sum,
                self._diag_shared_gate_sq_sum,
                samples,
            )
            shared_gate_min = self._diag_shared_gate_min
            shared_gate_max = self._diag_shared_gate_max
        else:
            shared_gate_mean = base_shared_alpha
            shared_gate_std = 0.0
            shared_gate_min = base_shared_alpha
            shared_gate_max = base_shared_alpha
        if self._diag_samples:
            residual_gate_mean, residual_gate_std = self._mean_std(
                self._diag_residual_gate_sum,
                self._diag_residual_gate_sq_sum,
                samples,
            )
            residual_gate_min = self._diag_residual_gate_min
            residual_gate_max = self._diag_residual_gate_max
        else:
            residual_gate_mean = self.residual_scale
            residual_gate_std = 0.0
            residual_gate_min = self.residual_scale
            residual_gate_max = self.residual_scale
        if self.adaptive_residual_gate and self.residual_gate_max_delta:
            residual_gate_mode = 'bounded_tanh'
            residual_gate_lower_bound = (
                self.residual_scale - self.residual_gate_max_delta
            )
            residual_gate_upper_bound = (
                self.residual_scale + self.residual_gate_max_delta
            )
        elif self.adaptive_residual_gate:
            residual_gate_mode = 'sigmoid'
            residual_gate_lower_bound = 0.0
            residual_gate_upper_bound = 1.0
        else:
            residual_gate_mode = 'fixed'
            residual_gate_lower_bound = self.residual_scale
            residual_gate_upper_bound = self.residual_scale
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
            'balance_proxy': self._diag_balance_proxy_sum / samples,
            'shared_output_norm': self._diag_shared_norm_sum / samples,
            'routed_output_norm': self._diag_routed_norm_sum / samples,
            'shared_alpha': shared_gate_mean,
            'shared_gate_mean': shared_gate_mean,
            'shared_gate_std': shared_gate_std,
            'shared_gate_min': shared_gate_min,
            'shared_gate_max': shared_gate_max,
            'residual_gate_mean': residual_gate_mean,
            'residual_gate_std': residual_gate_std,
            'residual_gate_min': residual_gate_min,
            'residual_gate_max': residual_gate_max,
            'residual_gate_mode': residual_gate_mode,
            'residual_gate_max_delta': self.residual_gate_max_delta,
            'residual_gate_lower_bound': residual_gate_lower_bound,
            'residual_gate_upper_bound': residual_gate_upper_bound,
            'aux_loss_free': self.aux_loss_free,
            'router_bias_update_rate': self.router_bias_update_rate,
            'router_bias_min': float(self._router_selection_bias.min().cpu().item()),
            'router_bias_max': float(self._router_selection_bias.max().cpu().item()),
            'adaptive_shared_gate': self.adaptive_shared_gate,
            'adaptive_residual_gate': self.adaptive_residual_gate,
            'expert_evaluations_per_sample': self.top_k + int(self.use_shared_expert),
            'single_adapter_mode': self.single_adapter_mode,
            'adaptive_grouping': self.adaptive_grouping,
            'target_num_groups': self.target_num_groups,
            'grouping_finalized': bool(self._grouping_finalized.item()),
            'grouping_batches': int(self._grouping_batches.item()),
            'expert_group_ids': group_ids.tolist(),
            'num_groups': int(group_ids.max().item()) + 1,
            'group_sizes': torch.bincount(group_ids).tolist(),
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

        router_context = self._router_input(local_state, final_state)
        if self.single_adapter_mode:
            normalized_context = router_context
            gate_weights = final_state.new_ones((final_state.size(0), 1))
        else:
            normalized_context = self.router_norm(router_context)
            gate_weights = F.softmax(self.router(normalized_context), dim=-1)
        self._update_adaptive_grouping(gate_weights)
        _, topk_indices = self._select_topk(self._selection_scores(gate_weights))
        self._update_router_selection_bias(topk_indices)
        # Selection bias must never change mixture weights or receive gradients.
        topk_weights = gate_weights.gather(1, topk_indices)
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
            shared_logits = self.shared_alpha
            if self.adaptive_shared_gate:
                shared_logits = shared_logits + self.shared_gate(normalized_context)
            shared_alpha = torch.sigmoid(shared_logits).to(final_state.dtype)
            if shared_alpha.ndim == 0:
                shared_alpha = shared_alpha.expand(final_state.size(0), 1)
            moe_delta = shared_alpha * shared_out + (1.0 - shared_alpha) * routed_out
        else:
            shared_out = torch.zeros_like(routed_out)
            shared_alpha = None
            moe_delta = routed_out

        top1_fraction = F.one_hot(
            topk_indices[:, 0], num_classes=self.num_experts
        ).float().mean(dim=0)
        mean_probability = gate_weights.mean(dim=0)
        balance_proxy = self.num_experts * torch.sum(
            top1_fraction * mean_probability
        )
        balance_loss = (
            final_state.sum() * 0.0 if self.aux_loss_free else balance_proxy
        )
        if self.adaptive_residual_gate:
            residual_gate_logits = self.residual_gate(normalized_context)
            if self.residual_gate_max_delta:
                residual_scale = (
                    self.residual_scale
                    + self.residual_gate_max_delta
                    * torch.tanh(residual_gate_logits)
                ).to(final_state.dtype)
            else:
                residual_scale = torch.sigmoid(
                    self._residual_scale_logit + residual_gate_logits
                ).to(final_state.dtype)
        else:
            residual_scale = final_state.new_full(
                (final_state.size(0), 1),
                self.residual_scale,
            )
        self._record_diagnostics(
            gate_weights,
            topk_indices,
            balance_loss,
            balance_proxy,
            shared_out,
            routed_out,
            shared_alpha,
            residual_scale,
        )
        return final_state + residual_scale * moe_delta, balance_loss
