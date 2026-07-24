import torch
from torch import nn
from layer import (
    CheckinEmbedding,
    EdgeEmbedding,
    HypergraphTransformer,
    TimeEncoder,
    DistanceEncoderHSTLSTM,
    DistanceEncoderSTAN,
    DistanceEncoderSimple
)
from .sparse_moe import HypergraphConditionedSharedSparseMoE


class STHGCN(nn.Module):
    def __init__(self, cfg):
        super(STHGCN, self).__init__()
        self.device = cfg.run_args.device
        self.batch_size = cfg.run_args.batch_size
        self.eval_batch_size = cfg.run_args.eval_batch_size
        self.do_traj2traj = cfg.model_args.do_traj2traj
        self.distance_encoder_type = cfg.model_args.distance_encoder_type
        self.dropout_rate = cfg.model_args.dropout_rate
        self.generate_edge_attr = cfg.model_args.generate_edge_attr
        self.num_conv_layers = len(cfg.model_args.sizes)
        self.num_poi = cfg.dataset_args.num_poi
        self.embed_fusion_type = cfg.model_args.embed_fusion_type
        self.use_moe = bool(getattr(cfg.model_args, 'use_moe', False))
        self.moe_loss_weight = float(getattr(cfg.model_args, 'moe_loss_weight', 0.0))
        self.moe_post_group_loss_weight = float(
            getattr(
                cfg.model_args,
                'moe_post_group_loss_weight',
                self.moe_loss_weight,
            )
        )
        self.last_moe_loss_weight = 0.0
        self.checkin_embedding_layer = CheckinEmbedding(
            embed_size=cfg.model_args.embed_size,
            fusion_type=self.embed_fusion_type,
            dataset_args=cfg.dataset_args
        )
        self.checkin_embed_size = self.checkin_embedding_layer.output_embed_size
        self.edge_type_embedding_layer = EdgeEmbedding(
            embed_size=self.checkin_embed_size,
            fusion_type=self.embed_fusion_type,
            num_edge_type=cfg.model_args.num_edge_type
        )

        if cfg.model_args.activation == 'elu':
            self.act = nn.ELU()
        elif cfg.model_args.activation == 'relu':
            self.act = nn.RReLU()
        elif cfg.model_args.activation == 'leaky_relu':
            self.act = nn.LeakyReLU()
        else:
            self.act = torch.tanh

        if cfg.conv_args.time_fusion_mode == 'add':
            continuous_encoder_dim = self.checkin_embed_size
        else:
            continuous_encoder_dim = cfg.model_args.st_embed_size

        if self.generate_edge_attr:
            # use edge_type to create edge_attr_embed
            self.edge_attr_embedding_layer = EdgeEmbedding(
                embed_size=self.checkin_embed_size,
                fusion_type=self.embed_fusion_type,
                num_edge_type=cfg.model_args.num_edge_type
            )
        else:
            # source_traj_size, target_traj_size, jaccard similarity as the raw features, and do linear transformation
            if cfg.conv_args.edge_fusion_mode == 'add':
                self.edge_attr_embedding_layer = nn.Linear(3, self.checkin_embed_size)
            else:
                self.edge_attr_embedding_layer = None

        self.conv_list = nn.ModuleList()

        # conv for ci2traj within which some ci2traj relations have been removed by time to prevent data leakage
        self.conv_for_time_filter = HypergraphTransformer(
            in_channels=self.checkin_embed_size,
            out_channels=self.checkin_embed_size,
            attn_heads=cfg.conv_args.num_attention_heads,
            residual_beta=cfg.conv_args.residual_beta,
            learn_beta=cfg.conv_args.learn_beta,
            dropout=cfg.conv_args.conv_dropout_rate,
            trans_method=cfg.conv_args.trans_method,
            edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
            time_fusion_mode=cfg.conv_args.time_fusion_mode,
            head_fusion_mode=cfg.conv_args.head_fusion_mode,
            residual_fusion_mode=None,
            edge_dim=None,
            rel_embed_dim=self.checkin_embed_size,
            time_embed_dim=continuous_encoder_dim,
            dist_embed_dim=continuous_encoder_dim,
            negative_slope=cfg.conv_args.negative_slope,
            have_query_feature=False
        )
        self.norms_for_time_filter = nn.BatchNorm1d(self.checkin_embed_size)
        self.dropout_for_time_filter = nn.Dropout(self.dropout_rate)

        if self.do_traj2traj:
            for i in range(self.num_conv_layers):
                if i == 0:
                    # ci2traj full
                    have_query_feature = False
                    residual_fusion_mode = None
                    edge_size = None
                else:
                    # traj2traj
                    have_query_feature = True
                    residual_fusion_mode = cfg.conv_args.residual_fusion_mode
                    if self.edge_attr_embedding_layer is None:
                        edge_size = 3
                    else:
                        edge_size = self.checkin_embed_size

                self.conv_list.append(
                    HypergraphTransformer(
                        in_channels=self.checkin_embed_size,
                        out_channels=self.checkin_embed_size,
                        attn_heads=cfg.conv_args.num_attention_heads,
                        residual_beta=cfg.conv_args.residual_beta,
                        learn_beta=cfg.conv_args.learn_beta,
                        dropout=cfg.conv_args.conv_dropout_rate,
                        trans_method=cfg.conv_args.trans_method,
                        edge_fusion_mode=cfg.conv_args.edge_fusion_mode,
                        time_fusion_mode=cfg.conv_args.time_fusion_mode,
                        head_fusion_mode=cfg.conv_args.head_fusion_mode,
                        residual_fusion_mode=residual_fusion_mode,
                        edge_dim=edge_size,
                        rel_embed_dim=self.checkin_embed_size,
                        time_embed_dim=continuous_encoder_dim,
                        dist_embed_dim=continuous_encoder_dim,
                        negative_slope=cfg.conv_args.negative_slope,
                        have_query_feature=have_query_feature
                    )
                )
            self.norms_list = nn.ModuleList()
            for i in range(self.num_conv_layers):
                self.norms_list.append(nn.BatchNorm1d(self.checkin_embed_size))

            self.dropout_list = nn.ModuleList()
            for i in range(self.num_conv_layers):
                self.dropout_list.append(nn.Dropout(self.dropout_rate))

        self.continuous_time_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)

        if self.distance_encoder_type == 'stan':
            self.continuous_distance_encoder = DistanceEncoderSTAN(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'time':
            self.continuous_distance_encoder = TimeEncoder(cfg.model_args, continuous_encoder_dim)
        elif self.distance_encoder_type == 'hstlstm':
            self.continuous_distance_encoder = DistanceEncoderHSTLSTM(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        elif self.distance_encoder_type == 'simple':
            self.continuous_distance_encoder = DistanceEncoderSimple(
                cfg.model_args,
                continuous_encoder_dim,
                cfg.dataset_args.spatial_slots
            )
        else:
            raise ValueError(f"Get wrong distance_encoder_type argument: {cfg.model_args.distance_encoder_type}!")

        # Keep the original prediction head initialization before optional R1
        # modules so R0 and R1 share all common-parameter initialization.
        self.linear = nn.Linear(self.checkin_embed_size, self.num_poi + 1)
        if self.use_moe:
            self.moe = HypergraphConditionedSharedSparseMoE(
                hidden_size=self.checkin_embed_size,
                rank=int(getattr(cfg.model_args, 'moe_rank', 32)),
                num_experts=int(getattr(cfg.model_args, 'moe_num_experts', 4)),
                top_k=int(getattr(cfg.model_args, 'moe_top_k', 2)),
                router_hidden_size=int(
                    getattr(cfg.model_args, 'moe_router_hidden_size', 128)
                ),
                dropout=float(getattr(cfg.model_args, 'moe_dropout', self.dropout_rate)),
                use_shared_expert=bool(
                    getattr(cfg.model_args, 'moe_use_shared_expert', True)
                ),
                shared_alpha_init=float(
                    getattr(cfg.model_args, 'moe_shared_alpha_init', 0.15)
                ),
                residual_scale=float(
                    getattr(cfg.model_args, 'moe_residual_scale', 1.0)
                ),
                router_context=str(
                    getattr(cfg.model_args, 'moe_router_context', 'hypergraph')
                ),
                adaptive_grouping=bool(
                    getattr(cfg.model_args, 'moe_adaptive_grouping', False)
                ),
                group_warmup_steps=int(
                    getattr(cfg.model_args, 'moe_group_warmup_steps', 3000)
                ),
                group_similarity_threshold=float(
                    getattr(cfg.model_args, 'moe_group_similarity_threshold', 0.5)
                ),
                target_num_groups=int(
                    getattr(cfg.model_args, 'moe_target_num_groups', 0)
                ),
                aux_loss_free=bool(
                    getattr(cfg.model_args, 'moe_aux_loss_free', False)
                ),
                router_bias_update_rate=float(
                    getattr(cfg.model_args, 'moe_router_bias_update_rate', 0.0)
                ),
                adaptive_shared_gate=bool(
                    getattr(cfg.model_args, 'moe_adaptive_shared_gate', False)
                ),
                adaptive_residual_gate=bool(
                    getattr(cfg.model_args, 'moe_adaptive_residual_gate', False)
                ),
                residual_gate_max_delta=float(
                    getattr(cfg.model_args, 'moe_residual_gate_max_delta', 0.0)
                ),
                use_hsid_lite=bool(
                    getattr(cfg.model_args, 'moe_use_hsid_lite', False)
                ),
                region_embed_size=int(
                    getattr(cfg.model_args, 'moe_region_embed_size', 16)
                ),
                coarse_region_degrees=float(
                    getattr(cfg.model_args, 'moe_coarse_region_degrees', 0.1)
                ),
                fine_region_degrees=float(
                    getattr(cfg.model_args, 'moe_fine_region_degrees', 0.001)
                ),
                coarse_region_buckets=int(
                    getattr(cfg.model_args, 'moe_coarse_region_buckets', 2048)
                ),
                fine_region_buckets=int(
                    getattr(cfg.model_args, 'moe_fine_region_buckets', 16384)
                ),
            )
        self.loss_func = nn.CrossEntropyLoss()
        self.last_moe_aux_loss = None

    def reset_moe_diagnostics(self):
        if self.use_moe:
            self.moe.reset_diagnostics()

    def get_moe_diagnostics(self, reset=False):
        if not self.use_moe:
            return None
        diagnostics = self.moe.diagnostics(reset=reset)
        diagnostics['balance_weight'] = self.current_moe_loss_weight()
        return diagnostics

    def current_moe_loss_weight(self):
        if self.use_moe and self.moe.aux_loss_free:
            return 0.0
        if (
            self.use_moe
            and self.moe.adaptive_grouping
            and bool(self.moe._grouping_finalized.item())
        ):
            return self.moe_post_group_loss_weight
        return self.moe_loss_weight

    def forward(self, data, label=None, mode='train'):
        input_x = data['x']  # [?, 8]
        split_idx = data['split_index']

        check_in_x = input_x[split_idx+1:]
        checkin_feature = self.checkin_embedding_layer(check_in_x)
        trajectory_feature = torch.zeros(
            split_idx+1,
            self.checkin_embed_size,
            device=checkin_feature.device
        )
        x = torch.cat([trajectory_feature, checkin_feature], dim=0)

        # change this line if you want to modify the time granularity when encode
        edge_time_embed = self.continuous_time_encoder(data['delta_ts'][0] / (60 * 60))
        if self.distance_encoder_type == 'stan':
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0], dist_type='ch2tj')
        else:
            edge_distance_embed = self.continuous_distance_encoder(data['delta_ss'][0])

        edge_attr_embed, edge_type_embed = None, None
        if data['edge_type'][0] is not None:
            if self.generate_edge_attr:
                edge_attr_embed = self.edge_attr_embedding_layer(data['edge_type'][0])
            edge_type_embed = self.edge_type_embedding_layer(data['edge_type'][0])

        x_for_time_filter = self.conv_for_time_filter(
            x,
            edge_index=data['edge_index'][0],
            edge_attr_embed=edge_attr_embed,
            edge_time_embed=edge_time_embed,
            edge_dist_embed=edge_distance_embed,
            edge_type_embed=edge_type_embed
        )
        x_for_time_filter = self.norms_for_time_filter(x_for_time_filter)
        x_for_time_filter = self.act(x_for_time_filter)
        x_for_time_filter = self.dropout_for_time_filter(x_for_time_filter)

        if data['edge_index'][-1] is not None and self.do_traj2traj:
            # all conv
            for idx, (edge_index, edge_attr, delta_ts, delta_dis, edge_type) in enumerate(
                    zip(data["edge_index"][1:], data["edge_attr"][1:], data["delta_ts"][1:], data["delta_ss"][1:],
                        data["edge_type"][1:])
            ):
                edge_time_embed = self.continuous_time_encoder(delta_ts / (60 * 60))
                if self.distance_encoder_type == 'stan':
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis, dist_type='tj2tj')
                else:
                    edge_distance_embed = self.continuous_distance_encoder(delta_dis)

                edge_attr_embed, edge_type_embed = None, None
                if edge_type is not None:
                    edge_type_embed = self.edge_type_embedding_layer(edge_type)
                    if self.generate_edge_attr:
                        edge_attr_embed = self.edge_attr_embedding_layer(edge_type)
                    else:
                        if self.edge_attr_embedding_layer:
                            edge_attr_embed = self.edge_attr_embedding_layer(edge_attr.to(torch.float32))
                        else:
                            edge_attr_embed = edge_attr.to(torch.float32)

                if idx == len(data['edge_index']) - 2:
                    if mode in ('test', 'validate'):
                        batch_size = self.eval_batch_size
                    else:
                        batch_size = self.batch_size
                    x_target = x_for_time_filter[:batch_size]
                else:
                    x_target = x[:edge_index.sparse_sizes()[0]]

                x = self.conv_list[idx](
                    (x, x_target),
                    edge_index=edge_index,
                    edge_attr_embed=edge_attr_embed,
                    edge_time_embed=edge_time_embed,
                    edge_dist_embed=edge_distance_embed,
                    edge_type_embed=edge_type_embed
                )
                x = self.norms_list[idx](x)
                x = self.act(x)
                x = self.dropout_list[idx](x)
        else:
            x = x_for_time_filter

        if self.use_moe:
            local_state = x_for_time_filter[:x.size(0)]
            region_coords = None
            if self.moe.use_hsid_lite:
                trajectory_context = data.get('x_target')
                if trajectory_context is None:
                    raise ValueError(
                        'HSID-lite requires leak-free trajectory context'
                    )
                if (
                    trajectory_context.ndim != 2
                    or trajectory_context.size(1) < 3
                    or trajectory_context.size(0) != x.size(0)
                ):
                    raise ValueError(
                        'x_target must align with routed trajectories and '
                        'contain mean longitude/latitude'
                    )
                # x_target is built from history check-ins after timestamp
                # filtering: [size, mean_lon, mean_lat, mean_time, min, max].
                region_coords = trajectory_context[:, 1:3]
            x, moe_aux_loss = self.moe(
                x,
                local_state,
                region_coords=region_coords,
            )
            self.last_moe_aux_loss = moe_aux_loss.detach()
        else:
            moe_aux_loss = x.sum() * 0.0
            self.last_moe_aux_loss = None

        logits = self.linear(x)
        loss = self.loss_func(logits, label.long())
        self.last_moe_loss_weight = self.current_moe_loss_weight()
        if self.use_moe and self.last_moe_loss_weight != 0.0:
            loss = loss + self.last_moe_loss_weight * moe_aux_loss
        return logits, loss
