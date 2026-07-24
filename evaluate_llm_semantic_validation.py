"""Paired validation-only evaluation for LLM semantic expert snapshots."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--yaml-file",
        default="best_conf/nyc_r1_6_aux_free_dual_gate_llm_semantic.yml",
    )
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--semantic-snapshot", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--rule-summary",
        default=None,
        help="Optional frozen rule_teacher_summary.json for paired composition.",
    )
    parser.add_argument(
        "--calibrate-multiplier",
        action="store_true",
        help=(
            "Choose a non-negative semantic-delta multiplier on the calibration "
            "partition, then report the frozen result on audit."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=20260724)
    parser.add_argument(
        "--data-split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--profile-ablation",
        choices=("none", "zero", "shuffle"),
        default="none",
    )
    parser.add_argument(
        "--semantic-delta-multiplier",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def ranking_metrics(ranks):
    ranks = np.asarray(ranks, dtype=np.int64)
    result = {
        f"recall_at_{cutoff}": float(np.mean(ranks <= cutoff))
        for cutoff in (1, 5, 10, 20)
    }
    result["mrr"] = float(np.mean(1.0 / ranks))
    result["objective"] = (
        4.0 * result["recall_at_1"] + result["recall_at_20"]
    )
    return result


def paired_partitions(num_rows, seed):
    generator = np.random.default_rng(seed)
    order = generator.permutation(num_rows)
    selection_end = int(round(0.6 * num_rows))
    calibration_end = int(round(0.8 * num_rows))
    return {
        "selection": order[:selection_end],
        "calibration": order[selection_end:calibration_end],
        "audit": order[calibration_end:],
        "all": order,
    }


def initialize(yaml_file, data_split):
    import torch

    from dataset import LBSNDataset
    from layer import NeighborSampler
    from model import STHGCN
    from preprocess import preprocess
    from utils import Cfg, seed_torch

    cfg = Cfg(yaml_file)
    cfg.model_args.sizes = [
        int(value) for value in cfg.model_args.sizes.split("-")
    ]
    requested_gpu = int(cfg.run_args.gpu)
    device = (
        f"cuda:{requested_gpu}"
        if requested_gpu >= 0 and torch.cuda.is_available()
        else "cpu"
    )
    cfg.run_args.device = device
    seed_torch(int(cfg.run_args.seed))
    preprocess(cfg)
    dataset = LBSNDataset(cfg)
    for name in (
        "spatial_slots",
        "num_user",
        "num_poi",
        "num_category",
        "padding_poi_id",
        "padding_user_id",
        "padding_poi_category",
        "padding_hour_id",
        "padding_weekday_id",
    ):
        setattr(cfg.dataset_args, name, getattr(dataset, name))

    if data_split == "validation":
        sample_idx = dataset.sample_idx_valid
        node_idx = dataset.node_idx_valid
        max_time = dataset.max_time_valid
        label = dataset.label_valid
    else:
        sample_idx = dataset.sample_idx_test
        node_idx = dataset.node_idx_test
        max_time = dataset.max_time_test
        label = dataset.label_test
    sampler = NeighborSampler(
        dataset.x,
        dataset.edge_index,
        dataset.edge_attr,
        intra_jaccard_threshold=cfg.model_args.intra_jaccard_threshold,
        inter_jaccard_threshold=cfg.model_args.inter_jaccard_threshold,
        edge_t=dataset.edge_t,
        edge_delta_t=dataset.edge_delta_t,
        edge_type=dataset.edge_type,
        sizes=cfg.model_args.sizes,
        sample_idx=sample_idx,
        node_idx=node_idx,
        edge_delta_s=dataset.edge_delta_s,
        max_time=max_time,
        label=label,
        batch_size=cfg.run_args.eval_batch_size,
        num_workers=0 if device == "cpu" else cfg.run_args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return cfg, STHGCN(cfg).to(device), sampler


def label_ranks(logits, labels):
    target_score = logits.gather(1, labels[:, None])
    return 1 + (logits > target_score).sum(dim=1)


def ranks_after_candidate_order(candidate_ids, labels, fallback_ranks):
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    fallback_ranks = np.asarray(fallback_ranks, dtype=np.int64)
    matches = candidate_ids == labels[:, None]
    found = matches.any(axis=1)
    ranks = fallback_ranks.copy()
    ranks[found] = np.argmax(matches[found], axis=1) + 1
    return ranks


def main():
    import torch

    args = parse_args()
    if args.data_split == "test" and args.calibrate_multiplier:
        raise ValueError("Multiplier calibration is forbidden on test.")
    cfg, model, sampler = initialize(args.yaml_file, args.data_split)
    backbone = torch.load(
        args.backbone_checkpoint,
        map_location=model.device,
    )
    model.load_backbone_state_dict(backbone["model_state_dict"])
    snapshot = torch.load(args.semantic_snapshot, map_location=model.device)
    model.llm_semantic_expert.load_state_dict(
        snapshot["semantic_state_dict"]
    )
    if args.profile_ablation == "zero":
        model.llm_semantic_expert.poi_profiles.zero_()
    elif args.profile_ablation == "shuffle":
        available = torch.nonzero(
            model.llm_semantic_expert.profile_available,
            as_tuple=False,
        ).flatten()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.split_seed)
        permutation = torch.randperm(
            int(available.numel()),
            generator=generator,
        ).to(available.device)
        original = model.llm_semantic_expert.poi_profiles[available].clone()
        model.llm_semantic_expert.poi_profiles[available] = original[
            permutation
        ]
    model.freeze_backbone_for_llm_semantic()
    model.eval()

    base_ranks = []
    semantic_ranks = []
    delta_abs_sum = 0.0
    delta_count = 0
    profile_loss_sum = 0.0
    query_count = 0
    base_candidate_ids = []
    base_candidate_scores = []
    semantic_candidate_ids = []
    semantic_candidate_scores = []
    labels_all = []
    sample_indices = []
    paired_base_logits = []
    paired_semantic_delta = []
    captured = []
    hook = model.llm_semantic_expert.register_forward_hook(
        lambda _module, _inputs, output: captured.append(
            output[0].detach()
        )
    )
    try:
        with torch.no_grad():
            for row in sampler:
                sample_indices.append(row.sample_idx.cpu().numpy())
                split_index = torch.max(
                    row.adjs_t[1].storage.row()
                ).tolist()
                row = row.to(model.device)
                input_data = {
                    "x": row.x,
                    "x_target": row.x_target,
                    "edge_index": row.adjs_t,
                    "edge_attr": row.edge_attrs,
                    "split_index": split_index,
                    "delta_ts": row.edge_delta_ts,
                    "delta_ss": row.edge_delta_ss,
                    "edge_type": row.edge_types,
                }
                logits, _loss = model(
                    input_data,
                    label=row.y[:, 0],
                    mode="validate",
                )
                model_semantic_delta = captured.pop(0)
                base_logits = logits - model_semantic_delta
                semantic_delta = (
                    float(args.semantic_delta_multiplier)
                    * model_semantic_delta
                )
                logits = base_logits + semantic_delta
                labels = row.y[:, 0].long()
                base_ranks.append(label_ranks(base_logits, labels).cpu())
                semantic_ranks.append(label_ranks(logits, labels).cpu())
                base_scores, base_ids = torch.topk(
                    base_logits,
                    k=20,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                semantic_scores, semantic_ids = torch.topk(
                    logits,
                    k=20,
                    dim=1,
                    largest=True,
                    sorted=True,
                )
                base_candidate_ids.append(base_ids.cpu().numpy())
                base_candidate_scores.append(base_scores.cpu().numpy())
                semantic_candidate_ids.append(semantic_ids.cpu().numpy())
                semantic_candidate_scores.append(
                    semantic_scores.cpu().numpy()
                )
                labels_all.append(labels.cpu().numpy())
                if args.calibrate_multiplier:
                    paired_base_logits.append(base_logits.cpu())
                    paired_semantic_delta.append(semantic_delta.cpu())
                delta_abs_sum += float(semantic_delta.abs().sum().cpu())
                delta_count += int(semantic_delta.numel())
                diagnostics = model.get_llm_semantic_diagnostics()
                profile_loss_sum += (
                    float(diagnostics["profile_loss"].cpu())
                    * int(labels.numel())
                )
                query_count += int(labels.numel())
    finally:
        hook.remove()

    base_ranks = torch.cat(base_ranks).numpy()
    semantic_ranks = torch.cat(semantic_ranks).numpy()
    labels_all = np.concatenate(labels_all)
    sample_indices = np.concatenate(sample_indices)
    if args.data_split == "validation":
        partitions = paired_partitions(len(base_ranks), args.split_seed)
    else:
        partitions = {"all": np.arange(len(base_ranks))}
    results = {}
    for name, indices in partitions.items():
        base_metrics = ranking_metrics(base_ranks[indices])
        semantic_metrics = ranking_metrics(semantic_ranks[indices])
        results[name] = {
            "num_queries": int(len(indices)),
            "base": base_metrics,
            "semantic": semantic_metrics,
            "delta": {
                key: semantic_metrics[key] - base_metrics[key]
                for key in base_metrics
            },
        }

    payload = {
        "protocol": (
            "paired_validation_only_v1"
            if args.data_split == "validation"
            else "paired_frozen_test_once_v1"
        ),
        "yaml_file": args.yaml_file,
        "backbone_checkpoint": args.backbone_checkpoint,
        "semantic_snapshot": args.semantic_snapshot,
        "semantic_step": int(snapshot["step"]),
        "split_seed": int(args.split_seed),
        "profile_ablation": args.profile_ablation,
        "semantic_delta_multiplier": float(
            args.semantic_delta_multiplier
        ),
        "data_split": args.data_split,
        "test_data_loaded": bool(args.data_split == "test"),
        "effective_scale": float(
            model.llm_semantic_expert.effective_scale.detach().cpu()
        ),
        "training_batches_seen": int(
            model.llm_semantic_expert.training_batches_seen.cpu()
        ),
        "mean_abs_semantic_delta": delta_abs_sum / max(delta_count, 1),
        "mean_profile_loss": profile_loss_sum / max(query_count, 1),
        "partitions": results,
    }
    if args.calibrate_multiplier:
        base_logits_cpu = torch.cat(paired_base_logits)
        semantic_delta_cpu = torch.cat(paired_semantic_delta)
        labels_cpu = torch.as_tensor(labels_all, dtype=torch.long)
        calibration_indices = partitions["calibration"]
        base_calibration = results["calibration"]["base"]
        curve = []
        multipliers = np.linspace(0.0, 4.0, 17)
        for multiplier in multipliers:
            ranks = label_ranks(
                base_logits_cpu
                + float(multiplier) * semantic_delta_cpu,
                labels_cpu,
            ).numpy()
            metrics = ranking_metrics(ranks[calibration_indices])
            curve.append(
                {
                    "multiplier": float(multiplier),
                    "metrics": metrics,
                    "mrr_non_decreasing": bool(
                        metrics["mrr"] + 1e-12
                        >= base_calibration["mrr"]
                    ),
                }
            )
        eligible = [row for row in curve if row["mrr_non_decreasing"]]
        selected = max(
            eligible,
            key=lambda row: (
                row["metrics"]["objective"],
                row["metrics"]["mrr"],
                -row["multiplier"],
            ),
        )
        selected_ranks = label_ranks(
            base_logits_cpu
            + selected["multiplier"] * semantic_delta_cpu,
            labels_cpu,
        ).numpy()
        calibrated_partitions = {}
        for name, indices in partitions.items():
            base_metrics = ranking_metrics(base_ranks[indices])
            calibrated_metrics = ranking_metrics(selected_ranks[indices])
            calibrated_partitions[name] = {
                "num_queries": int(len(indices)),
                "base": base_metrics,
                "calibrated_semantic": calibrated_metrics,
                "delta": {
                    key: calibrated_metrics[key] - base_metrics[key]
                    for key in base_metrics
                },
            }
        payload["multiplier_calibration"] = {
            "grid": curve,
            "selection_partition_used": False,
            "calibration_partition_used": True,
            "audit_partition_used_for_selection": False,
            "selected_multiplier": selected["multiplier"],
            "partitions": calibrated_partitions,
        }
    if args.rule_summary:
        import pandas as pd

        from rule_teacher import (
            DatasetRuleStatistics,
            RankingBatch,
            rerank_candidates,
        )

        rule_summary = json.loads(
            Path(args.rule_summary).read_text(encoding="utf-8")
        )
        preprocessed_dir = (
            Path("data")
            / cfg.dataset_args.dataset_name
            / "preprocessed"
        )
        statistics = DatasetRuleStatistics.from_preprocessed_dir(
            cfg.dataset_args.dataset_name,
            str(preprocessed_dir),
        )
        query_file = (
            "validate_sample.csv"
            if args.data_split == "validation"
            else "test_sample.csv"
        )
        queries = pd.read_csv(
            preprocessed_dir / query_file,
            low_memory=False,
        ).iloc[sample_indices].reset_index(drop=True)
        rule_search = rule_summary["rule_search"]
        rule_names = tuple(rule_search["rule_names"])
        rule_weights = np.asarray(
            [
                rule_search["weights"][name]
                for name in rule_names
            ],
            dtype=np.float64,
        )
        rule_strength = float(rule_search["rule_strength"])

        def frozen_rule_ranks(
            candidate_ids,
            candidate_scores,
            fallback_ranks,
        ):
            candidate_ids = np.concatenate(candidate_ids)
            batch = RankingBatch(
                candidate_ids=candidate_ids,
                model_scores=np.concatenate(candidate_scores),
                labels=labels_all,
                base_label_ranks=fallback_ranks,
                sample_indices=sample_indices,
            )
            features = statistics.score_candidates(
                queries,
                candidate_ids,
            )
            reranked, _scores = rerank_candidates(
                batch,
                features,
                rule_names,
                rule_weights,
                rule_strength,
            )
            return ranks_after_candidate_order(
                reranked,
                labels_all,
                fallback_ranks,
            )

        base_rule_ranks = frozen_rule_ranks(
            base_candidate_ids,
            base_candidate_scores,
            base_ranks,
        )
        semantic_rule_ranks = frozen_rule_ranks(
            semantic_candidate_ids,
            semantic_candidate_scores,
            semantic_ranks,
        )
        rule_results = {}
        for name, indices in partitions.items():
            base_metrics = ranking_metrics(base_rule_ranks[indices])
            semantic_metrics = ranking_metrics(
                semantic_rule_ranks[indices]
            )
            rule_results[name] = {
                "num_queries": int(len(indices)),
                "frozen_rule_on_base": base_metrics,
                "frozen_rule_on_semantic": semantic_metrics,
                "delta": {
                    key: semantic_metrics[key] - base_metrics[key]
                    for key in base_metrics
                },
            }
        payload["rule_summary"] = args.rule_summary
        payload["rule_partitions"] = rule_results
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
