import argparse
import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
import pandas as pd

from rule_teacher import (
    DEFAULT_RULE_NAMES,
    DatasetRuleStatistics,
    RankingBatch,
    evaluate_ranking,
    rerank_candidates,
    search_rule_weights,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export backbone Top-20 candidates once, then fit and evaluate "
            "a dataset-calibrated offline rule teacher."
        )
    )
    parser.add_argument(
        "-f",
        "--yaml_file",
        required=True,
        help="Configuration below conf/, for example best_conf/nyc_r1_6_aux_free_dual_gate.yml.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Backbone checkpoint.pt path or the directory containing it.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for candidates, weights, predictions, and summary.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--weight-samples", type=int, default=256)
    parser.add_argument("--search-seed", type=int, default=20260723)
    parser.add_argument(
        "--reuse-candidates",
        action="store_true",
        help="Skip GPU inference and reuse validation_candidates.npz and test_candidates.npz.",
    )
    return parser.parse_args()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _checkpoint_file(path):
    if not path:
        raise ValueError("--checkpoint is required unless --reuse-candidates is used.")
    checkpoint = Path(path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    return checkpoint


def _initialize_inference(yaml_file):
    import torch

    from dataset import LBSNDataset
    from layer import NeighborSampler
    from model import STHGCN
    from preprocess import preprocess
    from utils import Cfg, seed_torch

    cfg = Cfg(yaml_file)
    sizes = [int(value) for value in cfg.model_args.sizes.split("-")]
    cfg.model_args.sizes = sizes
    requested_gpu = int(cfg.run_args.gpu)
    if requested_gpu >= 0 and torch.cuda.is_available():
        device = f"cuda:{requested_gpu}"
    else:
        device = "cpu"
    cfg.run_args.device = device
    seed_torch(int(cfg.run_args.seed))

    preprocess(cfg)
    lbsn_dataset = LBSNDataset(cfg)
    cfg.dataset_args.spatial_slots = lbsn_dataset.spatial_slots
    cfg.dataset_args.num_user = lbsn_dataset.num_user
    cfg.dataset_args.num_poi = lbsn_dataset.num_poi
    cfg.dataset_args.num_category = lbsn_dataset.num_category
    cfg.dataset_args.padding_poi_id = lbsn_dataset.padding_poi_id
    cfg.dataset_args.padding_user_id = lbsn_dataset.padding_user_id
    cfg.dataset_args.padding_poi_category = lbsn_dataset.padding_poi_category
    cfg.dataset_args.padding_hour_id = lbsn_dataset.padding_hour_id
    cfg.dataset_args.padding_weekday_id = lbsn_dataset.padding_weekday_id

    common = {
        "x": lbsn_dataset.x,
        "edge_index": lbsn_dataset.edge_index,
        "edge_attr": lbsn_dataset.edge_attr,
        "intra_jaccard_threshold": cfg.model_args.intra_jaccard_threshold,
        "inter_jaccard_threshold": cfg.model_args.inter_jaccard_threshold,
        "edge_t": lbsn_dataset.edge_t,
        "edge_delta_t": lbsn_dataset.edge_delta_t,
        "edge_type": lbsn_dataset.edge_type,
        "sizes": sizes,
        "edge_delta_s": lbsn_dataset.edge_delta_s,
        "batch_size": cfg.run_args.eval_batch_size,
        "num_workers": 0 if device == "cpu" else cfg.run_args.num_workers,
        "shuffle": False,
        "pin_memory": True,
    }
    sampler_validate = NeighborSampler(
        sample_idx=lbsn_dataset.sample_idx_valid,
        node_idx=lbsn_dataset.node_idx_valid,
        max_time=lbsn_dataset.max_time_valid,
        label=lbsn_dataset.label_valid,
        **common,
    )
    sampler_test = NeighborSampler(
        sample_idx=lbsn_dataset.sample_idx_test,
        node_idx=lbsn_dataset.node_idx_test,
        max_time=lbsn_dataset.max_time_test,
        label=lbsn_dataset.label_test,
        **common,
    )
    model = STHGCN(cfg).to(device)
    return cfg, model, sampler_validate, sampler_test


def _export_candidates(model, sampler, top_k):
    import torch
    from tqdm import tqdm

    model.eval()
    candidate_ids = []
    model_scores = []
    labels = []
    base_label_ranks = []
    sample_indices = []
    with torch.no_grad():
        for row in tqdm(sampler, desc="Export candidates", unit="batch"):
            sample_indices.append(row.sample_idx.cpu().numpy())
            split_index = torch.max(row.adjs_t[1].storage.row()).tolist()
            row = row.to(model.device)
            input_data = {
                "x": row.x,
                "edge_index": row.adjs_t,
                "edge_attr": row.edge_attrs,
                "split_index": split_index,
                "delta_ts": row.edge_delta_ts,
                "delta_ss": row.edge_delta_ss,
                "edge_type": row.edge_types,
            }
            logits, _ = model(
                input_data,
                label=row.y[:, 0],
                mode="test",
            )
            batch_top_k = min(int(top_k), int(logits.shape[1]))
            scores, indices = torch.topk(
                logits,
                k=batch_top_k,
                dim=1,
                largest=True,
                sorted=True,
            )
            target = row.y[:, 0].long()
            target_score = logits.gather(1, target[:, None])
            target_rank = 1 + torch.sum(logits > target_score, dim=1)
            candidate_ids.append(indices.cpu().numpy())
            model_scores.append(scores.cpu().numpy())
            labels.append(target.cpu().numpy())
            base_label_ranks.append(target_rank.cpu().numpy())

    return RankingBatch(
        candidate_ids=np.concatenate(candidate_ids, axis=0),
        model_scores=np.concatenate(model_scores, axis=0),
        labels=np.concatenate(labels, axis=0),
        base_label_ranks=np.concatenate(base_label_ranks, axis=0),
        sample_indices=np.concatenate(sample_indices, axis=0),
    )


def _aligned_queries(preprocessed_dir, split, batch):
    file_name = (
        "validate_sample.csv"
        if split == "validation"
        else "test_sample.csv"
    )
    frame = pd.read_csv(osp.join(preprocessed_dir, file_name), low_memory=False)
    if batch.sample_indices is not None:
        frame = frame.iloc[np.asarray(batch.sample_indices, dtype=np.int64)]
    return frame.reset_index(drop=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    validation_path = osp.join(args.output_dir, "validation_candidates.npz")
    test_path = osp.join(args.output_dir, "test_candidates.npz")

    if args.reuse_candidates:
        validation_batch = RankingBatch.load(validation_path)
        test_batch = RankingBatch.load(test_path)
        from utils import Cfg

        cfg = Cfg(args.yaml_file)
    else:
        checkpoint_file = _checkpoint_file(args.checkpoint)
        cfg, model, sampler_validate, sampler_test = _initialize_inference(
            args.yaml_file
        )
        import torch

        checkpoint = torch.load(checkpoint_file, map_location=model.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation_batch = _export_candidates(
            model,
            sampler_validate,
            args.top_k,
        )
        test_batch = _export_candidates(model, sampler_test, args.top_k)
        validation_batch.save(validation_path)
        test_batch.save(test_path)

    dataset_name = cfg.dataset_args.dataset_name
    preprocessed_dir = osp.join(
        "data",
        dataset_name,
        "preprocessed",
    )
    statistics = DatasetRuleStatistics.from_preprocessed_dir(
        dataset_name,
        preprocessed_dir,
    )
    validation_queries = _aligned_queries(
        preprocessed_dir,
        "validation",
        validation_batch,
    )
    test_queries = _aligned_queries(
        preprocessed_dir,
        "test",
        test_batch,
    )
    validation_features = statistics.score_candidates(
        validation_queries,
        validation_batch.candidate_ids,
    )
    test_features = statistics.score_candidates(
        test_queries,
        test_batch.candidate_ids,
    )

    base_validation_metrics = evaluate_ranking(
        validation_batch.candidate_ids,
        validation_batch.labels,
        validation_batch.base_label_ranks,
    )
    base_test_metrics = evaluate_ranking(
        test_batch.candidate_ids,
        test_batch.labels,
        test_batch.base_label_ranks,
    )
    search_result = search_rule_weights(
        validation_batch,
        validation_features,
        rule_names=DEFAULT_RULE_NAMES,
        num_weight_samples=args.weight_samples,
        seed=args.search_seed,
    )
    reranked_test, reranked_scores = rerank_candidates(
        test_batch,
        test_features,
        search_result.rule_names,
        search_result.weights,
        search_result.rule_strength,
    )
    rule_test_metrics = evaluate_ranking(
        reranked_test,
        test_batch.labels,
        test_batch.base_label_ranks,
    )
    np.savez_compressed(
        osp.join(args.output_dir, "test_reranked.npz"),
        candidate_ids=reranked_test.astype(np.int64),
        scores=reranked_scores.astype(np.float32),
        labels=np.asarray(test_batch.labels, dtype=np.int64),
        base_label_ranks=np.asarray(
            test_batch.base_label_ranks,
            dtype=np.int64,
        ),
    )

    summary = {
        "dataset": dataset_name,
        "yaml_file": args.yaml_file,
        "checkpoint": args.checkpoint,
        "top_k": int(args.top_k),
        "selection_objective": (
            "4 * Recall@1 + Recall@5 + MRR + NDCG@5"
        ),
        "statistics": statistics.summary(),
        "validation_previous_context_coverage": statistics.context_coverage(
            validation_queries
        ),
        "test_previous_context_coverage": statistics.context_coverage(
            test_queries
        ),
        "base_validation_metrics": base_validation_metrics,
        "rule_search": search_result.as_dict(),
        "base_test_metrics": base_test_metrics,
        "rule_test_metrics": rule_test_metrics,
        "test_delta": {
            key: rule_test_metrics[key] - base_test_metrics[key]
            for key in base_test_metrics
        },
        "test_was_not_used_for_weight_selection": True,
    }
    summary_path = osp.join(args.output_dir, "rule_teacher_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            _json_ready(summary),
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
