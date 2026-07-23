import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rule_teacher import (
    DEFAULT_RULE_NAMES,
    DatasetRuleStatistics,
    RankingBatch,
    evaluate_ranking,
    rerank_candidates,
)


EXPERIMENTS = (
    {
        "dataset": "nyc",
        "backbone": "A4",
        "candidate_dir": Path(
            "server_artifacts/nyc_rule_teacher_20260723/nyc_a4"
        ),
    },
    {
        "dataset": "ca",
        "backbone": "R0",
        "candidate_dir": Path(
            "server_artifacts/rule_teacher_cross_dataset_20260723/ca_r0"
        ),
    },
    {
        "dataset": "tky",
        "backbone": "R0",
        "candidate_dir": Path(
            "server_artifacts/rule_teacher_cross_dataset_20260723/tky_r0"
        ),
    },
)

METRIC_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "mrr",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_20",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled rule-teacher ablations on frozen Top-20 "
            "candidate caches without retuning after observing test."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="docs/results/rule_teacher_ablation",
    )
    return parser.parse_args()


def load_queries(dataset, split, batch):
    file_name = (
        "validate_sample.csv"
        if split == "validation"
        else "test_sample.csv"
    )
    path = Path("data") / dataset / "preprocessed" / file_name
    frame = pd.read_csv(path, low_memory=False)
    if batch.sample_indices is not None:
        frame = frame.iloc[np.asarray(batch.sample_indices, dtype=np.int64)]
    return frame.reset_index(drop=True)


def normalized_subset_weights(full_weights, included_names):
    weights = np.array(
        [
            full_weights[index] if name in included_names else 0.0
            for index, name in enumerate(DEFAULT_RULE_NAMES)
        ],
        dtype=np.float64,
    )
    total = float(weights.sum())
    if total <= 1e-12:
        return np.zeros_like(weights), 0.0
    return weights / total, total


def make_variants(full_weights, rule_strength):
    variants = [
        ("base", np.zeros_like(full_weights), 0.0),
        ("full", full_weights.copy(), rule_strength),
    ]
    for index, name in enumerate(DEFAULT_RULE_NAMES):
        weights = np.zeros_like(full_weights)
        weights[index] = 1.0
        variants.append((f"only_{name}", weights, rule_strength))

    for removed_name in DEFAULT_RULE_NAMES:
        included = [
            name
            for name in DEFAULT_RULE_NAMES
            if name != removed_name
        ]
        weights, retained_mass = normalized_subset_weights(
            full_weights,
            included,
        )
        strength = rule_strength if retained_mass > 0.0 else 0.0
        variants.append((f"without_{removed_name}", weights, strength))

    cumulative_subsets = (
        ("user_spatial", ("user_preference", "spatial")),
        ("user_temporal", ("user_preference", "temporal")),
        (
            "user_spatial_temporal",
            ("user_preference", "spatial", "temporal"),
        ),
    )
    for label, included in cumulative_subsets:
        weights, retained_mass = normalized_subset_weights(
            full_weights,
            included,
        )
        strength = rule_strength if retained_mass > 0.0 else 0.0
        variants.append((label, weights, strength))
    return variants


def metrics_for_variant(batch, features, weights, rule_strength):
    if rule_strength == 0.0:
        candidates = batch.candidate_ids
    else:
        candidates, _ = rerank_candidates(
            batch,
            features,
            DEFAULT_RULE_NAMES,
            weights,
            rule_strength,
        )
    return evaluate_ranking(
        candidates,
        batch.labels,
        batch.base_label_ranks,
    )


def selection_objective(metrics):
    return float(
        4.0 * metrics["recall_at_1"]
        + metrics["recall_at_5"]
        + metrics["mrr"]
        + metrics["ndcg_at_5"]
    )


def assert_metrics_close(actual, expected, label):
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        if not np.isclose(actual[key], expected_value, atol=1e-12, rtol=0.0):
            raise AssertionError(
                f"{label} mismatch for {key}: "
                f"{actual[key]} != {expected_value}"
            )


def run_experiment(spec, output_dir):
    started_at = time.perf_counter()
    dataset = spec["dataset"]
    backbone = spec["backbone"]
    candidate_dir = spec["candidate_dir"]
    summary = json.loads(
        (candidate_dir / "rule_teacher_summary.json").read_text(
            encoding="utf-8"
        )
    )
    validation_batch = RankingBatch.load(
        str(candidate_dir / "validation_candidates.npz")
    )
    test_batch = RankingBatch.load(
        str(candidate_dir / "test_candidates.npz")
    )
    statistics = DatasetRuleStatistics.from_preprocessed_dir(
        dataset,
        str(Path("data") / dataset / "preprocessed"),
    )
    validation_queries = load_queries(
        dataset,
        "validation",
        validation_batch,
    )
    test_queries = load_queries(dataset, "test", test_batch)
    validation_features = statistics.score_candidates(
        validation_queries,
        validation_batch.candidate_ids,
    )
    test_features = statistics.score_candidates(
        test_queries,
        test_batch.candidate_ids,
    )

    rule_search = summary["rule_search"]
    full_weights = np.array(
        [rule_search["weights"][name] for name in DEFAULT_RULE_NAMES],
        dtype=np.float64,
    )
    rule_strength = float(rule_search["rule_strength"])
    if not np.isclose(full_weights.sum(), 1.0, atol=1e-12):
        raise AssertionError(f"{dataset}: full rule weights do not sum to one.")

    records = []
    variants = make_variants(full_weights, rule_strength)
    base_validation = None
    base_test = None
    for variant_name, weights, strength in variants:
        validation_metrics = metrics_for_variant(
            validation_batch,
            validation_features,
            weights,
            strength,
        )
        test_metrics = metrics_for_variant(
            test_batch,
            test_features,
            weights,
            strength,
        )
        if variant_name == "base":
            base_validation = validation_metrics
            base_test = test_metrics
            assert_metrics_close(
                test_metrics,
                summary["base_test_metrics"],
                f"{dataset} base test",
            )
            assert_metrics_close(
                validation_metrics,
                summary["base_validation_metrics"],
                f"{dataset} base validation",
            )
        if variant_name == "full":
            assert_metrics_close(
                test_metrics,
                summary["rule_test_metrics"],
                f"{dataset} full test",
            )
            assert_metrics_close(
                validation_metrics,
                rule_search["validation_metrics"],
                f"{dataset} full validation",
            )
        if base_test is not None and not np.isclose(
            test_metrics["recall_at_20"],
            base_test["recall_at_20"],
            atol=1e-12,
        ):
            raise AssertionError(
                f"{dataset} {variant_name}: Recall@20 changed."
            )

        record = {
            "dataset": dataset,
            "backbone": backbone,
            "variant": variant_name,
            "rule_strength": float(strength),
        }
        for index, name in enumerate(DEFAULT_RULE_NAMES):
            record[f"weight_{name}"] = float(weights[index])
        for split_name, metrics in (
            ("validation", validation_metrics),
            ("test", test_metrics),
        ):
            for key in METRIC_KEYS:
                record[f"{split_name}_{key}"] = float(metrics[key])
            record[f"{split_name}_objective"] = selection_objective(metrics)
        records.append(record)

    if base_validation is None or base_test is None:
        raise RuntimeError(f"{dataset}: base metrics were not evaluated.")
    for record in records:
        for key in METRIC_KEYS:
            record[f"test_delta_{key}"] = (
                record[f"test_{key}"] - base_test[key]
            )
        for k in (1, 5, 10, 20):
            delta = record[f"test_delta_recall_at_{k}"]
            record[f"test_added_hits_at_{k}"] = int(
                round(delta * len(test_batch.labels))
            )

    sensitivity_records = []
    for strength in np.linspace(0.0, 1.0, 21):
        metrics = metrics_for_variant(
            validation_batch,
            validation_features,
            full_weights,
            float(strength),
        )
        row = {
            "dataset": dataset,
            "backbone": backbone,
            "rule_strength": float(strength),
            "selected_strength": bool(
                np.isclose(strength, rule_strength, atol=1e-12)
            ),
            "validation_objective": selection_objective(metrics),
        }
        for key in METRIC_KEYS:
            row[f"validation_{key}"] = float(metrics[key])
        sensitivity_records.append(row)

    sensitivity_frame = pd.DataFrame(sensitivity_records)
    best_sensitivity_row = sensitivity_frame.loc[
        sensitivity_frame["validation_objective"].idxmax()
    ]
    if not bool(best_sensitivity_row["selected_strength"]):
        raise AssertionError(
            f"{dataset}: frozen lambda is not optimal for its frozen weights."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_path = output_dir / f"{dataset}_{backbone.lower()}_ablation.csv"
    sensitivity_path = (
        output_dir / f"{dataset}_{backbone.lower()}_lambda_sensitivity.csv"
    )
    pd.DataFrame(records).to_csv(ablation_path, index=False)
    sensitivity_frame.to_csv(sensitivity_path, index=False)

    return {
        "dataset": dataset,
        "backbone": backbone,
        "candidate_dir": str(candidate_dir),
        "ablation_csv": str(ablation_path),
        "lambda_sensitivity_csv": str(sensitivity_path),
        "num_validation_samples": int(len(validation_batch.labels)),
        "num_test_samples": int(len(test_batch.labels)),
        "previous_context_coverage": {
            "validation": statistics.context_coverage(validation_queries),
            "test": statistics.context_coverage(test_queries),
        },
        "frozen_rule_strength": rule_strength,
        "frozen_weights": {
            name: float(weight)
            for name, weight in zip(DEFAULT_RULE_NAMES, full_weights)
        },
        "best_frozen_weight_strength_on_validation": float(
            best_sensitivity_row["rule_strength"]
        ),
        "duration_s": time.perf_counter() - started_at,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    summaries = [
        run_experiment(spec, output_dir)
        for spec in EXPERIMENTS
    ]
    combined = []
    for summary in summaries:
        frame = pd.read_csv(summary["ablation_csv"])
        combined.append(frame)
    combined_frame = pd.concat(combined, ignore_index=True)
    combined_path = output_dir / "combined_ablation.csv"
    combined_frame.to_csv(combined_path, index=False)

    payload = {
        "protocol": (
            "Frozen Top-20 candidates, frozen full-teacher weights, "
            "frozen dataset-specific lambda; no post-test retuning."
        ),
        "experiments": summaries,
        "combined_ablation_csv": str(combined_path),
    }
    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
