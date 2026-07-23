import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_rule_teacher_ablation import EXPERIMENTS, load_queries
from rule_teacher import (
    DEFAULT_RULE_NAMES,
    DatasetRuleStatistics,
    RankingBatch,
    normalize_rows,
)


EARTH_RADIUS_KM = 6371.0088


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze which frozen-candidate samples are corrected or harmed "
            "by the frozen rule teacher and define validation-derived LLM gates."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="docs/results/rule_teacher_error_analysis",
    )
    parser.add_argument(
        "--full-sample-dir",
        default="server_artifacts/rule_teacher_error_analysis",
    )
    parser.add_argument("--representatives-per-outcome", type=int, default=10)
    return parser.parse_args()


def haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(np.asarray(lat1, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1, dtype=np.float64))
    lat2 = np.radians(np.asarray(lat2, dtype=np.float64))
    lon2 = np.radians(np.asarray(lon2, dtype=np.float64))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0.0, 1.0)))


def rank_from_candidates(candidate_ids, labels, fallback_ranks):
    matches = candidate_ids == labels[:, None]
    found = matches.any(axis=1)
    positions = np.argmax(matches, axis=1) + 1
    ranks = np.where(found, positions, fallback_ranks)
    return found, positions, ranks


def gather_target(values, found, positions):
    row_indices = np.arange(values.shape[0])
    safe_positions = np.maximum(positions - 1, 0)
    gathered = values[row_indices, safe_positions].astype(np.float64)
    gathered[~found] = np.nan
    return gathered


def margins(scores):
    sorted_scores = np.sort(scores, axis=1)
    if scores.shape[1] < 2:
        return np.zeros(scores.shape[0], dtype=np.float64)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def category_name_maps(statistics):
    if "PoiCategoryName" not in statistics.train_events.columns:
        return {}
    category_names = (
        statistics.train_events[
            ["PoiCategoryId", "PoiCategoryName"]
        ]
        .dropna()
        .groupby("PoiCategoryId")["PoiCategoryName"]
        .agg(lambda values: str(values.mode().iloc[0]))
    )
    return {
        int(category_id): name
        for category_id, name in category_names.items()
    }


def poi_categories(candidate_ids, poi_category):
    result = np.full(candidate_ids.shape, -1, dtype=np.int64)
    for row_index in range(candidate_ids.shape[0]):
        for column_index in range(candidate_ids.shape[1]):
            result[row_index, column_index] = poi_category.get(
                int(candidate_ids[row_index, column_index]),
                -1,
            )
    return result


def analyze_split(
    dataset,
    backbone,
    split,
    batch,
    queries,
    features,
    statistics,
    weights,
    rule_strength,
):
    model_scores = normalize_rows(batch.model_scores)
    stacked_features = np.stack(
        [features[name] for name in DEFAULT_RULE_NAMES],
        axis=-1,
    )
    rule_scores = np.tensordot(stacked_features, weights, axes=([-1], [0]))
    final_scores = (
        (1.0 - rule_strength) * model_scores
        + rule_strength * rule_scores
    )
    final_order = np.argsort(-final_scores, axis=1, kind="stable")
    final_candidates = np.take_along_axis(
        batch.candidate_ids,
        final_order,
        axis=1,
    )

    found, target_positions, base_ranks = rank_from_candidates(
        batch.candidate_ids,
        batch.labels,
        batch.base_label_ranks,
    )
    _, final_positions, final_ranks = rank_from_candidates(
        final_candidates,
        batch.labels,
        batch.base_label_ranks,
    )
    if not np.array_equal(found, final_candidates.__eq__(batch.labels[:, None]).any(axis=1)):
        raise AssertionError(f"{dataset} {split}: candidate membership changed.")

    model_top1 = batch.candidate_ids[:, 0]
    rule_top1_positions = np.argmax(rule_scores, axis=1)
    rule_top1 = batch.candidate_ids[
        np.arange(len(batch.labels)),
        rule_top1_positions,
    ]
    final_top1 = final_candidates[:, 0]
    model_rule_disagree = model_top1 != rule_top1
    teacher_changes_top1 = model_top1 != final_top1

    outcomes = np.full(len(batch.labels), "unchanged", dtype=object)
    outcomes[(base_ranks == 1) & (final_ranks == 1)] = "kept_top1"
    outcomes[(base_ranks > 1) & (final_ranks == 1)] = "corrected_top1"
    outcomes[(base_ranks == 1) & (final_ranks > 1)] = "broken_top1"
    outcomes[
        (base_ranks > 1)
        & (final_ranks > 1)
        & (final_ranks < base_ranks)
    ] = "improved_non_top1"
    outcomes[
        (base_ranks > 1)
        & (final_ranks > 1)
        & (final_ranks > base_ranks)
    ] = "worsened_non_top1"

    category_names = category_name_maps(statistics)
    candidate_categories = poi_categories(
        batch.candidate_ids,
        statistics.poi_category,
    )
    final_categories = np.take_along_axis(
        candidate_categories,
        final_order,
        axis=1,
    )

    users = pd.to_numeric(
        queries["UserId"],
        errors="coerce",
    ).fillna(-1).astype(np.int64).to_numpy()
    previous_epochs = pd.to_numeric(
        queries["last_checkin_epoch_time"],
        errors="coerce",
    ).fillna(-1).astype(np.int64).to_numpy()
    previous_categories = np.full(len(queries), -1, dtype=np.int64)
    previous_latitudes = np.full(len(queries), np.nan, dtype=np.float64)
    previous_longitudes = np.full(len(queries), np.nan, dtype=np.float64)
    for index, (user_id, epoch) in enumerate(zip(users, previous_epochs)):
        previous = statistics.previous_event_lookup.get(
            (int(user_id), int(epoch))
        )
        if previous is not None:
            previous_categories[index] = int(previous[1])
            previous_latitudes[index] = float(previous[2])
            previous_longitudes[index] = float(previous[3])

    target_latitudes = pd.to_numeric(
        queries["Latitude"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    target_longitudes = pd.to_numeric(
        queries["Longitude"],
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    target_distance_km = haversine_km(
        previous_latitudes,
        previous_longitudes,
        target_latitudes,
        target_longitudes,
    )

    target_categories = pd.to_numeric(
        queries["PoiCategoryId"],
        errors="coerce",
    ).fillna(-1).astype(np.int64).to_numpy()
    query_labels = pd.to_numeric(
        queries["PoiId"],
        errors="coerce",
    ).fillna(-1).astype(np.int64).to_numpy()
    if not np.array_equal(query_labels, batch.labels.astype(np.int64)):
        raise AssertionError(f"{dataset} {split}: query labels are misaligned.")

    sample_indices = (
        np.asarray(batch.sample_indices, dtype=np.int64)
        if batch.sample_indices is not None
        else np.arange(len(batch.labels), dtype=np.int64)
    )
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "backbone": backbone,
            "split": split,
            "sample_index": sample_indices,
            "user_id": users,
            "timestamp": queries["UTCTimeOffset"].astype(str).to_numpy(),
            "label_poi_id": batch.labels.astype(np.int64),
            "label_category_id": target_categories,
            "previous_category_id": previous_categories,
            "model_top1_poi_id": model_top1.astype(np.int64),
            "model_top1_category_id": candidate_categories[:, 0],
            "rule_top1_poi_id": rule_top1.astype(np.int64),
            "rule_top1_category_id": candidate_categories[
                np.arange(len(batch.labels)),
                rule_top1_positions,
            ],
            "final_top1_poi_id": final_top1.astype(np.int64),
            "final_top1_category_id": final_categories[:, 0],
            "target_in_top20": found,
            "base_rank": base_ranks.astype(np.int64),
            "final_rank": final_ranks.astype(np.int64),
            "rank_improvement": (base_ranks - final_ranks).astype(np.int64),
            "outcome": outcomes,
            "model_rule_top1_disagree": model_rule_disagree,
            "teacher_changes_top1": teacher_changes_top1,
            "model_margin": margins(model_scores),
            "rule_margin": margins(rule_scores),
            "final_margin": margins(final_scores),
            "target_model_score": gather_target(
                model_scores,
                found,
                target_positions,
            ),
            "target_rule_score": gather_target(
                rule_scores,
                found,
                target_positions,
            ),
            "target_distance_km": target_distance_km,
        }
    )
    for name in DEFAULT_RULE_NAMES:
        frame[f"target_{name}"] = gather_target(
            features[name],
            found,
            target_positions,
        )

    for column, categories in (
        ("label_category_name", target_categories),
        ("previous_category_name", previous_categories),
        ("model_top1_category_name", candidate_categories[:, 0]),
        ("final_top1_category_name", final_categories[:, 0]),
    ):
        frame[column] = [
            category_names.get(int(category_id), str(int(category_id)))
            for category_id in categories
        ]
    return frame


def validation_thresholds(frame):
    return {
        "model_margin_q25": float(frame["model_margin"].quantile(0.25)),
        "model_margin_q50": float(frame["model_margin"].quantile(0.50)),
        "final_margin_q50": float(frame["final_margin"].quantile(0.50)),
    }


def gate_masks(frame, thresholds):
    disagreement = frame["model_rule_top1_disagree"].to_numpy(dtype=bool)
    changed = frame["teacher_changes_top1"].to_numpy(dtype=bool)
    low_model_q25 = (
        frame["model_margin"].to_numpy()
        <= thresholds["model_margin_q25"]
    )
    low_model_q50 = (
        frame["model_margin"].to_numpy()
        <= thresholds["model_margin_q50"]
    )
    low_final_q50 = (
        frame["final_margin"].to_numpy()
        <= thresholds["final_margin_q50"]
    )
    return {
        "all_samples": np.ones(len(frame), dtype=bool),
        "model_rule_disagree": disagreement,
        "disagree_and_low_model_margin_q25": disagreement & low_model_q25,
        "disagree_and_low_model_margin_q50": disagreement & low_model_q50,
        "teacher_changes_top1": changed,
        "changed_and_low_model_margin_q50": changed & low_model_q50,
        "changed_and_low_final_margin_q50": changed & low_final_q50,
    }


def summarize_gates(dataset, backbone, frame, thresholds):
    outcomes = frame["outcome"].to_numpy()
    total_corrected = int(np.sum(outcomes == "corrected_top1"))
    total_broken = int(np.sum(outcomes == "broken_top1"))
    records = []
    for gate_name, mask in gate_masks(frame, thresholds).items():
        size = int(mask.sum())
        corrected = int(np.sum(mask & (outcomes == "corrected_top1")))
        broken = int(np.sum(mask & (outcomes == "broken_top1")))
        records.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "gate": gate_name,
                "num_samples": size,
                "sample_fraction": size / len(frame),
                "corrected_top1": corrected,
                "broken_top1": broken,
                "net_top1": corrected - broken,
                "corrected_capture_rate": (
                    corrected / total_corrected
                    if total_corrected
                    else 0.0
                ),
                "broken_capture_rate": (
                    broken / total_broken
                    if total_broken
                    else 0.0
                ),
                "top1_change_density": (
                    (corrected + broken) / size
                    if size
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_outcomes(dataset, backbone, frame):
    counts = frame["outcome"].value_counts()
    corrected = int(counts.get("corrected_top1", 0))
    broken = int(counts.get("broken_top1", 0))
    return {
        "dataset": dataset,
        "backbone": backbone,
        "num_samples": int(len(frame)),
        "target_in_top20": int(frame["target_in_top20"].sum()),
        "target_in_top20_fraction": float(frame["target_in_top20"].mean()),
        "model_rule_top1_disagree": int(
            frame["model_rule_top1_disagree"].sum()
        ),
        "model_rule_top1_disagree_fraction": float(
            frame["model_rule_top1_disagree"].mean()
        ),
        "teacher_changes_top1": int(frame["teacher_changes_top1"].sum()),
        "teacher_changes_top1_fraction": float(
            frame["teacher_changes_top1"].mean()
        ),
        "base_top1_correct": int((frame["base_rank"] == 1).sum()),
        "final_top1_correct": int((frame["final_rank"] == 1).sum()),
        "corrected_top1": corrected,
        "broken_top1": broken,
        "net_top1": corrected - broken,
        "kept_top1": int(counts.get("kept_top1", 0)),
        "improved_non_top1": int(counts.get("improved_non_top1", 0)),
        "worsened_non_top1": int(counts.get("worsened_non_top1", 0)),
        "unchanged": int(counts.get("unchanged", 0)),
    }


def summarize_features(dataset, backbone, frame):
    columns = [
        "model_margin",
        "rule_margin",
        "final_margin",
        "target_distance_km",
        "target_model_score",
        "target_rule_score",
        *[f"target_{name}" for name in DEFAULT_RULE_NAMES],
    ]
    records = []
    for outcome, group in frame.groupby("outcome", sort=False):
        record = {
            "dataset": dataset,
            "backbone": backbone,
            "outcome": outcome,
            "num_samples": int(len(group)),
            "model_rule_top1_disagree_fraction": float(
                group["model_rule_top1_disagree"].mean()
            ),
            "target_in_top20_fraction": float(
                group["target_in_top20"].mean()
            ),
            "base_rank_mean": float(group["base_rank"].mean()),
            "final_rank_mean": float(group["final_rank"].mean()),
        }
        for column in columns:
            record[f"{column}_mean"] = float(group[column].mean())
            record[f"{column}_median"] = float(group[column].median())
        records.append(record)
    return pd.DataFrame(records)


def representative_cases(frame, per_outcome):
    selected = []
    for outcome, group in frame.groupby("outcome", sort=False):
        if outcome in ("corrected_top1", "improved_non_top1"):
            group = group.sort_values(
                ["rank_improvement", "target_rule_score"],
                ascending=[False, False],
            )
        elif outcome in ("broken_top1", "worsened_non_top1"):
            group = group.sort_values(
                ["rank_improvement", "model_margin"],
                ascending=[True, False],
            )
        else:
            continue
        selected.append(group.head(per_outcome))
    if not selected:
        return frame.head(0)
    columns = [
        "dataset",
        "backbone",
        "sample_index",
        "user_id",
        "timestamp",
        "outcome",
        "label_poi_id",
        "label_category_name",
        "previous_category_name",
        "model_top1_poi_id",
        "model_top1_category_name",
        "final_top1_poi_id",
        "final_top1_category_name",
        "base_rank",
        "final_rank",
        "rank_improvement",
        "model_rule_top1_disagree",
        "model_margin",
        "rule_margin",
        "final_margin",
        "target_distance_km",
        *[f"target_{name}" for name in DEFAULT_RULE_NAMES],
    ]
    return pd.concat(selected, ignore_index=True)[columns]


def run_experiment(spec, output_dir, full_sample_dir, per_outcome):
    started_at = time.perf_counter()
    dataset = spec["dataset"]
    backbone = spec["backbone"]
    candidate_dir = spec["candidate_dir"]
    summary = json.loads(
        (candidate_dir / "rule_teacher_summary.json").read_text(
            encoding="utf-8"
        )
    )
    rule_search = summary["rule_search"]
    weights = np.array(
        [rule_search["weights"][name] for name in DEFAULT_RULE_NAMES],
        dtype=np.float64,
    )
    rule_strength = float(rule_search["rule_strength"])
    statistics = DatasetRuleStatistics.from_preprocessed_dir(
        dataset,
        str(Path("data") / dataset / "preprocessed"),
    )

    frames = {}
    for split, file_name in (
        ("validation", "validation_candidates.npz"),
        ("test", "test_candidates.npz"),
    ):
        batch = RankingBatch.load(str(candidate_dir / file_name))
        queries = load_queries(dataset, split, batch)
        features = statistics.score_candidates(
            queries,
            batch.candidate_ids,
        )
        frames[split] = analyze_split(
            dataset,
            backbone,
            split,
            batch,
            queries,
            features,
            statistics,
            weights,
            rule_strength,
        )

    thresholds = validation_thresholds(frames["validation"])
    test_frame = frames["test"]
    outcome_summary = summarize_outcomes(dataset, backbone, test_frame)
    expected_net = int(
        round(
            summary["test_delta"]["recall_at_1"]
            * outcome_summary["num_samples"]
        )
    )
    if outcome_summary["net_top1"] != expected_net:
        raise AssertionError(
            f"{dataset}: top-1 correction accounting does not match summary."
        )

    gate_summary = summarize_gates(
        dataset,
        backbone,
        test_frame,
        thresholds,
    )
    feature_summary = summarize_features(
        dataset,
        backbone,
        test_frame,
    )
    representatives = representative_cases(test_frame, per_outcome)

    full_sample_dir.mkdir(parents=True, exist_ok=True)
    full_sample_path = full_sample_dir / f"{dataset}_{backbone.lower()}_samples.csv"
    test_frame.to_csv(full_sample_path, index=False)

    return {
        "dataset": dataset,
        "backbone": backbone,
        "thresholds_from_validation": thresholds,
        "outcome_summary": outcome_summary,
        "gate_summary": gate_summary,
        "feature_summary": feature_summary,
        "representatives": representatives,
        "full_sample_csv": str(full_sample_path),
        "duration_s": time.perf_counter() - started_at,
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    full_sample_dir = Path(args.full_sample_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        run_experiment(
            spec,
            output_dir,
            full_sample_dir,
            args.representatives_per_outcome,
        )
        for spec in EXPERIMENTS
    ]
    outcome_frame = pd.DataFrame(
        [result["outcome_summary"] for result in results]
    )
    gate_frame = pd.concat(
        [result["gate_summary"] for result in results],
        ignore_index=True,
    )
    feature_frame = pd.concat(
        [result["feature_summary"] for result in results],
        ignore_index=True,
    )
    representative_frame = pd.concat(
        [result["representatives"] for result in results],
        ignore_index=True,
    )

    outcome_path = output_dir / "outcome_summary.csv"
    gate_path = output_dir / "llm_gate_summary.csv"
    feature_path = output_dir / "outcome_feature_summary.csv"
    representative_path = output_dir / "representative_cases.csv"
    outcome_frame.to_csv(outcome_path, index=False)
    gate_frame.to_csv(gate_path, index=False)
    feature_frame.to_csv(feature_path, index=False)
    representative_frame.to_csv(representative_path, index=False)

    payload = {
        "protocol": (
            "Frozen candidates and frozen rule teacher. Margin thresholds "
            "are derived from validation only; test is used for reporting."
        ),
        "experiments": [
            {
                "dataset": result["dataset"],
                "backbone": result["backbone"],
                "thresholds_from_validation": result[
                    "thresholds_from_validation"
                ],
                "full_sample_csv": result["full_sample_csv"],
                "duration_s": result["duration_s"],
            }
            for result in results
        ],
        "outputs": {
            "outcome_summary": str(outcome_path),
            "llm_gate_summary": str(gate_path),
            "outcome_feature_summary": str(feature_path),
            "representative_cases": str(representative_path),
        },
    }
    summary_path = output_dir / "error_analysis_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
