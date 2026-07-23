from __future__ import annotations

import os.path as osp
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DEFAULT_RULE_NAMES = (
    "spatial",
    "temporal",
    "category_transition",
    "user_preference",
    "popularity",
)

EARTH_RADIUS_KM = 6371.0088
EPSILON = 1e-12


def _haversine_km(lat1, lon1, lat2, lon2):
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


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """Min-max normalize each query independently."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional score matrix.")
    row_min = np.nanmin(values, axis=1, keepdims=True)
    row_max = np.nanmax(values, axis=1, keepdims=True)
    spread = row_max - row_min
    normalized = np.divide(
        values - row_min,
        spread,
        out=np.zeros_like(values, dtype=np.float64),
        where=spread > EPSILON,
    )
    return np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)


@dataclass(frozen=True)
class RankingBatch:
    candidate_ids: np.ndarray
    model_scores: np.ndarray
    labels: np.ndarray
    base_label_ranks: np.ndarray
    sample_indices: Optional[np.ndarray] = None

    def __post_init__(self):
        candidate_ids = np.asarray(self.candidate_ids)
        model_scores = np.asarray(self.model_scores)
        labels = np.asarray(self.labels).reshape(-1)
        base_label_ranks = np.asarray(self.base_label_ranks).reshape(-1)
        if candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must have shape [num_queries, top_k].")
        if model_scores.shape != candidate_ids.shape:
            raise ValueError("model_scores must have the same shape as candidate_ids.")
        if labels.shape[0] != candidate_ids.shape[0]:
            raise ValueError("labels must contain one POI id per query.")
        if base_label_ranks.shape[0] != candidate_ids.shape[0]:
            raise ValueError("base_label_ranks must contain one rank per query.")

    @classmethod
    def load(cls, path: str) -> "RankingBatch":
        payload = np.load(path)
        sample_indices = (
            payload["sample_indices"]
            if "sample_indices" in payload.files
            else None
        )
        return cls(
            candidate_ids=payload["candidate_ids"],
            model_scores=payload["model_scores"],
            labels=payload["labels"],
            base_label_ranks=payload["base_label_ranks"],
            sample_indices=sample_indices,
        )

    def save(self, path: str) -> None:
        payload = {
            "candidate_ids": np.asarray(self.candidate_ids, dtype=np.int64),
            "model_scores": np.asarray(self.model_scores, dtype=np.float32),
            "labels": np.asarray(self.labels, dtype=np.int64),
            "base_label_ranks": np.asarray(self.base_label_ranks, dtype=np.int64),
        }
        if self.sample_indices is not None:
            payload["sample_indices"] = np.asarray(
                self.sample_indices,
                dtype=np.int64,
            )
        np.savez_compressed(path, **payload)


@dataclass(frozen=True)
class RuleSearchResult:
    rule_names: Sequence[str]
    weights: np.ndarray
    rule_strength: float
    objective: float
    validation_metrics: Mapping[str, float]

    def as_dict(self) -> Dict[str, object]:
        return {
            "rule_names": list(self.rule_names),
            "weights": {
                name: float(weight)
                for name, weight in zip(self.rule_names, self.weights)
            },
            "rule_strength": float(self.rule_strength),
            "objective": float(self.objective),
            "validation_metrics": {
                key: float(value)
                for key, value in self.validation_metrics.items()
            },
        }


class DatasetRuleStatistics:
    """Training-only rule statistics with dataset-specific calibration."""

    required_columns = {
        "UserId",
        "PoiId",
        "PoiCategoryId",
        "Latitude",
        "Longitude",
        "UTCTimeOffset",
        "UTCTimeOffsetEpoch",
    }

    def __init__(
        self,
        dataset_name: str,
        train_events: pd.DataFrame,
        history_events: pd.DataFrame,
        transition_backoff: float = 5.0,
        temporal_backoff: float = 5.0,
    ):
        self.dataset_name = str(dataset_name)
        self.transition_backoff = float(transition_backoff)
        self.temporal_backoff = float(temporal_backoff)
        self.train_events = self._prepare_events(train_events)
        self.history_events = self._prepare_events(history_events)
        if self.train_events.empty:
            raise ValueError("Training events are empty.")

        self._build_poi_metadata()
        self._build_global_priors()
        self._build_transition_statistics()
        self._build_spatial_statistics()
        self._build_user_preferences()
        self._build_previous_event_lookup()

    @classmethod
    def from_sample_frame(
        cls,
        dataset_name: str,
        sample_frame: pd.DataFrame,
    ) -> "DatasetRuleStatistics":
        if "SplitTag" not in sample_frame.columns:
            raise ValueError("sample.csv must contain SplitTag.")
        split = sample_frame["SplitTag"].astype(str).str.lower()
        train_events = sample_frame.loc[split == "train"].copy()
        return cls(dataset_name, train_events, sample_frame)

    @classmethod
    def from_preprocessed_dir(
        cls,
        dataset_name: str,
        preprocessed_dir: str,
    ) -> "DatasetRuleStatistics":
        sample_frame = pd.read_csv(
            osp.join(preprocessed_dir, "sample.csv"),
            low_memory=False,
        )
        return cls.from_sample_frame(dataset_name, sample_frame)

    def _prepare_events(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = self.required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing event columns: {sorted(missing)}")
        frame = frame.copy()
        for column in ("UserId", "PoiId", "PoiCategoryId"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["UTCTimeOffsetEpoch"] = pd.to_numeric(
            frame["UTCTimeOffsetEpoch"],
            errors="coerce",
        )
        frame["Latitude"] = pd.to_numeric(frame["Latitude"], errors="coerce")
        frame["Longitude"] = pd.to_numeric(frame["Longitude"], errors="coerce")
        timestamps = pd.to_datetime(frame["UTCTimeOffset"], errors="coerce")
        frame["_hour"] = timestamps.dt.hour
        frame["_weekday"] = timestamps.dt.weekday
        frame = frame.dropna(
            subset=[
                "UserId",
                "PoiId",
                "PoiCategoryId",
                "UTCTimeOffsetEpoch",
                "Latitude",
                "Longitude",
                "_hour",
                "_weekday",
            ]
        )
        for column in (
            "UserId",
            "PoiId",
            "PoiCategoryId",
            "UTCTimeOffsetEpoch",
            "_hour",
            "_weekday",
        ):
            frame[column] = frame[column].astype(np.int64)
        frame["_time_bucket"] = (
            frame["_hour"] + 24 * (frame["_weekday"] >= 5).astype(np.int64)
        )
        return frame.sort_values(
            ["UserId", "UTCTimeOffsetEpoch", "check_ins_id"]
            if "check_ins_id" in frame.columns
            else ["UserId", "UTCTimeOffsetEpoch"]
        ).reset_index(drop=True)

    def _build_poi_metadata(self) -> None:
        poi_groups = self.train_events.groupby("PoiId", sort=False)
        poi_category = poi_groups["PoiCategoryId"].agg(
            lambda values: int(values.mode().iloc[0])
        )
        poi_latitude = poi_groups["Latitude"].median()
        poi_longitude = poi_groups["Longitude"].median()
        self.poi_category = {
            int(key): int(value)
            for key, value in poi_category.items()
        }
        self.poi_latitude = {
            int(key): float(value)
            for key, value in poi_latitude.items()
        }
        self.poi_longitude = {
            int(key): float(value)
            for key, value in poi_longitude.items()
        }

    def _build_global_priors(self) -> None:
        category_counts = Counter(
            self.train_events["PoiCategoryId"].astype(int).tolist()
        )
        poi_counts = Counter(self.train_events["PoiId"].astype(int).tolist())
        total_events = float(sum(category_counts.values()))
        self.category_prior = {
            key: value / total_events
            for key, value in category_counts.items()
        }
        self.poi_popularity = {
            key: np.log1p(value)
            for key, value in poi_counts.items()
        }

        self.temporal_category_counts = Counter(
            zip(
                self.train_events["_time_bucket"].astype(int),
                self.train_events["PoiCategoryId"].astype(int),
            )
        )
        self.temporal_bucket_totals = Counter(
            self.train_events["_time_bucket"].astype(int).tolist()
        )

    def _build_transition_statistics(self) -> None:
        group_columns = ["UserId"]
        if "pseudo_session_trajectory_id" in self.train_events.columns:
            group_columns.append("pseudo_session_trajectory_id")
        previous_category = self.train_events.groupby(group_columns)[
            "PoiCategoryId"
        ].shift(1)
        valid = previous_category.notna()
        pairs = zip(
            previous_category.loc[valid].astype(int),
            self.train_events.loc[valid, "PoiCategoryId"].astype(int),
        )
        self.category_transition_counts = Counter(pairs)
        self.category_transition_totals = Counter(
            previous_category.loc[valid].astype(int).tolist()
        )

    def _build_spatial_statistics(self) -> None:
        group_columns = ["UserId"]
        if "pseudo_session_trajectory_id" in self.train_events.columns:
            group_columns.append("pseudo_session_trajectory_id")
        grouped = self.train_events.groupby(group_columns, sort=False)
        previous_latitude = grouped["Latitude"].shift(1)
        previous_longitude = grouped["Longitude"].shift(1)
        valid = previous_latitude.notna() & previous_longitude.notna()
        distances = _haversine_km(
            previous_latitude.loc[valid],
            previous_longitude.loc[valid],
            self.train_events.loc[valid, "Latitude"],
            self.train_events.loc[valid, "Longitude"],
        )
        distance_frame = pd.DataFrame(
            {
                "UserId": self.train_events.loc[valid, "UserId"].to_numpy(),
                "distance": distances,
            }
        )
        finite_positive = distance_frame[
            np.isfinite(distance_frame["distance"])
            & (distance_frame["distance"] > 0.01)
        ]
        if finite_positive.empty:
            self.dataset_distance_scale_km = 1.0
            self.user_distance_scale_km = {}
            return
        self.dataset_distance_scale_km = max(
            float(finite_positive["distance"].median()),
            0.1,
        )
        user_medians = finite_positive.groupby("UserId")["distance"].median()
        lower = self.dataset_distance_scale_km * 0.2
        upper = self.dataset_distance_scale_km * 5.0
        self.user_distance_scale_km = {
            int(user_id): float(np.clip(value, lower, upper))
            for user_id, value in user_medians.items()
        }

    def _build_user_preferences(self) -> None:
        self.user_category_counts = defaultdict(Counter)
        self.user_poi_counts = defaultdict(Counter)
        self.user_totals = Counter()
        for row in self.train_events[
            ["UserId", "PoiId", "PoiCategoryId"]
        ].itertuples(index=False):
            user_id = int(row.UserId)
            self.user_category_counts[user_id][int(row.PoiCategoryId)] += 1
            self.user_poi_counts[user_id][int(row.PoiId)] += 1
            self.user_totals[user_id] += 1

    def _build_previous_event_lookup(self) -> None:
        latest = self.history_events.drop_duplicates(
            ["UserId", "UTCTimeOffsetEpoch"],
            keep="last",
        )
        self.previous_event_lookup = {
            (int(row.UserId), int(row.UTCTimeOffsetEpoch)): (
                int(row.PoiId),
                int(row.PoiCategoryId),
                float(row.Latitude),
                float(row.Longitude),
            )
            for row in latest[
                [
                    "UserId",
                    "UTCTimeOffsetEpoch",
                    "PoiId",
                    "PoiCategoryId",
                    "Latitude",
                    "Longitude",
                ]
            ].itertuples(index=False)
        }

    def score_candidates(
        self,
        query_frame: pd.DataFrame,
        candidate_ids: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
        if candidate_ids.ndim != 2:
            raise ValueError("candidate_ids must be two-dimensional.")
        if len(query_frame) != candidate_ids.shape[0]:
            raise ValueError("Query rows and candidate rows are not aligned.")
        required = {
            "UserId",
            "UTCTimeOffset",
            "last_checkin_epoch_time",
        }
        missing = required.difference(query_frame.columns)
        if missing:
            raise ValueError(f"Missing query columns: {sorted(missing)}")

        query_frame = query_frame.reset_index(drop=True).copy()
        timestamps = pd.to_datetime(
            query_frame["UTCTimeOffset"],
            errors="coerce",
        )
        hours = timestamps.dt.hour.fillna(0).astype(int).to_numpy()
        weekdays = timestamps.dt.weekday.fillna(0).astype(int).to_numpy()
        time_buckets = hours + 24 * (weekdays >= 5).astype(np.int64)
        users = pd.to_numeric(
            query_frame["UserId"],
            errors="coerce",
        ).fillna(-1).astype(int).to_numpy()
        previous_epochs = pd.to_numeric(
            query_frame["last_checkin_epoch_time"],
            errors="coerce",
        ).fillna(-1).astype(np.int64).to_numpy()

        shape = candidate_ids.shape
        spatial = np.zeros(shape, dtype=np.float64)
        temporal = np.zeros(shape, dtype=np.float64)
        transition = np.zeros(shape, dtype=np.float64)
        preference = np.zeros(shape, dtype=np.float64)
        popularity = np.zeros(shape, dtype=np.float64)

        for row_index, (user_id, previous_epoch, bucket) in enumerate(
            zip(users, previous_epochs, time_buckets)
        ):
            previous = self.previous_event_lookup.get(
                (int(user_id), int(previous_epoch))
            )
            if previous is None:
                previous_poi = -1
                previous_category = -1
                previous_latitude = np.nan
                previous_longitude = np.nan
            else:
                (
                    previous_poi,
                    previous_category,
                    previous_latitude,
                    previous_longitude,
                ) = previous

            user_total = max(self.user_totals.get(int(user_id), 0), 1)
            distance_scale = self.user_distance_scale_km.get(
                int(user_id),
                self.dataset_distance_scale_km,
            )
            transition_total = self.category_transition_totals.get(
                int(previous_category),
                0,
            )
            temporal_total = self.temporal_bucket_totals.get(int(bucket), 0)

            for column_index, poi_id in enumerate(candidate_ids[row_index]):
                poi_id = int(poi_id)
                category_id = self.poi_category.get(poi_id, -1)
                latitude = self.poi_latitude.get(poi_id, np.nan)
                longitude = self.poi_longitude.get(poi_id, np.nan)
                category_prior = self.category_prior.get(category_id, 0.0)

                if (
                    np.isfinite(previous_latitude)
                    and np.isfinite(previous_longitude)
                    and np.isfinite(latitude)
                    and np.isfinite(longitude)
                ):
                    distance = float(
                        _haversine_km(
                            previous_latitude,
                            previous_longitude,
                            latitude,
                            longitude,
                        )
                    )
                    spatial[row_index, column_index] = np.exp(
                        -distance / max(distance_scale, 0.1)
                    )

                temporal_count = self.temporal_category_counts.get(
                    (int(bucket), int(category_id)),
                    0,
                )
                temporal[row_index, column_index] = (
                    temporal_count
                    + self.temporal_backoff * category_prior
                ) / (temporal_total + self.temporal_backoff)

                transition_count = self.category_transition_counts.get(
                    (int(previous_category), int(category_id)),
                    0,
                )
                transition[row_index, column_index] = (
                    transition_count
                    + self.transition_backoff * category_prior
                ) / (transition_total + self.transition_backoff)

                category_preference = (
                    self.user_category_counts[int(user_id)].get(
                        int(category_id),
                        0,
                    )
                    / user_total
                )
                poi_preference = (
                    self.user_poi_counts[int(user_id)].get(poi_id, 0)
                    / user_total
                )
                preference[row_index, column_index] = (
                    0.7 * category_preference + 0.3 * poi_preference
                )
                popularity[row_index, column_index] = self.poi_popularity.get(
                    poi_id,
                    0.0,
                )

        return {
            "spatial": normalize_rows(spatial),
            "temporal": normalize_rows(temporal),
            "category_transition": normalize_rows(transition),
            "user_preference": normalize_rows(preference),
            "popularity": normalize_rows(popularity),
        }

    def context_coverage(self, query_frame: pd.DataFrame) -> float:
        required = {"UserId", "last_checkin_epoch_time"}
        missing = required.difference(query_frame.columns)
        if missing:
            raise ValueError(f"Missing query columns: {sorted(missing)}")
        users = pd.to_numeric(
            query_frame["UserId"],
            errors="coerce",
        ).fillna(-1).astype(np.int64)
        epochs = pd.to_numeric(
            query_frame["last_checkin_epoch_time"],
            errors="coerce",
        ).fillna(-1).astype(np.int64)
        hits = sum(
            (int(user_id), int(epoch)) in self.previous_event_lookup
            for user_id, epoch in zip(users, epochs)
        )
        return float(hits / len(query_frame)) if len(query_frame) else 0.0

    def summary(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "num_train_events": int(len(self.train_events)),
            "num_history_events": int(len(self.history_events)),
            "num_users": int(self.train_events["UserId"].nunique()),
            "num_pois": int(len(self.poi_category)),
            "num_categories": int(
                self.train_events["PoiCategoryId"].nunique()
            ),
            "dataset_distance_scale_km": float(
                self.dataset_distance_scale_km
            ),
        }


def evaluate_ranking(
    candidate_ids: np.ndarray,
    labels: np.ndarray,
    base_label_ranks: np.ndarray,
    ks: Sequence[int] = (1, 5, 10, 20),
) -> Dict[str, float]:
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    base_label_ranks = np.asarray(base_label_ranks, dtype=np.int64).reshape(-1)
    if candidate_ids.ndim != 2:
        raise ValueError("candidate_ids must be two-dimensional.")
    matches = candidate_ids == labels[:, None]
    found = matches.any(axis=1)
    candidate_positions = np.argmax(matches, axis=1) + 1
    ranks = np.where(found, candidate_positions, base_label_ranks)

    metrics: Dict[str, float] = {}
    for k in ks:
        within = ranks <= int(k)
        metrics[f"recall_at_{k}"] = float(np.mean(within))
        discounted = np.where(
            within,
            1.0 / np.log2(ranks.astype(np.float64) + 1.0),
            0.0,
        )
        reciprocal = np.where(
            within,
            1.0 / ranks.astype(np.float64),
            0.0,
        )
        metrics[f"ndcg_at_{k}"] = float(np.mean(discounted))
        metrics[f"map_at_{k}"] = float(np.mean(reciprocal))
    metrics["mrr"] = float(
        np.mean(1.0 / np.maximum(ranks.astype(np.float64), 1.0))
    )
    return metrics


def rerank_candidates(
    batch: RankingBatch,
    rule_features: Mapping[str, np.ndarray],
    rule_names: Sequence[str],
    weights: np.ndarray,
    rule_strength: float,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(rule_names),):
        raise ValueError("Rule weight count does not match rule_names.")
    if not 0.0 <= float(rule_strength) <= 1.0:
        raise ValueError("rule_strength must be between 0 and 1.")

    stacked = np.stack(
        [np.asarray(rule_features[name], dtype=np.float64) for name in rule_names],
        axis=-1,
    )
    if stacked.shape[:2] != batch.candidate_ids.shape:
        raise ValueError("Rule feature shape does not match candidates.")
    rule_score = np.tensordot(stacked, weights, axes=([-1], [0]))
    model_score = normalize_rows(batch.model_scores)
    final_score = (
        (1.0 - float(rule_strength)) * model_score
        + float(rule_strength) * rule_score
    )
    order = np.argsort(-final_score, axis=1, kind="stable")
    candidates = np.take_along_axis(batch.candidate_ids, order, axis=1)
    scores = np.take_along_axis(final_score, order, axis=1)
    return candidates, scores


def _selection_objective(metrics: Mapping[str, float]) -> float:
    return float(
        4.0 * metrics["recall_at_1"]
        + metrics["recall_at_5"]
        + metrics["mrr"]
        + metrics["ndcg_at_5"]
    )


def search_rule_weights(
    batch: RankingBatch,
    rule_features: Mapping[str, np.ndarray],
    rule_names: Sequence[str] = DEFAULT_RULE_NAMES,
    num_weight_samples: int = 256,
    strength_grid: Optional[Sequence[float]] = None,
    seed: int = 20260723,
) -> RuleSearchResult:
    if strength_grid is None:
        strength_grid = np.linspace(0.0, 1.0, 21)
    rng = np.random.default_rng(seed)
    num_rules = len(rule_names)
    candidates = [
        np.full(num_rules, 1.0 / num_rules, dtype=np.float64),
        *np.eye(num_rules, dtype=np.float64),
    ]
    if num_weight_samples > 0:
        candidates.extend(
            rng.dirichlet(
                np.ones(num_rules, dtype=np.float64),
                size=num_weight_samples,
            )
        )

    stacked = np.stack(
        [np.asarray(rule_features[name], dtype=np.float64) for name in rule_names],
        axis=-1,
    )
    if stacked.shape[:2] != batch.candidate_ids.shape:
        raise ValueError("Rule feature shape does not match candidates.")
    model_score = normalize_rows(batch.model_scores)
    best_result = None
    best_key = None
    evaluated_zero_strength = False
    for weights in candidates:
        rule_score = np.tensordot(stacked, weights, axes=([-1], [0]))
        for rule_strength in strength_grid:
            rule_strength = float(rule_strength)
            if rule_strength == 0.0 and evaluated_zero_strength:
                continue
            final_score = (
                (1.0 - rule_strength) * model_score
                + rule_strength * rule_score
            )
            order = np.argsort(-final_score, axis=1, kind="stable")
            reranked = np.take_along_axis(
                batch.candidate_ids,
                order,
                axis=1,
            )
            metrics = evaluate_ranking(
                reranked,
                batch.labels,
                batch.base_label_ranks,
            )
            objective = _selection_objective(metrics)
            key = (
                round(objective, 12),
                round(metrics["mrr"], 12),
                -round(float(rule_strength), 12),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_result = RuleSearchResult(
                    rule_names=tuple(rule_names),
                    weights=np.asarray(weights, dtype=np.float64),
                    rule_strength=float(rule_strength),
                    objective=objective,
                    validation_metrics=metrics,
                )
            if rule_strength == 0.0:
                evaluated_zero_strength = True
    if best_result is None:
        raise RuntimeError("No rule-teacher candidate was evaluated.")
    return best_result
