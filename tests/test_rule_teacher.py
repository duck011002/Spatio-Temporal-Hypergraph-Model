import unittest

import numpy as np
import pandas as pd

from rule_teacher import (
    DatasetRuleStatistics,
    RankingBatch,
    evaluate_ranking,
    rerank_candidates,
    search_rule_weights,
)


class RuleTeacherTest(unittest.TestCase):
    def test_reranking_preserves_recall_at_candidate_boundary(self):
        batch = RankingBatch(
            candidate_ids=np.array([[1, 2, 3], [4, 5, 6]]),
            model_scores=np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]]),
            labels=np.array([2, 6]),
            base_label_ranks=np.array([2, 3]),
        )
        features = {"rule": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])}
        reranked, _ = rerank_candidates(
            batch,
            features,
            ("rule",),
            np.array([1.0]),
            1.0,
        )
        base = evaluate_ranking(
            batch.candidate_ids,
            batch.labels,
            batch.base_label_ranks,
            ks=(1, 3),
        )
        improved = evaluate_ranking(
            reranked,
            batch.labels,
            batch.base_label_ranks,
            ks=(1, 3),
        )
        self.assertEqual(base["recall_at_3"], improved["recall_at_3"])
        self.assertGreater(improved["recall_at_1"], base["recall_at_1"])

    def test_zero_rule_strength_returns_original_order(self):
        batch = RankingBatch(
            candidate_ids=np.array([[7, 8, 9]]),
            model_scores=np.array([[3.0, 2.0, 1.0]]),
            labels=np.array([7]),
            base_label_ranks=np.array([1]),
        )
        features = {"rule": np.array([[0.0, 0.0, 1.0]])}
        reranked, _ = rerank_candidates(
            batch,
            features,
            ("rule",),
            np.array([1.0]),
            0.0,
        )
        np.testing.assert_array_equal(reranked, batch.candidate_ids)

    def test_statistics_fit_only_training_split(self):
        frame = pd.DataFrame(
            [
                self._event(1, 10, 1, 0.0, 0.0, 100, "train"),
                self._event(1, 11, 1, 0.0, 0.1, 200, "train"),
                self._event(1, 99, 9, 0.0, 0.2, 300, "validation"),
                self._event(1, 99, 9, 0.0, 0.2, 400, "test"),
            ]
        )
        statistics = DatasetRuleStatistics.from_sample_frame("synthetic", frame)
        self.assertNotIn(99, statistics.poi_popularity)
        self.assertEqual(statistics.summary()["num_train_events"], 2)

    def test_weight_search_can_recover_helpful_rule(self):
        batch = RankingBatch(
            candidate_ids=np.array([[1, 2], [3, 4]]),
            model_scores=np.array([[2.0, 1.0], [2.0, 1.0]]),
            labels=np.array([2, 4]),
            base_label_ranks=np.array([2, 2]),
        )
        features = {
            "helpful": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "noise": np.array([[1.0, 0.0], [1.0, 0.0]]),
        }
        result = search_rule_weights(
            batch,
            features,
            rule_names=("helpful", "noise"),
            num_weight_samples=16,
            strength_grid=(0.0, 1.0),
            seed=7,
        )
        self.assertEqual(result.rule_strength, 1.0)
        self.assertGreater(result.weights[0], result.weights[1])
        self.assertEqual(result.validation_metrics["recall_at_1"], 1.0)

    @staticmethod
    def _event(user, poi, category, latitude, longitude, epoch, split):
        return {
            "UserId": user,
            "PoiId": poi,
            "PoiCategoryId": category,
            "Latitude": latitude,
            "Longitude": longitude,
            "UTCTimeOffset": pd.Timestamp(epoch, unit="s"),
            "UTCTimeOffsetEpoch": epoch,
            "SplitTag": split,
            "check_ins_id": epoch,
        }


if __name__ == "__main__":
    unittest.main()
