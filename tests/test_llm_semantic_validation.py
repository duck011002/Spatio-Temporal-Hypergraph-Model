import unittest

import numpy as np

from evaluate_llm_semantic_validation import (
    paired_partitions,
    ranks_after_candidate_order,
    ranking_metrics,
)


class LlmSemanticValidationTest(unittest.TestCase):
    def test_ranking_metrics(self):
        metrics = ranking_metrics(np.asarray([1, 2, 10, 30]))
        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_5"], 0.5)
        self.assertEqual(metrics["recall_at_10"], 0.75)
        self.assertEqual(metrics["recall_at_20"], 0.75)
        self.assertAlmostEqual(
            metrics["objective"],
            4.0 * 0.25 + 0.75,
        )

    def test_partitions_are_reproducible_and_disjoint(self):
        first = paired_partitions(100, 7)
        second = paired_partitions(100, 7)
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])
        combined = np.concatenate(
            [
                first["selection"],
                first["calibration"],
                first["audit"],
            ]
        )
        self.assertEqual(len(np.unique(combined)), 100)
        self.assertEqual(set(combined.tolist()), set(range(100)))

    def test_candidate_ranks_fall_back_when_label_is_outside_topk(self):
        candidate_ids = np.asarray([[4, 2, 1], [3, 5, 7]])
        labels = np.asarray([2, 8])
        fallback = np.asarray([10, 25])
        np.testing.assert_array_equal(
            ranks_after_candidate_order(candidate_ids, labels, fallback),
            [2, 25],
        )


if __name__ == "__main__":
    unittest.main()
