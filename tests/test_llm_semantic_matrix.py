import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_llm_semantic_matrix import (
    FEATURE_NAMES,
    build_semantic_matrix,
    save_semantic_matrix,
)
from generate_llm_category_profiles import (
    INTENT_NAMES,
    PROFILE_VERSION,
    ROLE_NAMES,
    TIME_NAMES,
)


def semantic_profile(value):
    return {
        "intents": {name: value for name in INTENT_NAMES},
        "time_affinity": {name: value for name in TIME_NAMES},
        "mobility_roles": {name: value for name in ROLE_NAMES},
        "repeat_likelihood": value,
        "routine_likelihood": value,
        "distance_tolerance": value,
        "confidence": value,
        "summary": "Synthetic semantic profile for testing.",
    }


class LlmSemanticMatrixTest(unittest.TestCase):
    def write_fixture(self, directory, include_second=True):
        sample_path = directory / "sample.csv"
        pd.DataFrame(
            [
                {
                    "PoiId": 0,
                    "PoiCategoryId": 10,
                    "PoiCategoryName": "Cafe",
                    "SplitTag": "train",
                },
                {
                    "PoiId": 1,
                    "PoiCategoryId": 11,
                    "PoiCategoryName": "Park",
                    "SplitTag": "train",
                },
                {
                    "PoiId": 0,
                    "PoiCategoryId": 12,
                    "PoiCategoryName": "Restaurant",
                    "SplitTag": "train",
                },
                {
                    "PoiId": 2,
                    "PoiCategoryId": 99,
                    "PoiCategoryName": "Unknown",
                    "SplitTag": "validation",
                },
            ]
        ).to_csv(sample_path, index=False)
        rows = [
            {
                "dataset": "nyc",
                "category_name": "Cafe",
                "profile": semantic_profile(0.2),
            }
        ]
        if include_second:
            rows.append(
                {
                    "dataset": "nyc",
                    "category_name": "Park",
                    "profile": semantic_profile(0.8),
                }
            )
        profile_path = directory / "profiles.json"
        profile_path.write_text(
            json.dumps(
                {
                    "protocol": {"profile_version": PROFILE_VERSION},
                    "profiles": rows,
                }
            ),
            encoding="utf-8",
        )
        return sample_path, profile_path

    def test_matrix_aligns_real_pois_and_reserves_final_padding_row(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            sample_path, profile_path = self.write_fixture(directory)
            matrix, available, category_ids = build_semantic_matrix(
                sample_path,
                profile_path,
                "nyc",
            )
            self.assertEqual(matrix.shape, (4, len(FEATURE_NAMES)))
            np.testing.assert_allclose(matrix[0], 0.2)
            np.testing.assert_allclose(matrix[1], 0.8)
            np.testing.assert_allclose(matrix[2], 0.0)
            np.testing.assert_allclose(matrix[3], 0.0)
            np.testing.assert_array_equal(
                available,
                [True, True, False, False],
            )
            np.testing.assert_array_equal(category_ids, [10, 11, -1, -1])

            output = save_semantic_matrix(
                directory / "matrix.npz",
                matrix,
                available,
                category_ids,
                "nyc",
            )
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["poi_profiles"].shape, matrix.shape)
                self.assertEqual(payload["feature_names"].tolist(), list(FEATURE_NAMES))

    def test_missing_category_profile_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            sample_path, profile_path = self.write_fixture(
                directory,
                include_second=False,
            )
            with self.assertRaisesRegex(ValueError, "Missing LLM profiles"):
                build_semantic_matrix(sample_path, profile_path, "nyc")


if __name__ == "__main__":
    unittest.main()
