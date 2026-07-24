import unittest

from generate_llm_category_profiles import (
    CategoryRequest,
    INTENT_NAMES,
    ROLE_NAMES,
    TIME_NAMES,
    build_prompt,
    parse_profile_response,
)


class LlmCategoryProfilesTest(unittest.TestCase):
    def valid_payload(self):
        return {
            "intents": {name: 0.2 for name in INTENT_NAMES},
            "time_affinity": {name: 0.3 for name in TIME_NAMES},
            "mobility_roles": {name: 0.4 for name in ROLE_NAMES},
            "repeat_likelihood": 0.5,
            "routine_likelihood": 0.6,
            "distance_tolerance": 0.7,
            "confidence": 0.8,
            "summary": "A routine category with broad time affinity.",
        }

    def test_prompt_contains_category_but_no_user_or_label(self):
        prompt = build_prompt("Train Station")
        self.assertIn("Category=Train Station", prompt)
        self.assertNotIn("UserId", prompt)
        self.assertNotIn("ground_truth", prompt.lower())

    def test_profile_schema_is_strict(self):
        import json

        parsed = parse_profile_response(
            json.dumps(self.valid_payload())
        )
        self.assertEqual(set(parsed["intents"]), set(INTENT_NAMES))
        invalid = self.valid_payload()
        invalid["confidence"] = 1.5
        with self.assertRaises(ValueError):
            parse_profile_response(json.dumps(invalid))

    def test_cache_key_deduplicates_same_category_name(self):
        first = CategoryRequest(
            "nyc",
            (1,),
            "Gym / Fitness Center",
            10,
            build_prompt("Gym / Fitness Center"),
        )
        second = CategoryRequest(
            "nyc",
            (2,),
            "  gym / FITNESS center ",
            20,
            build_prompt("  gym / FITNESS center "),
        )
        self.assertEqual(first.cache_key, second.cache_key)

    def test_long_summary_is_safely_truncated(self):
        import json

        payload = self.valid_payload()
        payload["summary"] = " ".join(["word"] * 25)
        parsed = parse_profile_response(json.dumps(payload))
        self.assertEqual(len(parsed["summary"].split()), 20)


if __name__ == "__main__":
    unittest.main()
