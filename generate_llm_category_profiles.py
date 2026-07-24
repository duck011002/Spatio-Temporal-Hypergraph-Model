"""Generate cached LLM semantic profiles for POI category names.

The LLM is used as a category-knowledge teacher, not as a next-POI
classifier. Profiles are independent of user labels and contain only
general semantic priors that a small local reranker can consume.
"""

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_rule_teacher_ablation import EXPERIMENTS
from rule_teacher import DatasetRuleStatistics, RankingBatch


PROFILE_VERSION = "category_semantic_profile_v2_deduplicated"
INPUT_PRICE_PER_MILLION = 0.15
OUTPUT_PRICE_PER_MILLION = 1.50
MAX_OUTPUT_TOKENS = 320
MAX_RESERVED_COST_PER_REQUEST = 0.0015
INTENT_NAMES = (
    "home_lodging",
    "work_education",
    "transport",
    "food_drink",
    "shopping_errands",
    "health_care",
    "leisure_culture",
    "nightlife",
    "outdoor_sports",
    "personal_services",
)
TIME_NAMES = (
    "early_morning",
    "morning",
    "midday",
    "afternoon",
    "evening",
    "late_night",
)
ROLE_NAMES = (
    "routine_anchor",
    "transit_connector",
    "purposeful_destination",
    "spontaneous_stop",
)


def read_api_key(path):
    value = Path(path).read_text(encoding="utf-8").strip()
    if not value or any(character.isspace() for character in value):
        raise ValueError("API key file must contain one non-empty token.")
    return value


def category_name_map(statistics):
    if "PoiCategoryName" not in statistics.train_events.columns:
        return {}
    names = (
        statistics.train_events[["PoiCategoryId", "PoiCategoryName"]]
        .dropna()
        .groupby("PoiCategoryId")["PoiCategoryName"]
        .agg(lambda values: str(values.mode().iloc[0]))
    )
    return {
        int(category_id): name
        for category_id, name in names.items()
    }


@dataclass(frozen=True)
class CategoryRequest:
    dataset: str
    category_ids: tuple
    category_name: str
    candidate_frequency: int
    prompt: str

    @property
    def cache_key(self):
        normalized_name = " ".join(
            self.category_name.casefold().split()
        )
        raw = (
            f"{self.dataset}|{normalized_name}|"
            f"{PROFILE_VERSION}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate offline Qwen semantic profiles for POI categories. "
            "No query labels or test examples are sent."
        )
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("nyc", "ca", "tky"),
        default=["nyc"],
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=20,
        help="Profile the most frequent validation-candidate categories; zero means all.",
    )
    parser.add_argument(
        "--include-all-poi-categories",
        action="store_true",
        help=(
            "Include categories attached to every encoded POI, even when a "
            "category never appears in the saved validation candidate lists."
        ),
    )
    parser.add_argument("--model", default="qwen-flash")
    parser.add_argument(
        "--base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    parser.add_argument("--api-key-file", default="key.txt")
    parser.add_argument(
        "--cache-dir",
        default="server_artifacts/llm_category_profiles",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/results/llm_category_profiles",
    )
    parser.add_argument("--max-cost-cny", type=float, default=0.05)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_prompt(category_name):
    intent_example = ",".join(f'"{name}":0.0' for name in INTENT_NAMES)
    time_example = ",".join(f'"{name}":0.0' for name in TIME_NAMES)
    role_example = ",".join(f'"{name}":0.0' for name in ROLE_NAMES)
    return "\n".join(
        [
            "Create a general semantic profile for one Foursquare/Gowalla POI category.",
            "This is world-knowledge annotation, not prediction of a particular user or visit.",
            "All numeric values must be between 0 and 1.",
            "Intent values are independent multi-label affinities.",
            "Time values describe typical visit-time affinity.",
            "Mobility roles describe how the category behaves in an activity chain.",
            "repeat_likelihood estimates repeated visits by the same user.",
            "routine_likelihood estimates whether visits are part of a routine.",
            "distance_tolerance estimates willingness to travel farther for this category.",
            f"Category={category_name}",
            "Return JSON only with exactly this structure:",
            (
                '{"intents":{'
                f"{intent_example}"
                '},"time_affinity":{'
                f"{time_example}"
                '},"mobility_roles":{'
                f"{role_example}"
                '},"repeat_likelihood":0.0,"routine_likelihood":0.0,'
                '"distance_tolerance":0.0,"confidence":0.0,'
                '"summary":"at most 20 English words"}'
            ),
        ]
    )


def _validate_score_map(payload, field, expected_names):
    values = payload.get(field)
    if not isinstance(values, dict) or set(values) != set(expected_names):
        raise ValueError(f"{field} must contain exactly the required keys.")
    result = {}
    for name in expected_names:
        score = float(values[name])
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"{field}.{name} must be between zero and one.")
        result[name] = score
    return result


def parse_profile_response(content):
    content = str(content).strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    payload = json.loads(content)
    result = {
        "intents": _validate_score_map(payload, "intents", INTENT_NAMES),
        "time_affinity": _validate_score_map(
            payload,
            "time_affinity",
            TIME_NAMES,
        ),
        "mobility_roles": _validate_score_map(
            payload,
            "mobility_roles",
            ROLE_NAMES,
        ),
    }
    for field in (
        "repeat_likelihood",
        "routine_likelihood",
        "distance_tolerance",
        "confidence",
    ):
        value = float(payload.get(field, math.nan))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be between zero and one.")
        result[field] = value
    summary = str(payload.get("summary", "")).strip()
    if not summary:
        raise ValueError("summary must be non-empty.")
    result["summary"] = " ".join(summary.split()[:20])
    return result


def category_requests(spec, limit, include_all_poi_categories=False):
    dataset = spec["dataset"]
    candidate_dir = spec["candidate_dir"]
    batch = RankingBatch.load(
        str(candidate_dir / "validation_candidates.npz")
    )
    statistics = DatasetRuleStatistics.from_preprocessed_dir(
        dataset,
        str(Path("data") / dataset / "preprocessed"),
    )
    names = category_name_map(statistics)
    frequencies = {}
    for poi_id in batch.candidate_ids.reshape(-1):
        category_id = int(statistics.poi_category.get(int(poi_id), -1))
        if category_id < 0:
            continue
        frequencies[category_id] = frequencies.get(category_id, 0) + 1
    if include_all_poi_categories:
        sample_path = (
            Path("data") / dataset / "preprocessed" / "sample.csv"
        )
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                category_id = int(row["PoiCategoryId"])
                category_name = str(row["PoiCategoryName"])
                previous_name = names.get(category_id)
                if previous_name is not None and previous_name != category_name:
                    raise ValueError(
                        f"Category ID {category_id} has conflicting names: "
                        f"{previous_name!r} and {category_name!r}."
                    )
                names[category_id] = category_name
                frequencies.setdefault(category_id, 0)
    grouped = {}
    for category_id, frequency in frequencies.items():
        category_name = str(
            names.get(category_id, f"Category {category_id}")
        )
        normalized_name = " ".join(category_name.casefold().split())
        payload = grouped.setdefault(
            normalized_name,
            {
                "category_name": category_name,
                "category_ids": [],
                "frequency": 0,
            },
        )
        payload["category_ids"].append(category_id)
        payload["frequency"] += frequency
    ordered = sorted(
        grouped.values(),
        key=lambda payload: (
            -payload["frequency"],
            payload["category_name"],
            min(payload["category_ids"]),
        ),
    )
    if limit > 0:
        ordered = ordered[:limit]
    return [
        CategoryRequest(
            dataset=dataset,
            category_ids=tuple(sorted(payload["category_ids"])),
            category_name=payload["category_name"],
            candidate_frequency=int(payload["frequency"]),
            prompt=build_prompt(payload["category_name"]),
        )
        for payload in ordered
    ], {
        "dataset": dataset,
        "backbone": spec["backbone"],
        "all_candidate_category_ids": len(frequencies),
        "all_candidate_category_names": len(grouped),
        "include_all_poi_categories": bool(include_all_poi_categories),
        "selected_category_names": len(ordered),
        "candidate_occurrences_covered": int(
            sum(payload["frequency"] for payload in ordered)
        ),
        "candidate_occurrences_total": int(sum(frequencies.values())),
    }


def response_cost(usage):
    input_tokens = int(
        usage.get("prompt_tokens", usage.get("input_tokens", 0))
    )
    output_tokens = int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    return (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000.0


def estimated_tokens(requests):
    input_tokens = int(
        sum(math.ceil(len(request.prompt) / 4.0) for request in requests)
    )
    output_tokens = len(requests) * MAX_OUTPUT_TOKENS
    cost = (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000.0
    return input_tokens, output_tokens, cost


def request_profile(request, args, api_key):
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a conservative POI category ontology "
                    "annotator. Return only valid JSON."
                ),
            },
            {"role": "user", "content": request.prompt},
        ],
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
    }
    http_request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
        },
    )
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            with urllib.request.urlopen(
                http_request,
                timeout=args.timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
            content = result["choices"][0]["message"]["content"]
            profile = parse_profile_response(content)
            usage = result.get("usage", {})
            return {
                "profile": profile,
                "content": content,
                "usage": usage,
                "request_id": result.get("id"),
                "cost_cny": response_cost(usage),
            }
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = error
            if attempt >= args.max_retries:
                break
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(
        f"Profile request failed for {request.category_name} after "
        f"{args.max_retries + 1} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def cache_path(cache_dir, model, request):
    safe_model = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in model
    )
    return (
        cache_dir
        / "profile_cache"
        / request.dataset
        / PROFILE_VERSION
        / safe_model
        / f"{request.cache_key}.json"
    )


def load_cached(path, request):
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    prompt_hash = hashlib.sha256(
        request.prompt.encode("utf-8")
    ).hexdigest()
    if payload.get("prompt_hash") != prompt_hash:
        return None
    parse_profile_response(
        json.dumps(payload["profile"], ensure_ascii=False)
    )
    return payload


def save_cached(path, request, model, response):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile_version": PROFILE_VERSION,
        "dataset": request.dataset,
        "category_ids": list(request.category_ids),
        "category_name": request.category_name,
        "candidate_frequency": request.candidate_frequency,
        "model": model,
        "prompt_hash": hashlib.sha256(
            request.prompt.encode("utf-8")
        ).hexdigest(),
        "profile": response["profile"],
        "content": response["content"],
        "usage": response.get("usage", {}),
        "request_id": response.get("request_id"),
        "cost_cny": float(response.get("cost_cny", 0.0)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main():
    args = parse_args()
    if args.limit_per_dataset < 0 or args.max_cost_cny <= 0:
        raise SystemExit(
            "--limit-per-dataset must be non-negative and budget positive."
        )
    selected = set(args.datasets)
    requests = []
    metadata = []
    for spec in EXPERIMENTS:
        if spec["dataset"] not in selected:
            continue
        dataset_requests, dataset_metadata = category_requests(
            spec,
            args.limit_per_dataset,
            include_all_poi_categories=args.include_all_poi_categories,
        )
        requests.extend(dataset_requests)
        metadata.append(dataset_metadata)

    cache_dir = Path(args.cache_dir)
    cached_by_key = {}
    missing = []
    for request in requests:
        path = cache_path(cache_dir, args.model, request)
        cached = load_cached(path, request)
        if cached is None:
            missing.append((request, path))
        else:
            cached_by_key[request.cache_key] = cached
    input_tokens, output_tokens, estimated_cost = estimated_tokens(
        [request for request, _ in missing]
    )
    preflight = {
        "profile_version": PROFILE_VERSION,
        "datasets": metadata,
        "total_selected_categories": len(requests),
        "new_requests": len(missing),
        "cached_requests": len(requests) - len(missing),
        "estimated_input_tokens": input_tokens,
        "estimated_output_token_cap": output_tokens,
        "estimated_cost_cny": estimated_cost,
        "reserved_cost_cny": (
            len(missing) * MAX_RESERVED_COST_PER_REQUEST
        ),
        "hard_budget_cny": args.max_cost_cny,
        "dry_run": args.dry_run,
    }
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    if preflight["reserved_cost_cny"] > args.max_cost_cny:
        raise RuntimeError(
            "Hard budget reserve exceeds --max-cost-cny before requests."
        )

    api_key = read_api_key(args.api_key_file)
    new_cost = 0.0
    for index, (request, path) in enumerate(missing, start=1):
        if (
            new_cost + MAX_RESERVED_COST_PER_REQUEST
            > args.max_cost_cny
        ):
            raise RuntimeError(
                "Hard budget reserve reached before the next request."
            )
        response = request_profile(request, args, api_key)
        cached = save_cached(path, request, args.model, response)
        cached_by_key[request.cache_key] = cached
        new_cost += float(cached["cost_cny"])
        print(
            f"profile_progress={index}/{len(missing)} "
            f"new_cost_cny={new_cost:.6f}",
            flush=True,
        )

    profiles = []
    for request in requests:
        cached = cached_by_key.get(request.cache_key)
        if cached is None:
            raise RuntimeError(
                f"Missing cached profile for {request.category_name}."
            )
        profiles.append(
            {
                "dataset": request.dataset,
                "category_ids": list(request.category_ids),
                "category_name": request.category_name,
                "candidate_frequency": request.candidate_frequency,
                "profile": cached["profile"],
                "cost_cny": float(cached["cost_cny"]),
                "cache_file": str(
                    cache_path(cache_dir, args.model, request)
                ),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{PROFILE_VERSION}_profiles.json"
    output_path.write_text(
        json.dumps(
            {
                "protocol": {
                    "profile_version": PROFILE_VERSION,
                    "model": args.model,
                    "labels_in_prompt": False,
                    "user_or_poi_ids_in_prompt": False,
                    "test_examples_in_prompt": False,
                    "temperature": 0,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                "request_summary": {
                    **preflight,
                    "new_cost_cny": new_cost,
                },
                "profiles": profiles,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
