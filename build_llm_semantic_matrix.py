"""Build a POI-aligned matrix from cached LLM category profiles.

The output contains no user histories or labels. It only joins each encoded
POI to the training-only modal category used by the frozen rule teacher.
Unseen POIs, the padding/unknown sentinel, and STHGCN's legacy extra output row
remain unavailable all-zero vectors.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from generate_llm_category_profiles import (
    INTENT_NAMES,
    PROFILE_VERSION,
    ROLE_NAMES,
    TIME_NAMES,
    parse_profile_response,
)


SCALAR_FEATURE_NAMES = (
    "repeat_likelihood",
    "routine_likelihood",
    "distance_tolerance",
    "confidence",
)
FEATURE_NAMES = (
    tuple(f"intent.{name}" for name in INTENT_NAMES)
    + tuple(f"time.{name}" for name in TIME_NAMES)
    + tuple(f"role.{name}" for name in ROLE_NAMES)
    + SCALAR_FEATURE_NAMES
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Join LLM category profiles to encoded POI IDs."
    )
    parser.add_argument("--dataset", default="nyc")
    parser.add_argument(
        "--profiles",
        default=(
            "docs/results/llm_category_profiles/"
            f"{PROFILE_VERSION}_profiles.json"
        ),
    )
    parser.add_argument(
        "--sample-csv",
        default=None,
        help="Defaults to data/<dataset>/preprocessed/sample.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Defaults to server_artifacts/llm_category_profiles/"
            "<dataset>_semantic_profile_matrix_v2.npz."
        ),
    )
    return parser.parse_args()


def normalize_category_name(value):
    return " ".join(str(value).casefold().split())


def profile_vector(profile):
    validated = parse_profile_response(
        json.dumps(profile, ensure_ascii=False)
    )
    values = [validated["intents"][name] for name in INTENT_NAMES]
    values.extend(validated["time_affinity"][name] for name in TIME_NAMES)
    values.extend(validated["mobility_roles"][name] for name in ROLE_NAMES)
    values.extend(validated[name] for name in SCALAR_FEATURE_NAMES)
    vector = np.asarray(values, dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),):
        raise ValueError("Unexpected semantic profile width.")
    return vector


def load_profiles(path, dataset):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocol = payload.get("protocol", {})
    version = protocol.get("profile_version")
    if version != PROFILE_VERSION:
        raise ValueError(
            f"Expected profile version {PROFILE_VERSION!r}, got {version!r}."
        )
    profiles = {}
    display_names = {}
    for row in payload.get("profiles", []):
        if str(row.get("dataset")) != dataset:
            continue
        name = str(row["category_name"])
        normalized = normalize_category_name(name)
        vector = profile_vector(row["profile"])
        previous = profiles.get(normalized)
        if previous is not None and not np.array_equal(previous, vector):
            raise ValueError(
                f"Conflicting profiles for normalized category {normalized!r}."
            )
        profiles[normalized] = vector
        display_names[normalized] = name
    if not profiles:
        raise ValueError(f"No profiles found for dataset {dataset!r}.")
    return profiles, display_names


def build_semantic_matrix(sample_csv, profiles_path, dataset):
    profiles, _ = load_profiles(profiles_path, dataset)
    frame = pd.read_csv(
        sample_csv,
        usecols=["PoiId", "PoiCategoryId", "PoiCategoryName", "SplitTag"],
    )
    if frame.empty:
        raise ValueError("The preprocessed sample file has no POIs.")

    frame["PoiId"] = pd.to_numeric(frame["PoiId"], errors="raise").astype(int)
    frame["PoiCategoryId"] = pd.to_numeric(
        frame["PoiCategoryId"],
        errors="raise",
    ).astype(int)
    sentinel_poi_id = int(frame["PoiId"].max())
    extra_output_id = sentinel_poi_id + 1
    observed_ids = set(int(value) for value in frame["PoiId"].unique())
    expected_ids = set(range(sentinel_poi_id + 1))
    missing_encoded_ids = sorted(expected_ids - observed_ids)
    if missing_encoded_ids:
        preview = ", ".join(str(value) for value in missing_encoded_ids[:20])
        raise ValueError(f"Encoded POI IDs are not contiguous; missing: {preview}")

    split = frame["SplitTag"].astype(str).str.casefold()
    train_frame = frame.loc[split == "train"].copy()
    if train_frame.empty:
        raise ValueError("No training events are available for POI metadata.")
    if bool((train_frame["PoiId"] >= sentinel_poi_id).any()):
        raise ValueError("Unknown or padding POI IDs unexpectedly appear in train.")

    modal_categories = train_frame.groupby("PoiId")["PoiCategoryId"].agg(
        lambda values: int(values.mode().iloc[0])
    )

    poi_rows = {}
    missing = set()
    for poi_id, category_id in modal_categories.items():
        matching = train_frame.loc[
            (train_frame["PoiId"] == int(poi_id))
            & (train_frame["PoiCategoryId"] == int(category_id)),
            "PoiCategoryName",
        ].astype(str)
        category_name = str(matching.mode().iloc[0])
        normalized = normalize_category_name(category_name)
        vector = profiles.get(normalized)
        if vector is None:
            missing.add(category_name)
            continue
        poi_rows[int(poi_id)] = (int(category_id), normalized, vector)

    if missing:
        formatted = ", ".join(sorted(missing))
        raise ValueError(
            f"Missing LLM profiles for {len(missing)} category names: "
            f"{formatted}"
        )

    # The label encoder reserves the maximum observed sample ID as its
    # padding/unknown sentinel. Legacy STHGCN emits num_poi + 1 logits, adding
    # one extra output row after that sentinel.
    matrix = np.zeros(
        (extra_output_id + 1, len(FEATURE_NAMES)),
        dtype=np.float32,
    )
    available = np.zeros(extra_output_id + 1, dtype=np.bool_)
    category_ids = np.full(extra_output_id + 1, -1, dtype=np.int64)
    for poi_id, (category_id, _normalized, vector) in poi_rows.items():
        matrix[poi_id] = vector
        available[poi_id] = True
        category_ids[poi_id] = category_id

    if bool(available[sentinel_poi_id]):
        raise ValueError("The padding/unknown POI row must remain unavailable.")
    if bool(available[extra_output_id]):
        raise ValueError("The legacy extra output row must remain unavailable.")
    return matrix, available, category_ids


def save_semantic_matrix(output_path, matrix, available, category_ids, dataset):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        poi_profiles=matrix,
        profile_available=available,
        poi_category_ids=category_ids,
        feature_names=np.asarray(FEATURE_NAMES, dtype=np.str_),
        profile_version=np.asarray(PROFILE_VERSION, dtype=np.str_),
        dataset=np.asarray(dataset, dtype=np.str_),
    )
    return output_path


def main():
    args = parse_args()
    sample_csv = args.sample_csv or (
        Path("data") / args.dataset / "preprocessed" / "sample.csv"
    )
    output = args.output or (
        Path("server_artifacts")
        / "llm_category_profiles"
        / f"{args.dataset}_semantic_profile_matrix_v2.npz"
    )
    matrix, available, category_ids = build_semantic_matrix(
        sample_csv=sample_csv,
        profiles_path=args.profiles,
        dataset=args.dataset,
    )
    path = save_semantic_matrix(
        output,
        matrix,
        available,
        category_ids,
        args.dataset,
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "shape": list(matrix.shape),
                "profile_width": len(FEATURE_NAMES),
                "known_poi_coverage": float(available[:-2].mean()),
                "sentinel_row_available": bool(available[-2]),
                "extra_output_row_available": bool(available[-1]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
