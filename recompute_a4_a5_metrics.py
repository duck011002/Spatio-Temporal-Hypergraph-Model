"""Recompute all available A4/A5 metrics from frozen local artifacts.

This script is offline-only. It never creates a Jev client and never trains a
model. It recomputes ranking metrics, paired changes, user-cluster bootstrap
intervals, and the TKY validation grid diagnostic.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np

from a5_jev.core import GRID, fuse
from a5_jev.data import Dataset
from rule_teacher.core import evaluate_ranking


def _install_scipy_compat() -> None:
    """Allow historical replay in the lean local environment.

    The historical source imports only scipy.stats.binomtest. Current A4/A5
    recomputation does not require scipy, so keep this compatibility shim local
    to the offline replay instead of changing the experiment environment.
    """
    try:
        from scipy.stats import binomtest  # noqa: F401
        scipy_available = True
    except ImportError:
        scipy_available = False

    if not scipy_available:
        stats = types.ModuleType("scipy.stats")

        def binomtest(k: int, n: int):
            if n <= 0:
                value = 1.0
            else:
                observed = math.comb(n, k) * 0.5**n
                value = sum(
                    math.comb(n, i) * 0.5**n
                    for i in range(n + 1)
                    if math.comb(n, i) * 0.5**n <= observed + 1e-15
                )
            return types.SimpleNamespace(pvalue=min(float(value), 1.0))

        stats.binomtest = binomtest
        scipy = types.ModuleType("scipy")
        scipy.stats = stats
        sys.modules["scipy"] = scipy
        sys.modules["scipy.stats"] = stats

    try:
        import sklearn  # noqa: F401
    except ImportError:
        linear = types.ModuleType("sklearn.linear_model")
        pipeline = types.ModuleType("sklearn.pipeline")
        preprocessing = types.ModuleType("sklearn.preprocessing")

        class LogisticRegression:  # pragma: no cover - import-only shim
            pass

        class StandardScaler:  # pragma: no cover - import-only shim
            pass

        linear.LogisticRegression = LogisticRegression
        preprocessing.StandardScaler = StandardScaler
        pipeline.make_pipeline = lambda *items: items
        sklearn = types.ModuleType("sklearn")
        sklearn.linear_model = linear
        sklearn.pipeline = pipeline
        sklearn.preprocessing = preprocessing
        sys.modules["sklearn"] = sklearn
        sys.modules["sklearn.linear_model"] = linear
        sys.modules["sklearn.pipeline"] = pipeline
        sys.modules["sklearn.preprocessing"] = preprocessing


ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / "server_artifacts" / "a5_review_20260921"
OUT = ROOT / "docs" / "results" / "a4_a5_recomputed_20260921"
BOOTSTRAP_SEED = 20260920
BOOTSTRAP_REPEATS = 2000


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rank_values(ranked: np.ndarray, labels: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    matches = np.asarray(ranked) == np.asarray(labels)[:, None]
    found = matches.any(axis=1)
    positions = np.argmax(matches, axis=1) + 1
    return np.where(found, positions, np.asarray(fallback, dtype=np.int64))


def bootstrap_delta(ranks: np.ndarray, base_ranks: np.ndarray, users: np.ndarray) -> dict:
    d1 = (ranks == 1).astype(float) - (base_ranks == 1).astype(float)
    dm = 1.0 / ranks.astype(float) - 1.0 / base_ranks.astype(float)
    unique, inverse = np.unique(users, return_inverse=True)
    counts = np.bincount(inverse)
    sums1 = np.bincount(inverse, weights=d1)
    sumsm = np.bincount(inverse, weights=dm)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values1 = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    valuesm = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for index in range(BOOTSTRAP_REPEATS):
        draws = rng.integers(0, len(unique), len(unique))
        denominator = counts[draws].sum()
        values1[index] = sums1[draws].sum() / denominator
        valuesm[index] = sumsm[draws].sum() / denominator
    rescued = int((d1 > 0).sum())
    lost = int((d1 < 0).sum())
    result = {
        "net_top1": rescued - lost,
        "rescued": rescued,
        "lost": lost,
        "delta_r1": float(d1.mean()),
        "delta_mrr": float(dm.mean()),
        "user_cluster_ci_r1": np.quantile(values1, [0.025, 0.975]).tolist(),
        "user_cluster_ci_mrr": np.quantile(valuesm, [0.025, 0.975]).tolist(),
        "n_users": int(len(unique)),
        "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    try:
        from scipy.stats import binomtest

        result["mcnemar_exact_p_descriptive"] = float(
            binomtest(rescued, rescued + lost).pvalue if rescued + lost else 1.0
        )
    except ImportError:
        result["mcnemar_exact_p_descriptive"] = None
    return result


def pair_summary(
    ranked: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    fallback: np.ndarray,
    users: np.ndarray,
) -> dict:
    after = evaluate_ranking(ranked, labels, fallback)
    before = evaluate_ranking(base, labels, fallback)
    ranks = rank_values(ranked, labels, fallback)
    base_ranks = rank_values(base, labels, fallback)
    delta = {key: float(after[key] - before[key]) for key in after}
    return {
        "n": int(len(labels)),
        "metrics": after,
        "backbone": before,
        "delta": delta,
        "paired": bootstrap_delta(ranks, base_ranks, users),
    }


def current_predictions(path: Path) -> list[dict]:
    payload = load_json(path)
    if isinstance(payload, dict) and "predictions" in payload:
        return payload["predictions"]
    raise ValueError(f"Unexpected prediction payload: {path}")


def formal_pair(
    dataset: str,
    split: str,
    weight: float,
    temperature: float,
    candidate_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    candidate_dir = candidate_dir or REVIEW / "artifacts" / f"{dataset}_a4_candidates"
    output_dir = output_dir or REVIEW / "artifacts" / f"{dataset}_a5_jev"
    data = Dataset(dataset, ROOT / "data" / dataset / "preprocessed", candidate_dir, split)
    predictions = current_predictions(output_dir / f"{split}_predictions.json")
    ranked = fuse(
        data.batch.candidate_ids,
        data.batch.model_scores,
        data.categories,
        predictions,
        weight,
        temperature,
    )
    return pair_summary(
        ranked,
        data.batch.candidate_ids,
        data.batch.labels,
        data.batch.base_label_ranks,
        data.queries.UserId.to_numpy(),
    )


def tky_grid() -> dict:
    candidate_dir = REVIEW / "artifacts" / "tky_a4_candidates"
    output_dir = REVIEW / "artifacts" / "tky_a5_jev"
    data = Dataset("tky", ROOT / "data" / "tky" / "preprocessed", candidate_dir, "validation")
    predictions = current_predictions(output_dir / "validation_predictions.json")
    records = []
    for weight, temperature in GRID:
        ranked = fuse(
            data.batch.candidate_ids,
            data.batch.model_scores,
            data.categories,
            predictions,
            weight,
            temperature,
        )
        all_pair = pair_summary(
            ranked,
            data.batch.candidate_ids,
            data.batch.labels,
            data.batch.base_label_ranks,
            data.queries.UserId.to_numpy(),
        )
        partitions = {}
        for part in ("selection", "calibration", "audit"):
            mask = data.partitions == part
            partitions[part] = pair_summary(
                ranked[mask],
                data.batch.candidate_ids[mask],
                data.batch.labels[mask],
                data.batch.base_label_ranks[mask],
                data.queries.UserId.to_numpy()[mask],
            )
        selection_metrics = partitions["selection"]["metrics"]
        selection_objective = selection_metrics["recall_at_1"] + 0.25 * selection_metrics["mrr"]
        calibration = partitions["calibration"]["paired"]
        audit = partitions["audit"]["paired"]
        records.append(
            {
                "weight": float(weight),
                "temperature": float(temperature),
                "selection_objective": float(selection_objective),
                "selection": partitions["selection"],
                "calibration": partitions["calibration"],
                "audit": partitions["audit"],
                "all": all_pair,
                "calibration_ok": bool(calibration["net_top1"] >= 0 and calibration["delta_mrr"] >= 0),
                "audit_ok": bool(audit["net_top1"] > 0 and audit["delta_mrr"] >= 0),
            }
        )
    safe = [item for item in records if item["calibration_ok"]]
    best_safe = max(safe, key=lambda item: (item["selection_objective"], -item["weight"], -item["temperature"]))
    current = next(item for item in records if item["weight"] == 0.2 and item["temperature"] == 1.0)
    return {
        "current_frozen": current,
        "best_calibration_safe_candidate": best_safe,
        "records": records,
    }


def historical_nyc() -> dict:
    _install_scipy_compat()
    source = ROOT / "server_artifacts" / "jev_category_nyc_20260920" / "source" / "experiment.py"
    spec = importlib.util.spec_from_file_location("historical_jev_category", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import historical source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = module.Data()
    loaded = data.load("test")
    predictions = load_json(ROOT / "server_artifacts" / "jev_category_nyc_20260920" / "test_predictions.json")
    distribution = module.poi_distribution(data, loaded, predictions)
    report = module.base.assess(loaded, distribution, 0.35)
    return {
        "protocol": "historical_category_fusion",
        "alpha": 0.35,
        "recomputed": report["all"],
    }


def main() -> None:
    _install_scipy_compat()
    OUT.mkdir(parents=True, exist_ok=True)
    ca_validation = formal_pair("ca", "validation", 0.35, 2.0)
    ca_test = formal_pair("ca", "test", 0.35, 2.0)
    tky_validation = formal_pair("tky", "validation", 0.2, 1.0)
    final_remote = REVIEW / "final_remote" / "artifacts"
    tky_safe_frozen = load_json(final_remote / "tky_a5_jev_v1" / "frozen.json")
    nyc_frozen = load_json(final_remote / "nyc_a5_jev_formal" / "frozen.json")
    nyc_force_frozen = load_json(final_remote / "nyc_a5_jev_force_test" / "frozen.json")
    tky_safe_test = formal_pair(
        "tky",
        "test",
        float(tky_safe_frozen["weight"]),
        float(tky_safe_frozen["temperature"]),
        candidate_dir=REVIEW / "artifacts" / "tky_a4_candidates",
        output_dir=final_remote / "tky_a5_jev_v1",
    )
    nyc_validation = formal_pair(
        "nyc",
        "validation",
        float(nyc_frozen["weight"]),
        float(nyc_frozen["temperature"]),
        candidate_dir=final_remote / "nyc_a4_candidates",
        output_dir=final_remote / "nyc_a5_jev_formal",
    )
    nyc_force_test = formal_pair(
        "nyc",
        "test",
        float(nyc_force_frozen["weight"]),
        float(nyc_force_frozen["temperature"]),
        candidate_dir=final_remote / "nyc_a4_candidates",
        output_dir=final_remote / "nyc_a5_jev_force_test",
    )
    summary = {
        "version": "offline-a4-a5-recompute-v1",
        "offline_only": True,
        "sources": {
            "review_bundle": str(REVIEW.relative_to(ROOT)),
            "ca_candidates": str((REVIEW / "artifacts" / "ca_a4_candidates").relative_to(ROOT)),
            "tky_candidates": str((REVIEW / "artifacts" / "tky_a4_candidates").relative_to(ROOT)),
        },
        "ca": {"validation": ca_validation, "test": ca_test},
        "tky": {"validation": tky_validation, "grid": tky_grid(), "a5_1_test": tky_safe_test},
        "nyc": historical_nyc(),
        "final_remote": {
            "tky_a5_1": {"frozen": tky_safe_frozen, "test": tky_safe_test},
            "nyc_formal": {
                "frozen": nyc_frozen,
                "validation": nyc_validation,
                "test_override": {"frozen": nyc_force_frozen, "test": nyc_force_test},
            },
        },
        "gpu_pending": [
            "NYC test was executed only as an explicitly labeled override after the formal validation safety gate rejected the selected fusion.",
        ],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "output": str((OUT / "summary.json").relative_to(ROOT)),
        "ca_test": ca_test["paired"],
        "tky_current_validation": summary["tky"]["grid"]["current_frozen"]["all"]["paired"],
        "tky_safe_candidate": {
            "weight": summary["tky"]["grid"]["best_calibration_safe_candidate"]["weight"],
            "temperature": summary["tky"]["grid"]["best_calibration_safe_candidate"]["temperature"],
            "calibration": summary["tky"]["grid"]["best_calibration_safe_candidate"]["calibration"]["paired"],
            "audit": summary["tky"]["grid"]["best_calibration_safe_candidate"]["audit"]["paired"],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
