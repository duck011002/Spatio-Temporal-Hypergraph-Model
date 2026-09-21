"""Freeze an A5.1 safety-gated parameter choice from cached validation only.

This is an offline preparation step. It never calls Jev and never modifies the
original A5 run directory. The policy chooses the highest selection objective
among grid points that do not reduce calibration Top-1 or MRR; audit remains a
strict final eligibility gate.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from a5_jev.core import GRID, fuse
from a5_jev.data import Dataset, save, sha256
from run_a5_jev import assess, load_json, source_hashes, verify_predictions


def freeze(args: argparse.Namespace) -> None:
    source = args.source_output
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("prepared.json", "validation_predictions.json", "example_request.json"):
        if (source / name).exists():
            shutil.copy2(source / name, output / name)

    prepared = load_json(output / "prepared.json")
    data = Dataset(args.dataset, args.data, args.candidates, "validation")
    if prepared["dataset"] != args.dataset:
        raise ValueError("Prepared dataset mismatch")
    if prepared["candidate_manifest_hash"] != sha256(args.candidates / "candidate_manifest.json"):
        raise ValueError("Candidate manifest changed")
    if prepared["splits"]["validation"] != data.provenance():
        raise ValueError("Validation provenance changed")
    predictions = verify_predictions(data, output / "validation_predictions.json", prepared)

    trials = []
    for weight, temperature in GRID:
        ranked = fuse(
            data.batch.candidate_ids,
            data.batch.model_scores,
            data.categories,
            predictions,
            weight,
            temperature,
        )
        result = assess(data, predictions, weight, temperature)
        selection = result["selection"]
        calibration = result["calibration"]
        audit = result["audit"]
        objective = selection["metrics"]["recall_at_1"] + 0.25 * selection["metrics"]["mrr"]
        trials.append(
            dict(
                weight=weight,
                temperature=temperature,
                objective=objective,
                result=result,
                calibration_ok=(calibration["net_top1"] >= 0 and calibration["delta_mrr"] >= 0),
                audit_ok=(audit["net_top1"] > 0 and audit["delta_mrr"] >= 0),
            )
        )

    safe = [trial for trial in trials if trial["calibration_ok"]]
    if not safe:
        raise RuntimeError("No A5.1 candidate satisfies the calibration safety gate")
    best = max(safe, key=lambda trial: (trial["objective"], -trial["weight"], -trial["temperature"]))
    eligible = bool(best["audit_ok"])
    frozen = dict(
        version="A5-Jev-safe-v1",
        policy="calibration_safe_selection_v1",
        weight=best["weight"],
        temperature=best["temperature"],
        smoothing=0.02,
        eligible_for_test=eligible,
        prepared_hash=sha256(output / "prepared.json"),
        validation_predictions_hash=sha256(output / "validation_predictions.json"),
        source_output=str(source),
        source_hashes=source_hashes(),
    )
    save(output / "frozen.json", frozen)
    save(
        output / "validation_report.json",
        dict(
            version="A5-Jev-safe-v1",
            policy="calibration_safe_selection_v1",
            selected=best["result"],
            selected_weight=best["weight"],
            selected_temperature=best["temperature"],
            eligible_for_test=eligible,
            trials=trials,
            source_output=str(source),
        ),
    )
    print(
        dict(
            output=str(output),
            weight=best["weight"],
            temperature=best["temperature"],
            calibration=best["result"]["calibration"],
            audit=best["result"]["audit"],
            eligible_for_test=eligible,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("ca", "tky", "nyc"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    freeze(parser.parse_args())


if __name__ == "__main__":
    main()
