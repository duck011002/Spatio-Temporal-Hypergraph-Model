"""Create an A5 manifest for a verified, already-exported A4 candidate cache.

This is intentionally separate from ``export_a5_candidates.py``: legacy A4
checkpoints may not contain the newer experiment metadata, while their saved
candidate arrays can still be used without rerunning model inference.
"""
import argparse
from pathlib import Path

from a5_jev.data import Dataset, save, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["ca", "tky", "nyc"])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-note", required=True)
    args = parser.parse_args()

    manifest_path = args.candidates / "candidate_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite {manifest_path}")
    for split in ("validation", "test"):
        path = args.candidates / f"{split}_candidates.npz"
        if not path.is_file():
            raise FileNotFoundError(path)
        # Load through the same checks used by A5 so malformed or misaligned
        # legacy arrays fail before a network request can be started.
        Dataset(args.dataset, args.data, args.candidates, split)

    root = Path(__file__).resolve().parent
    events = args.data / "sample.csv"
    manifest = dict(
        backbone="A4",
        dataset=args.dataset,
        smoke_only=False,
        checkpoint="legacy_cached_a4_candidate_export",
        checkpoint_hash=None,
        config_hash=sha256(args.config),
        candidate_hashes={
            split: sha256(args.candidates / f"{split}_candidates.npz")
            for split in ("validation", "test")
        },
        sample_hash=sha256(events),
        query_hashes={
            "validation": sha256(args.data / "validate_sample.csv"),
            "test": sha256(args.data / "test_sample.csv"),
        },
        seed=args.seed,
        note=args.source_note,
    )
    save(manifest_path, manifest)
    print(f"A5 manifest created: {manifest_path}")


if __name__ == "__main__":
    main()
