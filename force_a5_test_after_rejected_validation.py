"""Create an explicitly overridden A5 test directory after validation rejection.

This does not alter the original frozen run.  It is only for a user-authorized
diagnostic test whose validation safety gate was rejected; the override is
recorded in the new frozen file and must be reported separately from formal
results.
"""
import argparse
import json
import shutil
from pathlib import Path

from a5_jev.data import save, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--acknowledge-validation-rejection", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_validation_rejection:
        parser.error("explicitly acknowledge the rejected validation gate")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")

    source_frozen_path = args.source_output / "frozen.json"
    source_frozen = json.loads(source_frozen_path.read_text(encoding="utf-8"))
    if source_frozen.get("eligible_for_test"):
        raise ValueError("Source run already passed validation; use the normal test stage")
    for name in ("prepared.json", "example_request.json", "validation_predictions.json"):
        if not (args.source_output / name).is_file():
            raise FileNotFoundError(args.source_output / name)

    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("prepared.json", "example_request.json", "validation_predictions.json"):
        shutil.copy2(args.source_output / name, args.output / name)
    forced = dict(source_frozen)
    forced.update(
        version="A5-Jev-explicit-test-override-v1",
        eligible_for_test=True,
        validation_gate_overridden=True,
        override_reason=args.reason,
        source_frozen_hash=sha256(source_frozen_path),
    )
    save(args.output / "frozen.json", forced)
    print(f"Created explicit override test directory: {args.output}")


if __name__ == "__main__":
    main()
