"""Benchmark the sampler's full and rectangular local Jaccard calculations.

This is a read-only smoke test.  It uses the already preprocessed TKY
sample.csv, creates a realistic target-plus-neighbor trajectory batch, and
checks that the rectangular computation is numerically equivalent to the
slice consumed by NeighborSampler.filter_traj2traj_with_leakage.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def build_membership(path: Path) -> tuple[csr_matrix, int, int]:
    frame = pd.read_csv(
        path,
        usecols=["PoiId", "pseudo_session_trajectory_id"],
    )
    poi_codes, _ = pd.factorize(frame["PoiId"], sort=False)
    traj_codes, _ = pd.factorize(
        frame["pseudo_session_trajectory_id"],
        sort=False,
    )
    membership = coo_matrix(
        (
            np.ones(frame.shape[0], dtype=np.float64),
            (poi_codes, traj_codes),
        ),
        shape=(int(poi_codes.max()) + 1, int(traj_codes.max()) + 1),
    ).tocsr()
    return membership, membership.shape[0], membership.shape[1]


def choose_local_trajectories(
    membership: csr_matrix,
    batch_size: int,
    fanout: int,
    seed: int,
) -> np.ndarray:
    """Approximate one sampled traj2traj hop using real POI overlaps."""
    rng = np.random.default_rng(seed)
    target = rng.choice(membership.shape[1], size=batch_size, replace=False)
    target_overlap = (membership[:, target].T @ membership).tocsr()
    neighbor_parts = []
    for row_index, traj_id in enumerate(target):
        candidates = target_overlap.indices[
            target_overlap.indptr[row_index]:target_overlap.indptr[row_index + 1]
        ]
        candidates = candidates[candidates != traj_id]
        if candidates.size > fanout:
            candidates = rng.choice(candidates, size=fanout, replace=False)
        neighbor_parts.append(candidates)

    neighbors = np.unique(np.concatenate(neighbor_parts))
    neighbors = neighbors[~np.isin(neighbors, target)]
    return np.concatenate([target, neighbors])


def original_full_then_slice(
    membership: csr_matrix,
    batch_size: int,
) -> tuple[csr_matrix, np.ndarray, int]:
    """The current implementation: calculate all local trajectory pairs."""
    gram = (membership.T @ membership).tocoo()
    diagonal = gram.diagonal()
    gram.data = gram.data / (
        diagonal[gram.col] + diagonal[gram.row] - gram.data
    )
    result = gram.tocsc()[:batch_size, :].tocsr()
    return result, diagonal, gram.nnz


def rectangular_only(
    membership: csr_matrix,
    batch_size: int,
) -> tuple[csr_matrix, np.ndarray, int]:
    """Equivalent result for the only rectangle consumed by the sampler."""
    diagonal = np.asarray(membership.multiply(membership).sum(axis=0)).ravel()
    rectangle = (membership[:, :batch_size].T @ membership).tocoo()
    rectangle.data = rectangle.data / (
        diagonal[rectangle.col] + diagonal[rectangle.row] - rectangle.data
    )
    return rectangle.tocsr(), diagonal, rectangle.nnz


def timed(callable_object):
    start = time.perf_counter()
    result = callable_object()
    return result, time.perf_counter() - start


def same_sparse_values(left: csr_matrix, right: csr_matrix) -> tuple[bool, float]:
    difference = (left - right).tocoo()
    max_abs_difference = (
        0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data)))
    )
    return left.shape == right.shape and max_abs_difference == 0.0, max_abs_difference


def threshold_decisions_match(
    left: csr_matrix,
    right: csr_matrix,
    threshold: float,
) -> bool:
    left = left.tocsr()
    right = right.tocsr()
    if not (
        np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
    ):
        return False
    epsilon = 1e-6
    left_keep = (1.0 + epsilon - left.data) <= (1.0 - threshold + epsilon)
    right_keep = (1.0 + epsilon - right.data) <= (1.0 - threshold + epsilon)
    return bool(np.array_equal(left_keep, right_keep))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path("data/tky/preprocessed/sample.csv"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--fanout", type=int, default=400)
    parser.add_argument("--seed", type=int, default=54607333)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not args.sample_file.is_file():
        raise FileNotFoundError(args.sample_file)
    if args.batch_size <= 0 or args.fanout <= 0 or args.repeats <= 0:
        raise ValueError("batch-size, fanout, and repeats must be positive")

    membership, poi_count, trajectory_count = build_membership(args.sample_file)
    selected = choose_local_trajectories(
        membership,
        args.batch_size,
        args.fanout,
        args.seed,
    )
    local_membership = membership[:, selected].tocsr()
    local_nnz = int(local_membership.nnz)
    maximum_checkins = int(np.max(np.asarray(local_membership.sum(axis=0))))

    original_times = []
    rectangular_times = []
    reference = None
    candidate = None
    full_nnz = None
    rectangle_nnz = None
    original_diagonal = None
    rectangular_diagonal = None

    for _ in range(args.repeats):
        (reference, original_diagonal, full_nnz), elapsed = timed(
            lambda: original_full_then_slice(local_membership, args.batch_size)
        )
        original_times.append(elapsed)
        gc.collect()
        (candidate, rectangular_diagonal, rectangle_nnz), elapsed = timed(
            lambda: rectangular_only(local_membership, args.batch_size)
        )
        rectangular_times.append(elapsed)
        gc.collect()

    assert reference is not None
    assert candidate is not None
    exact_values, max_abs_difference = same_sparse_values(reference, candidate)
    exact_diagonal = bool(np.array_equal(original_diagonal, rectangular_diagonal))
    same_threshold_decisions = threshold_decisions_match(
        reference,
        candidate,
        args.threshold,
    )
    original_median = float(np.median(original_times))
    rectangular_median = float(np.median(rectangular_times))

    report = {
        "cpu_count": os.cpu_count(),
        "source": str(args.sample_file),
        "source_pois": poi_count,
        "source_trajectories": trajectory_count,
        "batch_size": args.batch_size,
        "fanout": args.fanout,
        "local_trajectories": int(local_membership.shape[1]),
        "local_checkin_nnz": local_nnz,
        "max_checkins_per_local_trajectory": maximum_checkins,
        "full_gram_nnz": int(full_nnz),
        "rectangle_nnz": int(rectangle_nnz),
        "full_seconds": original_times,
        "rectangle_seconds": rectangular_times,
        "full_median_seconds": original_median,
        "rectangle_median_seconds": rectangular_median,
        "speedup": original_median / rectangular_median,
        "exact_jaccard_values": exact_values,
        "max_abs_jaccard_difference": max_abs_difference,
        "exact_diagonal": exact_diagonal,
        "identical_threshold_decisions": same_threshold_decisions,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not (exact_values and exact_diagonal and same_threshold_decisions):
        raise SystemExit("equivalence check failed")


if __name__ == "__main__":
    main()
