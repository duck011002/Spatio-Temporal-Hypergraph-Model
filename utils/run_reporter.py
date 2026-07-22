"""Structured, per-run experiment reporting without changing training semantics."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from tqdm import tqdm


def _json_value(value: Any) -> Any:
    """Convert common runtime values into JSON-compatible primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _git_metadata() -> dict[str, Any]:
    """Record the source revision when the project is run from a Git checkout."""
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None}


class _TqdmConsoleHandler(logging.Handler):
    """Keep log lines readable while tqdm owns the terminal line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


class _ConsoleNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not message.startswith("[Training] Parameter ")


def configure_console_logging(level: int = logging.INFO) -> None:
    """Add one concise console handler while retaining the existing file logger."""
    root = logging.getLogger()
    if any(getattr(handler, "_sthgcn_console", False) for handler in root.handlers):
        return

    handler = _TqdmConsoleHandler(level=level)
    handler._sthgcn_console = True
    handler.addFilter(_ConsoleNoiseFilter())
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
    root.addHandler(handler)


class RunReporter:
    """Write a portable, structured record for one training invocation."""

    EPOCH_FIELDS = (
        "epoch",
        "global_step",
        "train_loss",
        "learning_rate",
        "duration_s",
        "validation_step",
        "validation_loss",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "mrr",
    )

    def __init__(self, run_dir: Path, metadata: Mapping[str, Any]) -> None:
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.epoch_metrics_path = run_dir / "epoch_metrics.csv"
        self._metadata = dict(metadata)

    @classmethod
    def create(
        cls,
        *,
        root: str | Path,
        dataset: str,
        model: str,
        seed: int,
        hparams: Mapping[str, Any],
        log_path: str,
        tensorboard_path: str,
    ) -> "RunReporter":
        now = dt.datetime.now()
        base_run_id = f"{now:%Y%m%d_%H%M%S}_{dataset}_{model}_seed{seed}"
        parent = Path(root)
        run_dir = parent / base_run_id
        suffix = 1
        while run_dir.exists():
            run_dir = parent / f"{base_run_id}_{suffix:02d}"
            suffix += 1
        run_dir.mkdir(parents=True, exist_ok=False)

        metadata = {
            "run_id": run_dir.name,
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "started_at": now.isoformat(timespec="seconds"),
            "log_path": log_path,
            "tensorboard_path": tensorboard_path,
        }
        reporter = cls(run_dir, metadata)
        reporter._write_json("run.json", metadata)
        reporter._write_json("config.json", hparams)
        reporter._write_json(
            "environment.json",
            {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": sys.version,
                "git": _git_metadata(),
            },
        )
        reporter.event("run_started", **metadata)
        return reporter

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        with (self.run_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(_json_value(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    def event(self, event: str, **payload: Any) -> None:
        row = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "run_id": self._metadata["run_id"],
            "event": event,
            **_json_value(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def stage_started(self, stage: str, **payload: Any) -> None:
        self.event("stage_started", stage=stage, **payload)

    def stage_finished(self, stage: str, duration_s: float, **payload: Any) -> None:
        self.event("stage_finished", stage=stage, duration_s=round(duration_s, 3), **payload)

    def validation(self, *, epoch: int, global_step: int, metrics: Mapping[str, Any]) -> None:
        self.event(
            "validation",
            epoch=epoch,
            global_step=global_step,
            metrics=metrics,
        )

    def epoch(self, *, epoch: int, global_step: int, train_loss: float,
              learning_rate: float, duration_s: float,
              validation: Mapping[str, Any] | None) -> None:
        metrics = validation or {}
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "learning_rate": learning_rate,
            "duration_s": duration_s,
            "validation_step": metrics.get("global_step", ""),
            "validation_loss": metrics.get("loss", ""),
            "recall_at_1": metrics.get("recall_at_1", ""),
            "recall_at_5": metrics.get("recall_at_5", ""),
            "recall_at_10": metrics.get("recall_at_10", ""),
            "recall_at_20": metrics.get("recall_at_20", ""),
            "mrr": metrics.get("mrr", ""),
        }
        is_new_file = not self.epoch_metrics_path.exists()
        with self.epoch_metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.EPOCH_FIELDS)
            if is_new_file:
                writer.writeheader()
            writer.writerow(_json_value(row))
        self.event("epoch_finished", **row)

    def finish(self, **summary: Any) -> None:
        payload = {**self._metadata, "finished_at": dt.datetime.now().isoformat(timespec="seconds"), **summary}
        self._write_json("summary.json", payload)
        self.event("run_finished", **summary)
