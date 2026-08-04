"""Persistent, source-bound cache for one-click VR scan evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from jasna.one_click_vr.planner import OneClickVrPlan, build_one_click_vr_plan
from jasna.one_click_vr.projection import ProjectionEvidence

if TYPE_CHECKING:
    from jasna.gui.models import AppSettings


SCAN_CACHE_SCHEMA_VERSION = 2
SCAN_ALGORITHM_VERSION = "jasna-one-click-vr-scan-v2"


def scan_cache_path(
    source: str | Path,
    output_path: str | Path,
    settings: "AppSettings",
) -> Path:
    """Return a stable sidecar path without writing to the source directory."""

    source_path = Path(source).resolve()
    output = Path(output_path)
    base = (
        Path(settings.working_directory).expanduser()
        if str(settings.working_directory).strip()
        else output.parent
    )
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", source_path.stem).strip("._-")
    slug = (slug or "video")[:80]
    source_key = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()[:12]
    return base / ".jasna-one-click-vr" / f"{slug}-{source_key}.scan.json"


def _source_identity(source: str | Path) -> dict[str, Any]:
    path = Path(source).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@lru_cache(maxsize=16)
def _file_sha256(path_text: str, size_bytes: int, mtime_ns: int) -> str:
    del size_bytes, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_signature(settings: "AppSettings") -> dict[str, Any]:
    from jasna.gui.mosaic_scan import SCAN_SCORE_FLOOR
    from jasna.mosaic.detection_registry import (
        coerce_detection_model_name,
        require_detection_model_weights,
    )

    model_name = coerce_detection_model_name(str(settings.detection_model))
    model_path = Path(require_detection_model_weights(model_name)).resolve()
    model_stat = model_path.stat()
    return {
        "algorithm_version": SCAN_ALGORITHM_VERSION,
        "model_name": model_name,
        "model_path": str(model_path),
        "model_size_bytes": model_stat.st_size,
        "model_mtime_ns": model_stat.st_mtime_ns,
        "model_sha256": _file_sha256(
            str(model_path), model_stat.st_size, model_stat.st_mtime_ns
        ),
        "batch_size": int(settings.batch_size),
        "fp16": bool(settings.fp16_mode),
        "vr_mode": str(settings.vr_mode),
        "projection_analysis": str(settings.vr_projection) == "auto",
        "requested_interval_seconds": float(settings.one_click_scan_interval),
        "score_floor": float(SCAN_SCORE_FLOOR),
    }


def _plan_payload(plan: OneClickVrPlan) -> dict[str, Any]:
    return {
        "sample_times": list(plan.sample_times),
        "sample_scores": list(plan.sample_scores),
        "effective_interval_seconds": plan.scan_interval_seconds,
        "duration_seconds": plan.duration_seconds,
        "completed_until_seconds": plan.completed_until_seconds,
        "projection_evidence": (
            plan.projection_evidence.to_dict()
            if plan.projection_evidence is not None
            else None
        ),
    }


def write_scan_cache(
    path: str | Path,
    source: str | Path,
    settings: "AppSettings",
    plan: OneClickVrPlan,
) -> None:
    """Atomically persist scan samples; restoration ranges remain derived data."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCAN_CACHE_SCHEMA_VERSION,
        "source": _source_identity(source),
        "scan": _scan_signature(settings),
        "evidence": _plan_payload(plan),
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_scan_cache(
    path: str | Path,
    source: str | Path,
    settings: "AppSettings",
) -> OneClickVrPlan | None:
    """Load compatible evidence and re-plan it using the current threshold."""

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            return None
        if payload.get("schema_version") != SCAN_CACHE_SCHEMA_VERSION:
            return None
        if payload.get("source") != _source_identity(source):
            return None
        if payload.get("scan") != _scan_signature(settings):
            return None
        evidence = payload["evidence"]
        if not isinstance(evidence, dict):
            return None
        plan = build_one_click_vr_plan(
            evidence["sample_times"],
            evidence["sample_scores"],
            threshold=float(settings.one_click_scan_threshold),
            scan_interval_seconds=float(evidence["effective_interval_seconds"]),
            duration_seconds=float(evidence["duration_seconds"]),
            completed_until_seconds=float(evidence["completed_until_seconds"]),
            minimum_consecutive_hits=int(
                settings.one_click_min_consecutive_hits
            ),
        )
        projection_payload = evidence.get("projection_evidence")
        if projection_payload is not None:
            plan = replace(
                plan,
                projection_evidence=ProjectionEvidence.from_dict(
                    projection_payload
                ),
            )
        return plan
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
