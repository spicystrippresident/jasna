from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

from jasna.gui.models import AppSettings
from jasna.one_click_vr.cache import (
    load_scan_cache,
    scan_cache_path,
    write_scan_cache,
)
from jasna.one_click_vr.planner import build_one_click_vr_plan
from jasna.one_click_vr.projection import ProjectionScoreSample, choose_projection
from jasna.segments import SegmentRange


def _plan():
    return build_one_click_vr_plan(
        (0.0, 1.0, 2.0),
        (0.1, 0.8, 0.1),
        threshold=0.5,
        scan_interval_seconds=1.0,
        duration_seconds=4.0,
        completed_until_seconds=2.0,
    )


def test_scan_cache_round_trip_and_rethreshold(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-v1")
    settings = AppSettings(
        processing_mode="one_click_vr",
        detection_model="rfdetr-v6",
        detection_score_threshold=0.5,
        one_click_scan_threshold=0.5,
        one_click_min_consecutive_hits=1,
        working_directory=str(tmp_path / "work"),
    )
    cache_path = scan_cache_path(source, tmp_path / "output.mp4", settings)

    with patch(
        "jasna.mosaic.detection_registry.require_detection_model_weights",
        return_value=model,
    ):
        write_scan_cache(cache_path, source, settings, _plan())
        loaded = load_scan_cache(cache_path, source, settings)
        stricter = load_scan_cache(
            cache_path,
            source,
            replace(settings, one_click_scan_threshold=0.9),
        )

    assert cache_path.parent == tmp_path / "work" / ".jasna-one-click-vr"
    assert loaded is not None
    assert loaded.sample_times == (0.0, 1.0, 2.0)
    assert loaded.sample_scores == (0.1, 0.8, 0.1)
    assert loaded.segments == (SegmentRange(0.5, 2.5),)
    assert stricter is not None
    assert stricter.segments == ()


def test_scan_cache_rejects_changed_source_or_scan_contract(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-v1")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-v1")
    settings = AppSettings(processing_mode="one_click_vr")
    cache_path = scan_cache_path(source, tmp_path / "output.mp4", settings)

    with patch(
        "jasna.mosaic.detection_registry.require_detection_model_weights",
        return_value=model,
    ):
        write_scan_cache(cache_path, source, settings, _plan())
        assert load_scan_cache(
            cache_path,
            source,
            replace(settings, one_click_scan_interval=0.5),
        ) is None

        source.write_bytes(b"source-v2-with-a-different-size")
        assert load_scan_cache(cache_path, source, settings) is None


def test_scan_cache_treats_invalid_json_as_a_miss(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    settings = AppSettings(processing_mode="one_click_vr")
    cache_path = scan_cache_path(source, tmp_path / "output.mp4", settings)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not json", encoding="utf-8")

    assert load_scan_cache(cache_path, source, settings) is None

    cache_path.write_text("[]", encoding="utf-8")
    assert load_scan_cache(cache_path, source, settings) is None


def test_scan_cache_round_trips_projection_evidence(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    model = tmp_path / "model.pt"
    model.write_bytes(b"model-v1")
    settings = AppSettings(processing_mode="one_click_vr")
    cache_path = scan_cache_path(source, tmp_path / "output.mp4", settings)
    plan = replace(
        _plan(),
        projection_evidence=choose_projection(
            (
                ProjectionScoreSample(1.0, (1, 2, 3, 4), 0.8, 0.4, 0.7, 0.5),
                ProjectionScoreSample(2.0, (1, 2, 3, 4), 0.9, 0.4, 0.7, 0.5),
            )
        ),
    )

    with patch(
        "jasna.mosaic.detection_registry.require_detection_model_weights",
        return_value=model,
    ):
        write_scan_cache(cache_path, source, settings, plan)
        loaded = load_scan_cache(cache_path, source, settings)

    assert loaded is not None
    assert loaded.projection_evidence == plan.projection_evidence
