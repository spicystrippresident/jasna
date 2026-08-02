from __future__ import annotations

import torch

from jasna.gui.mosaic_scan import MosaicScanResult
from jasna.one_click_vr.projection import (
    ProjectionEvidence,
    ProjectionScoreSample,
    candidates_from_scan,
    choose_projection,
)


def _sample(seconds: float, *, raw: float, fisheye: float, gnomonic: float):
    return ProjectionScoreSample(
        seconds=seconds,
        bbox_xyxy=(100.0, 300.0, 180.0, 380.0),
        source_score=0.8,
        raw_score=raw,
        fisheye_score=fisheye,
        gnomonic_score=gnomonic,
    )


def test_candidates_are_mapped_from_per_eye_scan_masks() -> None:
    masks = torch.zeros((3, 10, 20), dtype=torch.uint8)
    masks[1, 7:9, 2:5] = 1
    masks[2, 6:10, 13:18] = 1
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0),
        scores=(0.2, 0.8, 0.9),
        masks=masks,
        stride=1.0,
        duration=3.0,
        completed_until=2.0,
    )

    candidates = candidates_from_scan(
        result,
        threshold=0.5,
        video_width=2000,
        video_height=1000,
    )

    assert len(candidates) == 2
    assert candidates[0].seconds == 2.0
    assert candidates[0].bbox_xyxy == (1300.0, 600.0, 1800.0, 1000.0)
    assert candidates[1].bbox_xyxy == (200.0, 700.0, 500.0, 900.0)


def test_candidates_require_valid_sbs_masks_and_threshold_hits() -> None:
    result = MosaicScanResult(
        times=(0.0,),
        scores=(0.4,),
        masks=torch.ones((1, 10, 20), dtype=torch.uint8),
        stride=1.0,
        duration=1.0,
        completed_until=0.0,
    )

    assert candidates_from_scan(
        result,
        threshold=0.5,
        video_width=2000,
        video_height=1000,
    ) == ()
    assert candidates_from_scan(
        result,
        threshold=0.3,
        video_width=1999,
        video_height=1000,
    ) == ()


def test_projection_requires_consistent_multi_time_evidence() -> None:
    evidence = choose_projection(
        (
            _sample(1.0, raw=0.55, fisheye=0.72, gnomonic=0.60),
            _sample(2.0, raw=0.58, fisheye=0.70, gnomonic=0.59),
        )
    )

    assert evidence.selected == "fisheye"
    assert evidence.confidence > 0.0


def test_projection_does_not_select_from_one_time_or_inconsistent_wins() -> None:
    one_time = choose_projection(
        (
            _sample(1.0, raw=0.40, fisheye=0.80, gnomonic=0.30),
            _sample(1.0, raw=0.42, fisheye=0.78, gnomonic=0.31),
        )
    )
    inconsistent = choose_projection(
        (
            _sample(1.0, raw=0.80, fisheye=0.40, gnomonic=0.30),
            _sample(2.0, raw=0.30, fisheye=0.85, gnomonic=0.40),
        )
    )

    assert one_time.selected is None
    assert inconsistent.selected is None


def test_projection_uses_strongest_roi_at_each_time_but_keeps_all_evidence() -> None:
    samples = (
        _sample(1.0, raw=0.90, fisheye=0.93, gnomonic=0.89),
        _sample(1.0, raw=0.04, fisheye=0.06, gnomonic=0.03),
        _sample(2.0, raw=0.91, fisheye=0.94, gnomonic=0.92),
        _sample(2.0, raw=0.20, fisheye=0.19, gnomonic=0.18),
    )

    evidence = choose_projection(samples)

    assert evidence.selected == "fisheye"
    assert evidence.samples == samples


def test_projection_evidence_json_round_trip() -> None:
    original = choose_projection(
        (
            _sample(1.0, raw=0.40, fisheye=0.30, gnomonic=0.70),
            _sample(2.0, raw=0.42, fisheye=0.31, gnomonic=0.72),
        )
    )

    restored = ProjectionEvidence.from_dict(original.to_dict())

    assert restored == original
