from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from jasna.crop_buffer import compute_enlarged_bbox
from jasna.vr180 import resolve_vr_mode
from jasna.vr_mask_geometry import (
    SBS_CENTER_DILATION_RATIO,
    SBS_RIM_DILATION_RATIO,
    sbs_fisheye_dilation_ratios,
)


def _metadata(width: int = 4096, height: int = 2048):
    return SimpleNamespace(
        video_width=width,
        video_height=height,
        sample_aspect_ratio=Fraction(1, 1),
        stereo_layout="",
        spherical_projection="",
    )


def test_fisheye_center_keeps_legacy_dilation() -> None:
    ratios = sbs_fisheye_dilation_ratios(
        (900.0, 900.0, 1100.0, 1100.0),
        frame_height=2048,
        eye_width=2048,
    )

    assert ratios == pytest.approx(
        (SBS_CENTER_DILATION_RATIO, SBS_CENTER_DILATION_RATIO)
    )


def test_fisheye_rim_expands_more_but_stays_capped() -> None:
    ratio_y, ratio_x = sbs_fisheye_dilation_ratios(
        (900.0, 1800.0, 1100.0, 2000.0),
        frame_height=2048,
        eye_width=2048,
    )

    assert SBS_CENTER_DILATION_RATIO < ratio_y <= SBS_RIM_DILATION_RATIO
    assert ratio_y < ratio_x <= SBS_RIM_DILATION_RATIO


def test_fisheye_crop_adds_context_without_crossing_eye_seam() -> None:
    bbox = (1900, 1800, 2038, 2000)
    plain = compute_enlarged_bbox(
        bbox,
        2048,
        4096,
        (0, 2048),
    )
    fisheye = compute_enlarged_bbox(
        bbox,
        2048,
        4096,
        (0, 2048),
        fisheye_geometry=True,
    )

    assert fisheye[0] <= plain[0]
    assert fisheye[1] <= plain[1]
    assert fisheye[2] <= 2048
    assert fisheye[3] <= 2048


def test_explicit_fisheye_mode_enables_geometry_only_for_sbs() -> None:
    fisheye = resolve_vr_mode("sbs-fisheye", _metadata(), Path("video.mp4"))
    flat = resolve_vr_mode("off", _metadata(), Path("video.mp4"))

    assert fisheye.is_sbs
    assert fisheye.fisheye_mask_geometry
    assert not flat.fisheye_mask_geometry
