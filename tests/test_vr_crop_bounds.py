import numpy as np
import torch

from jasna.crop_buffer import compute_enlarged_bbox, extract_crop


def test_left_eye_crop_expansion_cannot_cross_sbs_seam(monkeypatch) -> None:
    import jasna.crop_buffer as crop_buffer

    monkeypatch.setattr(crop_buffer, "MIN_BORDER", 20)
    frame = torch.zeros((3, 64, 128), dtype=torch.uint8)

    crop = extract_crop(
        frame,
        np.array([58.0, 20.0, 63.0, 30.0], dtype=np.float32),
        64,
        128,
        x_bounds=(0, 64),
    )

    assert crop.enlarged_bbox[2] <= 64


def test_right_eye_crop_expansion_cannot_cross_sbs_seam(monkeypatch) -> None:
    import jasna.crop_buffer as crop_buffer

    monkeypatch.setattr(crop_buffer, "MIN_BORDER", 20)
    frame = torch.zeros((3, 64, 128), dtype=torch.uint8)

    crop = extract_crop(
        frame,
        np.array([65.0, 20.0, 70.0, 30.0], dtype=np.float32),
        64,
        128,
        x_bounds=(64, 128),
    )

    assert crop.enlarged_bbox[0] >= 64


def test_normal_sbs_crop_does_not_use_fisheye_rim_expansion() -> None:
    bbox = np.array([1748.0, 3400.0, 2348.0, 4000.0], dtype=np.float32)

    normal = compute_enlarged_bbox(bbox, 4096, 8192, (0, 4096))
    fisheye = compute_enlarged_bbox(
        bbox, 4096, 8192, (0, 4096), fisheye_geometry=True
    )

    normal_area = (normal[2] - normal[0]) * (normal[3] - normal[1])
    fisheye_area = (fisheye[2] - fisheye[0]) * (fisheye[3] - fisheye[1])
    assert fisheye_area > normal_area


def test_fisheye_rim_crop_stays_inside_right_eye() -> None:
    bbox = np.array([4098.0, 3400.0, 4510.0, 4000.0], dtype=np.float32)

    crop = compute_enlarged_bbox(
        bbox, 4096, 8192, (4096, 8192), fisheye_geometry=True
    )

    assert crop[0] >= 4096
    assert crop[2] <= 8192
