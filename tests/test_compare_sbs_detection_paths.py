from __future__ import annotations

import numpy as np
import torch

from jasna.mosaic.detections import Detections
from scripts.compare_sbs_detection_paths import (
    ComparisonStats,
    combine_sbs_detections,
    compare_detection_batches,
)


def _eye(box_x: float, mask_value: bool) -> Detections:
    return Detections(
        boxes_xyxy=[np.array([[box_x, 2, box_x + 3, 6]], dtype=np.float32)],
        masks=[torch.full((1, 2, 3), mask_value, dtype=torch.bool)],
    )


def test_combine_sbs_detections_offsets_right_boxes_and_masks() -> None:
    combined = combine_sbs_detections(
        _eye(1, True), _eye(2, True), target_eye_width=10
    )

    assert combined.boxes_xyxy[0].tolist() == [
        [1.0, 2.0, 4.0, 6.0],
        [12.0, 2.0, 15.0, 6.0],
    ]
    assert combined.masks[0].shape == (2, 2, 6)
    assert combined.masks[0][0, :, :3].all()
    assert not combined.masks[0][0, :, 3:].any()
    assert not combined.masks[0][1, :, :3].any()
    assert combined.masks[0][1, :, 3:].all()


def test_compare_detection_batches_records_exact_and_changed_results() -> None:
    baseline = combine_sbs_detections(
        _eye(1, True), _eye(2, False), target_eye_width=10
    )
    identical = combine_sbs_detections(
        _eye(1, True), _eye(2, False), target_eye_width=10
    )
    changed = combine_sbs_detections(
        _eye(1.5, False), _eye(2, False), target_eye_width=10
    )
    stats = ComparisonStats()

    compare_detection_batches(baseline, identical, stats)
    compare_detection_batches(baseline, changed, stats)

    assert stats.frames == 2
    assert stats.legacy_detections == stats.batched_detections == 4
    assert stats.count_mismatch_frames == 0
    assert stats.box_mismatch_frames == 1
    assert stats.mask_mismatch_frames == 1
    assert stats.mask_mismatch_pixels == 6
    assert stats.max_box_abs_error == 0.5
