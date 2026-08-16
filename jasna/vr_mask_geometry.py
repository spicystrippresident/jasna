from __future__ import annotations

import math


SBS_CENTER_DILATION_RATIO = 0.028
SBS_RIM_DILATION_RATIO = 0.060
SBS_RIM_START_RADIUS = 0.45
SBS_RADIAL_EXPANSION_FRACTION = 0.60


def sbs_fisheye_dilation_ratios(
    bbox_xyxy: tuple[float, float, float, float],
    frame_height: int,
    eye_width: int,
) -> tuple[float, float]:
    """Return (y, x) source-space dilation ratios for one SBS eye.

    The source projection stretches content increasingly toward an eye's rim.
    Most of that stretch is tangential, so a lower-edge region grows more in X
    while a side-edge region grows more in Y. The centre retains the legacy
    2.8% dilation and the strongest rim axis is capped at 6%.
    """
    frame_height = int(frame_height)
    eye_width = int(eye_width)
    if frame_height <= 0 or eye_width <= 0:
        raise ValueError(
            f"Invalid SBS geometry: height={frame_height}, eye_width={eye_width}"
        )

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    eye_index = math.floor(center_x / eye_width)
    eye_center_x = (eye_index + 0.5) * eye_width

    rx = (center_x - eye_center_x) / (eye_width * 0.5)
    ry = (center_y - frame_height * 0.5) / (frame_height * 0.5)
    radius = math.hypot(rx, ry)
    if radius <= SBS_RIM_START_RADIUS:
        return SBS_CENTER_DILATION_RATIO, SBS_CENTER_DILATION_RATIO

    t = min(
        1.0,
        (radius - SBS_RIM_START_RADIUS) / (1.0 - SBS_RIM_START_RADIUS),
    )
    t = t * t * (3.0 - 2.0 * t)
    extra = (SBS_RIM_DILATION_RATIO - SBS_CENTER_DILATION_RATIO) * t

    tangent_x = abs(ry) / radius
    tangent_y = abs(rx) / radius
    floor = SBS_RADIAL_EXPANSION_FRACTION
    ratio_x = SBS_CENTER_DILATION_RATIO + extra * (
        floor + (1.0 - floor) * tangent_x
    )
    ratio_y = SBS_CENTER_DILATION_RATIO + extra * (
        floor + (1.0 - floor) * tangent_y
    )
    return ratio_y, ratio_x
