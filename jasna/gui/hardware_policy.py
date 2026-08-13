"""Conservative GUI defaults derived from lightweight hardware telemetry."""

DEFAULT_DETECTION_BATCH_SIZE = 4
HIGH_VRAM_DETECTION_BATCH_SIZE = 8

# Board-advertised 24 GB cards can report slightly less through the driver.
# This stays above the capacity of 20 GB-class cards while accepting that
# reporting difference.
HIGH_VRAM_MIN_BYTES = 23 * 1024**3


def recommended_detection_batch_size(
    detection_model: str,
    total_vram_bytes: int | None,
) -> int:
    """Return the real-video-validated GUI detection batch.

    Only RF-DETR v6 has passed the 8/10-bit, 8K validation at batch 8. Keep all
    other models and unknown/smaller GPUs on the established batch 4 path.
    """
    if (
        str(detection_model).strip().lower() == "rfdetr-v6"
        and total_vram_bytes is not None
        and int(total_vram_bytes) >= HIGH_VRAM_MIN_BYTES
    ):
        return HIGH_VRAM_DETECTION_BATCH_SIZE
    return DEFAULT_DETECTION_BATCH_SIZE
