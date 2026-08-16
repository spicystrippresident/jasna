"""Conservative GUI defaults derived from lightweight hardware telemetry."""

DEFAULT_DETECTION_BATCH_SIZE = 4
HIGH_VRAM_DETECTION_BATCH_SIZE = 8

# Board-advertised 24 GiB cards can report slightly less through the driver.
# This stays above 20 GiB-class cards while accepting that reporting difference.
HIGH_VRAM_MIN_BYTES = 23 * 1024**3


def recommended_detection_batch_size(
    detection_model: str,
    total_vram_bytes: int | None,
) -> int:
    """Return the RX 7900 XTX-validated RF-DETR v6 detection batch."""

    if (
        str(detection_model).strip().lower() == "rfdetr-v6"
        and total_vram_bytes is not None
        and int(total_vram_bytes) >= HIGH_VRAM_MIN_BYTES
    ):
        return HIGH_VRAM_DETECTION_BATCH_SIZE
    return DEFAULT_DETECTION_BATCH_SIZE
