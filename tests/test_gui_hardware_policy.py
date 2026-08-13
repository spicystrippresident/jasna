from types import SimpleNamespace

import pytest

from jasna.gui.hardware_policy import (
    HIGH_VRAM_MIN_BYTES,
    recommended_detection_batch_size,
)
from jasna.gui.settings_panel import SettingsPanel


@pytest.mark.parametrize(
    ("model", "total_vram_bytes", "expected"),
    [
        ("rfdetr-v6", HIGH_VRAM_MIN_BYTES, 8),
        ("rfdetr-v6", HIGH_VRAM_MIN_BYTES - 1, 4),
        ("rfdetr-v6", None, 4),
        ("rfdetr-v6-large", 24 * 1024**3, 4),
        ("lada-yolo-v4", 24 * 1024**3, 4),
    ],
)
def test_recommended_detection_batch_size_is_conservative(
    model, total_vram_bytes, expected
) -> None:
    assert recommended_detection_batch_size(model, total_vram_bytes) == expected


def test_settings_panel_resolves_hidden_batch_from_current_model_and_vram() -> None:
    section = SimpleNamespace(collect=lambda: {"detection_model": "rfdetr-v6"})
    panel = SimpleNamespace(
        _sections=[section],
        _total_vram_bytes=24 * 1024**3,
    )

    settings = SettingsPanel.get_settings(panel)

    assert settings.detection_model == "rfdetr-v6"
    assert settings.batch_size == 8


def test_settings_panel_does_not_apply_batch_8_to_unvalidated_model() -> None:
    section = SimpleNamespace(
        collect=lambda: {"detection_model": "rfdetr-v6-large"}
    )
    panel = SimpleNamespace(
        _sections=[section],
        _total_vram_bytes=24 * 1024**3,
    )

    assert SettingsPanel.get_settings(panel).batch_size == 4
