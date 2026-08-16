from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from jasna.gui.app import JasnaApp
from jasna.gui.hardware_policy import (
    DEFAULT_DETECTION_BATCH_SIZE,
    HIGH_VRAM_DETECTION_BATCH_SIZE,
    HIGH_VRAM_MIN_BYTES,
    recommended_detection_batch_size,
)
from jasna.gui.models import AppSettings
from jasna.gui.settings_panel import SettingsPanel


def test_only_rfdetr_v6_on_at_least_23_gib_uses_batch_eight() -> None:
    assert recommended_detection_batch_size("rfdetr-v6", HIGH_VRAM_MIN_BYTES) == (
        HIGH_VRAM_DETECTION_BATCH_SIZE
    )
    assert recommended_detection_batch_size("RFDETR-V6", 24 * 1024**3) == 8
    assert recommended_detection_batch_size("rfdetr-v6", HIGH_VRAM_MIN_BYTES - 1) == (
        DEFAULT_DETECTION_BATCH_SIZE
    )
    assert recommended_detection_batch_size("rfdetr-v6", None) == 4
    assert recommended_detection_batch_size("rfdetr-v6-large", 24 * 1024**3) == 4


def test_settings_panel_derives_hidden_batch_from_latest_vram_telemetry() -> None:
    panel = SimpleNamespace(
        _sections=[SimpleNamespace(collect=lambda: {"detection_model": "rfdetr-v6"})],
        _total_vram_bytes=24 * 1024**3,
    )

    settings = SettingsPanel.get_settings(panel)

    assert isinstance(settings, AppSettings)
    assert settings.batch_size == 8


def test_app_keeps_last_known_vram_capacity_when_sample_is_missing() -> None:
    app = SimpleNamespace(
        _control_bar=SimpleNamespace(set_system_stats=MagicMock()),
        _settings_panel=SimpleNamespace(set_total_vram_bytes=MagicMock()),
    )
    known = SimpleNamespace(total_vram_bytes=24 * 1024**3)
    missing = SimpleNamespace(total_vram_bytes=None)

    JasnaApp._apply_system_stats(app, known)
    JasnaApp._apply_system_stats(app, missing)

    assert app._control_bar.set_system_stats.call_count == 2
    app._settings_panel.set_total_vram_bytes.assert_called_once_with(24 * 1024**3)
