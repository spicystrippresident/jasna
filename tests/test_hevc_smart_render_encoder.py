from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import AcceleratorVendor
from jasna.media import (
    VideoMetadata,
    hevc_level_to_amf_option,
    parse_hevc_level_idc,
)


def _metadata(**overrides) -> VideoMetadata:
    metadata = VideoMetadata(
        video_file="input.mkv",
        video_height=1080,
        video_width=1920,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="hevc",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=True,
        hevc_level=183,
    )
    return replace(metadata, **overrides)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ({"codec_name": "hevc", "level": 183}, 183),
        ({"codec_name": "HEVC", "level": "186"}, 186),
        ({"codec_name": "hevc", "level": 180.0}, 180),
        ({"codec_name": "hevc", "level": 180.5}, None),
        ({"codec_name": "hevc", "level": True}, None),
        ({"codec_name": "h264", "level": 183}, None),
    ],
)
def test_parse_hevc_level(stream, expected) -> None:
    assert parse_hevc_level_idc(stream) == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [(30, "1.0"), (183, "6.1"), ("186", "6.2"), (181, None), (None, None)],
)
def test_map_hevc_level_to_amf(level, expected) -> None:
    assert hevc_level_to_amf_option(level) == expected


def test_linux_amd_hevc_fragment_uses_stable_cqp_and_source_level(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 28},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["level"] == "6.1"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert "vbaq" not in encoder.encoder_options


def test_fragment_policy_is_not_applied_to_windows_amd(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 28, "preanalysis": 1},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["qp_i"] == "28"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert "level" not in encoder.encoder_options


def test_resolve_hevc_vui_uses_decoded_source_values(monkeypatch) -> None:
    import jasna.media.video_encoder as module

    original = _metadata(
        video_fps=19001 / 317,
        average_fps=19001 / 317,
        video_fps_exact=Fraction(19001, 317),
        color_primaries="",
        color_transfer="",
    )
    stream = SimpleNamespace(
        codec_context=SimpleNamespace(
            framerate=Fraction(60_000, 1_001),
            rate=Fraction(60_000, 1_001),
        )
    )
    frame = SimpleNamespace(
        color_range=1,
        colorspace=1,
        color_primaries=9,
        color_trc=16,
    )
    source = MagicMock()
    source.streams.video = [stream]
    source.decode.return_value = iter([frame])
    opened = MagicMock()
    opened.__enter__.return_value = source
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=opened))

    resolved, output_fps = module.resolve_hevc_smart_render_vui(original)

    assert output_fps == Fraction(60_000, 1_001)
    assert resolved.video_fps_exact == output_fps
    assert resolved.color_range == AvColorRange.MPEG
    assert resolved.color_space == AvColorspace.ITU709
    assert resolved.color_primaries == "bt2020"
    assert resolved.color_transfer == "smpte2084"
    assert original.video_fps_exact == Fraction(19001, 317)


def test_resolve_hevc_vui_fails_closed_to_existing_metadata(monkeypatch) -> None:
    import jasna.media.video_encoder as module

    original = _metadata()
    monkeypatch.setattr(module.av, "open", MagicMock(side_effect=OSError("unreadable")))

    resolved, output_fps = module.resolve_hevc_smart_render_vui(original)

    assert output_fps == original.video_fps_exact
    assert resolved == original
    assert resolved is not original
