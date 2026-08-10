from __future__ import annotations

import sys
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    capabilities_for_device,
    vendor_for_device,
)
from jasna.media import VideoMetadata, validate_encoder_settings


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=False,
    )


def test_rocm_uses_cuda_device_api_but_reports_amd(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.1")
    assert vendor_for_device("cuda:0") is AcceleratorVendor.AMD
    capabilities = capabilities_for_device("cuda:0")
    assert capabilities.amf is True
    assert capabilities.tensorrt is False
    assert capabilities.nvcodec is False


def test_amd_basicvsrpp_skips_tensorrt_compilation(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda _device: False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(
        compiler,
        "_basicvsrpp_engines_exist",
        MagicMock(side_effect=AssertionError("TensorRT probe on AMD")),
    )
    result = compiler.ensure_engines_compiled(
        compiler.EngineCompilationRequest(
            device="cuda:0",
            fp16=True,
            basicvsrpp=True,
            basicvsrpp_model_path="model.pth",
        )
    )
    assert result.use_basicvsrpp_tensorrt is False


def test_amf_encoder_settings_are_vendor_specific() -> None:
    assert validate_encoder_settings(
        {"preanalysis": 1, "cq": 24},
        codec="h264",
        vendor=AcceleratorVendor.AMD,
    ) == {"preanalysis": 1, "cq": 24}
    with pytest.raises(ValueError, match="temporal-aq"):
        validate_encoder_settings(
            {"temporal-aq": 1},
            codec="h264",
            vendor=AcceleratorVendor.AMD,
        )
    assert validate_encoder_settings(
        {"aq_mode": "caq"},
        codec="av1",
        vendor=AcceleratorVendor.AMD,
    ) == {"aq_mode": "caq"}
    with pytest.raises(ValueError, match="vbaq"):
        validate_encoder_settings(
            {"vbaq": 1},
            codec="av1",
            vendor=AcceleratorVendor.AMD,
        )


def test_video_encoder_selects_amf_and_normalizes_cq(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="h264",
        encoder_settings={"cq": 21},
    )
    assert encoder.encoder_name == "h264_amf"
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["qvbr_quality_level"] == "21"
    assert "cq" not in encoder.encoder_options


@pytest.mark.parametrize(
    ("is_10bit", "frame_format", "codec_pixel_format"),
    [
        (True, "p010le", "yuv420p10le"),
        (False, "nv12", "yuv420p"),
    ],
)
def test_software_reference_encoder_is_explicit_and_matches_source_depth(
    monkeypatch, tmp_path, is_10bit, frame_format, codec_pixel_format
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.is_10bit = is_10bit
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 21},
        encoder_backend="software-reference",
        match_input_bit_depth=True,
    )

    assert encoder.encoder_name == "libx265"
    assert encoder.software_reference is True
    assert encoder.spec.frame_format == frame_format
    assert encoder.spec.codec_pixel_format == codec_pixel_format
    assert encoder.encoder_options["crf"] == "21"
    assert "cq" not in encoder.encoder_options


def test_software_reference_encoder_rejects_non_hevc(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    with pytest.raises(ValueError, match="only HEVC"):
        module.NvidiaVideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="h264",
            encoder_settings={},
            encoder_backend="software-reference",
        )


def test_amf_p010_host_input_reinterprets_signed_storage() -> None:
    import jasna.media.video_encoder as module

    packed = torch.tensor([-32768, -1, 0, 32767], dtype=torch.int16)
    host_input = module._amf_host_input(packed, ten_bit=True)

    assert host_input.dtype is torch.uint16
    assert torch.equal(host_input, packed.view(torch.uint16))


def test_amf_uses_its_native_forced_idr_option_name() -> None:
    import jasna.media.video_encoder as module

    assert module._forced_idr_option(AcceleratorVendor.AMD) == "forced_idr"
    assert module._forced_idr_option(AcceleratorVendor.NVIDIA) == "forced-idr"


def test_linux_smart_render_maps_nvenc_b_reference_setting_to_amf(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="h264",
        encoder_settings={"bf": 2, "b_ref_mode": "disabled"},
        smart_fragment=True,
    )

    assert encoder.encoder_options["bf"] == "2"
    assert encoder.encoder_options["bf_ref"] == "0"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert "b_ref_mode" not in encoder.encoder_options


def test_linux_amd_smart_render_accepts_av1_but_keeps_windows_protected(
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
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="av1",
        encoder_settings={},
        smart_fragment=True,
    )
    assert encoder.encoder_name == "av1_amf"
    assert encoder.encoder_options["forced_idr"] == "1"

    monkeypatch.setattr(module.sys, "platform", "win32")
    with pytest.raises(ValueError, match="validated only on Linux"):
        module.NvidiaVideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="h264",
            encoder_settings={},
            smart_fragment=True,
        )


def test_linux_amd_p010_av1_uses_source_rate_peak_vbr(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    metadata = _metadata()
    metadata.codec_name = "av1"
    metadata.is_10bit = True
    metadata.video_bitrate = 17_000_000

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="av1",
        encoder_settings={"cq": 35},
    )

    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert encoder._target_bit_rate == 17_000_000


def test_linux_amd_8bit_av1_smart_fragment_keeps_source_ceiling(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    metadata = _metadata()
    metadata.codec_name = "av1"
    metadata.video_bitrate = 17_000_000

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="av1",
        encoder_settings={"cq": 35},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.spec.ten_bit is False
    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["qvbr_quality_level"] == "35"
    assert encoder.encoder_options["maxrate"] == "17000000"
    assert encoder.encoder_options["bufsize"] == "34000000"
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize("is_10bit", [False, True])
@pytest.mark.parametrize("portable_cq", [False, True])
def test_linux_amd_hevc_source_ceiling_uses_stable_peak_vbr(
    monkeypatch, tmp_path, is_10bit, portable_cq
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = is_10bit
    metadata.video_bitrate = 22_002_901

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 28} if portable_cq else {},
        match_input_bit_depth=True,
    )

    expected_rate = int(metadata.video_bitrate * 1.25)
    assert encoder.spec.ten_bit is is_10bit
    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["maxrate"] == str(expected_rate)
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate == expected_rate


@pytest.mark.parametrize("is_10bit", [False, True])
def test_linux_amd_hevc_smart_fragment_maps_portable_cq_to_cqp(
    monkeypatch, tmp_path, is_10bit
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = is_10bit
    metadata.video_bitrate = 8_828_873
    metadata.video_width = 8192
    metadata.video_height = 4096

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 28},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert "vbaq" not in encoder.encoder_options
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize(
    ("is_10bit", "source_level", "expected_level", "frame_format"),
    [
        (False, 183, "6.1", "nv12"),
        (True, 186, "6.2", "p010le"),
    ],
)
def test_linux_amd_hevc_smart_fragment_propagates_source_level(
    monkeypatch,
    tmp_path,
    is_10bit,
    source_level,
    expected_level,
    frame_format,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = is_10bit
    metadata.hevc_level = source_level

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 28},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.spec.frame_format == frame_format
    assert encoder.encoder_options["level"] == expected_level


def test_linux_amd_hevc_smart_fragment_preserves_explicit_level(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.hevc_level = 183

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"level": "5.1"},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["level"] == "5.1"


@pytest.mark.parametrize(
    ("platform", "vendor", "codec"),
    [
        ("win32", AcceleratorVendor.AMD, "hevc"),
        ("linux", AcceleratorVendor.NVIDIA, "hevc"),
        ("linux", AcceleratorVendor.AMD, "h264"),
    ],
)
def test_source_hevc_level_is_not_added_outside_linux_amd_hevc_fragments(
    monkeypatch, platform, vendor, codec
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", platform)
    metadata = _metadata()
    metadata.hevc_level = 183

    assert module.add_amd_hevc_smart_fragment_source_level(
        {},
        metadata,
        codec=codec,
        vendor=vendor,
    ) == {}


@pytest.mark.parametrize("is_10bit", [False, True])
def test_linux_amd_hevc_smart_fragment_preserves_qvbr_but_disables_preanalysis(
    monkeypatch, tmp_path, is_10bit
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = is_10bit
    metadata.video_bitrate = 8_828_873

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"qvbr_quality_level": 28, "preanalysis": 1},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["qvbr_quality_level"] == "28"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


def test_linux_amd_hevc_smart_fragment_uses_cqp_at_smaller_resolution(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_bitrate = 8_828_873

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "fragment.nut"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 28},
        match_input_bit_depth=True,
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options


def test_linux_amd_hevc_explicit_rate_control_is_preserved(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_bitrate = 20_000_000

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"rc": "qvbr"},
        match_input_bit_depth=True,
    )

    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["qvbr_quality_level"] == "25"
    assert encoder.encoder_options["bufsize"] == "50000000"
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize("smart_fragment", [False, True])
@pytest.mark.parametrize("portable_cq", [False, True])
def test_linux_amd_h264_source_ceiling_keeps_native_qvbr(
    monkeypatch, tmp_path, smart_fragment, portable_cq
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.video_bitrate = 20_000_000

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="h264",
        encoder_settings={"cq": 28} if portable_cq else {},
        smart_fragment=smart_fragment,
    )

    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert encoder.encoder_options["qvbr_quality_level"] == (
        "28" if portable_cq else "24"
    )
    assert encoder.encoder_options["maxrate"] == "20000000"
    assert encoder.encoder_options["bufsize"] == "40000000"
    assert encoder._target_bit_rate is None
    assert ("forced_idr" in encoder.encoder_options) is smart_fragment


def test_windows_amd_hevc_source_ceiling_keeps_native_qvbr(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_bitrate = 20_000_000

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert encoder.encoder_options["bufsize"] == "50000000"
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize("source_bitrate", [0, 1_504_546_792, 3_000_000_000])
def test_linux_amd_hevc_without_valid_source_ceiling_keeps_qvbr(
    monkeypatch, tmp_path, source_bitrate
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_bitrate = source_bitrate

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize("source_bitrate", [0, 1_504_546_792, 3_000_000_000])
def test_linux_amd_hevc_gui_cq_without_valid_source_ceiling_uses_cqp(
    monkeypatch, tmp_path, source_bitrate
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_bitrate = source_bitrate

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={"cq": 28},
        match_input_bit_depth=True,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


def test_streaming_encoder_selects_amf(monkeypatch, tmp_path) -> None:
    import jasna.streaming_encoder as module

    monkeypatch.setattr(module, "find_executable", lambda _name: "/ffmpeg")
    popen = MagicMock()
    popen.stderr = []
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=popen))
    encoder = module.StreamingEncoder(
        tmp_path,
        4.0,
        _metadata(),
        "missing.mp4",
        torch.device("cuda:0"),
    )
    encoder._vendor = AcceleratorVendor.AMD
    encoder._launch_ffmpeg(0)
    cmd = module.subprocess.Popen.call_args.args[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_amf"
    assert "-qvbr_quality_level" in cmd
    assert "h264_nvenc" not in cmd


def test_amf_decoder_context_is_created(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("h264_amf", "r")
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_windows_av1_amf_decoder_uses_stream_codec_instead_of_libdav1d_name(
    monkeypatch,
) -> None:
    import jasna.media.video_decoder as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    metadata = _metadata()
    metadata.codec_name = "av1"
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata,
    )
    source = SimpleNamespace(
        name="libdav1d",
        extradata=b"header",
        width=16,
        height=16,
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )

    reader._setup_amf_decoder(source)

    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("av1_amf", "r")
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_linux_av1_input_uses_faster_software_decoder(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    create = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=create),
    )
    metadata = _metadata()
    metadata.codec_name = "av1"
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata,
    )
    source = SimpleNamespace(name="libdav1d", thread_type=None)

    reader._setup_amf_decoder(source)

    create.assert_not_called()
    assert reader._decoder_ctx is None
    assert reader._amd_hardware_decode is False
    assert source.thread_type == "AUTO"


def test_linux_8k_hevc_scan_uses_faster_software_decoder(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    create = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=create),
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_width = 8192
    metadata.video_height = 4096
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata,
        prefer_software_decode=True,
    )
    source = SimpleNamespace(name="hevc", thread_type=None)

    reader._setup_amf_decoder(source)

    create.assert_not_called()
    assert reader._decoder_ctx is None
    assert reader._amd_hardware_decode is False
    assert source.thread_type == "AUTO"


def test_linux_8k_hevc_regular_reader_keeps_amf_decoder(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.video_width = 8192
    metadata.video_height = 4096
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata,
    )
    source = SimpleNamespace(
        name="hevc",
        extradata=b"header",
        width=8192,
        height=4096,
        framerate=Fraction(60, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )

    reader._setup_amf_decoder(source)

    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("hevc_amf", "r")
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_amf_decoder_survives_pyav18_time_base_regression(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    class FakeDecoder:
        def __init__(self):
            object.__setattr__(self, "opened", False)

        def __setattr__(self, name, value):
            if name == "time_base":
                raise RuntimeError("Cannot access 'time_base' as a decoder")
            object.__setattr__(self, name, value)

        def open(self, strict=False):
            object.__setattr__(self, "opened", True)

    decoder = FakeDecoder()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        metadata,
    )
    source = SimpleNamespace(
        name="hevc",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("hevc_amf", "r")
    assert decoder.opened is True
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_amf_decoder_allows_missing_sample_aspect_ratio(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = SimpleNamespace(open=MagicMock())
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        framerate=Fraction(30, 1),
        sample_aspect_ratio=None,
        thread_type=None,
    )

    reader._setup_amf_decoder(source)

    decoder.open.assert_called_once_with(strict=False)
    assert not hasattr(decoder, "sample_aspect_ratio")
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_linux_10bit_input_skips_unreliable_pyav_amf_decoder(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    create = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=create),
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = True
    reader = module.NvidiaVideoReader(
        "input.mkv", 4, torch.device("cuda:0"), metadata
    )
    source = SimpleNamespace(name="hevc", thread_type=None)

    reader._setup_amf_decoder(source)

    create.assert_not_called()
    assert reader._decoder_ctx is None
    assert reader._amd_hardware_decode is False
    assert source.thread_type == "AUTO"


def test_linux_10bit_amf_encoding_disables_preanalysis(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    metadata = _metadata()
    metadata.codec_name = "hevc"
    metadata.is_10bit = True

    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mkv"),
        torch.device("cuda:0"),
        metadata,
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.spec.ten_bit is True
    assert encoder.encoder_options["preanalysis"] == "0"


def test_amf_av1_uses_its_native_aq_option(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mkv"),
        torch.device("cuda:0"),
        _metadata(),
        codec="av1",
        encoder_settings={},
    )

    assert encoder.encoder_options["aq_mode"] == "caq"
    assert "vbaq" not in encoder.encoder_options


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_yuv_eager_converter_runs_on_gpu_planes(monkeypatch) -> None:
    import jasna.media.yuv_to_rgb as module

    monkeypatch.setattr(module, "is_nvidia_device", lambda _device: False)
    H = W = 16
    generator = torch.Generator().manual_seed(0)
    y = torch.randint(16, 236, (H, W), dtype=torch.uint8, generator=generator)
    uv = torch.randint(16, 240, (H // 2, W // 2, 2), dtype=torch.uint8, generator=generator)

    cpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cpu")
    )
    expected = torch.empty((3, H, W), dtype=torch.uint8)
    cpu.convert_into(y, uv, expected)

    gpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cuda:0")
    )
    out = torch.empty((3, H, W), dtype=torch.uint8, device="cuda:0")
    gpu.convert_into(y.cuda(), uv.cuda(), out)

    assert (out.cpu().int() - expected.int()).abs().max() <= 1


def test_amf_8bit_downgrade_drops_bitdepth(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["profile"] == "main"
    assert "bitdepth" not in encoder.encoder_options



def test_rfdetr_torch_runner_maps_outputs(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")

    class FakeCore:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, x):
            batch = x.shape[0]
            return {
                "pred_boxes": torch.zeros(batch, 5, 4),
                "pred_logits": torch.zeros(batch, 5, 3),
                "pred_masks": torch.zeros(batch, 5, 8, 8),
            }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = SimpleNamespace(model=FakeCore())

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=FakeModel))
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda *_a, **_k: {"model": {"class_embed.weight": torch.zeros(3, 256)}},
    )

    runner = module.RfDetrTorchRunner(
        weights,
        input_shapes=[(2, 3, 576, 576)],
        device=torch.device("cpu"),
        fp16=False,
        resolution=576,
        variant="medium",
    )

    assert runner.input_names == ["input"]
    assert runner.input_dtypes == {"input": torch.float32}
    assert runner.output_names == ["dets", "labels", "masks"]
    assert runner.outputs["dets"].ndim == 3
    assert runner.outputs["dets"].shape[-1] == 4
    assert runner.outputs["labels"].ndim == 3
    assert runner.outputs["masks"].ndim == 4
    # num_classes derived from class_embed rows - 1 (rfdetr adds a reserve slot).
    assert runner._wrapper.kwargs["num_classes"] == 2

    out = runner.infer({"input": torch.zeros(2, 3, 576, 576)})
    assert set(out) == {"dets", "labels", "masks"}
    assert out["dets"].shape == (2, 5, 4)
    assert out["labels"].shape == (2, 5, 3)
    assert out["masks"].shape == (2, 5, 8, 8)

    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.infer({"input": torch.zeros(2, 3, 576, 576)})


def test_rfdetr_torch_runner_rejects_unknown_variant(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace())

    with pytest.raises(RuntimeError, match="unsupported variant"):
        module.RfDetrTorchRunner(
            weights,
            input_shapes=[(1, 3, 576, 576)],
            device=torch.device("cpu"),
            fp16=False,
            resolution=576,
            variant="mystery",
        )


def test_amd_rfdetr_needs_no_detection_engine(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device=None: True)

    assert compiler._detection_engine_exists(
        "rfdetr-v6",
        "rfdetr-v6.pt",
        batch_size=4,
        fp16=True,
        device="cpu",
    )
