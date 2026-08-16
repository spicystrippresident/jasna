from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from jasna.segments import SegmentRange
from jasna.session_config import SessionConfig
from jasna.session_factory import RestorationSession, build_pipeline, build_restoration_session


def _config(**overrides) -> SessionConfig:
    base = dict(
        device="cuda:0",
        fp16=True,
        batch_size=4,
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        detection_score_threshold=0.25,
        max_detection_gap=2,
        min_detection_duration=2,
        scene_detection=True,
        restoration_model_path=Path("restore.pth"),
        compile_basicvsrpp=True,
        max_clip_size=90,
        temporal_overlap=8,
        enable_crossfade=True,
        denoise_strength="none",
        denoise_step="after_primary",
        secondary_restoration="none",
        tvai_ffmpeg_path="ffmpeg.exe",
        tvai_model="iris-2",
        tvai_scale=4,
        tvai_args="noise=0",
        tvai_workers=2,
        rtx_scale=4,
        rtx_quality="high",
        rtx_denoise="medium",
        rtx_deblur="none",
        vr_mode="auto",
        codec="hevc",
        encoder_settings={"cq": 25},
        lut_path=None,
        retarget_high_fps=False,
        disable_progress=False,
        working_dir=None,
    )
    base.update(overrides)
    return SessionConfig(**base)


def _build_session(
    config: SessionConfig,
    *,
    disable_basicvsrpp_tensorrt: bool = False,
    amd: bool = False,
):
    compile_result = MagicMock(use_basicvsrpp_tensorrt=True)
    tvai_cls = MagicMock(name="TvaiSecondaryRestorer")
    unet_cls = MagicMock(name="Unet4xSecondaryRestorer")
    rtx_cls = MagicMock(name="RtxSuperresSecondaryRestorer")
    secondary_modules = {}
    for module_name, class_name, class_mock in (
        ("tvai_secondary_restorer", "TvaiSecondaryRestorer", tvai_cls),
        ("unet4x_secondary_restorer", "Unet4xSecondaryRestorer", unet_cls),
        ("rtx_superres_secondary_restorer", "RtxSuperresSecondaryRestorer", rtx_cls),
    ):
        module = ModuleType(f"jasna.restorer.{module_name}")
        setattr(module, class_name, class_mock)
        secondary_modules[module.__name__] = module
    with (
        patch.dict(sys.modules, secondary_modules),
        patch("jasna.accelerator.is_amd_device", return_value=amd),
        patch(
            "jasna.engine_compiler.ensure_engines_compiled",
            return_value=compile_result,
        ) as compiled,
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer") as restorer_cls,
        patch("jasna.restorer.restoration_pipeline.RestorationPipeline") as pipeline_cls,
    ):
        session = build_restoration_session(
            config,
            disable_basicvsrpp_tensorrt=disable_basicvsrpp_tensorrt,
            log_callback=None,
        )
    return session, compiled, restorer_cls, pipeline_cls, tvai_cls, unet_cls, rtx_cls


def _detection_cache_session(*, device: object = "cpu") -> RestorationSession:
    return RestorationSession(
        device=device,
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
    )


def _detection_request(**overrides) -> dict[str, object]:
    request: dict[str, object] = {
        "name": "rfdetr-v5",
        "path": Path("det.onnx"),
        "batch_size": 4,
        "score_threshold": 0.25,
        "fp16": True,
    }
    request.update(overrides)
    return request


def test_session_without_secondary() -> None:
    session, compiled, restorer_cls, pipeline_cls, *_ = _build_session(_config())

    assert session.secondary_restorer is None
    assert session.detection_model_name == "rfdetr-v5"
    assert session.detection_model_path == Path("det.onnx")
    request = compiled.call_args.args[0]
    assert request.basicvsrpp is True
    assert request.basicvsrpp_model_path == "restore.pth"
    assert request.detection is True
    assert request.detection_model_name == "rfdetr-v5"
    assert request.detection_batch_size == 4
    assert request.unet4x is False
    assert restorer_cls.call_args.kwargs["use_tensorrt"] is True
    assert restorer_cls.call_args.kwargs["max_clip_size"] == 90
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is None


def test_session_selects_tvai_secondary() -> None:
    session, _, _, pipeline_cls, tvai_cls, *_ = _build_session(
        _config(secondary_restoration="tvai", tvai_scale=2, tvai_workers=1, tvai_denoise=True)
    )

    assert session.secondary_restorer is tvai_cls.return_value
    kwargs = tvai_cls.call_args.kwargs
    assert kwargs["ffmpeg_path"] == "ffmpeg.exe"
    assert kwargs["tvai_args"] == "model=iris-2:scale=2:noise=0"
    assert kwargs["scale"] == 2
    assert kwargs["num_workers"] == 1
    assert kwargs["tvai_denoise"] is True
    assert pipeline_cls.call_args.kwargs["secondary_restorer"] is tvai_cls.return_value


def test_tvai_denoise_requires_tvai_secondary() -> None:
    with pytest.raises(ValueError, match="requires secondary restoration 'tvai'"):
        _build_session(_config(tvai_denoise=True))


def test_session_selects_unet_secondary() -> None:
    session, compiled, _, _, _, unet_cls, _ = _build_session(
        _config(secondary_restoration="unet-4x")
    )

    assert session.secondary_restorer is unet_cls.return_value
    assert compiled.call_args.args[0].unet4x is True
    assert unet_cls.call_args.kwargs["fp16"] is True


def test_session_selects_rtx_secondary_and_maps_none_levels() -> None:
    session, _, _, _, _, _, rtx_cls = _build_session(
        _config(secondary_restoration="rtx-super-res", rtx_denoise="none", rtx_deblur="low")
    )

    assert session.secondary_restorer is rtx_cls.return_value
    kwargs = rtx_cls.call_args.kwargs
    assert kwargs["scale"] == 4
    assert kwargs["quality"] == "high"
    assert kwargs["denoise"] is None
    assert kwargs["deblur"] == "low"


def test_disable_basicvsrpp_tensorrt_gates_compilation() -> None:
    _, compiled, *_ = _build_session(_config(), disable_basicvsrpp_tensorrt=True)

    assert compiled.call_args.args[0].basicvsrpp is False


def test_amd_rejects_secondary_restoration() -> None:
    with pytest.raises(ValueError, match="not available in the AMD build"):
        _build_session(_config(secondary_restoration="tvai"), amd=True)


def test_amd_disables_basicvsrpp_compilation() -> None:
    _, compiled, *_ = _build_session(_config(), amd=True)

    assert compiled.call_args.args[0].basicvsrpp is False


def test_session_close_closes_restorers() -> None:
    session, *_ = _build_session(_config(secondary_restoration="unet-4x"))

    session.close()

    session.restoration_pipeline.restorer.close.assert_called_once_with()
    session.secondary_restorer.close.assert_called_once_with()


def test_detection_model_cache_reuses_same_key() -> None:
    session = _detection_cache_session()
    detector = MagicMock()
    first_request = _detection_request(path=Path("models") / ".." / "det.onnx")
    second_request = _detection_request(path=Path("det.onnx"))

    with patch(
        "jasna.mosaic.detection_registry.build_detection_model",
        return_value=detector,
    ) as build_detection_model:
        first = session.get_detection_model(**first_request)
        second = session.get_detection_model(**second_request)

    assert first is detector
    assert second is detector
    build_detection_model.assert_called_once_with(
        "rfdetr-v5",
        Path("models") / ".." / "det.onnx",
        batch_size=4,
        device="cpu",
        score_threshold=0.25,
        fp16=True,
    )


@pytest.mark.parametrize(
    "changed_field, changed_value",
    [
        ("name", "lada-yolo-v4"),
        ("path", Path("other-det.onnx")),
        ("batch_size", 8),
        ("score_threshold", 0.5),
        ("fp16", False),
        ("device", "cuda:1"),
    ],
)
def test_detection_model_cache_rebuilds_when_construction_input_changes(
    changed_field: str,
    changed_value: object,
) -> None:
    session = _detection_cache_session()
    first_detector = MagicMock()
    second_detector = MagicMock()
    request = _detection_request()

    with patch(
        "jasna.mosaic.detection_registry.build_detection_model",
        side_effect=(first_detector, second_detector),
    ) as build_detection_model:
        assert session.get_detection_model(**request) is first_detector
        if changed_field == "device":
            session.device = changed_value
            changed_request = request
        else:
            changed_request = _detection_request(**{changed_field: changed_value})
        assert session.get_detection_model(**changed_request) is second_detector

    assert build_detection_model.call_count == 2
    first_detector.close.assert_called_once_with()
    second_detector.close.assert_not_called()
    assert session.detection_model is second_detector
    assert session.detection_model_key is not None


def test_detection_model_cache_clears_closed_detector_if_rebuild_fails() -> None:
    session = _detection_cache_session()
    old_detector = MagicMock()
    session.detection_model = old_detector
    session.detection_model_key = ("old", "det.onnx", 4, "cpu", 0.25, True)

    with (
        patch(
            "jasna.mosaic.detection_registry.build_detection_model",
            side_effect=RuntimeError("load failed"),
        ),
        pytest.raises(RuntimeError, match="load failed"),
    ):
        session.get_detection_model(**_detection_request())

    old_detector.close.assert_called_once_with()
    assert session.detection_model is None
    assert session.detection_model_key is None


def test_session_close_closes_cached_detector_once_and_clears_cache() -> None:
    session = _detection_cache_session()
    detector = MagicMock()
    session.detection_model = detector
    session.detection_model_key = ("cached", "det.onnx", 4, "cpu", 0.25, True)

    session.close()
    session.close()

    detector.close.assert_called_once_with()
    assert session.detection_model is None
    assert session.detection_model_key is None


def test_build_pipeline_passes_through_config_and_session() -> None:
    config = _config(
        lut_path="lut.cube",
        retarget_high_fps=True,
        disable_progress=True,
        working_dir=Path("/scratch"),
        vr_projection="gnomonic",
    )
    session = RestorationSession(
        device=MagicMock(),
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
    )
    segments = (SegmentRange(1, 2),)
    splice_plan = MagicMock()
    progress_callback = MagicMock()

    with patch("jasna.pipeline.Pipeline") as pipeline_cls:
        pipeline = build_pipeline(
            config,
            session,
            Path("in.mp4"),
            Path("out.mp4"),
            progress_callback=progress_callback,
            segments=segments,
            splice_plan=splice_plan,
        )

    assert pipeline is pipeline_cls.return_value
    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["input_video"] == Path("in.mp4")
    assert kwargs["output_video"] == Path("out.mp4")
    assert kwargs["detection_model_name"] == "rfdetr-v5"
    assert kwargs["detection_model_path"] == Path("det.onnx")
    assert kwargs["detection_score_threshold"] == 0.25
    assert kwargs["detection_session"] is session
    assert kwargs["restoration_pipeline"] is session.restoration_pipeline
    assert kwargs["codec"] == "hevc"
    assert kwargs["encoder_settings"] == {"cq": 25}
    assert kwargs["batch_size"] == 4
    assert kwargs["device"] is session.device
    assert kwargs["max_clip_size"] == 90
    assert kwargs["temporal_overlap"] == 8
    assert kwargs["max_detection_gap"] == 2
    assert kwargs["min_detection_duration"] == 2
    assert kwargs["enable_crossfade"] is True
    assert kwargs["scene_detection"] is True
    assert kwargs["vr_mode"] == "auto"
    assert kwargs["vr_projection"] == "gnomonic"
    assert kwargs["fp16"] is True
    assert kwargs["disable_progress"] is True
    assert kwargs["progress_callback"] is progress_callback
    assert kwargs["lut_path"] == "lut.cube"
    assert kwargs["retarget_high_fps"] is True
    assert kwargs["segments"] == segments
    assert kwargs["splice_plan"] is splice_plan
    assert kwargs["working_dir"] == Path("/scratch")
    signature = kwargs["processing_signature"]
    assert signature["device"] == "cuda:0"
    assert signature["denoise_strength"] == "none"
    assert signature["denoise_step"] == "after_primary"
    assert signature["secondary_restoration"] == "none"
    assert signature["vr_projection"] == "gnomonic"


def test_build_pipeline_defaults_optional_runtime_inputs() -> None:
    session = RestorationSession(
        device=MagicMock(),
        detection_model_name="rfdetr-v5",
        detection_model_path=Path("det.onnx"),
        restoration_pipeline=MagicMock(),
        secondary_restorer=None,
    )

    with patch("jasna.pipeline.Pipeline") as pipeline_cls:
        build_pipeline(_config(), session, Path("in.mp4"), Path("out.mp4"))

    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["progress_callback"] is None
    assert kwargs["segments"] is None
    assert kwargs["splice_plan"] is None
    assert kwargs["detection_session"] is session


def test_build_pipeline_reuses_session_detector_across_sequential_videos() -> None:
    session = _detection_cache_session()
    detector = MagicMock()

    with patch(
        "jasna.mosaic.detection_registry.build_detection_model",
        return_value=detector,
    ) as build_detection_model:
        first = build_pipeline(_config(), session, Path("first.mp4"), Path("first-out.mp4"))
        second = build_pipeline(_config(), session, Path("second.mp4"), Path("second-out.mp4"))

    assert first.detection_model is detector
    assert second.detection_model is detector
    build_detection_model.assert_called_once()

    first.close()
    second.close()
    detector.close.assert_not_called()

    session.close()
    detector.close.assert_called_once_with()
