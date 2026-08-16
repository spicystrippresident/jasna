from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from jasna.mosaic.detections import Detections
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
from jasna.vr180 import SbsDetectionAdapter


def _runner() -> MagicMock:
    runner = MagicMock()
    runner.input_names = ["input"]
    runner.input_dtypes = {"input": torch.float32}
    runner.output_names = ["pred_boxes", "pred_logits", "pred_masks"]
    runner.outputs = {
        "pred_boxes": torch.zeros(1, 1, 4),
        "pred_logits": torch.zeros(1, 1, 1),
        "pred_masks": torch.zeros(1, 1, 2, 2),
    }
    return runner


def _build_model_for_backend(monkeypatch, *, amd: bool) -> RfDetrMosaicDetectionModel:
    import jasna.mosaic.rfdetr as module

    runner = _runner()
    monkeypatch.setattr(module, "is_amd_device", lambda _device: amd)
    monkeypatch.setattr(module, "is_nvidia_device", lambda _device: not amd)
    if amd:
        import jasna.mosaic.rfdetr_torch_runner as torch_runner_module

        monkeypatch.setattr(
            torch_runner_module,
            "RfDetrTorchRunner",
            lambda *args, **kwargs: runner,
        )
        return RfDetrMosaicDetectionModel(
            weights_path=Path("rfdetr-v6.pt"),
            batch_size=2,
            device=torch.device("cpu"),
            resolution=8,
            dynamic_batch=True,
            torch_variant="medium",
            fp16=False,
        )

    engine_path = MagicMock()
    engine_path.exists.return_value = True
    monkeypatch.setattr(
        module,
        "get_onnx_tensorrt_engine_path",
        lambda *args, **kwargs: engine_path,
    )
    monkeypatch.setattr(module, "TrtRunner", lambda *args, **kwargs: runner)
    return RfDetrMosaicDetectionModel(
        weights_path=Path("rfdetr-v6.onnx"),
        batch_size=2,
        device=torch.device("cpu"),
        resolution=8,
        dynamic_batch=True,
        fp16=False,
    )


def test_sbs_eye_batching_is_advertised_only_for_amd(monkeypatch) -> None:
    assert _build_model_for_backend(
        monkeypatch,
        amd=True,
    ).supports_sbs_eye_batching
    assert not _build_model_for_backend(
        monkeypatch,
        amd=False,
    ).supports_sbs_eye_batching


def _combined_dispatch_model() -> tuple[RfDetrMosaicDetectionModel, list[torch.Tensor]]:
    model = object.__new__(RfDetrMosaicDetectionModel)
    dispatches: list[torch.Tensor] = []
    model._preprocess = lambda frames: frames.to(dtype=torch.float32)
    model.boxes_out = "boxes"
    model.logits_out = "logits"
    model.masks_out = "masks"
    model.score_threshold = 0.25
    model.max_select = 1

    def infer(frames: torch.Tensor) -> dict[str, torch.Tensor]:
        dispatches.append(frames.clone())
        values = frames[:, 0, 0, 0]
        batch_size = int(values.shape[0])
        boxes = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
        boxes[:, 0, 0] = values / 100.0
        boxes[:, 0, 1] = 0.5
        boxes[:, 0, 2] = 0.1
        boxes[:, 0, 3] = 0.2
        logits = (values / 10.0).reshape(batch_size, 1, 1)
        masks = torch.ones((batch_size, 1, 2, 2), dtype=torch.float32)
        masks[1::2] = -1.0
        return {"boxes": boxes, "logits": logits, "masks": masks}

    model._infer = infer
    return model, dispatches


def _eyes() -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.zeros((2, 3, 2, 2), dtype=torch.uint8)
    right = torch.zeros((2, 3, 2, 2), dtype=torch.uint8)
    left[0] = 10
    left[1] = 20
    right[0] = 70
    right[1] = 80
    return left, right


def test_rfdetr_combined_sbs_dispatch_preserves_left_right_order() -> None:
    model, dispatches = _combined_dispatch_model()
    left_frames, right_frames = _eyes()

    left, right = model.detect_sbs_eyes(
        left_frames,
        right_frames,
        target_hw=(100, 100),
    )

    assert len(dispatches) == 1
    assert dispatches[0][:, 0, 0, 0].tolist() == [10.0, 20.0, 70.0, 80.0]
    np.testing.assert_allclose(
        [boxes[0, 0] for boxes in left.boxes_xyxy],
        [5.0, 15.0],
    )
    np.testing.assert_allclose(
        [boxes[0, 0] for boxes in right.boxes_xyxy],
        [65.0, 75.0],
    )
    assert left.masks[0].all()
    assert not left.masks[1].any()
    assert right.masks[0].all()
    assert not right.masks[1].any()

    (left_scores, left_masks), (right_scores, right_masks) = model.scan_sbs_eyes(
        left_frames,
        right_frames,
        mask_hw=(3, 4),
    )

    assert len(dispatches) == 2
    assert dispatches[1][:, 0, 0, 0].tolist() == [10.0, 20.0, 70.0, 80.0]
    torch.testing.assert_close(
        left_scores,
        torch.sigmoid(torch.tensor([1.0, 2.0])),
    )
    torch.testing.assert_close(
        right_scores,
        torch.sigmoid(torch.tensor([7.0, 8.0])),
    )
    assert left_masks[0].all()
    assert not left_masks[1].any()
    assert right_masks[0].all()
    assert not right_masks[1].any()


def test_rfdetr_sbs_eye_preprocess_rejects_mismatched_shapes() -> None:
    model = object.__new__(RfDetrMosaicDetectionModel)

    with pytest.raises(ValueError, match="identical shapes"):
        model._preprocess_sbs_eyes(
            torch.zeros((2, 3, 2, 2), dtype=torch.uint8),
            torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
        )


class _BatchableDetector:
    def __init__(
        self,
        *,
        detect_oom: bool = False,
        scan_oom: bool = False,
    ) -> None:
        self.supports_sbs_eye_batching = True
        self.detect_oom = detect_oom
        self.scan_oom = scan_oom
        self.detect_calls: list[tuple[torch.Tensor, tuple[int, int]]] = []
        self.batched_detect_calls: list[
            tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
        ] = []
        self.scan_calls: list[tuple[torch.Tensor, tuple[int, int]]] = []
        self.batched_scan_calls: list[
            tuple[torch.Tensor, torch.Tensor, tuple[int, int]]
        ] = []

    @staticmethod
    def _detections(frames: torch.Tensor) -> Detections:
        boxes = [
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            for _ in range(int(frames.shape[0]))
        ]
        masks = [
            torch.full(
                (1, 2, 3),
                bool(value),
                dtype=torch.bool,
                device=frames.device,
            )
            for value in frames[:, 0, 0, 0]
        ]
        return Detections(boxes_xyxy=boxes, masks=masks)

    def __call__(
        self,
        frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        self.detect_calls.append((frames.clone(), target_hw))
        return self._detections(frames)

    def detect_sbs_eyes(
        self,
        left_frames: torch.Tensor,
        right_frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> tuple[Detections, Detections]:
        self.batched_detect_calls.append(
            (left_frames.clone(), right_frames.clone(), target_hw)
        )
        if self.detect_oom:
            raise torch.OutOfMemoryError("simulated SBS eye batch OOM")
        return self._detections(left_frames), self._detections(right_frames)

    @staticmethod
    def _scan(
        frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = frames[:, 0, 0, 0].float()
        masks = torch.ones(
            (frames.shape[0], *mask_hw),
            dtype=torch.bool,
            device=frames.device,
        )
        return scores, masks

    def scan_scores_masks(
        self,
        frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.scan_calls.append((frames.clone(), mask_hw))
        return self._scan(frames, mask_hw=mask_hw)

    def scan_sbs_eyes(
        self,
        left_frames: torch.Tensor,
        right_frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        self.batched_scan_calls.append(
            (left_frames.clone(), right_frames.clone(), mask_hw)
        )
        if self.scan_oom:
            raise torch.OutOfMemoryError("simulated SBS eye batch OOM")
        return (
            self._scan(left_frames, mask_hw=mask_hw),
            self._scan(right_frames, mask_hw=mask_hw),
        )


def _sbs_frames() -> torch.Tensor:
    frames = torch.zeros((2, 3, 4, 8), dtype=torch.uint8)
    frames[0, :, :, :4] = 3
    frames[0, :, :, 4:] = 7
    frames[1, :, :, :4] = 8
    frames[1, :, :, 4:] = 2
    return frames


def test_sbs_adapter_uses_combined_detection_and_keeps_eye_offsets() -> None:
    detector = _BatchableDetector()

    result = SbsDetectionAdapter(detector)(_sbs_frames(), target_hw=(4, 8))

    assert len(detector.batched_detect_calls) == 1
    assert detector.detect_calls == []
    left_frames, right_frames, target_hw = detector.batched_detect_calls[0]
    assert target_hw == (4, 4)
    assert left_frames[:, 0, 0, 0].tolist() == [3, 8]
    assert right_frames[:, 0, 0, 0].tolist() == [7, 2]
    np.testing.assert_array_equal(
        result.boxes_xyxy[0],
        np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 4.0]],
            dtype=np.float32,
        ),
    )
    assert result.masks[0].shape == (2, 2, 6)
    assert result.masks[0][0, :, :3].all()
    assert not result.masks[0][0, :, 3:].any()
    assert result.masks[0][1, :, 3:].all()
    assert not result.masks[0][1, :, :3].any()


def test_sbs_adapter_keeps_nvidia_style_two_eye_calls_when_batching_is_off() -> None:
    detector = _BatchableDetector()
    detector.supports_sbs_eye_batching = False
    adapter = SbsDetectionAdapter(detector)

    adapter(_sbs_frames(), target_hw=(4, 8))
    scores, masks = adapter.scan_scores_masks(_sbs_frames(), mask_hw=(2, 8))

    assert detector.batched_detect_calls == []
    assert len(detector.detect_calls) == 2
    assert detector.batched_scan_calls == []
    assert [target for _, target in detector.scan_calls] == [(2, 4), (2, 4)]
    assert scores.tolist() == [7.0, 8.0]
    assert masks.shape == (2, 2, 8)


def test_sbs_scan_uses_combined_dispatch_only_for_even_mask_width() -> None:
    detector = _BatchableDetector()
    adapter = SbsDetectionAdapter(detector)

    scores, masks = adapter.scan_scores_masks(_sbs_frames(), mask_hw=(2, 8))

    assert len(detector.batched_scan_calls) == 1
    assert detector.scan_calls == []
    assert detector.batched_scan_calls[0][2] == (2, 4)
    assert scores.tolist() == [7.0, 8.0]
    assert masks.shape == (2, 2, 8)

    adapter.scan_scores_masks(_sbs_frames(), mask_hw=(2, 9))

    assert len(detector.batched_scan_calls) == 1
    assert [target for _, target in detector.scan_calls] == [(2, 4), (2, 5)]


def test_sbs_detection_oom_retries_separately_and_disables_batching(monkeypatch) -> None:
    import jasna.vr180 as vr180

    detector = _BatchableDetector(detect_oom=True)
    empty_cache = MagicMock()
    monkeypatch.setattr(vr180.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vr180.torch.cuda, "empty_cache", empty_cache)
    adapter = SbsDetectionAdapter(detector)

    adapter(_sbs_frames(), target_hw=(4, 8))

    assert not detector.supports_sbs_eye_batching
    assert len(detector.batched_detect_calls) == 1
    assert len(detector.detect_calls) == 2
    empty_cache.assert_called_once_with()

    adapter(_sbs_frames(), target_hw=(4, 8))

    assert len(detector.batched_detect_calls) == 1
    assert len(detector.detect_calls) == 4
    empty_cache.assert_called_once_with()


def test_sbs_scan_oom_retries_separately_and_disables_batching(monkeypatch) -> None:
    import jasna.vr180 as vr180

    detector = _BatchableDetector(scan_oom=True)
    empty_cache = MagicMock()
    monkeypatch.setattr(vr180.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(vr180.torch.cuda, "empty_cache", empty_cache)
    adapter = SbsDetectionAdapter(detector)

    scores, masks = adapter.scan_scores_masks(_sbs_frames(), mask_hw=(2, 8))

    assert not detector.supports_sbs_eye_batching
    assert len(detector.batched_scan_calls) == 1
    assert [target for _, target in detector.scan_calls] == [(2, 4), (2, 4)]
    assert scores.tolist() == [7.0, 8.0]
    assert masks.shape == (2, 2, 8)
    empty_cache.assert_called_once_with()
