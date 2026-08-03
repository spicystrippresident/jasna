import pytest
import torch
import torch.nn.functional as F

from jasna.media.resize_normalize import ResizeNormalizer
from jasna.mosaic.rfdetr import _IMAGENET_MEAN, _IMAGENET_STD
from jasna.mosaic.yolo import _YOLO_LETTERBOX_PAD_VALUE, _letterbox_geometry


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs accelerator"),
    pytest.mark.skipif(
        getattr(torch.version, "hip", None) is None,
        reason="tests the ROCm Triton fused preprocess kernel",
    ),
]


def _reference(frames: torch.Tensor, resolution: int) -> torch.Tensor:
    values = frames.to(dtype=torch.float32).div_(255.0)
    values = F.interpolate(
        values,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )
    mean = values.new_tensor(_IMAGENET_MEAN)[:, None, None]
    std = values.new_tensor(_IMAGENET_STD)[:, None, None]
    return (values - mean) / std


@pytest.mark.parametrize("shape", [(2, 3, 73, 121), (1, 3, 128, 128)])
def test_rocm_fused_rfdetr_preprocess_matches_torch(shape) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17)
    frames = torch.randint(
        0,
        256,
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    normalizer = ResizeNormalizer(
        device=torch.device("cuda"),
        dtype=torch.float32,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        fill=(0.0, 0.0, 0.0),
    )

    result = normalizer.run(
        frames,
        out_hw=(57, 57),
        content=(0, 0, 57, 57),
    )

    assert normalizer.backend == "triton-rocm"
    assert result.shape == (shape[0], 3, 57, 57)
    assert result.dtype is torch.float32
    assert (result - _reference(frames, 57)).abs().max().item() < 2e-5


def test_rocm_fused_preprocess_accepts_an_sbs_eye_view() -> None:
    generator = torch.Generator(device="cuda").manual_seed(23)
    frames = torch.randint(
        0,
        256,
        (2, 3, 80, 160),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    right_eye = frames[:, :, :, 80:]
    normalizer = ResizeNormalizer(
        device=torch.device("cuda"),
        dtype=torch.float32,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        fill=(0.0, 0.0, 0.0),
    )

    result = normalizer.run(
        right_eye,
        out_hw=(48, 48),
        content=(0, 0, 48, 48),
    )

    assert right_eye.stride(0) != right_eye.numel() // right_eye.shape[0]
    assert (result - _reference(right_eye, 48)).abs().max().item() < 2e-5


def test_rocm_kernel_failure_falls_back_to_torch() -> None:
    class BrokenKernel:
        def launch(self, *args, **kwargs) -> None:
            raise RuntimeError("compile failed")

    frames = torch.arange(
        3 * 32 * 48,
        device="cuda",
        dtype=torch.int64,
    ).remainder(256).to(torch.uint8).reshape(1, 3, 32, 48)
    normalizer = ResizeNormalizer(
        device=torch.device("cuda"),
        dtype=torch.float32,
        mean=_IMAGENET_MEAN,
        std=_IMAGENET_STD,
        fill=(0.0, 0.0, 0.0),
    )
    normalizer._kernel = BrokenKernel()
    normalizer._backend = "triton-rocm"

    result = normalizer.run(
        frames,
        out_hw=(24, 24),
        content=(0, 0, 24, 24),
    )

    assert normalizer.backend == "torch"
    assert not normalizer.available
    assert torch.equal(result, _reference(frames, 24))
    assert torch.equal(
        normalizer.run(
            frames,
            out_hw=(24, 24),
            content=(0, 0, 24, 24),
        ),
        _reference(frames, 24),
    )


def test_rocm_fused_fp16_yolo_letterbox_matches_torch() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    frames = torch.randint(
        0,
        256,
        (2, 3, 73, 121),
        generator=generator,
        device="cuda",
        dtype=torch.uint8,
    )
    output_size = 96
    _gain, left, top, content_width, content_height = _letterbox_geometry(
        73,
        121,
        (output_size, output_size),
    )
    normalizer = ResizeNormalizer(
        device=torch.device("cuda"),
        dtype=torch.float16,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        fill=(_YOLO_LETTERBOX_PAD_VALUE,) * 3,
    )

    result = normalizer.run(
        frames,
        out_hw=(output_size, output_size),
        content=(left, top, content_width, content_height),
    )
    resized = F.interpolate(
        frames.to(dtype=torch.float16).div_(255.0),
        size=(content_height, content_width),
        mode="bilinear",
        align_corners=False,
    )
    reference = torch.full(
        (2, 3, output_size, output_size),
        _YOLO_LETTERBOX_PAD_VALUE,
        device="cuda",
        dtype=torch.float16,
    )
    reference[
        :,
        :,
        top : top + content_height,
        left : left + content_width,
    ] = resized

    assert torch.equal(result, reference)
