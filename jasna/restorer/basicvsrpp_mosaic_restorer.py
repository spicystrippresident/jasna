import logging

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import is_amd_device, is_nvidia_device
from jasna.models.basicvsrpp.inference import load_model

INFERENCE_SIZE = 256
AMD_INDEPENDENT_CLIP_BATCH_SIZE = 2
AMD_INDEPENDENT_CLIP_BATCH_MIN_FRAMES = 60
AMD_INDEPENDENT_CLIP_BATCH_MAX_PADDING_FRAMES = 4
AMD_INDEPENDENT_CLIP_BATCH_MIN_FREE_BYTES = 768 * 1024 * 1024


class BasicvsrppMosaicRestorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        max_clip_size: int,
        use_tensorrt: bool,
        fp16: bool,
        config: str | dict | None = None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device)
        self.max_clip_size = int(max_clip_size)
        self.use_tensorrt = bool(use_tensorrt)
        self.dtype = torch.float16 if fp16 else torch.float32
        self.input_dtype = self.dtype

        self._split_forward = None
        self.model = None

        if self.use_tensorrt and is_nvidia_device(self.device):
            from jasna.restorer.basicvsrpp_sub_engines import create_split_forward

            pytorch_model = load_model(config, checkpoint_path, self.device, fp16)
            self._split_forward = create_split_forward(
                model=pytorch_model,
                model_weights_path=checkpoint_path,
                device=self.device,
                fp16=fp16,
            )
            if self._split_forward is not None:
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        else:
            self.use_tensorrt = False
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

        # Independent long clips can share the model batch dimension on ROCm.
        # Short one-off shapes are slower because of MIOpen setup overhead. The
        # TensorRT split path has batch-1 loop-body engines and stays unchanged.
        if is_amd_device(self.device):
            self.independent_clip_batch_size = AMD_INDEPENDENT_CLIP_BATCH_SIZE
            self.independent_clip_batch_min_frames = (
                AMD_INDEPENDENT_CLIP_BATCH_MIN_FRAMES
            )
            self.independent_clip_batch_max_padding_frames = (
                AMD_INDEPENDENT_CLIP_BATCH_MAX_PADDING_FRAMES
            )
            self.independent_clip_batch_min_free_bytes = (
                AMD_INDEPENDENT_CLIP_BATCH_MIN_FREE_BYTES
            )
        else:
            self.independent_clip_batch_size = 1
            self.independent_clip_batch_min_frames = 0
            self.independent_clip_batch_max_padding_frames = 0
            self.independent_clip_batch_min_free_bytes = 0

    def close(self) -> None:
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = None
        self.model = None

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (C, H, W) tensors in RGB format, [0, 255]
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            stacked = torch.stack(video).to(device=self.device, dtype=self.input_dtype, memory_format=torch.contiguous_format).div_(255.0)

            if self._split_forward is not None:
                result = self._split_forward(stacked.unsqueeze(0))
            else:
                result = self.model(inputs=stacked.unsqueeze(0))
            return result.squeeze(0)

    def raw_process_batch(self, videos: list[list[Tensor]]) -> list[torch.Tensor]:
        """Restore equal-length, independent clips in one model invocation."""
        if not videos:
            return []
        frame_count = len(videos[0])
        if frame_count <= 0 or any(len(video) != frame_count for video in videos):
            raise ValueError("batched restoration requires equal non-empty clip lengths")
        if len(videos) == 1 or self._split_forward is not None:
            return [self.raw_process(video) for video in videos]

        with torch.inference_mode():
            stacked = torch.stack(
                [torch.stack(video) for video in videos]
            ).to(
                device=self.device,
                dtype=self.input_dtype,
                memory_format=torch.contiguous_format,
            ).div_(255.0)
            result = self.model(inputs=stacked)
            return list(result.unbind(0))

    def restore(self, video: list[Tensor]) -> list[Tensor]:
        """
        Args:
            video: list of (H, W, C) uint8 tensors in RGB format
        Returns:
            list of (256, 256, C) uint8 tensors in RGB format
        """
        result = self.raw_process([frame.permute(2, 0, 1) for frame in video])
        result = result.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).permute(0, 2, 3, 1)
        return list(torch.unbind(result, 0))
