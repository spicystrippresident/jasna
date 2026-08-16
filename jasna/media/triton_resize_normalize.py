from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _resize_normalize_kernel(
    source,
    destination,
    mean,
    standard_deviation,
    fill,
    source_batch_stride: tl.constexpr,
    source_channel_stride: tl.constexpr,
    source_row_stride: tl.constexpr,
    destination_batch_stride: tl.constexpr,
    destination_channel_stride: tl.constexpr,
    destination_row_stride: tl.constexpr,
    batch: tl.constexpr,
    source_height: tl.constexpr,
    source_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    content_left: tl.constexpr,
    content_top: tl.constexpr,
    content_width: tl.constexpr,
    content_height: tl.constexpr,
    output_fp16: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    element_count = batch * 3 * output_height * output_width
    valid = offsets < element_count

    x = offsets % output_width
    remaining = offsets // output_width
    y = remaining % output_height
    remaining = remaining // output_height
    channel = remaining % 3
    batch_index = remaining // 3

    local_x = x - content_left
    local_y = y - content_top
    inside = (
        valid
        & (local_x >= 0)
        & (local_x < content_width)
        & (local_y >= 0)
        & (local_y < content_height)
    )

    source_x = tl.maximum(
        (local_x.to(tl.float32) + 0.5) * (source_width / content_width) - 0.5,
        0.0,
    )
    source_y = tl.maximum(
        (local_y.to(tl.float32) + 0.5) * (source_height / content_height) - 0.5,
        0.0,
    )
    x0 = tl.minimum(tl.floor(source_x).to(tl.int64), source_width - 1)
    y0 = tl.minimum(tl.floor(source_y).to(tl.int64), source_height - 1)
    x1 = tl.minimum(x0 + 1, source_width - 1)
    y1 = tl.minimum(y0 + 1, source_height - 1)
    wx = source_x - x0.to(tl.float32)
    wy = source_y - y0.to(tl.float32)

    source_base = (
        batch_index * source_batch_stride
        + channel * source_channel_stride
    )
    offset00 = source_base + y0 * source_row_stride + x0
    offset01 = source_base + y0 * source_row_stride + x1
    offset10 = source_base + y1 * source_row_stride + x0
    offset11 = source_base + y1 * source_row_stride + x1
    value00 = tl.load(source + offset00, mask=inside, other=0.0).to(tl.float32) / 255.0
    value01 = tl.load(source + offset01, mask=inside, other=0.0).to(tl.float32) / 255.0
    value10 = tl.load(source + offset10, mask=inside, other=0.0).to(tl.float32) / 255.0
    value11 = tl.load(source + offset11, mask=inside, other=0.0).to(tl.float32) / 255.0
    if output_fp16:
        value00 = value00.to(tl.float16).to(tl.float32)
        value01 = value01.to(tl.float16).to(tl.float32)
        value10 = value10.to(tl.float16).to(tl.float32)
        value11 = value11.to(tl.float16).to(tl.float32)

    interpolated = (
        (1.0 - wy) * ((1.0 - wx) * value00 + wx * value01)
        + wy * ((1.0 - wx) * value10 + wx * value11)
    )
    channel_mean = tl.load(mean + channel, mask=valid, other=0.0)
    channel_std = tl.load(standard_deviation + channel, mask=valid, other=1.0)
    if output_fp16:
        interpolated = interpolated.to(tl.float16).to(tl.float32)
        channel_mean = channel_mean.to(tl.float16).to(tl.float32)
        channel_std = channel_std.to(tl.float16).to(tl.float32)
        normalized = (interpolated - channel_mean).to(tl.float16).to(tl.float32)
        normalized = (normalized / channel_std).to(tl.float16)
    else:
        normalized = (interpolated - channel_mean) / channel_std
    fill_value = tl.load(fill + channel, mask=valid, other=0.0)
    result = tl.where(inside, normalized, fill_value)

    destination_offset = (
        batch_index * destination_batch_stride
        + channel * destination_channel_stride
        + y * destination_row_stride
        + x
    )
    tl.store(destination + destination_offset, result, mask=valid)


class TritonResizeNormalizeKernel:
    def launch(
        self,
        frames: torch.Tensor,
        out: torch.Tensor,
        content: tuple[int, int, int, int],
        mean: torch.Tensor,
        std: torch.Tensor,
        fill: torch.Tensor,
    ) -> None:
        batch, _, source_height, source_width = frames.shape
        output_height, output_width = out.shape[2:]
        left, top, content_width, content_height = content
        block_size = 256
        grid = (triton.cdiv(out.numel(), block_size),)
        _resize_normalize_kernel[grid](
            frames,
            out,
            mean,
            std,
            fill,
            frames.stride(0),
            frames.stride(1),
            frames.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            batch,
            source_height,
            source_width,
            output_height,
            output_width,
            left,
            top,
            content_width,
            content_height,
            out.dtype is torch.float16,
            block_size,
            num_warps=4,
        )
