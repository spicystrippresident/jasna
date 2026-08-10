from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest
import torch

from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    _ScanTensorCollector,
    scan_sample_stride,
    segments_from_scores,
)
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange


def test_stride_follows_fps():
    assert scan_sample_stride(29.97) == 30
    assert scan_sample_stride(60.0, seconds=0.5) == 30
    assert scan_sample_stride(30.0, seconds=0.0) == 1
    assert scan_sample_stride(1.0) == 1
    assert scan_sample_stride(0.1) == 1


def test_consecutive_hits_merge_into_one_range():
    times = (0.0, 1.0, 2.0, 3.0, 4.0)
    scores = (0.0, 0.8, 0.9, 0.7, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0
    )
    assert segments == (SegmentRange(0.5, 4.5),)


def test_isolated_hits_stay_separate():
    times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    scores = (0.9, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0)
    segments = segments_from_scores(
        times, scores, threshold=0.5, stride=1.0, duration=10.0, pad=0.25
    )
    assert segments == (SegmentRange(0.0, 1.25), SegmentRange(2.75, 4.25))


def test_threshold_filters_hits():
    times = (0.0, 1.0)
    scores = (0.3, 0.6)
    assert segments_from_scores(times, scores, threshold=0.7, stride=1.0, duration=5.0) == ()
    low = segments_from_scores(times, scores, threshold=0.2, stride=1.0, duration=5.0)
    assert low == (SegmentRange(0.0, 2.5),)


def test_ranges_clamped_to_video():
    segments = segments_from_scores(
        (0.0, 9.0), (0.9, 0.9), threshold=0.5, stride=1.0, duration=9.8
    )
    assert segments[0].start == 0.0
    assert segments[-1].end == pytest.approx(9.8)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        segments_from_scores((0.0,), (0.5, 0.6), threshold=0.5, stride=1.0, duration=5.0)


def test_mask_lookup_only_returns_exact_sample():
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0),
        scores=(0.1, 0.9, 0.2),
        masks=["m0", "m1", "m2"],
        stride=1.0,
        duration=10.0,
        completed_until=2.0,
    )
    assert result.sample_at(1.01, tolerance=0.02) == (1.0, 0.9, "m1")
    assert result.sample_at(1.2, tolerance=0.02) is None
    assert result.sample_at(5.0, tolerance=0.02) is None


def test_scan_collector_recycles_gpu_chunk_when_free_memory_reaches_reserve(
    monkeypatch,
):
    import torch
    class FakeCuda:
        def __init__(self):
            self.free = [1_000, 100]
            self.empty_cache_calls = 0

        def mem_get_info(self):
            value = self.free.pop(0) if len(self.free) > 1 else self.free[0]
            return value, 2_000

        def empty_cache(self):
            self.empty_cache_calls += 1

    class FakeTorch:
        uint8 = torch.uint8
        float32 = torch.float32
        empty = staticmethod(torch.empty)
        cuda = FakeCuda()

    collector_globals = _ScanTensorCollector.__init__.__globals__
    monkeypatch.setitem(collector_globals, "SCAN_VRAM_RESERVE_BYTES", 100)
    monkeypatch.setitem(collector_globals, "SCAN_SPILL_CHUNK_BYTES", 16)
    spills = []
    collector = _ScanTensorCollector(
        FakeTorch,
        capacity=10,
        mask_hw=(2, 2),
        batch_size=2,
        device="cpu",
        on_spill=lambda: spills.append(True),
    )
    collector.add(
        torch.tensor([0.1, 0.2, 0.3, 0.4]),
        torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2),
        count=4,
    )
    scores, masks = collector.finish()

    assert collector.spilling
    assert spills == [True]
    assert scores == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert torch.equal(masks, torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2))
    assert FakeTorch.cuda.empty_cache_calls == 2


def test_add_many_is_single_undo_step():
    state = SegmentEditorState(duration=100.0, fps=25.0)
    added = state.add_many((SegmentRange(1.0, 2.0), SegmentRange(5.0, 7.0)))
    assert added == 2
    assert len(state.segments) == 2
    assert state.undo()
    assert state.segments == ()


def test_add_many_skips_already_covered_ranges():
    state = SegmentEditorState(duration=100.0, fps=25.0)
    state.add(0.0, 10.0)
    assert state.add_many((SegmentRange(2.0, 3.0),)) == 0
    added = state.add_many((SegmentRange(2.0, 3.0), SegmentRange(20.0, 21.0)))
    assert added == 1
    assert state.segments == (SegmentRange(0.0, 10.0), SegmentRange(20.0, 21.0))


def test_scan_decoder_count_parallel_only_for_4k_on_nvidia():
    from jasna.gui.mosaic_scan import scan_decoder_count

    assert scan_decoder_count(3840, 2160, 120.0, amd=False) == 2
    assert scan_decoder_count(8192, 4096, 15.0, amd=False) == 2
    assert scan_decoder_count(1920, 1080, 120.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 5.0, amd=False) == 1
    assert scan_decoder_count(3840, 2160, 120.0, amd=True) == 1
    assert scan_decoder_count(8192, 4096, 120.0, amd=True) == 2


def test_amd_scan_decode_strategies_only_apply_to_large_long_inputs():
    from jasna.gui.mosaic_scan import scan_decode_backends

    assert scan_decode_backends(
        8192, 4096, 120.0, amd=True, amd_strategy="dual-rocdecode"
    ) == ("auto", "auto")
    assert scan_decode_backends(
        8192, 4096, 120.0, amd=True, amd_strategy="rocdecode-software"
    ) == ("auto", "pyav-sw")
    assert scan_decode_backends(
        3840, 2160, 120.0, amd=True, amd_strategy="dual-rocdecode"
    ) == ("auto",)
    assert scan_decode_backends(
        8192, 4096, 5.0, amd=True, amd_strategy="dual-rocdecode"
    ) == ("auto",)


def test_scan_segment_bounds_balance_slower_software_reader():
    from jasna.gui.mosaic_scan import scan_segment_sample_bounds

    assert scan_segment_sample_bounds(100, ("auto", "auto")) == (0, 50, 100)
    assert scan_segment_sample_bounds(100, ("auto", "pyav-sw")) == (0, 62, 100)


def test_scan_decode_batch_synchronizes_gpu_producer_stream(monkeypatch):
    import jasna.accelerator
    from jasna.gui.mosaic_scan import _synchronize_scan_decode_batch

    stream = SimpleNamespace(synchronize=lambda: None)
    synchronized = []
    stream.synchronize = lambda: synchronized.append(True)
    device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(jasna.accelerator, "current_stream", lambda value: stream)

    _synchronize_scan_decode_batch(SimpleNamespace(device=device))
    _synchronize_scan_decode_batch(
        SimpleNamespace(device=SimpleNamespace(type="cpu"))
    )

    assert synchronized == [True]


def test_segment_sample_indices_ownership():
    from jasna.gui.mosaic_scan import segment_sample_indices

    times = [9.8, 9.9, 10.0, 10.1]
    assert segment_sample_indices(times, 0.0, 10.0, is_last=False) == [0, 1]
    assert segment_sample_indices(times, 10.0, 20.0, is_last=True) == [2, 3]
    assert segment_sample_indices([5.0], 10.0, 20.0, is_last=True) == []


@pytest.mark.parametrize("failure", [False, True])
def test_dual_scan_releases_both_readers_on_stop_or_error(monkeypatch, failure):
    import jasna.accelerator
    import jasna.media.video_decoder
    from jasna.gui.mosaic_scan import MosaicScanWorker

    readers = []

    class FakeReader:
        def __init__(self, *_args, **_kwargs):
            self.start_pts = 0
            self.closed = False
            readers.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def frames(self, seek_ts=None):
            if failure:
                raise RuntimeError("decode failed")
            yield torch.zeros((1, 3, 2, 2), dtype=torch.uint8), [0]

    monkeypatch.setattr(jasna.accelerator, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(jasna.media.video_decoder, "NvidiaVideoReader", FakeReader)
    metadata = SimpleNamespace(
        duration=120.0,
        time_base=Fraction(1, 30),
        video_fps=30.0,
        average_fps=30.0,
        num_frames=3600,
        video_width=8192,
        video_height=4096,
    )
    worker = MosaicScanWorker(
        "input.mp4",
        metadata,
        SimpleNamespace(batch_size=1),
        stride_seconds=1.0,
        decode_strategy="dual-rocdecode",
    )

    if failure:
        with pytest.raises(RuntimeError, match="decode failed"):
            worker._scan(object())
    else:
        worker.stop()
        worker._scan(object())

    assert len(readers) == 2
    assert all(reader.closed for reader in readers)
