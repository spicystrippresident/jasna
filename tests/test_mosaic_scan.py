from __future__ import annotations

from types import SimpleNamespace

import pytest

from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    MosaicScanWorker,
    _ScanTensorCollector,
    scan_sample_stride,
    segments_from_scores,
)
from jasna.gui.models import AppSettings
from jasna.gui.segment_editor_state import SegmentEditorState
from jasna.segments import SegmentRange


def test_stride_follows_fps():
    assert scan_sample_stride(29.97) == 30
    assert scan_sample_stride(60.0, seconds=0.5) == 30
    assert scan_sample_stride(30.0, seconds=0.0) == 1
    assert scan_sample_stride(1.0) == 1
    assert scan_sample_stride(0.1) == 1


def test_editor_scan_does_not_emit_checkpoint_events_by_default():
    worker = MosaicScanWorker(
        "video.mp4",
        object(),
        AppSettings(),
        stride_seconds=1.0,
    )

    assert worker.emit_checkpoints is False
    assert worker.known_sample_scores == {}
    assert worker.serve_only is False


def test_boundary_score_window_uses_one_reusable_decoder(monkeypatch):
    import torch
    import jasna.media.video_decoder as video_decoder

    opened = []

    class FakeReader:
        def __init__(self, *args, **kwargs):
            opened.append((args, kwargs))
            self.start_pts = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def frames(self, seek_ts=None):
            assert seek_ts == pytest.approx(0.1)
            yield torch.zeros((2, 3, 2, 2), dtype=torch.uint8), [100, 200]
            yield torch.zeros((2, 3, 2, 2), dtype=torch.uint8), [300, 400]

    class FakeDetector:
        def scan_scores_masks(self, batch, *, mask_hw):
            count = batch.shape[0]
            return torch.arange(1, count + 1, dtype=torch.float32) / 10, torch.zeros(
                (count, *mask_hw), dtype=torch.bool
            )

    monkeypatch.setattr(video_decoder, "NvidiaVideoReader", FakeReader)
    worker = MosaicScanWorker(
        "video.mp4",
        SimpleNamespace(duration=1.0, time_base=0.001),
        AppSettings(batch_size=2),
        stride_seconds=0.5,
    )
    reusable = object()
    worker._reusable_rocdecoder = reusable

    event = worker._detect_scores(
        FakeDetector(),
        SimpleNamespace(start_seconds=0.1, end_seconds=0.3, generation=7),
    )

    assert event.times == pytest.approx((0.1, 0.2, 0.3))
    assert event.scores == pytest.approx((0.1, 0.2, 0.1))
    assert event.generation == 7
    assert len(opened) == 1
    assert opened[0][1]["frame_stride"] == 1
    assert opened[0][1]["prefer_software_decode"] is True
    assert opened[0][1]["reusable_rocdecoder"] is reusable


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


def test_segment_sample_indices_ownership():
    from jasna.gui.mosaic_scan import segment_sample_indices

    times = [9.8, 9.9, 10.0, 10.1]
    assert segment_sample_indices(times, 0.0, 10.0, is_last=False) == [0, 1]
    assert segment_sample_indices(times, 10.0, 20.0, is_last=True) == [2, 3]
    assert segment_sample_indices([5.0], 10.0, 20.0, is_last=True) == []
