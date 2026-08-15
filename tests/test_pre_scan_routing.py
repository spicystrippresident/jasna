from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from jasna.gui.models import AppSettings
from jasna.gui.mosaic_scan import (
    MosaicScanResult,
    ScanCheckpoint,
    ScanCompleted,
    ScanFailed,
    ScanScoresReady,
)
from jasna.gui.pre_scan_routing import (
    PRE_SCAN_ALGORITHM_VERSION,
    PreScanFailed,
    PreScanCoordinator,
    PreScanStopped,
    _ScanCheckpointStore,
    _checkpoint_signature,
    coarse_route,
    hit_sample_groups,
    normalize_scan_segments,
    sampled_coverage,
    segment_coverage,
)
from jasna.segments import SegmentRange


def _result(scores, *, stride=2.0, duration=40.0):
    times = tuple(index * stride for index in range(len(scores)))
    return MosaicScanResult(
        times=times,
        scores=tuple(scores),
        masks=(),
        stride=stride,
        duration=duration,
        completed_until=times[-1] if times else 0.0,
    )


def test_coarse_route_uses_configured_85_percent_boundary():
    exact = _result([0.9] * 17 + [0.0] * 3)
    below = _result([0.9] * 16 + [0.0] * 4)

    assert sampled_coverage(exact, threshold=0.35) == pytest.approx(0.85)
    assert coarse_route(
        exact,
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "full"
    assert coarse_route(
        below,
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "fine"


def test_coarse_zero_hits_selects_source_copy():
    assert coarse_route(
        _result([0.0] * 20),
        detection_threshold=0.35,
        full_threshold=0.85,
    ) == "copy"


def test_sampled_coverage_duration_weights_irregular_samples():
    result = MosaicScanResult(
        times=(0.0, 1.0, 2.0, 10.0),
        scores=(0.9, 0.9, 0.0, 0.0),
        masks=(),
        stride=4.0,
        duration=20.0,
        completed_until=10.0,
    )

    # Midpoint cells are [0,.5), [.5,1.5), [1.5,6), [6,20): two
    # dense keyframe hits therefore account for 1.5 s, not 50% of samples.
    assert sampled_coverage(result, threshold=0.35) == pytest.approx(1.5 / 20.0)


def test_fine_hit_groups_split_at_non_hits():
    result = _result([0.0, 0.9, 0.8, 0.0, 0.7, 0.0], stride=0.5, duration=3.0)
    assert hit_sample_groups(result, threshold=0.35) == ((1, 2), (4, 4))


def test_old_timeline_padding_minimum_and_merge_rules_are_preserved():
    one = normalize_scan_segments(
        (SegmentRange(50.0, 51.0),),
        duration=120.0,
    )
    assert one == (SegmentRange(35.5, 65.5),)

    merged = normalize_scan_segments(
        (SegmentRange(10.0, 20.0), SegmentRange(50.0, 60.0)),
        duration=120.0,
    )
    assert merged == (SegmentRange(0.0, 70.0),)
    assert segment_coverage(merged, 120.0) == pytest.approx(70 / 120)


def test_scan_checkpoint_round_trip_uses_timebase_keys_and_completion(tmp_path):
    path = tmp_path / "scan" / "manifest.json"
    signature = {"source": "demo", "settings": {"fine": 0.5}}
    store = _ScanCheckpointStore(
        path,
        signature,
        fps=29.97,
        time_base=1 / 90_000,
    )
    store.add_stage_samples(
        "fine",
        0.5,
        (0.0, 0.5005, 1.001),
        (0.0, 0.8, 0.2),
    )
    store.mark_stage_complete("fine", 0.5)

    resumed = _ScanCheckpointStore(
        path,
        signature,
        fps=29.97,
        time_base=1 / 90_000,
    )
    samples = resumed.samples("fine", 0.5)
    assert sorted(samples) == [0, 45045, 90090]
    assert samples[45045][1] == pytest.approx(0.8)
    assert resumed.stage_complete("fine", 0.5)


def test_scan_checkpoint_prefers_exact_pts_keys(tmp_path):
    store = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo"},
        fps=29.97,
        time_base=1 / 90_000,
    )

    store.add_stage_samples(
        "fine",
        0.5,
        (0.5005,),
        (0.8,),
        sample_keys=(45_046,),
    )

    assert store.samples("fine", 0.5) == {45_046: (0.5005, 0.8)}


def test_boundary_checkpoint_reuses_requested_frame_index(tmp_path):
    path = tmp_path / "scan" / "manifest.json"
    signature = {"source": "demo"}
    store = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )
    store.add_boundary_sample(15, 0.5005, 0.8)
    store.flush()

    resumed = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )

    assert resumed.boundary_samples() == {15: (0.5005, 0.8)}


def test_boundary_refinement_batches_each_half_second_window(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo"},
        fps=10.0,
        time_base=0.001,
    )
    requested = []

    class FakeWorker:
        def __init__(self):
            self.events = queue.Queue()
            self.generation = 0

        def request_scores(self, start_seconds, end_seconds):
            self.generation += 1
            requested.append((start_seconds, end_seconds))
            first = round(start_seconds * 10)
            last = round(end_seconds * 10)
            times = tuple(index / 10 for index in range(first, last + 1))
            scores = tuple(0.9 if 0.4 <= seconds < 1.3 else 0.0 for seconds in times)
            self.events.put(ScanScoresReady(times, scores, self.generation))
            return self.generation

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.metadata = SimpleNamespace(video_fps=10.0, duration=2.0, time_base=0.001)
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator.checkpoint = checkpoint
    result = _result([0.0, 0.9, 0.9, 0.0], stride=0.5, duration=2.0)

    refined = coordinator._refine_boundaries(result, FakeWorker(), threshold=0.35)

    assert refined == (SegmentRange(0.4, 1.3),)
    assert requested == pytest.approx([(0.1, 0.5), (1.1, 1.5)])
    assert len(checkpoint.boundary_samples()) == 8


def test_corrupt_checkpoint_is_reset_instead_of_marked_complete(tmp_path):
    path = tmp_path / "scan" / "manifest.json"
    signature = {"source": "demo"}
    store = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )
    store.add_stage_samples("fine", 0.5, (0.0,), (0.8,))
    store.mark_stage_complete("fine", 0.5)
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["stages"]["fine"]["samples"] = [["broken"]]
    path.write_text(json.dumps(raw), encoding="utf-8")

    resumed = _ScanCheckpointStore(
        path,
        signature,
        fps=30.0,
        time_base=1 / 90_000,
    )

    assert resumed.samples("fine", 0.5) == {}
    assert not resumed.stage_complete("fine", 0.5)


def test_incomplete_stage_decodes_from_start_and_reuses_exact_pts(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo"},
        fps=30.0,
        time_base=1 / 90_000,
    )
    checkpoint.add_stage_samples(
        "fine",
        0.5,
        (0.5005,),
        (0.8,),
        sample_keys=(45_046,),
    )
    checkpoint.flush()
    created = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.update(kwargs)
            self.events = queue.Queue()
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0, 0.5005),
                        scores=(0.1, 0.8),
                        masks=(),
                        stride=0.5,
                        duration=2.0,
                        completed_until=0.5005,
                    ),
                    stopped=False,
                )
            )

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=30.0, duration=2.0)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint

    coordinator._run_stage(
        "fine",
        0.5,
        seed_samples={90_090: (1.001, 0.4)},
    )

    assert created["start_seconds"] == 0.0
    assert created["known_sample_scores"] == {45_046: 0.8, 90_090: 0.4}
    assert created["emit_checkpoints"] is True


def test_adaptive_coarse_checkpoint_keeps_exact_pts_and_reuses_completed_stage(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=59.94,
        time_base=0.01,
    )
    created = []
    plan = object()

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.events = queue.Queue()
            self.events.put(ScanCheckpoint((0, 401), (0.0, 4.01), (0.1, 0.8)))
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0, 4.01),
                        scores=(0.1, 0.8),
                        masks=(),
                        stride=4.0,
                        duration=8.0,
                        completed_until=4.01,
                    ),
                    stopped=False,
                )
            )

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=59.94, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: plan

    result, _worker = coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert result.stride == pytest.approx(4.0)
    assert created[0]["adaptive_coarse_plan"] is plan
    assert checkpoint.samples("coarse", 4.0) == {0: (0.0, 0.1), 401: (4.01, 0.8)}
    assert checkpoint.stage_complete("coarse", 4.0)

    # A complete stage neither re-probes its GOP policy nor opens a worker.
    reused, _worker = coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)
    assert reused.times == pytest.approx((0.0, 4.01))
    assert len(created) == 1


def test_adaptive_coarse_cancellation_persists_completed_batches(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=60.0,
        time_base=0.01,
    )

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            self.events = queue.Queue()
            self.events.put(ScanCheckpoint((0,), (0.0,), (0.8,)))
            self.events.put(
                ScanCompleted(
                    MosaicScanResult(
                        times=(0.0,),
                        scores=(0.8,),
                        masks=(),
                        stride=4.0,
                        duration=8.0,
                        completed_until=0.0,
                    ),
                    stopped=True,
                )
            )

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=60.0, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: object()

    with pytest.raises(PreScanStopped):
        coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert checkpoint.samples("coarse", 4.0) == {0: (0.0, 0.8)}
    assert not checkpoint.stage_complete("coarse", 4.0)


def test_adaptive_coarse_native_failure_is_reported_without_second_worker(tmp_path):
    checkpoint = _ScanCheckpointStore(
        tmp_path / "scan" / "manifest.json",
        {"source": "demo", "algorithm": "adaptive"},
        fps=60.0,
        time_base=0.01,
    )
    created = []

    class FakeWorker:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            self.events = queue.Queue()
            self.events.put(ScanFailed("rocDecode failed: native decoder error"))

        def start(self):
            pass

        def close(self):
            pass

        def join(self):
            pass

    coordinator = PreScanCoordinator.__new__(PreScanCoordinator)
    coordinator.source = tmp_path / "video.mp4"
    coordinator.output = tmp_path / "output.mp4"
    coordinator.metadata = SimpleNamespace(video_fps=60.0, duration=8.0, time_base=0.01)
    coordinator.settings = AppSettings()
    coordinator._stopped = lambda: False
    coordinator._log = lambda *_args: None
    coordinator._progress = None
    coordinator._worker_factory = FakeWorker
    coordinator._active_worker = None
    coordinator.checkpoint = checkpoint
    coordinator._adaptive_coarse_plan = lambda _interval: object()

    with pytest.raises(PreScanFailed, match="native decoder error"):
        coordinator._run_stage("coarse", 4.0, adaptive_coarse=True)

    assert len(created) == 1
    assert "adaptive_coarse_plan" in created[0]


def test_checkpoint_signature_versions_adaptive_coarse_policy(monkeypatch, tmp_path):
    import jasna.mosaic.detection_registry as registry

    source = tmp_path / "输入视频.mp4"
    source.write_bytes(b"source")
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(registry, "coerce_detection_model_name", lambda _name: "fake")
    monkeypatch.setattr(registry, "require_detection_model_weights", lambda _name: weights)

    signature = _checkpoint_signature(
        source,
        tmp_path / "output.mp4",
        AppSettings(pre_scan_coarse_interval=4.0),
    )

    assert signature["algorithm_version"] == PRE_SCAN_ALGORITHM_VERSION
    assert signature["scan"]["adaptive_coarse"] == {
        "policy_version": "keyframe-gop-v1",
        "tolerance_ratio": 0.25,
        "coverage_policy": "duration-weighted-midpoints-v1",
    }
