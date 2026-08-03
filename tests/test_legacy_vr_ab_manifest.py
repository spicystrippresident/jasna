import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_legacy_vr_ab_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_legacy_vr_ab_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
manifest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manifest)


def test_pair_videos_requires_same_relative_parent_and_exact_restored_stem(tmp_path) -> None:
    sources = tmp_path / "sources"
    outputs = tmp_path / "outputs"
    (sources / "title").mkdir(parents=True)
    (outputs / "title").mkdir(parents=True)
    (outputs / "other").mkdir(parents=True)
    source = sources / "title" / "movie_8K.mp4"
    paired = outputs / "title" / "movie_8K_SSTART_EEND_sbs.restored.mp4"
    wrong_parent = outputs / "other" / "movie_8K_SSTART_EEND_sbs.restored.mp4"
    unrelated = outputs / "title" / "movie_8K.preview.mp4"
    for path in (source, paired, wrong_parent, unrelated):
        path.touch()

    pairs, unmatched_sources, unmatched_outputs = manifest.pair_videos(sources, outputs)

    assert pairs == [(source, paired)]
    assert unmatched_sources == []
    assert unmatched_outputs == [wrong_parent, unrelated]


def test_parse_legacy_log_uses_last_attempt_and_last_scan(tmp_path) -> None:
    log = tmp_path / "output.mp4.log"
    log.write_text(
        "\n".join(
            [
                "[2026-07-31 20:00:00] [log] 正在保存一键处理运行日志: first.log",
                "[2026-07-31 20:01:00] [source-scan] 时间段: 1 个，覆盖 10.0s (1.0%)",
                "[2026-07-31 20:02:00] [resource] 阶段资源: source-scan 时间段检测 耗时=120.0s 采样=40 系统CPU均/峰=90.0%/99.0% GPU总均/峰=95.0%/100.0%",
                "[2026-07-31 20:03:00] [amd-ce] Lada lane 0 已结束，立即验收并重试 1/10: 剩余 5 个 ROI",
                "[2026-07-31 21:00:00] [log] 正在保存一键处理运行日志: second.log",
                "[2026-07-31 21:01:00] [source-scan] 时间段: 4 个，覆盖 842.8s (36.8%)",
                "[2026-07-31 21:10:00] [resource] 阶段资源: source-scan 时间段检测 耗时=60.0s 采样=20 系统CPU均/峰=10.0%/30.0% GPU总均/峰=20.0%/80.0%",
                "[2026-07-31 23:00:00] [amd-ce] Lada lane 0 已结束，立即验收并重试 2/10: 剩余 3 个 ROI",
                "[2026-07-31 23:00:00] 完成！输出文件: output.mp4",
            ]
        ),
        encoding="utf-8",
    )

    result = manifest.parse_legacy_log(log)

    assert result["completed"] is True
    assert result["run_attempt_count"] == 2
    assert result["wall_seconds"] == 7200
    assert result["scan"] == {
        "timestamp": "2026-07-31 21:01:00",
        "range_count": 4,
        "covered_seconds": 842.8,
        "covered_percent": 36.8,
    }
    assert result["internal_retry_event_count"] == 1
    assert result["resource_summary"]["system_cpu_peak_percent"] == 30.0
    assert result["resource_summary"]["gpu_total_peak_percent"] == 80.0
    assert result["resource_summary"]["reported_phase_seconds"] == 60.0


def test_bit_depth_supports_common_yuv_formats() -> None:
    assert manifest._bit_depth({"pix_fmt": "yuv420p"}) == 8
    assert manifest._bit_depth({"pix_fmt": "yuv420p10le"}) == 10
    assert manifest._bit_depth({"profile": "Main 10", "pix_fmt": None}) == 10
