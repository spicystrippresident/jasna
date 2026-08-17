from jasna.gui.windows_gpu_usage import (
    _aggregate_gpu_engine_util,
    _format_windows_luid,
)


def _instance(luid, physical, engine, pid=100, engine_type="Compute_0"):
    return (
        f"pid_{pid}_luid_{luid}_phys_{physical}_eng_{engine}_"
        f"engtype_{engine_type}"
    )


def test_format_windows_luid_uses_pdh_high_then_low_order() -> None:
    assert _format_windows_luid(bytes.fromhex("23f8000000000000")) == (
        "0x00000000_0x0000f823"
    )
    assert _format_windows_luid(bytes(8)) is None


def test_gpu_engine_util_filters_luid_and_uses_busiest_engine() -> None:
    target = "0x00000000_0x0000f823"
    other = "0x00000000_0x0001c552"
    items = [
        (_instance(target, 0, 0), 0, 31.2),
        (_instance(target, 0, 1), 0, 72.6),
        (_instance(other, 0, 0), 0, 99.0),
    ]

    assert _aggregate_gpu_engine_util(items, target) == 73


def test_gpu_engine_util_sums_processes_on_one_engine_and_caps_at_100() -> None:
    target = "0x00000000_0x0000f823"
    items = [
        (_instance(target, 0, 4, pid=10), 0, 61.0),
        (_instance(target, 0, 4, pid=20), 1, 52.0),
        (_instance(target, 0, 5, pid=30), 0, 40.0),
    ]

    assert _aggregate_gpu_engine_util(items, target) == 100


def test_gpu_engine_util_distinguishes_idle_from_unknown() -> None:
    target = "0x00000000_0x0000f823"

    assert _aggregate_gpu_engine_util(
        [(_instance(target, 0, 0), 0, 0.0)],
        target,
    ) == 0
    assert _aggregate_gpu_engine_util(
        [(_instance("0x00000000_0x0001c552", 0, 0), 0, 0.0)],
        target,
    ) is None


def test_gpu_engine_util_ignores_invalid_status_values_and_names() -> None:
    target = "0x00000000_0x0000f823"
    items = [
        (_instance(target, 0, 0), 0xC0000BC6, 90.0),
        (_instance(target, 0, 1), 0, float("nan")),
        ("malformed", 0, 80.0),
    ]

    assert _aggregate_gpu_engine_util(items, target) is None
