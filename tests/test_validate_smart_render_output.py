from fractions import Fraction

import pytest

from scripts.validate_smart_render_output import (
    _nal_units,
    _max_pts_delta_seconds,
    _relative_seconds,
    _strictly_increasing,
    _timestamp_step_stats,
    _timestamp_steps_within_reference,
    _vcl_digest,
)


def test_length_prefixed_vcl_digest_ignores_non_vcl_units() -> None:
    vps = bytes([(32 << 1), 1, 2])
    slice_unit = bytes([(1 << 1), 3, 4, 5])
    packet = b"".join(len(unit).to_bytes(4, "big") + unit for unit in (vps, slice_unit))

    assert _nal_units(packet, length_size=4, length_prefixed=True) == [vps, slice_unit]
    assert _vcl_digest(packet, length_size=4, length_prefixed=True) == _vcl_digest(
        len(slice_unit).to_bytes(4, "big") + slice_unit,
        length_size=4,
        length_prefixed=True,
    )


def test_annex_b_nal_units_accept_three_and_four_byte_start_codes() -> None:
    first = bytes([(19 << 1), 1, 2])
    second = bytes([(39 << 1), 3, 4])
    packet = b"\x00\x00\x00\x01" + first + b"\x00\x00\x01" + second

    assert _nal_units(packet, length_size=4, length_prefixed=False) == [first, second]


def test_pts_helpers() -> None:
    assert _strictly_increasing([1, 2, 3])
    assert not _strictly_increasing([1, 1, 2])
    assert not _strictly_increasing([2, 1, 3])
    assert _relative_seconds(90_090, 0, Fraction(1, 90_000)) == Fraction(1001, 1000)
    assert _max_pts_delta_seconds(
        [0, 1001, 2002],
        Fraction(1, 60_000),
        [0, 1502, 3003],
        Fraction(1, 90_000),
    ) == pytest.approx(1 / 180_000)
    assert _timestamp_step_stats(
        [0, 1501, 3003, 4504],
        Fraction(1, 90_000),
        Fraction(1001, 60_000),
    ) == pytest.approx(
        (
            1501 / 90_000,
            1502 / 90_000,
            0.5 / 90_000,
        )
    )


def test_dts_cadence_allows_source_quantization_and_existing_tail_gap() -> None:
    assert _timestamp_steps_within_reference(
        [0, 1501, 3003, 4504],
        Fraction(1, 90_000),
        [0, 17, 33, 50],
        Fraction(1, 1000),
        tolerance=Fraction(1, 1000),
    )
    assert _timestamp_steps_within_reference(
        [0, 1530, 3060, 7560, 10_530],
        Fraction(1, 90_000),
        [0, 17, 34, 84, 117],
        Fraction(1, 1000),
        tolerance=Fraction(1, 1000),
    )


def test_dts_cadence_rejects_gap_not_present_in_source() -> None:
    assert not _timestamp_steps_within_reference(
        [0, 1530, 3060, 12_600],
        Fraction(1, 90_000),
        [0, 17, 34, 84],
        Fraction(1, 1000),
        tolerance=Fraction(1, 1000),
    )


def test_invalid_length_prefixed_packet_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid length-prefixed"):
        _nal_units(b"\x00\x00\x00\x08short", length_size=4, length_prefixed=True)
