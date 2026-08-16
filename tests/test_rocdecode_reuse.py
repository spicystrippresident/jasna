from __future__ import annotations

import pytest

import jasna.media.video_decoder as video_decoder
from jasna.media.rocdecode import RocDecodeError
from jasna.media.video_decoder import ReusableRocDecoder


class _FakeDecoder:
    def __init__(self, device_id: int, codec_name: str) -> None:
        self.device_id = device_id
        self.codec_name = codec_name
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


def test_reusable_decoder_keeps_one_native_decoder_across_readers(monkeypatch) -> None:
    created: list[_FakeDecoder] = []

    def build(device_id: int, codec_name: str) -> _FakeDecoder:
        decoder = _FakeDecoder(device_id, codec_name)
        created.append(decoder)
        return decoder

    monkeypatch.setattr(video_decoder, "RocDecoder", build)
    reusable = ReusableRocDecoder()

    first = reusable.acquire(0, "hevc")
    reusable.release(first)
    second = reusable.acquire(0, "HEVC")
    reusable.release(second)
    reusable.close()

    assert first is second
    assert created == [first]
    assert first.close_calls == 1


def test_reusable_decoder_rejects_concurrent_or_mismatched_use(monkeypatch) -> None:
    monkeypatch.setattr(video_decoder, "RocDecoder", _FakeDecoder)
    reusable = ReusableRocDecoder()
    decoder = reusable.acquire(0, "hevc")

    with pytest.raises(RocDecodeError, match="already in use"):
        reusable.acquire(0, "hevc")
    reusable.release(decoder)
    with pytest.raises(RocDecodeError, match="cannot change device or codec"):
        reusable.acquire(0, "av1")
    reusable.close()


def test_discard_closes_failed_decoder_and_allows_replacement(monkeypatch) -> None:
    created: list[_FakeDecoder] = []

    def build(device_id: int, codec_name: str) -> _FakeDecoder:
        decoder = _FakeDecoder(device_id, codec_name)
        created.append(decoder)
        return decoder

    monkeypatch.setattr(video_decoder, "RocDecoder", build)
    reusable = ReusableRocDecoder()
    failed = reusable.acquire(0, "hevc")
    reusable.release(failed, discard=True)
    replacement = reusable.acquire(0, "hevc")
    reusable.release(replacement)
    reusable.close()

    assert failed is not replacement
    assert failed.close_calls == 1
    assert replacement.close_calls == 1
