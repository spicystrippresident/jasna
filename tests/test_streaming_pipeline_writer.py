from unittest.mock import MagicMock, patch

from jasna.streaming_pipeline import _StreamingFrameWriter


def test_streaming_writer_handles_zero_elapsed_time():
    encoder = MagicMock()
    hls_server = MagicMock()
    hls_server.frames_per_segment.return_value = 120

    with patch("jasna.streaming_pipeline.time.monotonic", return_value=42.0):
        writer = _StreamingFrameWriter(encoder, hls_server, start_segment=0)
        writer.after_write(100)

    encoder.raise_if_failed.assert_called_once_with()
    hls_server.update_production.assert_called_once_with(0)
    hls_server.wait_for_demand.assert_called_once()
