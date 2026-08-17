"""Read-only validation for preserved folder-batch resume candidates."""

from __future__ import annotations

from pathlib import Path

from jasna.media.splice import OutputValidationError, validate_video_output


class ResumeOutputValidationError(ValueError):
    """Raised when an existing output cannot represent a completed prior run."""


def validate_resume_video_output(
    source: str | Path,
    output: str | Path,
    *,
    configured_codec: str,
) -> None:
    """Accept outputs compatible with either source-copy or configured full render."""

    errors: list[str] = []
    for expected_codec in (None, configured_codec):
        try:
            validate_video_output(
                output,
                source=source,
                expected_codec=expected_codec,
            )
            return
        except OutputValidationError as exc:
            errors.append(str(exc))
    raise ResumeOutputValidationError("; ".join(errors))
