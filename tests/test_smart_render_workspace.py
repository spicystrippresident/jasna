from __future__ import annotations

import json
from fractions import Fraction

from jasna.media.splice import KeyframeIndex, SplicePlan, SpliceSpan
from jasna.segments import SegmentRange
from jasna.smart_render_workspace import (
    SmartRenderWorkspace,
    WORKSPACE_ALGORITHM_VERSION,
    workspace_signature,
)


def _plan() -> SplicePlan:
    return SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(
            SpliceSpan("render", 0, 60, ((15, 30),)),
            SpliceSpan("copy", 60, 120),
        ),
        segments=(SegmentRange(0.5, 1.0),),
    )


def _signature(
    tmp_path,
    *,
    projection="fisheye",
    codec="h264",
    encoder_settings=None,
):
    source = tmp_path / "input.mp4"
    if not source.exists():
        source.write_bytes(b"source video")
    model = tmp_path / "model.pt"
    if not model.exists():
        model.write_bytes(b"model")
    return workspace_signature(
        source=source,
        output=tmp_path / "output.mp4",
        plan=_plan(),
        processing={"fp16": True, "batch_size": 4},
        model_files={"detector": model},
        codec=codec,
        encoder_settings=(
            {"cq": 22, "g": 60}
            if encoder_settings is None
            else encoder_settings
        ),
        resolved_projection=projection,
    )


def test_workspace_reuses_only_complete_untampered_fragment(tmp_path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )
    fragment = workspace.fragment_path(0, ".ts")
    fragment.write_bytes(b"verified")
    workspace.mark_running(0)
    workspace.mark_complete(0, fragment)

    reopened = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )

    assert reopened.path == workspace.path
    assert reopened.reusable_fragment(0) == fragment.resolve()
    fragment.write_bytes(b"tampered")
    assert reopened.reusable_fragment(0) is None


def test_workspace_resets_interrupted_running_span(tmp_path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )
    workspace.mark_running(0)

    reopened = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )

    assert reopened.manifest["spans"][0]["status"] == "pending"
    assert reopened.reusable_fragment(0) is None


def test_workspace_signature_change_uses_separate_directory(tmp_path) -> None:
    first = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=_signature(tmp_path, projection="fisheye"),
    )
    second = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=_signature(tmp_path, projection="raw"),
    )

    assert first.path != second.path
    assert first.path.is_dir()
    assert second.path.is_dir()


def test_workspace_signature_changes_for_effective_hevc_level(tmp_path) -> None:
    first_signature = _signature(
        tmp_path,
        codec="hevc",
        encoder_settings={"cq": 28, "g": 60, "level": "6.1"},
    )
    second_signature = _signature(
        tmp_path,
        codec="hevc",
        encoder_settings={"cq": 28, "g": 60, "level": "6.2"},
    )

    assert first_signature["encoding"]["settings"]["level"] == "6.1"
    first = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=first_signature,
    )
    second = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=second_signature,
    )

    assert first.path != second.path


def test_encoder_policy_version_is_part_of_workspace_signature(
    monkeypatch, tmp_path
) -> None:
    import jasna.smart_render_workspace as module

    assert WORKSPACE_ALGORITHM_VERSION == "jasna-smart-render-workspace-v7"
    first = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=_signature(tmp_path),
    )
    monkeypatch.setattr(
        module,
        "WORKSPACE_ALGORITHM_VERSION",
        "jasna-smart-render-workspace-v7-test",
    )
    second = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=_signature(tmp_path),
    )

    assert first.path != second.path


def test_workspace_preserves_invalid_manifest_before_reinitializing(tmp_path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )
    workspace.manifest_path.write_text("not json", encoding="utf-8")

    reopened = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )

    assert json.loads(reopened.manifest_path.read_text(encoding="utf-8"))[
        "schema_version"
    ] == 1
    assert len(list(reopened.path.glob("manifest.invalid-*.json"))) == 1


def test_workspace_rejects_tampered_span_plan_and_external_artifact(tmp_path) -> None:
    signature = _signature(tmp_path)
    workspace = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )
    manifest = json.loads(workspace.manifest_path.read_text(encoding="utf-8"))
    manifest["spans"][0]["end_pts"] = 999
    workspace.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    reopened = SmartRenderWorkspace.open(
        tmp_path / "work",
        output=tmp_path / "output.mp4",
        signature=signature,
    )

    assert reopened.manifest["spans"][0]["end_pts"] == 60
    assert len(list(reopened.path.glob("manifest.invalid-*.json"))) == 1
    external = tmp_path / "external.ts"
    external.write_bytes(b"not a workspace fragment")
    try:
        reopened.mark_complete(0, external)
    except RuntimeError as exc:
        assert "outside its workspace" in str(exc)
    else:
        raise AssertionError("external artifact was accepted")
