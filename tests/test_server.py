"""Integration tests for the FastAPI editor backend (REST + WebSocket) over a temp song folder."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from agentlodge.editor.motion_bank import MotionBank  # noqa: E402
from agentlodge.editor.window_edit import MockWindowGenerator  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import server.app as A
    import server.rendering as R
    monkeypatch.delenv("AGENTLODGE_LIVE", raising=False)
    monkeypatch.delenv("MAESTRO_REQUIRE_MOTION_AUDIT", raising=False)
    # point the server at a temp media/sessions tree with one synthetic song
    media = tmp_path / "media" / "sng"
    media.mkdir(parents=True)
    motion = MockWindowGenerator().generate("edge", 0, 300, 0, energy=0.5, beats=None)
    np.save(media / "base_motion.npy", motion)
    beats = np.arange(0, 300, 15).astype(np.float32)
    strengths = np.full(len(beats), 0.1, dtype=np.float32)
    strengths[8] = 1.0
    np.save(media / "beats.npy", beats)
    np.save(media / "beat_strengths.npy", strengths)
    (media / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # tiny stub
    monkeypatch.setattr(A, "MEDIA", tmp_path / "media")
    monkeypatch.setattr(A, "SESSIONS", tmp_path / "sessions")
    monkeypatch.setattr(A, "MOTION_PREVIEWS", tmp_path / "motion-previews")
    A._sessions.clear()
    R._CJOBS.clear()
    return TestClient(A.app)


def test_live_editor_enables_the_blocking_motion_audit(monkeypatch):
    import server.app as A

    monkeypatch.delenv("AGENTLODGE_LIVE", raising=False)
    monkeypatch.delenv("MAESTRO_REQUIRE_MOTION_AUDIT", raising=False)
    monkeypatch.delenv("MAESTRO_ALLOW_UNAUDITED_RESEARCH", raising=False)
    assert not A._motion_audit_required()
    monkeypatch.setenv("AGENTLODGE_LIVE", "1")
    assert A._motion_audit_required()
    monkeypatch.setenv("MAESTRO_ALLOW_UNAUDITED_RESEARCH", "1")
    assert not A._motion_audit_required()


def test_openai_key_can_be_loaded_from_a_private_file(tmp_path, monkeypatch):
    import server.app as A

    key_file = tmp_path / ".oai_key"
    key_file.write_text("file-test-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OAI_KEY_FILE", str(key_file))

    assert A._openai_api_key() == "file-test-key"


def test_planner_live_verification_records_success(monkeypatch):
    import server.app as A

    monkeypatch.setenv("AGENTLODGE_VERIFY_LLM_PLANNER", "1")
    monkeypatch.setenv("AGENTLODGE_REQUIRE_LLM_PLANNER", "1")
    monkeypatch.setattr(A, "_openai_api_key", lambda: "test-key")
    monkeypatch.setattr(A, "_probe_llm_planner", lambda _key: "verified LLM planner")

    A._verify_planner()

    assert A._PLANNER_STATUS == {
        "configured": True,
        "verified": True,
        "model": os.environ.get("AGENTLODGE_PLANNER_MODEL", "gpt-4o"),
        "message": "verified LLM planner",
    }


def test_required_planner_fails_closed_when_live_call_fails(monkeypatch):
    import server.app as A

    monkeypatch.setenv("AGENTLODGE_VERIFY_LLM_PLANNER", "1")
    monkeypatch.setenv("AGENTLODGE_REQUIRE_LLM_PLANNER", "1")
    monkeypatch.setattr(A, "_openai_api_key", lambda: "test-key")

    def fail(_key):
        raise RuntimeError("provider rejected the request")

    monkeypatch.setattr(A, "_probe_llm_planner", fail)

    with pytest.raises(RuntimeError, match="live verification failed"):
        A._verify_planner()
    assert A._PLANNER_STATUS["configured"] is True
    assert A._PLANNER_STATUS["verified"] is False


def test_editor_prewarm_starts_the_blender_pool(monkeypatch):
    import server.app as A
    import server.rendering as rendering
    import server.warm_render as warm_render

    calls = []
    monkeypatch.setattr(rendering, "prewarm_pod", lambda: None)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda *, wait_ready=0: calls.append(wait_ready))

    A._prewarm()

    assert calls == [0]


def test_warm_daemon_liveness_requires_its_process(tmp_path, monkeypatch):
    import server.warm_render as warm_render

    (tmp_path / "daemon.pid").write_text("12345")
    (tmp_path / "daemon.ready").write_text(str(warm_render.PROTOCOL_VERSION))
    (tmp_path / "daemon.hb").touch()
    (tmp_path / "daemon.attestation.json").write_text(
        json.dumps(
            {
                "pid": 12345,
                "daemon_protocol_version": warm_render.PROTOCOL_VERSION,
                "render_contract_version": warm_render.RENDER_CONTRACT_VERSION,
            }
        )
    )
    monkeypatch.setattr(warm_render.os, "kill", lambda pid, sig: None)
    assert warm_render._alive(tmp_path)

    def missing_process(_pid, _sig):
        raise ProcessLookupError

    monkeypatch.setattr(warm_render.os, "kill", missing_process)
    assert not warm_render._alive(tmp_path)


def test_warm_pool_replaces_incompatible_daemon(tmp_path, monkeypatch):
    import server.warm_render as warm_render

    d = tmp_path / "d0"
    d.mkdir()
    (d / "daemon.pid").write_text("12345")
    (d / "daemon.ready").write_text(str(warm_render.PROTOCOL_VERSION - 1))
    process = {"alive": True}
    calls = []

    monkeypatch.setattr(warm_render, "DAEMON_ROOT", tmp_path)
    monkeypatch.setattr(warm_render, "POOL_SIZE", 1)
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "_alive", lambda _d: False)
    monkeypatch.setattr(warm_render, "_pid_alive", lambda _d: process["alive"])

    def fake_stop(_d):
        calls.append("stop")
        process["alive"] = False

    def fake_start(_i, **_kwargs):
        calls.append("start")

    monkeypatch.setattr(warm_render, "_stop_daemon", fake_stop)
    monkeypatch.setattr(warm_render, "_start_daemon", fake_start)

    warm_render.ensure_pool(width=320, height=320, samples=1)

    assert calls == ["stop", "start"]


def test_warm_daemon_launch_tracks_the_blender_pid(tmp_path, monkeypatch):
    import server.warm_render as warm_render

    class FakeProcess:
        pid = 24680

    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(warm_render, "DAEMON_ROOT", tmp_path)
    monkeypatch.setattr(warm_render.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(warm_render, "_blender", lambda: tmp_path / "blender")
    monkeypatch.setattr(warm_render, "_scene", lambda: tmp_path / "scene.blend")
    monkeypatch.setattr(warm_render, "_ybot", lambda: tmp_path / "ybot.fbx")
    monkeypatch.setattr(warm_render, "_daemon_script", lambda: tmp_path / "daemon.py")

    warm_render._start_daemon(0, width=320, height=240, samples=2)

    assert captured["command"][0] == str(tmp_path / "blender")
    assert captured["kwargs"]["start_new_session"] is True
    assert (tmp_path / "d0" / "daemon.pid").read_text() == "24680"


def test_warm_render_sends_sparse_direct_video_options(tmp_path, monkeypatch):
    import json
    import server.warm_render as warm_render

    d = tmp_path / "d0"
    d.mkdir()
    frames = tmp_path / "frames"
    raw_video = tmp_path / "raw.mp4"
    animated_glb = tmp_path / "animated.glb"
    rid = "r" + "a" * 10
    done = d / f"{rid}.done"

    monkeypatch.setattr(warm_render, "DAEMON_ROOT", tmp_path)
    monkeypatch.setattr(warm_render, "POOL_SIZE", 1)
    monkeypatch.setattr(warm_render, "_alive", lambda _d: True)
    monkeypatch.setattr(
        warm_render,
        "_attestation_matches",
        lambda _d, **_kwargs: True,
    )
    monkeypatch.setattr(warm_render.uuid, "uuid4", lambda: SimpleNamespace(hex="a" * 32))

    def finish_render(_delay):
        raw_video.write_bytes(b"video")
        animated_glb.write_bytes(b"glb")
        done.write_text("0.1")

    monkeypatch.setattr(warm_render.time, "sleep", finish_render)

    assert warm_render.warm_render(
        "poses.npz",
        str(frames),
        daemon=0,
        width=384,
        height=384,
        samples=1,
        engine="cycles",
        denoise=0,
        fast=True,
        stride=5,
        projection_only=True,
        batch_render=True,
        video_path=str(raw_video),
        export_glb=str(animated_glb),
        frame_start=15,
        frame_end=45,
        clear_frames=False,
        frame_format="tga",
    )
    request = json.loads((d / f"{rid}.req").read_text())
    assert request["fast"] is True
    assert request["stride"] == 5
    assert request["engine"] == "cycles"
    assert request["denoise"] == 0
    assert request["projection_only"] is True
    assert request["batch_render"] is True
    assert request["video_path"] == str(raw_video)
    assert request["export_glb"] == str(animated_glb)
    assert request["frame_start"] == 15
    assert request["frame_end"] == 45
    assert request["clear_frames"] is False
    assert request["frame_format"] == "tga"


def test_full_warm_render_preserves_quality_defaults_and_caches(tmp_path, monkeypatch):
    import server.fk as fk
    import server.rendering as rendering
    import server.warm_render as warm_render

    for name in (
        "AGENTLODGE_RENDER_FULL_W",
        "AGENTLODGE_RENDER_FULL_H",
        "AGENTLODGE_RENDER_FULL_SAMPLES",
        "AGENTLODGE_RENDER_ENGINE",
        "AGENTLODGE_RENDER_DENOISE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_WORKERS", "1")
    captured = {"renders": 0, "pool_calls": 0}
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"wav")

    monkeypatch.setattr(warm_render, "on_pod", lambda: True)

    def fake_ensure_pool(**kwargs):
        captured["pool_calls"] += 1
        captured["pool"] = kwargs
        return 2

    def fake_save(_motion, path):
        Path(path).write_bytes(b"poses")

    def fake_render(_poses, frames, **kwargs):
        captured["renders"] += 1
        captured["render"] = kwargs
        frames = Path(frames)
        frames.mkdir(parents=True, exist_ok=True)
        for frame in range(kwargs["frame_start"], kwargs["frame_end"]):
            (frames / f"frame_{frame:04d}.tga").write_bytes(b"tga")
        return True

    def fake_encode(frames, output, **kwargs):
        captured["encode"] = {"frames": frames, **kwargs}
        Path(output).write_bytes(b"video")
        return True

    monkeypatch.setattr(warm_render, "ensure_pool", fake_ensure_pool)
    monkeypatch.setattr(warm_render, "ready_daemons", lambda: [0, 1])
    monkeypatch.setattr(warm_render, "warm_render", fake_render)
    monkeypatch.setattr(warm_render, "render_provenance", lambda: {})
    monkeypatch.setattr(fk, "save_poses_npz", fake_save)
    monkeypatch.setattr(rendering, "_ffmpeg_frames", fake_encode)

    assert rendering._render_warm_local(
        "sng",
        np.zeros((120, 139), dtype=np.float32),
        tmp_path,
        "full",
        audio_wav=str(audio),
    )
    assert captured["pool"]["width"] == 1080
    assert captured["pool"]["height"] == 1080
    assert captured["pool"]["samples"] == 96
    assert captured["render"]["width"] == 1080
    assert captured["render"]["height"] == 1080
    assert captured["render"]["samples"] == 96
    assert captured["render"]["engine"] == "eevee"
    assert captured["render"]["denoise"] == 1
    assert captured["render"]["fast"] is False
    assert captured["render"]["stride"] == 1
    assert captured["render"]["batch_render"] is True
    assert captured["render"]["frame_start"] == 0
    assert captured["render"]["frame_end"] == 120
    assert captured["render"]["clear_frames"] is False
    assert captured["render"]["frame_format"] == "tga"
    assert "video_path" not in captured["render"]
    assert captured["encode"]["audio_wav"] == str(audio)
    assert (tmp_path / "edited.mp4").read_bytes() == b"video"

    assert rendering._render_warm_local(
        "sng",
        np.zeros((120, 139), dtype=np.float32),
        tmp_path,
        "full",
        audio_wav=str(audio),
    )
    assert captured["renders"] == 1
    assert captured["pool_calls"] == 1


def test_full_warm_render_shards_every_frame(tmp_path, monkeypatch):
    import server.fk as fk
    import server.rendering as rendering
    import server.warm_render as warm_render

    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_WORKERS", "3")
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda **_kwargs: 3)
    monkeypatch.setattr(warm_render, "ready_daemons", lambda: [0, 1, 2])
    calls = []

    def fake_save(_motion, path):
        Path(path).write_bytes(b"poses")

    def fake_render(_poses, frames, **kwargs):
        calls.append(kwargs)
        frames = Path(frames)
        frames.mkdir(parents=True, exist_ok=True)
        for frame in range(kwargs["frame_start"], kwargs["frame_end"]):
            (frames / f"frame_{frame:04d}.tga").write_bytes(b"tga")
        return True

    def fake_encode(_frames, output, **_kwargs):
        Path(output).write_bytes(b"video")
        return True

    monkeypatch.setattr(fk, "save_poses_npz", fake_save)
    monkeypatch.setattr(warm_render, "warm_render", fake_render)
    monkeypatch.setattr(warm_render, "render_provenance", lambda: {})
    monkeypatch.setattr(rendering, "_ffmpeg_frames", fake_encode)

    assert rendering._render_warm_local(
        "sharded",
        np.zeros((10, 139), dtype=np.float32),
        tmp_path,
        "full",
    )
    assert sorted(
        (call["daemon"], call["frame_start"], call["frame_end"])
        for call in calls
    ) == [(0, 0, 4), (1, 4, 7), (2, 7, 10)]
    assert all(call["clear_frames"] is False for call in calls)
    assert all(call["frame_format"] == "tga" for call in calls)


def test_compare_warm_preserves_window_quality_defaults(tmp_path, monkeypatch):
    import server.fk as fk
    import server.rendering as rendering
    import server.warm_render as warm_render

    for name in (
        "AGENTLODGE_RENDER_WIN_W",
        "AGENTLODGE_RENDER_WIN_H",
        "AGENTLODGE_RENDER_WIN_SAMPLES",
        "AGENTLODGE_RENDER_ENGINE",
        "AGENTLODGE_RENDER_DENOISE",
    ):
        monkeypatch.delenv(name, raising=False)
    calls = []

    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda **_kwargs: 2)
    monkeypatch.setattr(warm_render, "ready_daemons", lambda: [0, 1])

    def fake_save(motion, path):
        np.savez(path, fk_joints=np.zeros((motion.shape[0], 22, 3), dtype=np.float32))

    def fake_render(_poses, frames, **kwargs):
        calls.append(kwargs)
        Path(frames).mkdir(parents=True, exist_ok=True)
        if kwargs["rig_metrics"]:
            n = 6
            np.savez(
                kwargs["rig_metrics"],
                projected=np.ones((n, 22, 3), dtype=np.float32),
                rendered_frames=np.arange(n, dtype=np.int32),
            )
        return True

    def fake_encode(_frames, output, **_kwargs):
        Path(output).write_bytes(b"video")
        return True

    monkeypatch.setattr(fk, "save_poses_npz", fake_save)
    monkeypatch.setattr(warm_render, "warm_render", fake_render)
    monkeypatch.setattr(rendering, "_ffmpeg_frames", fake_encode)

    before = np.zeros((6, 139), dtype=np.float32)
    after = before.copy()
    after[:, 10] = 0.1
    assert rendering._compare_warm("sng", before, after, tmp_path)
    assert len(calls) == 2
    for call in calls:
        assert call["width"] == 448
        assert call["height"] == 448
        assert call["samples"] == 8
        assert call["engine"] == "eevee"
        assert call["denoise"] == 1
        assert call["fast"] is False
        assert call["stride"] == 1
        assert call["batch_render"] is True
        assert "video_path" not in call


def test_sparse_projected_joints_are_interpolated_for_highlights(tmp_path):
    import server.rendering as rendering

    n = 12
    before = np.zeros((n, 22, 3), dtype=np.float32)
    after = before.copy()
    after[:, [14, 17, 19, 21], 0] = np.linspace(0.0, 0.24, n)[:, None]
    projected = np.full((n, 22, 3), np.nan, dtype=np.float32)
    rendered = np.array([0, 5, 10], dtype=np.int32)
    for frame, offset in zip(rendered, (0.0, 0.05, 0.1)):
        projected[frame, :, 0] = 0.5
        projected[frame, :, 1] = 0.5
        projected[frame, :, 2] = 1.0
        projected[frame, 14, :2] = (0.61 + offset, 0.58)
        projected[frame, 17, :2] = (0.69 + offset, 0.55)
        projected[frame, 19, :2] = (0.76 + offset, 0.49)
        projected[frame, 21, :2] = (0.82 + offset, 0.44)

    before_path = tmp_path / "before.npz"
    after_path = tmp_path / "after.npz"
    rig_path = tmp_path / "after_rig.npz"
    np.savez(before_path, fk_joints=before)
    np.savez(after_path, fk_joints=after)
    np.savez(rig_path, projected=projected, rendered_frames=rendered)

    highlight = rendering._build_change_highlight(
        str(before_path),
        str(after_path),
        str(rig_path),
    )

    assert highlight is not None
    active_frames = [i for i, frame in enumerate(highlight["frames"]) if frame]
    assert any(frame not in rendered for frame in active_frames)
    assert {marker["part"] for frame in highlight["frames"] for marker in frame} == {"right_arm"}


def test_render_cache_keys_include_quality_settings():
    import server.rendering as rendering

    window = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)

    key = rendering._render_cache_key(
        window,
        width=448,
        height=448,
        samples=1,
        stride=3,
    )
    assert key == rendering._render_cache_key(
        window.copy(),
        width=448,
        height=448,
        samples=1,
        stride=3,
    )
    assert key != rendering._render_cache_key(
        window,
        width=512,
        height=448,
        samples=1,
        stride=3,
    )
    assert key != rendering._render_cache_key(
        window,
        width=448,
        height=448,
        samples=1,
        stride=3,
        context="preview:10:10:123:456",
    )


def test_render_identity_invalidates_cache_and_task_for_scene_and_root_motion():
    import server.rendering as rendering
    from server.distributed.render_contract import (
        RENDER_CONTRACT_VERSION,
        render_identity_digest,
    )
    from server.distributed.tasks import deterministic_task_id

    motion = np.arange(10 * 6, dtype=np.float32).reshape(10, 6)
    provenance = {
        "render_contract_version": RENDER_CONTRACT_VERSION,
        "daemon_protocol_version": 6,
        "scene": {
            "blend_sha256": "1" * 64,
            "ybot_sha256": "2" * 64,
        },
        "renderer": {
            "blender_version": "4.2.3 LTS",
            "blender_daemon_sha256": "3" * 64,
            "blender_render_ybot_sha256": "4" * 64,
            "blender_studio_sha256": "5" * 64,
            "render_root_motion_sha256": "6" * 64,
        },
        "selector": None,
    }
    quality = {
        "width": 1080,
        "height": 1080,
        "samples": 96,
        "engine": "eevee",
        "denoise": 1,
        "frame_format": "tga",
        "fps": 30,
    }
    changed_scene = {
        **provenance,
        "scene": {**provenance["scene"], "blend_sha256": "a" * 64},
    }
    changed_root_motion = {
        **provenance,
        "renderer": {
            **provenance["renderer"],
            "render_root_motion_sha256": "b" * 64,
        },
    }
    identities = [
        render_identity_digest(candidate, quality)
        for candidate in (provenance, changed_scene, changed_root_motion)
    ]
    assert len(set(identities)) == 3

    cache_keys = {
        rendering._render_cache_key(
            motion,
            width=1080,
            height=1080,
            samples=96,
            stride=1,
            engine="eevee",
            denoise=1,
            context=f"full:identity:{identity}",
        )
        for identity in identities
    }
    task_ids = {
        deterministic_task_id(
            "render.frames",
            {
                "frame_start": 0,
                "frame_end": 10,
                "render_identity_digest": identity,
            },
        )
        for identity in identities
    }
    assert len(cache_keys) == 3
    assert len(task_ids) == 3


def test_frame_sequence_accepts_blender_batch_numbering(tmp_path):
    import server.rendering as rendering

    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(3):
        (frames / f"frame_{index:04d}.png").write_bytes(b"png")

    assert rendering._frame_sequence(str(frames)) == (
        str(frames / "frame_%04d.png"),
        0,
        3,
    )
    (frames / "frame_0001.png").unlink()
    assert rendering._frame_sequence(str(frames)) is None

    for frame in frames.glob("*.png"):
        frame.unlink()
    for index in range(3):
        (frames / f"frame_{index:04d}.tga").write_bytes(b"tga")
    assert rendering._frame_sequence(str(frames)) == (
        str(frames / "frame_%04d.tga"),
        0,
        3,
    )


def test_render_ranges_cover_all_frames_once():
    import server.rendering as rendering

    assert rendering._render_ranges(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert rendering._render_ranges(3, 8) == [(0, 1), (1, 2), (2, 3)]
    assert rendering._render_ranges(0, 6) == []


def test_render_progress_counts_only_completed_frame_format(tmp_path):
    import server.rendering as rendering

    (tmp_path / "frame_0000.tga").write_bytes(b"frame")
    (tmp_path / "frame_0001.tga").write_bytes(b"frame")
    (tmp_path / "frame_0002.png").write_bytes(b"frame")
    (tmp_path / "notes.tga").write_bytes(b"not a frame")

    assert rendering._count_frame_files(tmp_path, "tga") == 2
    assert rendering._count_frame_files(tmp_path, "png") == 1


def test_upload_pipeline_uses_latest_structured_progress_event():
    import server.processing as processing

    event = processing._pipeline_event(
        "\n".join(
            [
                "MAESTRO_PROGRESS generation 40 Generating motion",
                "ordinary model output",
                "MAESTRO_SUBPROGRESS generation 45 LODGE generation complete",
            ]
        )
    )

    assert event == {
        "stage": "generation",
        "progress": None,
        "message": "LODGE generation complete",
        "subprogress": 45,
    }


def test_upload_pipeline_parses_remote_stage_timings():
    import server.processing as processing

    report = processing._pipeline_timing_report(
        "\n".join(
            [
                "MAESTRO_TIMING generation_lodge start 1000 k=1",
                "MAESTRO_TIMING generation_edge start 1100 k=1",
                "MAESTRO_TIMING generation_lodge end 4000 k=1",
                "MAESTRO_TIMING generation_edge end 5100 k=1",
                "MAESTRO_TIMING generation_lodge start 6000 retry",
                "MAESTRO_TIMING generation_lodge end 8000 retry",
            ]
        )
    )

    assert report["stages"]["generation_lodge"] == {
        "attempts": 2,
        "duration_seconds": 5.0,
        "first_started_at_ms": 1000,
        "last_ended_at_ms": 8000,
    }
    assert report["stages"]["generation_edge"]["duration_seconds"] == 4.0


def test_job_trace_persists_stages_and_browser_timing(tmp_path):
    import server.processing as processing

    processing._JOBS.clear()
    trace_path = tmp_path / "trace.json"
    processing._set(
        "trace-song",
        status="queued",
        stage="queued",
        started=100.0,
        request_id="request-1",
        source_bytes=123,
        trace_path=str(trace_path),
    )
    processing._set(
        "trace-song",
        status="processing",
        stage="generation",
        progress=40,
    )
    processing._set(
        "trace-song",
        status="done",
        stage="ready",
        progress=100,
        finished=105.0,
    )
    timing = processing.record_browser_timing(
        "trace-song",
        {
            "request_id": "request-1",
            "browser_started_at_ms": 1000,
            "browser_completed_at_ms": 6000,
            "browser_upload_seconds": 1.25,
            "browser_total_seconds": 5.0,
            "source_bytes": 123,
        },
    )

    job = processing.get_job("trace-song")
    persisted = json.loads(trace_path.read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in job["stage_timeline"]] == [
        "queued",
        "generation",
        "ready",
    ]
    assert all(stage["duration_seconds"] is not None for stage in job["stage_timeline"])
    assert timing["browser_total_seconds"] == 5.0
    assert persisted["browser_total_seconds"] == 5.0
    assert persisted["request_id"] == "request-1"


def test_upload_passes_request_and_source_metadata_to_processing(client, monkeypatch):
    import server.app as A

    captured = {}

    def fake_start(sid, wav, media_dir, display_name, **kwargs):
        captured.update(
            sid=sid,
            wav=wav,
            media_dir=media_dir,
            display_name=display_name,
            **kwargs,
        )

    monkeypatch.setattr(A.processing, "start_processing", fake_start)
    response = client.post(
        "/api/upload",
        files={"file": ("timed.wav", b"timed-audio", "audio/wav")},
        data={
            "request_id": "browser-request",
            "client_started_at_ms": "1234.5",
        },
        headers={"X-MAESTRO-Request-ID": "browser-request"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "browser-request"
    assert payload["source_bytes"] == len(b"timed-audio")
    assert captured["request_id"] == "browser-request"
    assert captured["client_started_at_ms"] == 1234.5
    assert captured["source_bytes"] == len(b"timed-audio")
    assert len(captured["source_sha256"]) == 64


def test_lists_songs_and_opens_session(client):
    songs = client.get("/api/songs").json()["songs"]
    assert any(s["sid"] == "sng" for s in songs)
    st = client.post("/api/session/sng").json()
    assert st["n_frames"] == 300 and st["duration"] == 10.0 and st["n_beats"] == 20
    assert st["head"] and len(st["timeline"]) == 1


def test_interview_mode_lists_only_curated_songs_in_order(client, monkeypatch):
    import server.app as A

    monkeypatch.setattr(A, "INTERVIEW_CATALOG", A.MEDIA / "missing-catalog.json")
    for sid, name, order, interview in (
        ("curated_b", "Second choice", 2, True),
        ("curated_a", "First choice", 1, True),
        ("internal", "Calibration", 0, False),
    ):
        song = A.MEDIA / sid
        song.mkdir()
        np.save(song / "base_motion.npy", np.zeros((30, 139), dtype=np.float32))
        (song / "meta.json").write_text(
            json.dumps({"name": name, "order": order, "interview": interview}),
            encoding="utf-8",
        )
    monkeypatch.setenv("MAESTRO_INTERVIEW_MODE", "1")

    songs = client.get("/api/songs").json()["songs"]

    assert [song["sid"] for song in songs] == ["curated_a", "curated_b"]
    assert [song["name"] for song in songs] == ["First choice", "Second choice"]
    assert [song["front_facing"] for song in songs] == [False, False]


def test_interview_catalog_is_an_ordered_whitelist(client, monkeypatch):
    import server.app as A

    for sid in ("full_dynamite", "segment_b", "segment_a", "other_song"):
        song = A.MEDIA / sid
        song.mkdir()
        np.save(song / "base_motion.npy", np.zeros((30, 139), dtype=np.float32))
        (song / "meta.json").write_text(
            json.dumps(
                {
                    "name": f"stale {sid}",
                    "order": 99,
                    "interview": True,
                    "front_facing": sid.startswith("segment_"),
                }
            ),
            encoding="utf-8",
        )
    catalog = A.MEDIA / "interview-catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "catalog": [
                    {"sid": "segment_a", "name": "First segment", "order": 1},
                    {"sid": "segment_b", "name": "Second segment", "order": 2},
                    {"sid": "other_song", "name": "Other approved song", "order": 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(A, "INTERVIEW_CATALOG", catalog)
    monkeypatch.setenv("MAESTRO_INTERVIEW_MODE", "1")

    songs = client.get("/api/songs").json()["songs"]

    assert [song["sid"] for song in songs] == [
        "segment_a",
        "segment_b",
        "other_song",
    ]
    assert [song["name"] for song in songs] == [
        "First segment",
        "Second segment",
        "Other approved song",
    ]
    assert [song["front_facing"] for song in songs] == [True, True, False]


def test_interview_catalog_fails_if_configured_media_is_missing(client, monkeypatch):
    import server.app as A

    song = A.MEDIA / "segment_a"
    song.mkdir()
    np.save(song / "base_motion.npy", np.zeros((30, 139), dtype=np.float32))
    catalog = A.MEDIA / "interview-catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "catalog": [
                    {"sid": "segment_a", "name": "First segment", "order": 1},
                    {"sid": "missing_segment", "name": "Missing segment", "order": 2},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(A, "INTERVIEW_CATALOG", catalog)
    monkeypatch.setenv("MAESTRO_INTERVIEW_MODE", "1")

    response = client.get("/api/songs")

    assert response.status_code == 503
    assert "missing_segment" in response.text


def test_study_player_is_mounted_and_linked_from_editor(client):
    study = client.get("/study/")
    assert study.status_code == 200
    assert "Blind comparison player" in study.text
    assert 'href="/study/"' in client.get("/").text


def test_song_lookup_resolves_the_media_root(tmp_path, monkeypatch):
    import server.app as A

    media = tmp_path / "media"
    song = media / "linked_song"
    song.mkdir(parents=True)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    monkeypatch.setattr(A, "MEDIA", shadow / ".." / "media")

    assert A._song_dir("linked_song") == song.resolve()


def test_uploaded_song_pipeline_uses_configured_pod_python(tmp_path, monkeypatch):
    import server.processing as P

    cfg = P.PodConfig(host="127.0.0.1", ws="/workspace")
    wav = tmp_path / "source.wav"
    wav.write_bytes(b"wav")
    media = tmp_path / "media"
    commands: list[str] = []

    monkeypatch.setenv(
        "AGENTLODGE_POD_PYTHON",
        "/workspace/AgentLODGE/.venv/bin/python",
    )
    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
    monkeypatch.setenv(
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
        "jukebox.extract,lodge.generate,edge.generate",
    )
    monkeypatch.setenv(
        "AGENTLODGE_WORKER_REGISTRY",
        "/workspace/maestro-workers/registry.json",
    )
    monkeypatch.setenv("AGENTLODGE_SHARED_ROOT", "/workspace")
    monkeypatch.setenv("AGENTLODGE_EARLY_LODGE_GENERATION", "1")
    monkeypatch.setenv("AGENTLODGE_EARLY_LODGE_WAIT_SECONDS", "120")
    monkeypatch.setattr(P, "pod_config", lambda: cfg)
    monkeypatch.setattr(P, "_co_located", lambda _cfg: False)

    def fake_ssh(_cfg, command, timeout=60):
        commands.append(command)
        stdout = "ok" if command == "echo ok" else ""
        if "scripts/process_song.sh" in command:
            stdout = "PROCESS_test_song_DONE"
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    def fake_scp_to(_cfg, _local, _remote, timeout=300):
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    def fake_scp_from(_cfg, _remote, local, timeout=300):
        path = Path(local)
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"result")
        return subprocess.CompletedProcess([], 0, stdout="", stderr="")

    monkeypatch.setattr(P, "_ssh", fake_ssh)
    monkeypatch.setattr(P, "_scp_to", fake_scp_to)
    monkeypatch.setattr(P, "_scp_from", fake_scp_from)

    P._process("test_song", wav, media, "Test song")

    process_command = next(c for c in commands if "scripts/process_song.sh" in c)
    assert "AL_PY=/workspace/AgentLODGE/.venv/bin/python" in process_command
    assert "AGENTLODGE_DISTRIBUTED=1" in process_command
    assert (
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES="
        "jukebox.extract,lodge.generate,edge.generate"
    ) in process_command
    assert (
        "AGENTLODGE_WORKER_REGISTRY="
        "/workspace/maestro-workers/registry.json"
    ) in process_command
    assert "AGENTLODGE_SHARED_ROOT=/workspace" in process_command
    assert "AGENTLODGE_EARLY_LODGE_GENERATION=1" in process_command
    assert "AGENTLODGE_EARLY_LODGE_WAIT_SECONDS=120" in process_command
    assert "MAESTRO_FRONT_FACING=0" in process_command
    assert P.get_job("test_song")["status"] == "done"


def test_hosted_upload_uses_warm_full_render(tmp_path, monkeypatch):
    import server.processing as processing
    import server.rendering as rendering

    workspace = tmp_path / "workspace"
    repo = workspace / "AgentLODGE-lossless"
    (repo / "scripts").mkdir(parents=True)
    output = workspace / "upload_hosted_song"
    output.mkdir(parents=True)
    np.save(output / "base_motion.npy", np.zeros((12, 139), dtype=np.float32))
    np.save(output / "beats.npy", np.array([0, 6], dtype=np.int32))
    np.save(output / "beat_strengths.npy", np.ones(2, dtype=np.float32))
    np.save(output / "bank_hosted_song_lodge_seed0.npy", np.zeros((12, 139)))
    cfg = processing.PodConfig(
        host="127.0.0.1",
        ws=str(workspace),
        bank_k="1",
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    media = tmp_path / "media"
    rendered = {}

    monkeypatch.setattr(processing, "pod_config", lambda: cfg)
    monkeypatch.setattr(processing, "_co_located", lambda _cfg: True)
    monkeypatch.setattr(processing, "_pod_repo", lambda _cfg: str(repo))

    def fake_ssh(_cfg, command, timeout=60):
        stdout = "PROCESS_hosted_song_DONE\n" if "process_song.sh" in command else "ok\n"
        return subprocess.CompletedProcess([], 0, stdout=stdout, stderr="")

    def fake_render(sid, motion, media_dir, scope, *, audio_wav=None):
        rendered.update(
            sid=sid,
            frames=len(motion),
            scope=scope,
            audio_wav=audio_wav,
        )
        (Path(media_dir) / "edited.mp4").write_bytes(b"video")
        return True

    monkeypatch.setattr(processing, "_ssh", fake_ssh)
    monkeypatch.setattr(processing, "_run_pod", fake_ssh)
    monkeypatch.setattr(rendering, "_render_warm_local", fake_render)

    processing._process("hosted_song", source, media, 'Hosted "song"')

    job = processing.get_job("hosted_song")
    assert job["status"] != "error", job
    assert rendered == {
        "sid": "hosted_song",
        "frames": 12,
        "scope": "full",
        "audio_wav": str(source),
    }
    assert (media / "preview.mp4").read_bytes() == b"video"
    assert (media / "bank" / "bank_hosted_song_lodge_seed0.npy").exists()
    assert json.loads((media / "meta.json").read_text()) == {
        "name": 'Hosted "song"',
        "front_facing": False,
    }
    assert job["status"] == "done"


def test_front_facing_file_normalization_is_atomic(tmp_path, monkeypatch):
    import server.processing as processing
    from agentlodge.dance.transition import _matrix_to_sixd, _sixd_to_matrix

    yaw = np.linspace(-1.2, 1.4, 60, dtype=np.float32)
    cos, sin = np.cos(yaw), np.sin(yaw)
    roots = np.stack(
        [
            cos, -sin, np.zeros_like(cos),
            sin, cos, np.zeros_like(cos),
            np.zeros_like(cos), np.zeros_like(cos), np.ones_like(cos),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    motion = np.zeros((len(yaw), 139), dtype=np.float32)
    motion[:, 3:9] = _matrix_to_sixd(roots)
    motion[:, :3] = np.arange(len(yaw) * 3, dtype=np.float32).reshape(-1, 3)
    path = tmp_path / "motion.npy"
    np.save(path, motion)

    monkeypatch.setenv("MAESTRO_FRONT_FACING", "1")
    canonical_yaw = processing._normalize_front_facing_file(path)
    assert canonical_yaw is not None

    normalized = np.load(path)
    normalized_roots = _sixd_to_matrix(normalized[:, 3:9])
    normalized_yaw = np.arctan2(normalized_roots[:, 1, 0], normalized_roots[:, 0, 0])
    assert np.max(np.abs(normalized_yaw - normalized_yaw[0])) < 1e-5
    np.testing.assert_array_equal(normalized[:, :3], motion[:, :3])
    assert not list(tmp_path.glob("*.front-facing.tmp"))


def test_cached_live_take_uses_the_song_canonical_heading(tmp_path, monkeypatch):
    import server.processing as processing
    from agentlodge.dance.transition import _matrix_to_sixd, _sixd_to_matrix

    song_dir = tmp_path / "song"
    bank_dir = song_dir / "bank"
    bank_dir.mkdir(parents=True)
    (song_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "Song",
                "front_facing": True,
                "front_facing_yaw": 0.6,
            }
        ),
        encoding="utf-8",
    )
    yaw = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    cos, sin = np.cos(yaw), np.sin(yaw)
    roots = np.stack(
        [
            cos, -sin, np.zeros_like(cos),
            sin, cos, np.zeros_like(cos),
            np.zeros_like(cos), np.zeros_like(cos), np.ones_like(cos),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    motion = np.zeros((len(yaw), 139), dtype=np.float32)
    motion[:, 3:9] = _matrix_to_sixd(roots)
    provider = processing.PodTakeProvider(
        "song",
        bank_dir,
        cfg=processing.PodConfig(host=None),
    )
    cached = provider._local_path("lodge", 7, (30, 60))
    np.save(cached, motion)
    monkeypatch.setenv("MAESTRO_FRONT_FACING", "0")

    normalized = provider("lodge", 7, 30, 60)

    normalized_roots = _sixd_to_matrix(normalized[:, 3:9])
    normalized_yaw = np.arctan2(normalized_roots[:, 1, 0], normalized_roots[:, 0, 0])
    np.testing.assert_allclose(normalized_yaw, 0.6, atol=1e-5)


def test_uploaded_song_pipeline_scripts_are_packaged():
    root = Path(__file__).resolve().parents[1]
    required = {
        "preprocess_song.py",
        "make_song_bestofk.py",
        "dispatch_backbone_generation.py",
        "dispatch_bank_generation.py",
        "build_window_bank.py",
        "normalize_front_facing.py",
        "process_song.sh",
        "runpod_worker.py",
    }
    assert required <= {path.name for path in (root / "scripts").iterdir()}
    process_script = (root / "scripts" / "process_song.sh").read_text(encoding="utf-8")
    generation_script = (root / "scripts" / "make_song_bestofk.py").read_text(
        encoding="utf-8"
    )
    assert (
        'progress preprocess 25 "Extracting music features and beat timing"'
        in process_script
    )
    assert 'progress generation 40 "Generating LODGE and EDGE motion"' in process_script
    assert "MAESTRO_SUBPROGRESS generation" in generation_script


def test_background_bank_launch_requires_every_seed(monkeypatch):
    import server.processing as processing

    cfg = processing.PodConfig(host="pod", ws="/workspace", bank_k="4")
    commands = []

    def fake_ssh(_cfg, command, timeout=60):
        commands.append((command, timeout))
        return subprocess.CompletedProcess([], 0, stdout="BANK_PID=123\n", stderr="")

    monkeypatch.setattr(processing, "_ssh", fake_ssh)

    assert processing._launch_background_bank(
        cfg,
        "song_123",
        "/workspace/upload_song_123",
        "/workspace/AgentLODGE-lossless",
        "/root/al_venv/bin/python",
    )
    command, timeout = commands[0]
    assert "AGENTLODGE_BANK_K=4" in command
    assert 'wc -l)\" -ge 8' in command
    assert "setsid bash -c" in command
    assert "/workspace/upload_song_123/bank.done" in command
    assert timeout == 60


def test_background_bank_uses_resident_generation_worker(monkeypatch):
    import server.processing as processing

    cfg = processing.PodConfig(host="pod", ws="/workspace", bank_k="4")
    commands = []

    def fake_ssh(_cfg, command, timeout=60):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, stdout="BANK_PID=123\n", stderr="")

    monkeypatch.setenv("AGENTLODGE_DISTRIBUTED", "1")
    monkeypatch.setenv(
        "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
        "dance.generate,lodge.generate,edge.generate",
    )
    monkeypatch.setattr(processing, "_ssh", fake_ssh)

    assert processing._launch_background_bank(
        cfg,
        "song_123",
        "/workspace/upload_song_123",
        "/workspace/AgentLODGE-lossless",
        "/root/al_venv/bin/python",
    )
    assert "dispatch_bank_generation.py" in commands[0]


def test_start_render_deduplicates_active_song(monkeypatch, tmp_path):
    import server.rendering as rendering

    starts = []

    class FakeThread:
        def __init__(self, **kwargs):
            starts.append(kwargs)

        def start(self):
            return None

    rendering._RJOBS.clear()
    monkeypatch.setattr(rendering.threading, "Thread", FakeThread)
    motion = np.zeros((4, 139), dtype=np.float32)

    rendering.start_render("song", motion, tmp_path, scope="full")
    rendering.start_render("song", motion, tmp_path, scope="full")

    assert len(starts) == 1
    assert rendering.get_render_job("song")["status"] == "queued"


def test_upload_critical_path_builds_only_seed_zero():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "process_song.sh"
    ).read_text()

    assert 'AGENTLODGE_BANK_K=1 "$PY" "$BANK" "$SID"' in script
    assert 'echo "BANK_DEFERRED $K"' in script
    assert 'AGENTLODGE_SKIP_RENDER' in script
    assert "MAESTRO_TIMING" in script
    assert "generation_total" in script


def test_packaged_song_generator_writes_expected_outputs(tmp_path, monkeypatch):
    import scripts.make_song_bestofk as M

    sid = "smoke"
    wav = tmp_path / f"LODGE/data/finedance/music_wav/{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(tmp_path / f"lodge_fd_{sid}_feats.npy", np.zeros((600, 35), dtype=np.float32))
    np.save(tmp_path / f"edge{sid}_slices.npy", np.zeros((2, 4, 8), dtype=np.float32))
    lodge = np.zeros((600, 139), dtype=np.float32)
    edge = np.ones((600, 139), dtype=np.float32)
    story = np.full((600, 139), 2.0, dtype=np.float32)

    monkeypatch.setattr(M, "WORKSPACE", tmp_path)
    monkeypatch.setattr(
        M,
        "extract_song_metadata",
        lambda _wav: SimpleNamespace(
            duration_seconds=20.0,
            beat_frames=np.array([], dtype=np.int64),
            wav_path=wav,
        ),
    )
    monkeypatch.setattr(
        M,
        "_run_lodge_job",
        lambda *_args, **_kwargs: {"motion": lodge, "summary": "lodge", "error": None},
    )
    monkeypatch.setattr(
        M,
        "_run_edge_job",
        lambda *_args, **_kwargs: {"motion": edge, "summary": "edge", "error": None},
    )
    monkeypatch.setattr(M, "release_torch_memory", lambda: None)
    monkeypatch.setattr(M, "_use_parallel_execution", lambda: True)
    monkeypatch.setattr(M, "analyze_structure", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(M.librosa, "load", lambda *_args, **_kwargs: (np.zeros(10), 22050))
    monkeypatch.setattr(M, "extract_audio_descriptor", lambda *_args: object())
    planner_call = {}

    def fake_storyboard(*_args, **kwargs):
        planner_call["api_key"] = kwargs.get("api_key")
        return SimpleNamespace(to_dict=lambda: {"plans": []})

    monkeypatch.setenv("OPENAI_API_KEY", "runtime-test-key")
    monkeypatch.setattr(M, "author_storyboard", fake_storyboard)
    monkeypatch.setattr(
        M,
        "build_story_dance",
        lambda *_args, **_kwargs: SimpleNamespace(
            motion=story,
            reasoning="assembled",
            schedule=[(0, 600, "lodge", "body")],
            section_scores=[{"common_motion_ids": ["wave"]}],
        ),
    )
    beat_calls = []
    monkeypatch.setattr(
        M,
        "write_beat_artifacts",
        lambda value: beat_calls.append(value),
    )

    report = M.generate_song(sid)

    assert report["frames"] == 600
    assert beat_calls == [sid]
    assert np.array_equal(np.load(tmp_path / f"lodge_fd_{sid}_full.npy"), lodge)
    assert np.array_equal(np.load(tmp_path / f"edge_fd_{sid}_full.npy"), edge)
    assert np.array_equal(np.load(tmp_path / f"fd_{sid}_STORY_bestofk.npy"), story)
    assert planner_call["api_key"] == "runtime-test-key"
    assert report["storyboard"] == {"plans": []}
    assert report["section_scores"] == [{"common_motion_ids": ["wave"]}]


def test_packaged_song_generator_reads_secure_key_file(tmp_path, monkeypatch):
    import scripts.make_song_bestofk as M

    key_file = tmp_path / ".oai_key"
    key_file.write_text("file-key\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OAI_KEY_FILE", str(key_file))

    assert M._openai_api_key() == "file-key"


def test_packaged_song_generator_rejects_partial_required_best_of_k(
    tmp_path,
    monkeypatch,
):
    import scripts.make_song_bestofk as M

    sid = "partial"
    wav = tmp_path / f"LODGE/data/finedance/music_wav/{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(tmp_path / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35)))
    np.save(tmp_path / f"edge{sid}_slices.npy", np.zeros((2, 4, 8)))
    motion = np.zeros((60, 139), dtype=np.float32)

    monkeypatch.setattr(M, "WORKSPACE", tmp_path)
    monkeypatch.setattr(
        M,
        "extract_song_metadata",
        lambda _wav: SimpleNamespace(
            duration_seconds=20.0,
            beat_frames=np.array([0, 30], dtype=np.int64),
            wav_path=wav,
        ),
    )
    monkeypatch.setattr(M, "_use_parallel_execution", lambda: False)
    monkeypatch.setattr(
        M,
        "best_of_k_job",
        lambda *_args, **_kwargs: {
            "motion": motion,
            "error": None,
            "summary": "partial",
            "best_of_k_requested": 10,
            "best_of_k_completed": 9,
            "best_of_k_selected_seed": None,
            "best_of_k_fallback": "one candidate failed",
        },
    )
    monkeypatch.setenv("AGENTLODGE_BEST_OF_K", "10")
    monkeypatch.setenv("AGENTLODGE_REQUIRE_FULL_BEST_OF_K", "1")

    with pytest.raises(RuntimeError, match="full best-of-10 generation required"):
        M.generate_song(sid)


def test_lists_named_motions_from_the_shared_manifest(client):
    data = client.get("/api/motions").json()
    assert data["version"] == MotionBank().version
    assert len(data["motions"]) == 19
    clap = next(m for m in data["motions"] if m["id"] == "clap_single")
    assert clap["name"] == "Single clap"
    assert "clap" in clap["aliases"]
    assert clap["default_anchor"] == "beat"
    assert clap["default_direction"] == "auto"
    assert clap["directions"] == ["forward", "left", "right"]
    assert clap["minimum_seconds"] > 0
    assert clap["description"]
    assert clap["preview_url"] == "/api/motions/clap_single/preview"
    assert clap["preview_available"] is False
    assert clap["source"] and clap["license"] and clap["attribution"]
    assert next(m for m in data["motions"] if m["id"] == "wave")["name"] == "Hand wave"
    overhead = next(m for m in data["motions"] if m["id"] == "clap_overhead")
    assert overhead["composition"]["maximum_intensity"] == 0.5
    assert "90°" in next(m for m in data["motions"] if m["id"] == "turn_quarter")["name"]
    assert "180°" in next(m for m in data["motions"] if m["id"] == "turn_half")["name"]


def test_motion_preview_endpoint_is_explicit_when_missing_and_serves_generated_mp4(
    client,
    tmp_path,
    monkeypatch,
):
    import server.app as A

    missing = client.get("/api/motions/turn_half/preview")
    assert missing.status_code == 404
    assert "has not been generated" in missing.text
    assert client.get("/api/motions/not_a_motion/preview").status_code == 404

    previews = tmp_path / "generated-previews"
    previews.mkdir()
    payload = b"\x00\x00\x00\x18ftypmp42motion-example"
    (previews / "turn_half.mp4").write_bytes(payload)
    monkeypatch.setattr(A, "MOTION_PREVIEWS", previews)
    available = client.get("/api/motions").json()["motions"]
    assert next(m for m in available if m["id"] == "turn_half")["preview_available"]
    response = client.get("/api/motions/turn_half/preview")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("video/mp4")
    assert response.content == payload


def test_named_beat_action_uses_the_strongest_window_beat(client):
    client.post("/api/session/sng")
    result = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 0, "b_sec": 6, "instruction": "add a clap here"},
    ).json()["result"]
    report = next(step["motion_bank"] for step in result["log"] if "motion_bank" in step)
    assert report["anchor"] == "beat"
    assert report["event_frame"] == 120
    assert result["ok"]


@pytest.mark.parametrize("motion_id", [spec.id for spec in MotionBank().specs])
def test_every_named_motion_accepts_a_natural_command_on_a_collapsed_window(
    client,
    motion_id,
):
    client.post("/api/session/sng")
    spec = MotionBank().resolve(motion_id)
    result = client.post(
        "/api/session/sng/edit",
        json={
            "a_sec": 5.0,
            "b_sec": 5.0,
            "instruction": f"add {spec.name.lower()} here",
        },
    ).json()["result"]

    assert result["ok"], (motion_id, result["feedback"], result["log"])
    assert result["window"][1] - result["window"][0] == spec.frames + 16
    step = next(item for item in result["log"] if item["tool"] == "motion_bank")
    assert step["status"] == "applied"
    assert step["motion_bank"]["id"] == motion_id


def test_short_jump_window_keeps_enough_slack_to_land_on_a_fractional_beat(
    client,
    tmp_path,
):
    beats = np.arange(0.483, 300, 15.325, dtype=np.float32)
    np.save(tmp_path / "media" / "sng" / "beats.npy", beats)
    client.post("/api/session/sng")
    result = client.post(
        "/api/session/sng/edit",
        json={
            "a_sec": 5.0,
            "b_sec": 5.0,
            "instruction": "add a two-foot jump here",
        },
    ).json()["result"]

    report = next(step["motion_bank"] for step in result["log"] if "motion_bank" in step)
    assert result["ok"], result["trace"]["final"]["checks"]
    assert report["beat_error_frames"] <= 0.51


@pytest.mark.parametrize(
    "checkpoint,expected",
    [
        ({"interval": {"start_sec": 1.25, "end_sec": 2.75}}, (1.25, 2.75)),
        ({"edit": {"window": [60, 150]}}, (2.0, 5.0)),
        ({"edit": {"interval": {"start_frame": 90, "end_frame": 180}}}, (3.0, 6.0)),
        ({"edit": {"window_sec": [4.5, 7.5]}}, (4.5, 7.5)),
        ({"edit": {"a_sec": 9, "b_sec": 8}}, (8.0, 9.0)),
    ],
)
def test_checkpoint_interval_normalizes_current_and_legacy_shapes(checkpoint, expected):
    from server.app import _checkpoint_interval

    result = _checkpoint_interval(checkpoint)
    assert result is not None
    assert (result["start_sec"], result["end_sec"]) == expected
    assert all(np.isfinite(value) for value in result.values())


def test_checkpoint_interval_omits_invalid_values_instead_of_emitting_nan():
    from server.app import _checkpoint_interval

    assert _checkpoint_interval({}) is None
    assert _checkpoint_interval({"edit": {"window": [None, "undefined"]}}) is None
    assert _checkpoint_interval({"interval": {"start_sec": float("nan"), "end_sec": 2}}) is None


def test_edit_commits_and_moves_metrics(client):
    client.post("/api/session/sng")
    r = client.post("/api/session/sng/edit",
                    json={"a_sec": 3, "b_sec": 6, "instruction": "make this calmer"}).json()
    res = r["result"]
    assert res["ok"] and res["goal"]["objective"] == "agent"
    assert res["log"] and res["log"][0]["tool"] == "energy"      # agent chose the energy tool
    assert res["agent_summary"]
    assert res["metrics_after"]["energy"] < res["metrics_before"]["energy"]
    assert len(r["state"]["timeline"]) == 2 and r["state"]["can_undo"]
    assert r["state"]["timeline"][-1]["interval"] == {
        "start_sec": 3.0,
        "end_sec": 6.0,
        "start_frame": 90,
        "end_frame": 180,
    }


def test_edit_from_older_checkpoint_creates_visible_branch_and_preserves_history_controls(client):
    initial = client.post("/api/session/sng").json()
    root = initial["head"]
    first = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"},
    ).json()["state"]["head"]
    second = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"},
    ).json()["state"]["head"]
    branched = client.post(
        "/api/session/sng/edit",
        json={
            "a_sec": 3,
            "b_sec": 6,
            "instruction": "tighten to the beat",
            "from_id": first,
        },
    ).json()["state"]

    by_id = {item["id"]: item for item in branched["timeline"]}
    branch_head = branched["head"]
    assert by_id[branch_head]["parent_id"] == first
    assert set(by_id[first]["children"]) == {second, branch_head}
    assert by_id[second]["is_branch"] and by_id[branch_head]["is_branch"]
    assert by_id[branch_head]["lineage"] == [root, first, branch_head]
    assert branched["can_undo"] and not branched["can_redo"]

    undone = client.post("/api/session/sng/undo").json()
    assert undone["head"] == first and undone["can_redo"]
    redone = client.post("/api/session/sng/redo").json()
    assert redone["head"] == branch_head
    restored = client.post("/api/session/sng/restore", json={"ckpt_id": second}).json()
    assert restored["head"] == second
    assert client.post(
        "/api/session/sng/edit",
        json={"a_sec": 1, "b_sec": 2, "instruction": "calmer", "from_id": "missing"},
    ).status_code == 404


def test_undo_redo_restore_endpoints(client):
    client.post("/api/session/sng")
    st0 = client.post("/api/session/sng").json()
    root = st0["head"]
    client.post("/api/session/sng/edit",
                json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    u = client.post("/api/session/sng/undo").json()
    assert u["head"] == root and u["can_redo"]
    rd = client.post("/api/session/sng/redo").json()
    assert rd["head"] != root
    rs = client.post("/api/session/sng/restore", json={"ckpt_id": root}).json()
    assert rs["head"] == root


def test_websocket_streams_progress_then_final(client):
    client.post("/api/session/sng")
    with client.websocket_connect("/api/session/sng/edit_ws") as ws:
        ws.send_json({"a_sec": 3, "b_sec": 6, "instruction": "tighten to the beat"})
        n_prog, final = 0, None
        while True:
            ev = ws.receive_json()
            if ev["type"] == "progress":
                n_prog += 1
            elif ev["type"] == "final":
                final = ev
                break
            elif ev["type"] == "error":
                pytest.fail(f"ws error: {ev}")
    assert n_prog >= 1
    assert final["result"]["goal"]["objective"] == "agent"
    assert final["result"]["log"][0]["tool"] == "beat_align"    # agent snapped to the beat
    assert final["state"]["head"]


def test_websocket_honors_selected_branch_base(client):
    client.post("/api/session/sng")
    first = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"},
    ).json()["state"]["head"]
    later = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"},
    ).json()["state"]["head"]

    with client.websocket_connect("/api/session/sng/edit_ws") as ws:
        ws.send_json({
            "a_sec": 3,
            "b_sec": 6,
            "instruction": "tighten to the beat",
            "from_id": first,
        })
        while True:
            event = ws.receive_json()
            if event["type"] == "error":
                pytest.fail(event["message"])
            if event["type"] == "final":
                break

    by_id = {item["id"]: item for item in event["state"]["timeline"]}
    assert by_id[event["state"]["head"]]["parent_id"] == first
    assert set(by_id[first]["children"]) == {later, event["state"]["head"]}


def test_unknown_song_404(client):
    assert client.post("/api/session/nope").status_code == 404


def test_editor_review_actions_explain_the_user_flow(client):
    """The UI should expose review tasks, not ambiguous renderer implementation details."""
    from server.app import STATIC

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    rendered = client.get("/").text

    assert "Compare original &amp; edited" in html
    assert "Synchronized side by side" in js
    assert "Review full song" in html
    assert "Combines all 5 sections" in html
    assert 'id="prevSection"' in html
    assert 'id="nextSection"' in html
    assert 'id="fullReview"' in html
    assert 'id="fullReviewSections"' in html
    assert 'id="fullReviewVideo"' in html
    assert 'id="fullReviewRender"' in html
    assert 'rel="icon"' in html
    assert (STATIC / "favicon.svg").is_file()
    assert 'id="renderBtn"' not in html
    assert "startFullSongReview" in js
    assert "/api/full-song-review" in js
    assert "openReviewSection" in js
    assert 'startRender("window")' not in js
    assert 'id="motionSuggestions"' in html
    assert 'id="motionPicker"' in html
    assert '<details id="motionPicker" class="motion-picker">' in html
    assert '<details id="motionPicker" class="motion-picker" open>' not in html
    assert 'id="motionPickerSummary"' in html
    assert "<strong>Common motions</strong>" in html
    assert 'aria-label="Common motions"' in html
    assert "20 supported common motions" not in html
    assert "/api/motions" in js
    assert 'id="motionPreview"' in html
    assert 'id="motionPreviewVideo"' in html
    assert "openMotionPreview" in js
    assert "preview_available" in js
    assert "not a placeholder" in js
    assert "/static/editor_utils.js" in html
    assert "/static/editor_utils.js?v=" in rendered
    assert (STATIC / "editor_utils.js").is_file()
    assert "clap to the right" in js
    assert "insertion is unavailable until it has its own visual audit" in js
    assert "follows the dance flow" in js
    assert 'id="compareBackdrop"' in html
    assert 'id="cmpContent"' in html
    assert 'id="cmpBefore"' in html
    assert 'id="cmpAfter"' in html
    assert 'id="cmpChangeSummary"' in html
    assert 'id="cmpTime"' in html
    assert "Both clips show the same edit window" in html
    assert "/static/compare_highlight.js" not in html
    assert (STATIC / "compare_highlight.js").is_file()
    assert "setupCompareVideos" in js
    assert "compareAncestors" in js
    assert "updateCompareAvailability" in js
    assert "renderCompareHighlight" not in js
    assert 'id="activityCenter"' in html
    assert 'id="agentProgWrap"' in html
    assert "activityStart" in js
    assert "activityUpdate" in js
    assert "XMLHttpRequest" in js
    assert "xhr.upload.onprogress" in js
    assert "X-MAESTRO-Request-ID" in js
    assert "/browser-timing" in js
    assert "waitForMediaReady" in js
    assert "action applied, quality warning" in js
    assert 'id="branchState"' in html
    assert 'id="branchCancel"' in html
    assert "selectBranchBase" in js
    assert "from_id: fromId" in js
    assert "checkpointLineage" in js
    assert "formatCheckpointInterval" in js
    assert "timelineFraction" in js
    assert 'addEventListener("pointerdown"' in js
    assert "setPointerCapture" in js
    assert "ResizeObserver" in js
    assert "requestAnimationFrame" in js
    assert "overflow-y: auto; overscroll-behavior: contain" in css
    assert "flex: 0 0 clamp(320px, 52vh, 720px)" in css
    assert ".motion-picker[open] summary::before" in css
    assert "@media (max-width: 900px)" in css

    tour = js[js.index("const TOUR_STEPS"):js.index("let tourIdx")]
    assert "\\u2014" not in tour and "—" not in tour
    assert 'el: "motionPickerSummary"' in tour
    assert "watch the exact editor example" in tour
    assert "strongest beat in the selected window" in tour


def test_compare_highlight_kernel_marks_only_changed_pixels():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required to execute comparison highlight JavaScript")
    from server.app import STATIC

    script = (STATIC / "compare_highlight.js").as_posix()
    harness = f"""
const fs = require("fs");
const vm = require("vm");
vm.runInThisContext(fs.readFileSync({script!r}, "utf8"));
const fn = globalThis.MAESTRO_COMPARE_HIGHLIGHT.colorizeChangedPixels;
const before = Uint8ClampedArray.from([
  30, 30, 30, 255,
  50, 50, 50, 255,
]);
const after = Uint8ClampedArray.from([
  32, 31, 30, 255,
  210, 200, 190, 255,
]);
const overlay = new Uint8ClampedArray(8);
overlay.fill(255);
const changed = fn(before, after, overlay, 42);
if (changed !== 1) throw new Error(`expected one changed pixel, got ${{changed}}`);
if (overlay[3] !== 0) throw new Error("unchanged pixel was highlighted");
if (overlay[4] !== 24 || overlay[5] !== 211 || overlay[6] !== 238 || overlay[7] === 0) {{
  throw new Error("changed pixel did not receive the cyan highlight");
}}
"""
    subprocess.run([node, "-e", harness], check=True, capture_output=True, text=True)


def test_change_highlight_localizes_a_changed_right_arm(tmp_path):
    import server.rendering as rendering

    n = 12
    before = np.zeros((n, 22, 3), dtype=np.float32)
    after = before.copy()
    after[:, [14, 17, 19, 21], 0] = np.linspace(0.0, 0.24, n)[:, None]
    projected = np.zeros((n, 22, 3), dtype=np.float32)
    projected[..., 0] = 0.5
    projected[..., 1] = 0.5
    projected[..., 2] = 1.0
    projected[:, 14, :2] = (0.61, 0.58)
    projected[:, 17, :2] = (0.69, 0.55)
    projected[:, 19, :2] = (0.76, 0.49)
    projected[:, 21, :2] = (0.82, 0.44)

    before_path = tmp_path / "before.npz"
    after_path = tmp_path / "after.npz"
    rig_path = tmp_path / "after_rig.npz"
    np.savez(before_path, fk_joints=before)
    np.savez(after_path, fk_joints=after)
    np.savez(rig_path, projected=projected)

    highlight = rendering._build_change_highlight(
        str(before_path),
        str(after_path),
        str(rig_path),
    )

    assert highlight is not None
    assert "Right arm" in highlight["parts"]
    assert "Left leg" not in highlight["parts"]
    active = [marker for frame in highlight["frames"] for marker in frame]
    assert active
    assert {marker["part"] for marker in active} == {"right_arm"}
    assert all(0 <= marker["x"] <= 1 and 0 <= marker["y"] <= 1 for marker in active)


def test_basic_auth_middleware_guards_when_env_set():
    """With MAESTRO_AUTH_* set, requests without/with-wrong credentials get 401; correct creds pass."""
    import importlib
    import server.app as A
    A = importlib.reload(A)  # re-evaluate module-level auth wiring with the env in place
    from fastapi.testclient import TestClient
    import base64
    try:
        import os
        os.environ["MAESTRO_AUTH_USER"] = "maestro"
        os.environ["MAESTRO_AUTH_PASS"] = "s3cret"
        A2 = importlib.reload(A)
        c = TestClient(A2.app)
        assert c.get("/project/").status_code in (401, 404)  # 401 when guarded (docs may be absent -> still 401 first)
        assert c.get("/api/songs").status_code == 401
        good = base64.b64encode(b"maestro:s3cret").decode()
        assert c.get("/api/songs", headers={"Authorization": f"Basic {good}"}).status_code == 200
        bad = base64.b64encode(b"maestro:wrong").decode()
        assert c.get("/api/songs", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    finally:
        import os
        os.environ.pop("MAESTRO_AUTH_USER", None)
        os.environ.pop("MAESTRO_AUTH_PASS", None)
        importlib.reload(A)  # restore the open (unguarded) app for other tests


def test_compare_requires_an_edit_first(client):
    client.post("/api/session/sng")
    # no edit yet -> nothing to compare
    assert client.post("/api/session/sng/compare").status_code == 400


def test_full_render_passes_song_audio_to_accelerated_renderer(client, monkeypatch, tmp_path):
    import server.app as A
    import server.rendering as R

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"wav")
    captured = {}

    def fake_start(sid, motion, media_dir, *, scope="window", a=None, b=None, audio_wav=None):
        captured.update(
            sid=sid,
            frames=np.asarray(motion).shape[0],
            scope=scope,
            audio_wav=audio_wav,
        )
        R._set(sid, status="rendering", progress=10)

    monkeypatch.setattr(R, "start_render", fake_start)
    monkeypatch.setattr(A, "_song_wav", lambda _sid: str(audio))
    client.post("/api/session/sng")

    response = client.post("/api/session/sng/render", json={"scope": "full"})

    assert response.status_code == 200
    assert captured["sid"] == "sng"
    assert captured["frames"] == 300
    assert captured["scope"] == "full"
    assert captured["audio_wav"] == str(audio)


def test_full_song_review_composes_current_heads_and_tracks_staleness(
    client,
    monkeypatch,
    tmp_path,
):
    import server.app as A
    import server.rendering as R

    media = A.MEDIA
    sections = [
        ("review_intro", 0, 6, 1.0),
        ("review_outro", 6, 12, 2.0),
    ]
    for sid, start, end, value in sections:
        directory = media / sid
        directory.mkdir(parents=True)
        np.save(directory / "base_motion.npy", np.full((end - start, 139), value, np.float32))
        (directory / "preview.mp4").write_bytes(b"preview")
        (directory / "meta.json").write_text(
            json.dumps({"name": sid, "front_facing": True}),
            encoding="utf-8",
        )
    source = media / "review_full"
    source.mkdir()
    np.save(source / "base_motion.npy", np.zeros((12, 139), np.float32))
    (source / "preview.mp4").write_bytes(b"preview")
    (source / "review_full.wav").write_bytes(b"wav")
    (source / "meta.json").write_text(
        json.dumps({"name": "Review full", "front_facing": True}),
        encoding="utf-8",
    )
    config = tmp_path / "review.json"
    config.write_text(
        json.dumps(
            {
                "source": {
                    "sid": "review_full",
                    "name": "Review full",
                    "artist": "Test",
                    "fps": 30,
                    "frames": 12,
                },
                "segments": [
                    {
                        "sid": sid,
                        "name": sid,
                        "order": index,
                        "start_frame": start,
                        "end_frame": end,
                    }
                    for index, (sid, start, end, _value) in enumerate(sections, start=1)
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(A, "INTERVIEW_CATALOG", config)
    monkeypatch.setattr(A, "_full_song_review_signature", None)
    R._RJOBS.clear()

    intro = A._load_session("review_intro")
    edited_intro = intro.current_motion().copy()
    edited_intro[:, 0] = 9
    intro.store.commit(edited_intro, label="edited intro")

    captured = {}

    def fake_start(sid, motion, media_dir, *, scope="window", a=None, b=None, audio_wav=None):
        captured.update(
            sid=sid,
            motion=np.asarray(motion).copy(),
            media_dir=Path(media_dir),
            scope=scope,
            audio_wav=audio_wav,
        )
        R._set(sid, status="queued", progress=3, message="queued")

    monkeypatch.setattr(R, "start_render", fake_start)

    before = client.get("/api/full-song-review")
    assert before.status_code == 200
    before_payload = before.json()
    assert before_payload["edited_sections"] == 1
    assert before_payload["total_sections"] == 2
    assert [section["edited"] for section in before_payload["sections"]] == [True, False]

    started = client.post("/api/full-song-review")
    assert started.status_code == 200
    assert captured["sid"] == A.FULL_SONG_REVIEW_JOB
    assert captured["scope"] == "full"
    assert captured["audio_wav"] == str(source / "review_full.wav")
    assert captured["motion"].shape == (12, 139)
    np.testing.assert_array_equal(captured["motion"][:6], edited_intro)
    np.testing.assert_array_equal(
        captured["motion"][6:],
        A._load_session("review_outro").current_motion(),
    )

    review_dir = A._full_song_review_dir()
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "edited.mp4").write_bytes(b"full review")
    R._set(A.FULL_SONG_REVIEW_JOB, status="done", progress=100, video="edited.mp4")
    ready = client.get("/api/full-song-review").json()
    assert ready["render"]["ready"] is True
    assert ready["render"]["stale"] is False
    video = client.get(
        ready["render"]["video_url"],
    )
    assert video.status_code == 200
    assert video.content == b"full review"
    assert A._full_song_review_manifest_path().is_file()

    A._full_song_review_signature = None
    R._RJOBS.clear()
    restored = client.get("/api/full-song-review").json()
    assert restored["render"]["ready"] is True
    assert restored["render"]["status"] == "done"

    outro = A._load_session("review_outro")
    edited_outro = outro.current_motion().copy()
    edited_outro[:, 1] = 8
    outro.store.commit(edited_outro, label="edited outro")
    stale = client.get("/api/full-song-review").json()
    assert stale["edited_sections"] == 2
    assert stale["render"]["ready"] is False
    assert stale["render"]["stale"] is True


def test_compare_after_edit_renders_before_and_after(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None,
                   audio_wav=None, audio_start=0.0, audio_dur=0.0,
                   comparison_id=None, head_id=None, before_id=None):
        captured["sid"] = sid
        captured["before"] = np.asarray(before)
        captured["after"] = np.asarray(after)
        captured["metrics"] = metrics
        captured["audio"] = (audio_wav, audio_start, audio_dur)
        captured["context"] = (comparison_id, head_id, before_id)
        R._cset(
            sid,
            status="rendering",
            progress=10,
            comparison_id=comparison_id,
            head_id=head_id,
            before_id=before_id,
        )

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit",
                json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    j = client.post("/api/session/sng/compare").json()
    assert captured["sid"] == "sng"
    # window 3-6s at 30 fps -> 90 frames for BOTH the pre-edit and current clips
    assert captured["before"].shape[0] == 90 and captured["after"].shape[0] == 90
    # the before/after clips differ (the edit changed the window)
    assert not np.allclose(captured["before"], captured["after"])
    m = captured["metrics"]
    assert set(("before", "after", "window", "window_sec", "before_id", "before_label")) <= set(m)
    assert "bas" in m["before"] and "bas" in m["after"]
    assert m["window_sec"] == [3.0, 6.0]
    assert m["before_is_original"] is True
    assert captured["context"][0] == j["comparison_id"]
    assert captured["context"][1] == m["after_id"]
    assert captured["context"][2] == m["before_id"]
    # audio is requested for the window (start = a_sec, dur = window length)
    assert captured["audio"][1] == 3.0 and abs(captured["audio"][2] - 3.0) < 1e-6
    assert j["status"] in ("queued", "rendering", "idle")
    assert client.get("/api/session/sng/compare").json()["status"] in ("queued", "rendering", "idle")


def test_reset_clears_edit_history(client):
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit", json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    st = client.post("/api/session/sng/edit",
                     json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"}).json()["state"]
    assert len(st["timeline"]) >= 3 and st["can_undo"]
    r = client.post("/api/session/sng/reset").json()
    assert len(r["timeline"]) == 1                       # only the original root remains
    assert r["timeline"][0]["label"] == "original"
    assert not r["can_undo"] and not r["can_redo"]
    # a follow-up compare now fails (no edit to compare) -> the history really is empty
    assert client.post("/api/session/sng/compare").status_code == 400


def test_compare_from_id_selects_chosen_prior_version(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None,
                   audio_wav=None, audio_start=0.0, audio_dur=0.0,
                   comparison_id=None, head_id=None, before_id=None):
        captured["before"] = np.asarray(before)
        captured["metrics"] = metrics
        R._cset(
            sid,
            status="rendering",
            progress=10,
            comparison_id=comparison_id,
            head_id=head_id,
            before_id=before_id,
        )

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit", json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    st = client.post("/api/session/sng/edit",
                     json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"}).json()["state"]
    tl = st["timeline"]
    root = next(c for c in tl if c.get("parent_id") is None)   # the pre-any-edit base state
    client.post("/api/session/sng/compare", json={"from_id": root["id"]})
    # comparing against the ROOT -> "before" is the untouched base window (chosen version honoured)
    assert captured["metrics"]["before_id"] == root["id"]
    assert captured["metrics"].get("before_label") is not None
    assert captured["before"].shape[0] == 90


def test_compare_defaults_to_root_and_rejects_non_ancestor(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None, **context):
        captured["metrics"] = metrics
        R._cset(
            sid,
            status="rendering",
            progress=10,
            comparison_id=context["comparison_id"],
            head_id=context["head_id"],
            before_id=context["before_id"],
        )

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    initial = client.post("/api/session/sng").json()
    root = initial["head"]
    first = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"},
    ).json()["state"]["head"]
    sibling = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"},
    ).json()["state"]["head"]
    current = client.post(
        "/api/session/sng/edit",
        json={
            "a_sec": 3,
            "b_sec": 6,
            "instruction": "tighten to the beat",
            "from_id": first,
        },
    ).json()["state"]["head"]

    response = client.post("/api/session/sng/compare")
    assert response.status_code == 200
    assert captured["metrics"]["before_id"] == root
    assert captured["metrics"]["before_is_original"] is True
    assert client.post(
        "/api/session/sng/compare",
        json={"from_id": first},
    ).status_code == 409

    rejected = client.post(
        "/api/session/sng/compare",
        json={"from_id": sibling},
    )
    assert rejected.status_code == 400
    assert "ancestor" in rejected.text
    assert current != sibling


def test_compare_status_is_stale_after_head_changes(client, monkeypatch):
    import server.rendering as R

    def fake_start(sid, before, after, media_dir, *, metrics=None, **context):
        R._cset(
            sid,
            status="done",
            progress=100,
            metrics=metrics,
            before_video="cmp_before.mp4",
            after_video="cmp_after.mp4",
            comparison_id=context["comparison_id"],
            head_id=context["head_id"],
            before_id=context["before_id"],
        )

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    client.post("/api/session/sng")
    client.post(
        "/api/session/sng/edit",
        json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"},
    )
    assert client.post("/api/session/sng/compare").json()["status"] == "done"

    client.post("/api/session/sng/reset")
    stale = client.get("/api/session/sng/compare").json()
    assert stale["status"] == "stale"
    assert stale["before_video"] is None
    assert stale["after_video"] is None
