import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _filament_root(tmp_path: Path) -> Path:
    root = tmp_path / "maestro-filament-poc"
    (root / "filament/bin/assets/ibl/lightroom_14b").mkdir(parents=True)
    (root / "vulkan-selector").mkdir()
    for path, data in (
        (root / "filament_bench", b"binary"),
        (root / "ybot_visible_static.glb", b"static"),
        (
            root / "filament/bin/assets/ibl/lightroom_14b/lightroom_14b_ibl.ktx",
            b"ibl",
        ),
        (root / "vulkan-selector/libvulkan.so.1", b"selector"),
        (root / "vulkan-selector/libvulkan.real.so.1", b"vulkan"),
        (root / "nvidia_icd.json", b"{}"),
    ):
        path.write_bytes(data)
    (root / "filament_bench").chmod(0o755)
    return root


def test_filament_full_render_exports_shards_validates_and_caches(tmp_path, monkeypatch):
    import server.filament_render as filament_render
    import server.fk as fk
    import server.warm_render as warm_render

    root = _filament_root(tmp_path)
    media = tmp_path / "media"
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("AGENTLODGE_FILAMENT_ROOT", str(root))
    monkeypatch.setenv("AGENTLODGE_NVIDIA_VK_ICD", str(root / "nvidia_icd.json"))
    monkeypatch.setenv("AGENTLODGE_RENDER_TMP", str(tmp_path))
    monkeypatch.setattr(filament_render, "_gpu_indices", lambda: [0, 1])
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "ensure_pool", lambda **_kwargs: 1)
    monkeypatch.setattr(warm_render, "ready_daemons", lambda: [0])

    captured = {"exports": 0, "export_kwargs": None, "workers": [], "concat": None}

    def fake_save(_motion, path):
        Path(path).write_bytes(b"poses")

    def fake_export(_poses, _frames, **kwargs):
        captured["exports"] += 1
        captured["export_kwargs"] = kwargs
        Path(kwargs["export_glb"]).write_bytes(b"animated")
        return True

    shard_frames = {}

    class FakeProcess:
        def __init__(self, _command, *, stdout, stderr, env):
            self.returncode = None
            self.environment = env
            stdout.write(
                (
                    "MAESTRO_VK_SELECTOR selected Vulkan physical device index "
                    f"{env['MAESTRO_VK_DEVICE_INDEX']} of 2\n"
                ).encode()
            )
            stdout.flush()
            shard = Path(env["MAESTRO_FILAMENT_ASYNC_VIDEO_PATH"])
            shard.write_bytes(b"shard")
            shard_frames[shard.resolve()] = int(env["MAESTRO_FILAMENT_ASYNC_FRAMES"])
            captured["workers"].append(
                {
                    "gpu": int(env["MAESTRO_VK_DEVICE_INDEX"]),
                    "offset": int(env["MAESTRO_FILAMENT_FRAME_OFFSET"]),
                    "frames": int(env["MAESTRO_FILAMENT_ASYNC_FRAMES"]),
                }
            )

        def wait(self, timeout):
            self.returncode = 0
            return 0

        def poll(self):
            self.returncode = 0
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    def fake_probe(path):
        path = Path(path).resolve()
        frames = shard_frames.get(path, 10)
        return {
            "codec": "h264",
            "width": 1080,
            "height": 1080,
            "frame_rate": "30/1",
            "frames": frames,
            "duration": frames / 30,
            "bytes": path.stat().st_size,
            "sha256": "a" * 64,
        }

    def fake_concat(shards, output, *, audio_wav, timeout):
        captured["concat"] = {
            "shards": list(shards),
            "audio_wav": audio_wav,
            "timeout": timeout,
        }
        Path(output).write_bytes(b"video")

    monkeypatch.setattr(fk, "save_poses_npz", fake_save)
    monkeypatch.setattr(warm_render, "warm_render", fake_export)
    monkeypatch.setattr(filament_render.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(filament_render, "_probe_video", fake_probe)
    monkeypatch.setattr(filament_render, "_concat_shards", fake_concat)

    updates = []
    motion = np.zeros((10, 139), dtype=np.float32)
    assert filament_render.render_full_motion(
        "song",
        motion,
        media,
        audio_wav=str(audio),
        update=lambda **fields: updates.append(fields),
    )
    assert captured["workers"] == [
        {"gpu": 0, "offset": 0, "frames": 5},
        {"gpu": 1, "offset": 5, "frames": 5},
    ]
    assert captured["concat"]["audio_wav"] == str(audio)
    assert captured["export_kwargs"]["frame_format"] == "tga"
    assert captured["export_kwargs"]["fast"] is False
    assert captured["export_kwargs"]["foot_grounding"] is False
    assert (media / "edited.mp4").read_bytes() == b"video"
    report_path = media / "filament_render_report.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text())
    assert set(report) == {
        "backend_version",
        "frames",
        "gpu_indices",
        "worker_gpu_indices",
        "workers_per_gpu",
        "settings",
        "pose_serialization_fk_seconds",
        "warm_pool_ensure_attestation_seconds",
        "blender_request_export_seconds",
        "worker_process_launch_seconds",
        "concurrent_worker_wait_seconds",
        "shard_probe_validation_seconds",
        "concat_mux_seconds",
        "final_probe_cache_publication_seconds",
        "export_seconds",
        "concat_seconds",
        "total_seconds",
        "animated_glb_bytes",
        "animated_glb_sha256",
        "workers",
        "final",
    }
    timing_fields = {
        "pose_serialization_fk_seconds",
        "warm_pool_ensure_attestation_seconds",
        "blender_request_export_seconds",
        "worker_process_launch_seconds",
        "concurrent_worker_wait_seconds",
        "shard_probe_validation_seconds",
        "concat_mux_seconds",
        "final_probe_cache_publication_seconds",
        "export_seconds",
        "concat_seconds",
        "total_seconds",
    }
    assert all(
        isinstance(report[field], float)
        and math.isfinite(report[field])
        and report[field] >= 0.0
        for field in timing_fields
    )
    assert report["blender_request_export_seconds"] == report["export_seconds"]
    assert report["concat_mux_seconds"] == report["concat_seconds"]
    assert report["worker_gpu_indices"] == [0, 1]
    assert report["workers_per_gpu"] == 1
    assert all(
        isinstance(worker["seconds"], float)
        and math.isfinite(worker["seconds"])
        and worker["seconds"] >= 0.0
        for worker in report["workers"]
    )
    assert updates[-1]["rendered_frames"] == 10

    assert filament_render.render_full_motion(
        "song",
        motion,
        media,
        audio_wav=str(audio),
    )
    assert captured["exports"] == 1
    assert len(captured["workers"]) == 2


def test_filament_worker_assignments_scale_per_gpu(monkeypatch):
    import server.filament_render as filament_render

    monkeypatch.setenv("AGENTLODGE_FILAMENT_WORKERS_PER_GPU", "3")
    assert filament_render._worker_assignments([0, 1]) == [
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
    ]
    assert filament_render._ranges(10, 6) == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
        (8, 9),
        (9, 10),
    ]


def test_filament_probe_uses_container_frame_count_without_decoding(
    tmp_path, monkeypatch
):
    import server.filament_render as filament_render

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1080,
                            "height": 1080,
                            "avg_frame_rate": "30/1",
                            "nb_frames": "5400",
                        }
                    ],
                    "format": {"duration": "180.0"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(filament_render.subprocess, "run", fake_run)
    assert filament_render._probe_video(video)["frames"] == 5400
    assert len(commands) == 1
    assert "-count_frames" not in commands[0]


def test_filament_probe_decodes_when_container_frame_count_is_missing(
    tmp_path, monkeypatch
):
    import server.filament_render as filament_render

    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        frame_fields = (
            {"nb_read_frames": "5400"}
            if "-count_frames" in command
            else {"nb_frames": "N/A"}
        )
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_name": "h264",
                            "width": 1080,
                            "height": 1080,
                            "avg_frame_rate": "30/1",
                            **frame_fields,
                        }
                    ],
                    "format": {"duration": "180.0"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(filament_render.subprocess, "run", fake_run)
    assert filament_render._probe_video(video)["frames"] == 5400
    assert len(commands) == 2
    assert "-count_frames" not in commands[0]
    assert "-count_frames" in commands[1]


def test_filament_validation_accepts_mux_duration_rounding(tmp_path):
    import server.filament_render as filament_render

    video = tmp_path / "video.mp4"
    probe = {
        "codec": "h264",
        "width": 1080,
        "height": 1080,
        "frame_rate": "81146880/2704927",
        "nominal_frame_rate": "30/1",
        "frames": 5283,
        "duration": 176.103,
    }

    filament_render._validate_probe(video, probe, expected_frames=5283)

    probe["frame_rate"] = "299/10"
    with pytest.raises(RuntimeError, match="validation failed"):
        filament_render._validate_probe(video, probe, expected_frames=5283)


def test_full_render_feature_flag_routes_to_filament(tmp_path, monkeypatch):
    import server.filament_render as filament_render
    import server.rendering as rendering

    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_BACKEND", "filament")
    captured = {}

    def fake_render(sid, motion, media_dir, **kwargs):
        captured.update(
            sid=sid,
            frames=motion.shape[0],
            media_dir=media_dir,
            audio_wav=kwargs["audio_wav"],
        )
        return True

    monkeypatch.setattr(filament_render, "render_full_motion", fake_render)
    assert rendering._render_warm_local(
        "filament",
        np.zeros((8, 139), dtype=np.float32),
        tmp_path,
        "full",
        audio_wav="song.wav",
    )
    assert captured == {
        "sid": "filament",
        "frames": 8,
        "media_dir": tmp_path,
        "audio_wav": "song.wav",
    }


def test_filament_failure_is_fail_closed(tmp_path, monkeypatch):
    import server.rendering as rendering

    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_BACKEND", "filament")
    monkeypatch.setattr(
        rendering,
        "pod_config",
        lambda: SimpleNamespace(host="should-not-be-used"),
    )
    monkeypatch.setattr(
        rendering,
        "_render_warm_local",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    rendering._RJOBS["filament-fail"] = {"started": 0.0}
    rendering._render(
        "filament-fail",
        np.zeros((8, 139), dtype=np.float32),
        tmp_path,
        "full",
        None,
        None,
    )
    job = rendering.get_render_job("filament-fail")
    assert job["status"] == "error"
    assert job["message"] == "Filament render failed: boom"
