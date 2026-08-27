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


def test_write_beat_artifacts_preserves_exact_pipeline_contract(
    tmp_path,
    monkeypatch,
):
    import scripts.make_song_bestofk as generator

    sid = "song_123"
    wav = tmp_path / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    waveform = np.arange(8, dtype=np.float32)
    beat_frames = np.array([[0, 2, 4, 6]], dtype=np.int32)
    frame_times = np.array([0.25, 1.5, 2.0, 3.25], dtype=np.float64)
    onset = np.array(
        [1.0, 2.0, 3.0, 4.0, np.nan, np.inf, -np.inf],
        dtype=np.float64,
    )
    observed = {}

    def fake_load(path, *, sr, mono):
        observed["load"] = (path, sr, mono)
        return waveform, 22_050

    def fake_beat_track(*, y, sr, hop_length, units):
        observed["beat_track"] = (y, sr, hop_length, units)
        return 120.0, beat_frames

    def fake_frames_to_time(frames, *, sr, hop_length):
        observed["frames_to_time"] = (
            np.asarray(frames).copy(),
            sr,
            hop_length,
        )
        return frame_times

    def fake_onset_strength(*, y, sr, hop_length):
        observed["onset_strength"] = (y, sr, hop_length)
        return onset

    monkeypatch.setattr(generator.librosa, "load", fake_load)
    monkeypatch.setattr(
        generator.librosa.beat,
        "beat_track",
        fake_beat_track,
    )
    monkeypatch.setattr(
        generator.librosa,
        "frames_to_time",
        fake_frames_to_time,
    )
    monkeypatch.setattr(
        generator.librosa.onset,
        "onset_strength",
        fake_onset_strength,
    )

    returned_beats, returned_strengths = generator.write_beat_artifacts(
        sid,
        workspace=tmp_path,
    )

    saved_beats = np.load(tmp_path / f"beats_{sid}.npy")
    saved_strengths = np.load(tmp_path / f"beat_strengths_{sid}.npy")
    expected_beats = np.asarray(frame_times * 30.0, dtype=np.float32)
    expected_strengths = np.array([0.5, 1.0, 0.0, 0.0], dtype=np.float32)
    np.testing.assert_array_equal(saved_beats, expected_beats)
    np.testing.assert_array_equal(saved_strengths, expected_strengths)
    np.testing.assert_array_equal(returned_beats, expected_beats)
    np.testing.assert_array_equal(returned_strengths, expected_strengths)
    assert saved_beats.dtype == np.float32
    assert saved_strengths.dtype == np.float32
    assert observed["load"] == (str(wav), 22_050, True)
    assert observed["beat_track"][0] is waveform
    assert observed["beat_track"][1:] == (22_050, 512, "frames")
    frames, sample_rate, hop_length = observed["frames_to_time"]
    np.testing.assert_array_equal(
        frames,
        np.array([0, 2, 4, 6], dtype=np.int64),
    )
    assert frames.dtype == np.int64
    assert (sample_rate, hop_length) == (22_050, 512)
    assert observed["onset_strength"][0] is waveform
    assert observed["onset_strength"][1:] == (22_050, 512)


def test_penetration_resolver_reuses_fk_module_and_skeleton(
    tmp_path,
    monkeypatch,
):
    import scripts.resolve_penetration as cleanup

    calls = []

    class FakeSkeleton:
        def __init__(self, **kwargs):
            calls.append(("skeleton", kwargs))

    fake_module = SimpleNamespace(
        SMPLX_Skeleton=FakeSkeleton,
    )

    def load_module(lodge_root):
        calls.append(("module", Path(lodge_root)))
        return fake_module

    monkeypatch.setattr(cleanup, "_fk_module", load_module)
    resolver = cleanup.PenetrationResolver(workspace=tmp_path)

    resolver.preload()
    resolver.preload()

    assert calls == [
        ("module", tmp_path / "LODGE"),
        (
            "skeleton",
            {
                "device": "cpu",
                "batch": 1,
                "Jpath": str(tmp_path / "LODGE/data/smplx_neu_J_1.npy"),
            },
        ),
    ]


def test_dance_generation_handler_emits_exact_seed_zero_bank(
    tmp_path,
    monkeypatch,
):
    import scripts.build_window_bank as bank
    from server.distributed.handlers import DanceGenerationHandler

    shared = tmp_path / "shared"
    sid = "song_123"
    wav = shared / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(shared / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35)))
    np.save(shared / f"edge{sid}_slices.npy", np.zeros((2, 3)))
    lodge = np.full((4, 139), 2.0, dtype=np.float32)
    edge = np.full((4, 139), 3.0, dtype=np.float32)
    monkeypatch.setattr(bank, "to_lodge_zup", lambda motion: motion + 10.0)
    monkeypatch.setattr(bank, "to_edge_zup", lambda motion: motion + 20.0)
    handler = DanceGenerationHandler(shared_root=shared)

    def fake_generate(value):
        assert value == sid
        np.save(shared / f"lodge_fd_{sid}_full.npy", lodge)
        np.save(shared / f"edge_fd_{sid}_full.npy", edge)
        np.save(
            shared / f"fd_{sid}_STORY_bestofk.npy",
            np.zeros((4, 139), dtype=np.float32),
        )
        return {"frames": 4, "best_of_k": 1, "generation_workers": {}}

    handler._generate_song = fake_generate

    result = handler({"sid": sid})

    assert result["frames"] == 4
    np.testing.assert_array_equal(
        np.load(shared / f"bank_{sid}_lodge_seed0.npy"),
        lodge + 10.0,
    )
    np.testing.assert_array_equal(
        np.load(shared / f"bank_{sid}_edge_seed0.npy"),
        edge + 20.0,
    )


def test_window_bank_retimes_every_take_to_the_story_length(tmp_path, monkeypatch):
    import scripts.build_window_bank as bank

    sid = "long_song"
    np.save(tmp_path / f"lodge_fd_{sid}_full.npy", np.zeros((96, 139), dtype=np.float32))
    np.save(tmp_path / f"edge_fd_{sid}_full.npy", np.zeros((104, 139), dtype=np.float32))
    np.save(tmp_path / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35), dtype=np.float32))
    np.save(tmp_path / f"edge{sid}_slices.npy", np.zeros((2, 3), dtype=np.float32))
    np.save(tmp_path / f"fd_{sid}_STORY_bestofk.npy", np.zeros((100, 139), dtype=np.float32))
    monkeypatch.setattr(bank, "to_lodge_zup", lambda motion: motion)
    monkeypatch.setattr(bank, "to_edge_zup", lambda motion: motion)

    result = bank.build_bank(sid, 1, workspace=tmp_path)

    assert len(result["files"]) == 2
    assert np.load(tmp_path / f"bank_{sid}_lodge_seed0.npy").shape == (100, 139)
    assert np.load(tmp_path / f"bank_{sid}_edge_seed0.npy").shape == (100, 139)


def test_dance_generation_handler_cleans_before_bank_and_marks_success(
    tmp_path,
):
    from server.distributed.handlers import DanceGenerationHandler

    shared = tmp_path / "shared"
    sid = "song_123"
    wav = shared / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(shared / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35)))
    np.save(shared / f"edge{sid}_slices.npy", np.zeros((2, 3)))
    motion_path = shared / f"fd_{sid}_STORY_bestofk.npy"
    marker = shared / f"penetration_cleanup_{sid}.done"
    source = np.zeros((4, 139), dtype=np.float32)
    sequence = []
    observed = {}

    class FakeResolver:
        def preload(self):
            sequence.append("preload")

        def resolve(self, motion, **kwargs):
            sequence.append("cleanup")
            assert not marker.exists()
            np.testing.assert_array_equal(motion, source)
            observed.update(kwargs)
            return motion + 1.0

    handler = DanceGenerationHandler(
        shared_root=shared,
        penetration_resolver=FakeResolver(),
    )

    def fake_generate(value):
        sequence.append("generate")
        assert value == sid
        np.save(motion_path, source)
        return {"frames": 4, "best_of_k": 1, "generation_workers": {}}

    def fake_build(value, bank_k, **kwargs):
        sequence.append("bank")
        assert marker.is_file()
        np.testing.assert_array_equal(np.load(motion_path), source + 1.0)
        return {"sid": value, "bank_k": bank_k, "files": []}

    handler._generate_song = fake_generate
    handler._build_bank = fake_build
    handler.preload()

    result = handler({"sid": sid, "penetration_cleanup": True})

    assert result["frames"] == 4
    assert sequence == ["preload", "generate", "cleanup", "bank"]
    assert observed == {
        "radius": 0.12,
        "margin": 0.03,
        "max_deg": 30.0,
    }
    assert marker.read_text(encoding="utf-8") == (
        "radius=0.12 margin=0.03 max_deg=30\n"
    )


def test_dance_generation_handler_does_not_mark_failed_cleanup_save(
    tmp_path,
    monkeypatch,
):
    import server.distributed.handlers as handlers

    shared = tmp_path / "shared"
    sid = "song_123"
    wav = shared / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    np.save(shared / f"lodge_fd_{sid}_feats.npy", np.zeros((2, 35)))
    np.save(shared / f"edge{sid}_slices.npy", np.zeros((2, 3)))
    motion_path = shared / f"fd_{sid}_STORY_bestofk.npy"
    marker = shared / f"penetration_cleanup_{sid}.done"
    marker.write_text("stale\n", encoding="utf-8")
    sequence = []

    class FakeResolver:
        def resolve(self, motion, **kwargs):
            sequence.append("cleanup")
            assert not marker.exists()
            return motion + 1.0

    handler = handlers.DanceGenerationHandler(
        shared_root=shared,
        penetration_resolver=FakeResolver(),
    )

    def fake_generate(value):
        sequence.append("generate")
        np.save(motion_path, np.zeros((4, 139), dtype=np.float32))
        return {"frames": 4, "best_of_k": 1, "generation_workers": {}}

    def fail_save(path, value):
        sequence.append("save")
        raise OSError("disk full")

    handler._generate_song = fake_generate
    handler._build_bank = lambda *args, **kwargs: pytest.fail(
        "bank must not run after cleanup save failure"
    )
    monkeypatch.setattr(handlers, "_atomic_save_array", fail_save)

    with pytest.raises(OSError, match="disk full"):
        handler({"sid": sid, "penetration_cleanup": True})

    assert sequence == ["generate", "cleanup", "save"]
    assert not marker.exists()


def test_beat_artifact_handler_writes_exact_resident_contract(
    tmp_path,
    monkeypatch,
):
    from agentlodge.audio.preprocess import AUDIO_TIMING_CONTRACT_VERSION
    from server.distributed.handlers import BeatArtifactHandler

    shared = tmp_path / "shared"
    sid = "song_123"
    wav = shared / "LODGE/data/finedance/music_wav" / f"{sid}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    beats = np.array([0.0, 15.5, 30.0], dtype=np.float32)
    strengths = np.array([0.2, 1.0, 0.4], dtype=np.float32)
    metadata = SimpleNamespace(
        duration_seconds=2.0,
        bpm=120.0,
        beat_frames=np.array([0, 15, 30], dtype=np.int64),
    )
    handler = BeatArtifactHandler(shared_root=shared)
    monkeypatch.setattr(
        handler,
        "_analyze",
        lambda path: (metadata, beats, strengths),
    )

    result = handler(
        {
            "sid": sid,
            "wav": str(wav),
            "metadata_output": str(shared / f"audio_timing_{sid}.json"),
            "beats_output": str(shared / f"beats_{sid}.npy"),
            "strengths_output": str(shared / f"beat_strengths_{sid}.npy"),
        }
    )

    np.testing.assert_array_equal(np.load(result["beats_output"]), beats)
    np.testing.assert_array_equal(np.load(result["strengths_output"]), strengths)
    payload = json.loads(
        Path(result["metadata_output"]).read_text(encoding="utf-8")
    )
    assert payload["contract_version"] == AUDIO_TIMING_CONTRACT_VERSION
    assert payload["duration_seconds"] == 2.0
    assert payload["bpm"] == 120.0
    assert payload["beat_frames"] == [0, 15, 30]
    assert payload["source"]["bytes"] == 3
    assert result["metadata_beats"] == 3
    assert result["editor_beats"] == 3


def test_dispatch_beat_tracking_requests_resident_worker(
    tmp_path,
    monkeypatch,
):
    import scripts.dispatch_beat_tracking as dispatch

    workspace = tmp_path / "workspace"
    wav = workspace / "LODGE/data/finedance/music_wav/song_123.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"wav")
    observed = {}

    class FakeRegistry:
        @classmethod
        def from_env(cls):
            return cls()

        def require(self, capability, *, max_age_seconds):
            observed["require"] = (capability, max_age_seconds)
            return ["worker-1"]

    class FakeCoordinator:
        def __init__(self, registry, *, heartbeat_max_age):
            observed["coordinator"] = (registry, heartbeat_max_age)

        def submit(self, kind, payload, *, worker):
            observed["submit"] = (kind, payload, worker)
            return payload

        def wait(self, handle, *, timeout):
            observed["wait"] = (handle, timeout)
            Path(handle["metadata_output"]).write_text("{}", encoding="utf-8")
            np.save(handle["beats_output"], np.array([0.0], dtype=np.float32))
            np.save(handle["strengths_output"], np.array([1.0], dtype=np.float32))
            return SimpleNamespace(
                output={"metadata_beats": 1, "editor_beats": 1},
            )

    monkeypatch.setattr(dispatch, "WorkerRegistry", FakeRegistry)
    monkeypatch.setattr(dispatch, "FileTaskCoordinator", FakeCoordinator)
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.setattr(
        sys,
        "argv",
        ["dispatch_beat_tracking.py", "song_123"],
    )

    assert dispatch.main() == 0

    kind, payload, worker = observed["submit"]
    assert kind == "audio.beats"
    assert worker == "worker-1"
    assert payload["sid"] == "song_123"
    assert payload["wav"] == str(wav)
    assert payload["source"]["bytes"] == 3


def test_dispatch_song_generation_requests_resident_cleanup(
    tmp_path,
    monkeypatch,
):
    import scripts.dispatch_song_generation as dispatch

    workspace = tmp_path / "workspace"
    inputs = (
        workspace / "LODGE/data/finedance/music_wav/song_123.wav",
        workspace / "lodge_fd_song_123_feats.npy",
        workspace / "edgesong_123_slices.npy",
    )
    for path in inputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))
    observed = {}

    class FakeRegistry:
        @classmethod
        def from_env(cls):
            return cls()

        def require(self, capability, *, max_age_seconds):
            observed["require"] = (capability, max_age_seconds)
            return ["worker-1"]

    class FakeCoordinator:
        def __init__(self, registry, *, heartbeat_max_age):
            observed["coordinator"] = (registry, heartbeat_max_age)

        def submit(self, kind, payload, *, worker):
            observed["submit"] = (kind, payload, worker)
            return "handle"

        def wait(self, handle, *, timeout):
            observed["wait"] = (handle, timeout)
            return SimpleNamespace(
                output={"frames": 12, "best_of_k": 1},
            )

    monkeypatch.setattr(dispatch, "WorkerRegistry", FakeRegistry)
    monkeypatch.setattr(dispatch, "FileTaskCoordinator", FakeCoordinator)
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.setattr(
        sys,
        "argv",
        ["dispatch_song_generation.py", "song_123"],
    )

    assert dispatch.main() == 0

    kind, payload, worker = observed["submit"]
    assert kind == "dance.generate"
    assert worker == "worker-1"
    assert dispatch.GENERATION_CONTRACT_VERSION == (
        "dance-generation-v3-resident-penetration-cleanup"
    )
    assert payload["contract_version"] == dispatch.GENERATION_CONTRACT_VERSION
    assert payload["penetration_cleanup"] is True


def test_dispatch_backbone_generation_publishes_fingerprinted_marker(
    tmp_path,
    monkeypatch,
):
    import scripts.dispatch_backbone_generation as dispatch

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sid = "song_123"
    features = workspace / f"lodge_fd_{sid}_feats.npy"
    np.save(features, np.arange(12, dtype=np.float32).reshape(3, 4))
    observed = {}

    class FakeRegistry:
        @classmethod
        def from_env(cls):
            return cls()

        def require(self, capability, *, max_age_seconds):
            observed["require"] = (capability, max_age_seconds)
            return ["worker-1"]

    class FakeCoordinator:
        def __init__(self, registry, *, heartbeat_max_age):
            observed["coordinator"] = (registry, heartbeat_max_age)

        def submit(self, kind, payload, *, worker):
            observed["submit"] = (kind, payload, worker)
            output = Path(payload["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            np.save(output, np.full((8, 139), 3.0, dtype=np.float32))
            return "handle"

        def wait(self, handle, *, timeout):
            observed["wait"] = (handle, timeout)
            return SimpleNamespace(
                output={"summary": "warm LODGE"},
                worker_id="worker-1",
            )

    monkeypatch.setattr(dispatch, "WorkerRegistry", FakeRegistry)
    monkeypatch.setattr(dispatch, "FileTaskCoordinator", FakeCoordinator)
    monkeypatch.setenv("WORKSPACE", str(workspace))
    monkeypatch.setattr(
        sys,
        "argv",
        ["dispatch_backbone_generation.py", sid, "lodge"],
    )

    assert dispatch.main() == 0

    kind, payload, worker = observed["submit"]
    assert kind == "lodge.generate"
    assert worker == "worker-1"
    assert payload["seed"] is None
    marker = json.loads(
        (workspace / f"lodge_early_{sid}.json").read_text(encoding="utf-8")
    )
    assert marker["contract_version"] == dispatch.EARLY_LODGE_CONTRACT_VERSION
    assert marker["source"] == dispatch._fingerprint(features)
    assert marker["output"] == dispatch._fingerprint(Path(payload["output"]))
    assert marker["shape"] == [8, 139]
    assert not (workspace / f"lodge_early_{sid}.pending").exists()


def test_early_lodge_reuse_requires_exact_source_and_output(
    tmp_path,
    monkeypatch,
):
    import scripts.make_song_bestofk as generator

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sid = "song_123"
    features = workspace / f"lodge_fd_{sid}_feats.npy"
    output = workspace / f"lodge_early_{sid}.npy"
    np.save(features, np.arange(12, dtype=np.float32).reshape(3, 4))
    expected = np.full((8, 139), 3.0, dtype=np.float32)
    np.save(output, expected)
    marker = workspace / f"lodge_early_{sid}.json"
    marker.write_text(
        json.dumps(
            {
                "contract_version": generator.EARLY_LODGE_CONTRACT_VERSION,
                "sid": sid,
                "seed": None,
                "source": generator._file_fingerprint(features),
                "output": generator._file_fingerprint(output),
                "shape": [8, 139],
                "summary": "warm LODGE",
                "worker_id": "lodge-0",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(generator, "WORKSPACE", workspace)

    reused = generator._load_early_lodge_result(sid, features)

    assert reused is not None
    assert reused["summary"] == "warm LODGE"
    assert reused["worker_id"] == "lodge-0"
    np.testing.assert_array_equal(reused["motion"], expected)

    np.save(features, np.zeros((3, 4), dtype=np.float32))
    assert generator._load_early_lodge_result(sid, features) is None
    assert not marker.exists()
    assert output.exists()


def _bash_executable() -> str:
    candidates = [
        shutil.which("bash"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    pytest.skip("bash is required for the process_song.sh contract test")


@pytest.mark.parametrize(
    (
        "artifact_mode",
        "resident_cleanup",
        "expect_artifact_fallback",
        "cleanup_failure",
        "best_of_k",
        "expect_early_lodge",
    ),
    [
        ("all", True, False, False, "1", True),
        ("partial", False, True, False, "1", True),
        ("none", False, True, False, "1", True),
        ("stale", False, True, False, "1", True),
        ("all", False, False, True, "1", True),
        ("all", True, False, False, "2", False),
    ],
)
def test_process_song_skips_complete_warm_artifacts_and_falls_back_otherwise(
    tmp_path,
    artifact_mode,
    resident_cleanup,
    expect_artifact_fallback,
    cleanup_failure,
    best_of_k,
    expect_early_lodge,
):
    root = Path(__file__).resolve().parents[1]
    workspace = tmp_path / "workspace"
    agent_root = workspace / "AgentLODGE"
    python_path = agent_root / ".venv/bin/python"
    bash_env = tmp_path / "bash_env.sh"
    home = tmp_path / "home"
    call_log = tmp_path / "calls.log"
    workspace.mkdir()
    python_path.parent.mkdir(parents=True)
    home.mkdir()

    python_path.write_text(
        """#!/usr/bin/env bash
set -eu
target="$1"
shift
name="$(basename "$target")"
printf '%s\n' "$name" >> "$CALL_LOG"
if [ "$name" = "resolve_penetration.py" ]; then
  printf '%s\n' "$*" > "$RESOLVE_ARGS_LOG"
  if [ "$FAKE_RESOLVE_FAILURE" = "1" ]; then
    exit 9
  fi
fi
case "$name" in
  preprocess_song.py)
    ;;
  dispatch_backbone_generation.py)
    sid="$1"
    printf 'early motion' > "$WORKSPACE/lodge_early_${sid}.npy"
    printf '{"contract_version":"lodge-early-generation-v1"}\n' \
      > "$WORKSPACE/lodge_early_${sid}.json"
    ;;
  make_song_bestofk.py)
    sid="$1"
    printf 'motion' > "fd_${sid}_STORY_bestofk.npy"
    if [ "$FAKE_ARTIFACT_MODE" = "all" ]; then
      printf 'beats' > "beats_${sid}.npy"
      printf 'strengths' > "beat_strengths_${sid}.npy"
      printf 'lodge' > "bank_${sid}_lodge_seed0.npy"
      printf 'edge' > "bank_${sid}_edge_seed0.npy"
    elif [ "$FAKE_ARTIFACT_MODE" = "partial" ]; then
      printf 'beats' > "beats_${sid}.npy"
      printf 'lodge' > "bank_${sid}_lodge_seed0.npy"
    fi
    if [ "$FAKE_RESIDENT_CLEANUP" = "1" ]; then
      printf 'complete\n' > "$WORKSPACE/penetration_cleanup_${sid}.done"
    fi
    ;;
  build_window_bank.py)
    sid="$1"
    printf 'lodge' > "bank_${sid}_lodge_seed0.npy"
    printf 'edge' > "bank_${sid}_edge_seed0.npy"
    ;;
  -)
    sid="$1"
    printf 'beats' > "beats_${sid}.npy"
    printf 'strengths' > "beat_strengths_${sid}.npy"
    ;;
esac
""",
        encoding="utf-8",
        newline="\n",
    )
    python_path.chmod(0o755)
    bash_env.write_text(
        "stat() { printf '10288727721\\n'; }\n",
        encoding="utf-8",
        newline="\n",
    )
    for name in (
        "preprocess_song.py",
        "make_song_bestofk.py",
        "dispatch_backbone_generation.py",
        "build_window_bank.py",
        "resolve_penetration.py",
        "render_one_ybot.sh",
    ):
        (workspace / name).write_text("", encoding="utf-8")

    sid = "warm_song"
    if artifact_mode == "stale":
        for name, value in (
            (f"beats_{sid}.npy", "old beats"),
            (f"beat_strengths_{sid}.npy", "old strengths"),
            (f"bank_{sid}_lodge_seed0.npy", "old lodge"),
            (f"bank_{sid}_edge_seed0.npy", "old edge"),
        ):
            (workspace / name).write_text(value, encoding="utf-8")
        (workspace / f"penetration_cleanup_{sid}.done").write_text(
            "stale\n",
            encoding="utf-8",
        )
    resolve_args_log = tmp_path / "resolve_args.log"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "AGENTLODGE_ROOT": str(agent_root),
            "AGENTLODGE_SKIP_RENDER": "1",
            "AGENTLODGE_JUKEBOX_PRIOR": str(workspace / "prior.pth.tar"),
            "HOME": str(home),
            "CALL_LOG": str(call_log),
            "RESOLVE_ARGS_LOG": str(resolve_args_log),
            "FAKE_ARTIFACT_MODE": artifact_mode,
            "FAKE_RESIDENT_CLEANUP": "1" if resident_cleanup else "0",
            "FAKE_RESOLVE_FAILURE": "1" if cleanup_failure else "0",
            "AGENTLODGE_DISTRIBUTED": "1",
            "AGENTLODGE_DISTRIBUTED_CAPABILITIES": "lodge.generate",
            "AGENTLODGE_EARLY_LODGE_GENERATION": "1",
            "AGENTLODGE_BEST_OF_K": best_of_k,
            "BASH_ENV": str(bash_env),
        }
    )
    result = subprocess.run(
        [
            _bash_executable(),
            str(root / "scripts/process_song.sh"),
            sid,
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if cleanup_failure:
        assert result.returncode != 0
        assert (
            f"PROCESS_{sid}_FAILED: penetration cleanup failed"
            in result.stderr
        )
        assert resolve_args_log.read_text(encoding="utf-8").strip() == (
            f"fd_{sid}_STORY_bestofk.npy "
            f"fd_{sid}_STORY_bestofk.npy "
            "--radius 0.12 --margin 0.03 --max-deg 30"
        )
        return

    assert result.returncode == 0, result.stdout + result.stderr
    calls = call_log.read_text(encoding="utf-8").splitlines()
    assert calls.count("preprocess_song.py") == 2
    assert calls.count("dispatch_backbone_generation.py") == (
        1 if expect_early_lodge else 0
    )
    assert calls.count("make_song_bestofk.py") == 1
    assert ("build_window_bank.py" in calls) is expect_artifact_fallback
    assert ("-" in calls) is expect_artifact_fallback
    assert ("resolve_penetration.py" in calls) is (not resident_cleanup)
    assert (
        "seed-0 bank already exists; skipping standalone build" in result.stdout
    ) is (not expect_artifact_fallback)
    assert (
        "beat artifacts already exist; skipping standalone tracking"
        in result.stdout
    ) is (not expect_artifact_fallback)
    assert (
        "resident penetration cleanup already complete; "
        "skipping standalone cleanup" in result.stdout
    ) is resident_cleanup
    if resident_cleanup:
        assert not resolve_args_log.exists()
    else:
        assert resolve_args_log.read_text(encoding="utf-8").strip() == (
            f"fd_{sid}_STORY_bestofk.npy "
            f"fd_{sid}_STORY_bestofk.npy "
            "--radius 0.12 --margin 0.03 --max-deg 30"
        )
    if artifact_mode == "stale":
        assert not (workspace / f"penetration_cleanup_{sid}.done").exists()
    assert (workspace / f"upload_{sid}/base_motion.npy").is_file()
    assert (workspace / f"upload_{sid}/beats.npy").is_file()
    assert (workspace / f"upload_{sid}/beat_strengths.npy").is_file()
    assert (workspace / f"upload_{sid}/bank_{sid}_lodge_seed0.npy").is_file()
    assert (workspace / f"upload_{sid}/bank_{sid}_edge_seed0.npy").is_file()
