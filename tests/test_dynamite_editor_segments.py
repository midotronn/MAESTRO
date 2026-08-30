from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.build_dynamite_editor_segments as builder


ROOT = Path(__file__).resolve().parents[1]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _hash(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = 300) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def _fixture(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    media = tmp_path / "media"
    source = media / "dynamite_fixture"
    bank = source / "bank"
    bank.mkdir(parents=True)

    base = np.arange(12 * 4, dtype=np.float32).reshape(12, 4)
    np.save(source / "base_motion.npy", base)
    beats = np.array([0.0, 3.5, 4.0, 7.99, 8.0, 11.5, 12.0], dtype=np.float32)
    strengths = np.arange(beats.size, dtype=np.float32) / 10
    np.save(source / "beats.npy", beats)
    np.save(source / "beat_strengths.npy", strengths)
    np.save(bank / "bank_dynamite_fixture_lodge_seed0.npy", base + 100)
    np.save(bank / "bank_dynamite_fixture_edge_seed1.npy", base + 200)
    np.save(bank / "diagnostic.npy", np.zeros((2, 4), dtype=np.float32))
    (source / "preview.mp4").write_bytes(b"fixture preview")
    (source / "meta.json").write_text(
        json.dumps(
            {
                "name": "Full Dynamite",
                "front_facing": True,
                "front_facing_yaw": 0.625,
                "interview": True,
            }
        ),
        encoding="utf-8",
    )
    pcm = np.arange(1205, dtype=np.int16)
    audio = source / "dynamite_fixture.wav"
    _write_wav(audio, pcm)

    segments = [
        {
            "sid": "fixture_intro",
            "name": "Fixture — Intro",
            "order": 1,
            "start_frame": 0,
            "end_frame": 4,
            "roles": ["intro"],
            "boundary_basis": "fixture boundary",
            "rationale": "fixture rationale",
        },
        {
            "sid": "fixture_chorus",
            "name": "Fixture — Chorus",
            "order": 2,
            "start_frame": 4,
            "end_frame": 8,
            "roles": ["chorus"],
            "boundary_basis": "fixture boundary",
            "rationale": "fixture rationale",
        },
        {
            "sid": "fixture_outro",
            "name": "Fixture — Outro",
            "order": 3,
            "start_frame": 8,
            "end_frame": 12,
            "roles": ["outro"],
            "boundary_basis": "fixture boundary",
            "rationale": "fixture rationale",
        },
    ]
    config = {
        "schema_version": 1,
        "source": {
            "sid": source.name,
            "name": "Dynamite",
            "artist": "BTS",
            "fps": 3,
            "frames": 12,
            "audio_sample_rate": 300,
            "audio_channels": 1,
            "audio_sample_width_bytes": 2,
            "audio_sha256": _hash(audio),
            "require_front_facing": True,
        },
        "duration_bounds_seconds": {"minimum": 1, "maximum": 2},
        "interview_policy": {"full_song_visible": False, "reason": "fixture"},
        "segments": segments,
        "catalog": [
            *[
                {
                    "sid": segment["sid"],
                    "name": segment["name"],
                    "order": segment["order"],
                    "kind": "segment",
                }
                for segment in segments
            ],
            {
                "sid": "other_approved",
                "name": "Other approved",
                "order": 4,
                "kind": "approved_full_song",
            },
        ],
    }
    config_path = tmp_path / "segments.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return source, config_path, base, pcm


def _fake_media_tools(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    wrong_frames_for: str | None = None,
) -> list[dict]:
    calls: list[dict] = []

    def fake_slice(
        source_video,
        output_video,
        audio_path,
        *,
        start_frame,
        end_frame,
        fps,
        audio_samples,
        sample_rate,
        channels,
        ffmpeg,
    ):
        payload = {
            "source": str(source_video),
            "audio": str(audio_path),
            "start_frame": start_frame,
            "end_frame": end_frame,
            "video_frames": end_frame - start_frame,
            "fps": fps,
            "audio_samples": audio_samples,
            "sample_rate": sample_rate,
            "channels": channels,
        }
        calls.append(payload)
        output_video.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def fake_probe(path, _ffprobe):
        if Path(path) == source / "preview.mp4":
            return {
                "video_frames": 12,
                "fps_numerator": 3,
                "fps_denominator": 1,
                "has_audio": False,
                "audio_sample_rate": None,
                "audio_channels": None,
                "audio_duration_seconds": None,
                "video_duration_seconds": 4.0,
                "format_duration_seconds": 4.0,
            }
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        frames = payload["video_frames"]
        if wrong_frames_for and Path(path).parent.name == wrong_frames_for:
            frames -= 1
        return {
            "video_frames": frames,
            "fps_numerator": payload["fps"],
            "fps_denominator": 1,
            "has_audio": True,
            "audio_sample_rate": payload["sample_rate"],
            "audio_channels": payload["channels"],
            "audio_duration_seconds": (
                payload["video_frames"] / payload["fps"] - 7 / payload["sample_rate"]
            ),
            "video_duration_seconds": payload["video_frames"] / payload["fps"],
            "format_duration_seconds": payload["video_frames"] / payload["fps"],
        }

    def fake_audio_samples(path, _ffmpeg, *, sample_rate, channels):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        assert payload["sample_rate"] == sample_rate
        assert payload["channels"] == channels
        return payload["audio_samples"]

    monkeypatch.setattr(builder, "_slice_preview", fake_slice)
    monkeypatch.setattr(builder, "_probe_preview", fake_probe)
    monkeypatch.setattr(builder, "_count_decoded_audio_samples", fake_audio_samples)
    return calls


def test_build_segments_slices_every_asset_and_is_idempotent(tmp_path, monkeypatch):
    source, config_path, base, pcm = _fixture(tmp_path)
    calls = _fake_media_tools(monkeypatch, source)
    before = _tree_hashes(source)

    report = builder.build_segments(source, config_path=config_path)

    assert [item["sid"] for item in report["built"]] == [
        "fixture_intro",
        "fixture_chorus",
        "fixture_outro",
    ]
    assert report["source_unchanged"] is True
    assert _tree_hashes(source) == before
    assert len(calls) == 3

    expected_beats = {
        "fixture_intro": np.array([0.0, 3.5], dtype=np.float32),
        "fixture_chorus": np.array([0.0, 3.99], dtype=np.float32),
        "fixture_outro": np.array([0.0, 3.5], dtype=np.float32),
    }
    for index, sid in enumerate(expected_beats):
        output = source.parent / sid
        start = index * 4
        end = start + 4
        np.testing.assert_array_equal(np.load(output / "base_motion.npy"), base[start:end])
        np.testing.assert_allclose(np.load(output / "beats.npy"), expected_beats[sid])
        np.testing.assert_array_equal(
            np.load(output / "beat_strengths.npy"),
            np.arange(7, dtype=np.float32)[
                np.array([0, 1]) if index == 0 else np.array([2, 3]) if index == 1 else np.array([4, 5])
            ]
            / 10,
        )
        np.testing.assert_array_equal(
            np.load(output / "bank" / f"bank_{sid}_lodge_seed0.npy"),
            base[start:end] + 100,
        )
        np.testing.assert_array_equal(
            np.load(output / "bank" / f"bank_{sid}_edge_seed1.npy"),
            base[start:end] + 200,
        )
        assert not (output / "bank" / "diagnostic.npy").exists()

        with wave.open(str(output / f"{sid}.wav"), "rb") as audio:
            assert audio.getnframes() == 400
            segment_pcm = np.frombuffer(audio.readframes(400), dtype="<i2")
        np.testing.assert_array_equal(segment_pcm, pcm[start * 100 : end * 100])

        metadata = json.loads((output / "meta.json").read_text(encoding="utf-8"))
        assert metadata["front_facing"] is True
        assert metadata["front_facing_yaw"] == 0.625
        assert metadata["order"] == index + 1
        manifest = json.loads(
            (output / builder.MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["segment"]["frames"] == 4
        assert manifest["segment"]["audio_samples"] == 400
        assert manifest["preview"]["video_frames"] == 4
        assert manifest["preview"]["decoded_audio_samples"] == 400
        assert manifest["preview"]["decoded_audio_padding_samples"] == 0
        assert manifest["skipped_incompatible_bank_arrays"] == [
            {
                "dtype": "<f4",
                "path": "diagnostic.npy",
                "shape": [2, 4],
            }
        ]
        for relative, detail in manifest["files"].items():
            assert detail["sha256"] == _hash(output / relative)

    output_hashes = {
        sid: _tree_hashes(source.parent / sid) for sid in expected_beats
    }
    second = builder.build_segments(source, config_path=config_path)
    assert second["built"] == []
    assert second["reused"] == list(expected_beats)
    assert len(calls) == 3
    assert {
        sid: _tree_hashes(source.parent / sid) for sid in expected_beats
    } == output_hashes

    np.save(source.parent / "fixture_chorus" / "base_motion.npy", np.zeros((4, 4)))
    repaired = builder.build_segments(source, config_path=config_path)
    assert [item["sid"] for item in repaired["built"]] == ["fixture_chorus"]
    assert len(calls) == 4
    np.testing.assert_array_equal(
        np.load(source.parent / "fixture_chorus" / "base_motion.npy"),
        base[4:8],
    )
    assert _tree_hashes(source) == before


def test_bad_preview_validation_publishes_nothing(tmp_path, monkeypatch):
    source, config_path, _, _ = _fixture(tmp_path)
    before = _tree_hashes(source)
    _fake_media_tools(monkeypatch, source, wrong_frames_for="fixture_chorus")

    with pytest.raises(builder.BuildError, match="fixture_chorus preview has 3 frames"):
        builder.build_segments(source, config_path=config_path)

    assert _tree_hashes(source) == before
    assert not any((source.parent / sid).exists() for sid in (
        "fixture_intro",
        "fixture_chorus",
        "fixture_outro",
    ))
    assert not list(source.parent.glob(".dynamite_fixture.segments-*.staging"))


def test_production_config_tracks_approved_story_and_dynamite_only_catalog():
    config = builder._load_config(builder.DEFAULT_CONFIG)
    story_path = (
        ROOT
        / "experiments"
        / "user_study"
        / "stimuli"
        / "story-reports"
        / "fd_dynamite_43089_STORY_bestofk.json"
    )
    story = json.loads(story_path.read_text(encoding="utf-8"))
    segments = config["segments"]

    assert [(segment["start_frame"], segment["end_frame"]) for segment in segments] == [
        (0, 1238),
        (1238, 2584),
        (2584, 3892),
        (3892, 4823),
        (4823, 6173),
    ]
    assert all(
        25 <= (segment["end_frame"] - segment["start_frame"]) / 30 <= 45
        for segment in segments
    )
    assert segments[0]["end_frame"] == story["schedule"][1][1]
    assert (
        segments[1]["end_frame"]
        == story["section_scores"][2]["common_motions"][0]["global_action_range"][1]
    )
    assert segments[2]["end_frame"] == story["schedule"][2][1]
    side_step_end = story["section_scores"][5]["common_motions"][0][
        "global_action_range"
    ][1]
    assert side_step_end < segments[3]["end_frame"]
    assert (segments[4]["end_frame"] - segments[4]["start_frame"]) / 30 == 45

    catalog = sorted(config["catalog"], key=lambda entry: entry["order"])
    assert [entry["sid"] for entry in catalog] == [segment["sid"] for segment in segments]
    assert all(entry["kind"] == "segment" for entry in catalog)
    assert config["interview_policy"]["full_song_visible"] is False
    assert config["source"]["sid"] not in {entry["sid"] for entry in catalog}


def test_ffmpeg_command_uses_frame_and_sample_exact_filters(tmp_path):
    command = builder._ffmpeg_preview_command(
        "ffmpeg",
        tmp_path / "source.mp4",
        tmp_path / "slice.wav",
        tmp_path / "preview.mp4",
        start_frame=123,
        end_frame=456,
        fps=30,
        audio_samples=244755,
        sample_rate=22050,
        channels=1,
    )

    assert "trim=start_frame=123:end_frame=456,setpts=N/(30*TB)" in command
    assert "atrim=start_sample=0:end_sample=244755,asetpts=PTS-STARTPTS" in command
    assert "-frames:v" not in command
    assert command[command.index("-t") + 1] == "11.100000000"
    assert command[command.index("-ar") + 1] == "22050"
    assert command[-1].endswith("preview.mp4")


def test_probe_prefers_nominal_cfr_rate_over_timestamp_derived_average(
    tmp_path,
    monkeypatch,
):
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "r_frame_rate": "30/1",
                "avg_frame_rate": "15802880/526767",
                "nb_frames": "6173",
                "nb_read_frames": "6173",
                "duration": "205.768359",
                "duration_ts": "526767",
                "time_base": "1/2560",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "sample_rate": "22050",
                "channels": 1,
                "duration": "205.775238",
            },
        ],
        "format": {"duration": "205.775238"},
    }
    monkeypatch.setattr(
        builder.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )

    probe = builder._probe_preview(tmp_path / "preview.mp4", "ffprobe")

    assert (probe["fps_numerator"], probe["fps_denominator"]) == (30, 1)
    assert probe["fps_source"] == "r_frame_rate"
    assert (
        probe["average_fps_numerator"],
        probe["average_fps_denominator"],
    ) == (15802880, 526767)
    builder._validate_video_timing(
        probe,
        expected_frames=6173,
        expected_fps=30,
        label="source preview",
    )
