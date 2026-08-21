from pathlib import Path

import pytest


def _write_ppm(path: Path, color: tuple[int, int, int]) -> None:
    width, height = 4, 4
    pixels = bytes(color) * width * height
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def test_ffv1_decoded_rgb_hash_matches_source_sequence(tmp_path):
    from server.distributed.handlers import _ffmpeg_executable, _package_ffv1
    from server.distributed.render_contract import (
        _ffprobe_executable,
        inspect_ffv1_shard,
        source_sequence_rgb_sha256,
    )

    try:
        _ffmpeg_executable()
        _ffprobe_executable()
    except (FileNotFoundError, RuntimeError):
        pytest.skip("ffmpeg and ffprobe required")

    frames = tmp_path / "frames"
    frames.mkdir()
    _write_ppm(frames / "frame_0003.ppm", (255, 0, 0))
    _write_ppm(frames / "frame_0004.ppm", (0, 255, 0))
    _write_ppm(frames / "frame_0005.ppm", (0, 0, 255))
    shard = tmp_path / "shard.mkv"
    _package_ffv1(
        frames,
        shard,
        frame_start=3,
        frame_end=6,
        frame_format="ppm",
        fps=30,
    )

    source_hash = source_sequence_rgb_sha256(
        frames,
        frame_start=3,
        frame_end=6,
        frame_format="ppm",
        width=4,
        height=4,
        fps=30,
    )
    shard_hash = inspect_ffv1_shard(
        shard,
        frame_start=3,
        frame_end=6,
        width=4,
        height=4,
        fps=30,
    )

    assert source_hash == shard_hash["decoded_rgb_sha256"]


def test_decoded_rgb_difference_reports_pixel_error(tmp_path):
    from scripts.validate_render_equivalence import decoded_rgb_difference
    from server.distributed.handlers import _ffmpeg_executable

    try:
        _ffmpeg_executable()
    except RuntimeError:
        pytest.skip("ffmpeg required")

    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    _write_ppm(reference / "frame_0000.ppm", (0, 0, 0))
    _write_ppm(candidate / "frame_0000.ppm", (0, 0, 1))

    difference = decoded_rgb_difference(
        reference,
        candidate,
        frame_count=1,
        reference_sequence_format="ppm",
        candidate_sequence_format="ppm",
    )

    assert difference["channels"] == 48
    assert difference["changed_channels"] == 16
    assert difference["changed_channel_fraction"] == pytest.approx(1 / 3)
    assert difference["mean_absolute_error_8bit"] == pytest.approx(1 / 3)
    assert difference["max_absolute_error_8bit"] == 1


def test_render_difference_tolerance_is_strict():
    from scripts.validate_render_equivalence import (
        render_difference_within_tolerance,
    )

    measured = {
        "changed_channel_fraction": 0.0001,
        "mean_absolute_error_8bit": 0.0002,
        "max_absolute_error_8bit": 4,
    }

    assert render_difference_within_tolerance(
        measured,
        max_changed_channel_fraction=0.0005,
        max_mean_absolute_error=0.001,
        max_channel_error=8,
    )
    assert not render_difference_within_tolerance(
        measured,
        max_changed_channel_fraction=0.00005,
        max_mean_absolute_error=0.001,
        max_channel_error=8,
    )
