from pathlib import Path

import pytest


def _write_ppm(path: Path, color: tuple[int, int, int]) -> None:
    width, height = 4, 4
    pixels = bytes(color) * width * height
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + pixels)


def test_ffv1_decoded_rgb_hash_matches_source_sequence(tmp_path):
    from scripts.validate_render_equivalence import decoded_rgb_hash
    from server.distributed.handlers import _ffmpeg_executable, _package_ffv1

    try:
        _ffmpeg_executable()
    except RuntimeError:
        pytest.skip("ffmpeg required")

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

    source_hash = decoded_rgb_hash(
        frames,
        frame_count=3,
        frame_start=3,
        sequence_format="ppm",
    )
    shard_hash = decoded_rgb_hash(shard, frame_count=3)

    assert source_hash == shard_hash
