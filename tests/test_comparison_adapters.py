"""Comparison adapter conversion and validation tests."""

from __future__ import annotations

import json

import numpy as np

from agentlodge.evaluation.adapters import (
    convert_bailando_motion,
    convert_finedance_motion,
    save_conversion,
)


def _identity_6d(frames: int, joints: int) -> np.ndarray:
    rotation = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    return np.tile(rotation, (frames, joints, 1))


def test_finedance_adapter_keeps_body_joints_contacts_and_30_fps():
    frames = 12
    source = np.zeros((frames, 319), dtype=np.float32)
    source[:, :4] = 1.0
    source[:, 4] = np.linspace(0, 1, frames)
    source[:, 7:] = _identity_6d(frames, 52).reshape(frames, -1)

    converted = convert_finedance_motion(source, source_up_axis="z")

    assert converted.motion.shape == (frames, 139)
    assert np.allclose(converted.motion[:, 135:139], 1.0)
    assert np.allclose(converted.motion[:, 3:9], [1, 0, 0, 0, 1, 0])
    assert converted.report["resampling"]["policy"] == "identity"


def test_bailando_adapter_downsamples_60_fps_rotation_matrices():
    frames = 20
    translation = np.zeros((frames, 3), dtype=np.float32)
    translation[:, 0] = np.arange(frames)
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, 24, 1, 1))

    converted = convert_bailando_motion(
        translation,
        rotations,
        source_up_axis="z",
    )

    assert converted.motion.shape == (10, 139)
    assert converted.report["target_fps"] == 30
    assert converted.report["contacts"] == "unavailable_zero_filled"
    assert np.allclose(converted.motion[:, 135:139], 0.0)
    assert converted.motion[-1, 0] == 19


def test_conversion_report_records_input_and_output_hashes(tmp_path):
    source = tmp_path / "source.npy"
    output = tmp_path / "normalized.npy"
    report = tmp_path / "conversion.json"
    frames = 4
    translation = np.zeros((frames, 3), dtype=np.float32)
    rotations = np.tile(np.eye(3, dtype=np.float32), (frames, 24, 1, 1))
    np.save(source, {"trans": translation, "rotations": rotations})
    converted = convert_bailando_motion(
        translation,
        rotations,
        source_up_axis="z",
        source_fps=30,
    )

    save_conversion(
        converted,
        input_path=source,
        output_path=output,
        report_path=report,
    )

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert len(payload["input_sha256"]) == 64
    assert len(payload["output_sha256"]) == 64
    assert np.load(output).shape == (frames, 139)
