import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _probe_payload():
    return {
        "streams": [
            {
                "codec_name": "ffv1",
                "width": 2,
                "height": 1,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "nb_read_packets": "2",
                "pix_fmt": "bgr0",
            }
        ],
        "packets": [
            {"pts_time": "0.000000", "dts_time": "0.000000", "flags": "K_"},
            {"pts_time": "0.033000", "dts_time": "0.033000", "flags": "K_"},
        ],
    }


def _inspect(monkeypatch, payload):
    import server.distributed.render_contract as contract

    monkeypatch.setattr(contract, "_ffprobe_executable", lambda: "ffprobe")
    monkeypatch.setattr(
        contract,
        "_decoded_rgb_sha256",
        lambda *_args, **_kwargs: "d" * 64,
    )
    monkeypatch.setattr(
        contract.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        ),
    )
    return contract.inspect_ffv1_shard(
        Path("shard.mkv"),
        frame_start=10,
        frame_end=12,
        width=2,
        height=1,
        fps=30,
    )


def test_ffv1_shard_validation_accepts_exact_contract(monkeypatch):
    result = _inspect(monkeypatch, _probe_payload())

    assert result == {
        "codec": "ffv1",
        "width": 2,
        "height": 1,
        "fps": 30,
        "frames": 2,
        "pixel_format": "bgr0",
        "decoded_rgb_digest_version": "rgb24-global-frame-v1",
        "decoded_rgb_sha256": "d" * 64,
    }


def test_ffv1_probe_does_not_decode_full_rgb(monkeypatch):
    import server.distributed.render_contract as contract

    commands = []
    monkeypatch.setattr(contract, "_ffprobe_executable", lambda: "ffprobe")
    monkeypatch.setattr(
        contract,
        "_decoded_rgb_sha256",
        lambda *_args, **_kwargs: pytest.fail(
            "metadata-only worker probe must not decode the shard"
        ),
    )
    monkeypatch.setattr(
        contract.subprocess,
        "run",
        lambda command, **_kwargs: (
            commands.append(command)
            or SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_probe_payload()),
                stderr="",
            )
        ),
    )

    result = contract.probe_ffv1_shard(
        Path("shard.mkv"),
        frame_start=10,
        frame_end=12,
        width=2,
        height=1,
        fps=30,
    )

    assert result == {
        "codec": "ffv1",
        "width": 2,
        "height": 1,
        "fps": 30,
        "frames": 2,
        "pixel_format": "bgr0",
    }
    assert len(commands) == 1
    assert commands[0][0] == "ffprobe"
    assert "-count_packets" in commands[0]
    assert "-show_packets" in commands[0]
    assert "-count_frames" not in commands[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["streams"][0].update(codec_name="h264"),
            "codec is not FFV1",
        ),
        (
            lambda payload: payload["streams"][0].update(width=3),
            "dimensions",
        ),
        (
            lambda payload: payload["streams"][0].update(
                avg_frame_rate="24/1"
            ),
            "frame rate",
        ),
        (
            lambda payload: (
                payload["streams"][0].update(nb_read_packets="1"),
                payload.update(packets=payload["packets"][:1]),
            ),
            "expected 2",
        ),
        (
            lambda payload: payload["packets"][1].update(
                pts_time="0.100000"
            ),
            "packets are not contiguous and ordered",
        ),
    ],
)
def test_ffv1_shard_validation_rejects_contract_violations(
    monkeypatch,
    mutate,
    message,
):
    payload = _probe_payload()
    mutate(payload)

    with pytest.raises(RuntimeError, match=message):
        _inspect(monkeypatch, payload)
