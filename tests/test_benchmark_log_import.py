import json

import pytest


def test_import_browser_log_extracts_completed_trace(tmp_path):
    from scripts.import_browser_benchmark_logs import import_logs

    log_path = tmp_path / "baseline.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"event": "accepted", "sid": "song_123"}),
                json.dumps(
                    {
                        "event": "done",
                        "sid": "song_123",
                        "wall_seconds": 123.4,
                        "trace": {
                            "status": "done",
                            "request_id": "request-1",
                            "service_state": "warm_assets_cold_models",
                            "browser_total_seconds": 120.0,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    imported = import_logs([log_path], tmp_path / "runs")

    assert imported[0]["sid"] == "song_123"
    trace = json.loads((tmp_path / "runs" / "song_123.json").read_text())
    assert trace["benchmark_sid"] == "song_123"
    assert trace["browser_wall_seconds"] == 123.4
    assert trace["benchmark_log"] == "baseline.jsonl"
    repeated = import_logs([log_path], tmp_path / "runs")
    assert repeated[0]["status"] == "existing"


def test_import_browser_log_rejects_sid_collision(tmp_path):
    from scripts.import_browser_benchmark_logs import import_logs

    output_dir = tmp_path / "runs"
    output_dir.mkdir()
    existing = output_dir / "song_123.json"
    existing.write_text(
        json.dumps(
            {
                "request_id": "request-1",
                "browser_total_seconds": 120.0,
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "collision.jsonl"
    log_path.write_text(
        json.dumps(
            {
                "event": "done",
                "sid": "song_123",
                "trace": {
                    "status": "done",
                    "request_id": "request-2",
                    "browser_total_seconds": 121.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="already belongs"):
        import_logs([log_path], output_dir)

    assert json.loads(existing.read_text())["request_id"] == "request-1"
