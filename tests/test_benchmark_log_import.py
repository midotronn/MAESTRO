import json


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
