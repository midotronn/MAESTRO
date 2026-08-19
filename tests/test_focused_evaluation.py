"""Focused comparison-protocol and performance-model regression tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_pipeline_traces import summarize_traces
from scripts.model_pipeline_capacity import model_capacity
from scripts.validate_focused_protocols import validate_protocols


ROOT = Path(__file__).resolve().parents[1]


def test_focused_protocols_are_internally_consistent():
    assert validate_protocols(ROOT) == []


def test_external_baseline_screening_selects_two_runnable_candidates():
    payload = json.loads(
        (ROOT / "experiments" / "comparisons" / "baselines.json").read_text(
            encoding="utf-8"
        )
    )
    selected = [
        candidate["id"]
        for candidate in payload["external_candidates"]
        if candidate["status"] == "selected_for_technical_pilot"
    ]
    rejected = {
        candidate["id"]: candidate["status"]
        for candidate in payload["external_candidates"]
        if candidate["status"].startswith("rejected_")
    }

    assert selected == ["bailando", "finedance"]
    assert rejected == {"beat_it": "rejected_no_reproducible_artifacts"}


def test_historical_capacity_model_exposes_the_scale_of_the_target():
    baseline = json.loads(
        (
            ROOT
            / "experiments"
            / "performance"
            / "historical_single_gpu.json"
        ).read_text(encoding="utf-8")
    )

    report = model_capacity(baseline)

    assert report["required_throughput"][
        "render_gpu_equivalents_under_ideal_linear_scaling"
    ] == 21
    assert report["required_throughput"][
        "generation_throughput_equivalents_under_ideal_linear_scaling"
    ] == 19
    assert report["historical_throughput"]["render_frames_per_second"] == 11.36


def test_trace_summary_uses_browser_latency_and_stage_timings():
    traces = [
        {
            "request_id": "a",
            "service_state": "warm",
            "browser_total_seconds": 50.0,
            "browser_upload_seconds": 2.0,
            "stage_timeline": [
                {"stage": "render", "duration_seconds": 20.0},
            ],
            "remote_pipeline_timings": {
                "stages": {
                    "generation_lodge": {"duration_seconds": 10.0},
                }
            },
        },
        {
            "request_id": "b",
            "service_state": "warm",
            "browser_total_seconds": 70.0,
            "browser_upload_seconds": 4.0,
            "stage_timeline": [
                {"stage": "render", "duration_seconds": 30.0},
            ],
            "remote_pipeline_timings": {
                "stages": {
                    "generation_lodge": {"duration_seconds": 14.0},
                }
            },
        },
        {
            "request_id": "cold",
            "service_state": "cold",
            "browser_total_seconds": 500.0,
        },
    ]

    report = summarize_traces(
        traces,
        excluded_runs=[{"sid": "failed", "reason": "browser exited"}],
    )

    assert report["measurement_attempts"] == 3
    assert report["accepted_runs"] == 2
    assert report["measurement_completion_rate"] == 0.6667
    assert report["excluded_runs"][0]["sid"] == "failed"
    assert report["service_states"] == {"warm": 2}
    assert report["browser_total_seconds"]["p50"] == 60.0
    assert report["browser_total_seconds"]["mean"] == 60.0
    assert report["browser_total_seconds"]["stddev"] == 10.0
    assert report["browser_total_seconds"]["coefficient_of_variation"] == 0.1667
    assert report["stages"]["render"]["p50"] == 25.0
    assert report["stages"]["remote:generation_lodge"]["p50"] == 12.0
    assert report["sla"]["p50_pass"]
    assert report["sla"]["p95_pass"]
