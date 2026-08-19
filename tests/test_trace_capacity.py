def _trace(request_id, render_seconds, preprocess, lodge, edge):
    return {
        "status": "done",
        "request_id": request_id,
        "benchmark_sid": request_id,
        "service_state": "warm_assets_cold_models",
        "browser_total_seconds": render_seconds + preprocess + 100,
        "rendered_frames": 5000,
        "stage_timeline": [
            {"stage": "render", "duration_seconds": render_seconds}
        ],
        "remote_pipeline_timings": {
            "stages": {
                "preprocess_edge": {"duration_seconds": preprocess},
                "generation_lodge": {"duration_seconds": lodge},
                "generation_edge": {"duration_seconds": edge},
            }
        },
    }


def test_trace_capacity_uses_low_tail_render_rate_for_p95():
    from scripts.model_trace_capacity import model_trace_capacity

    report = model_trace_capacity(
        [
            _trace("fast", 400, 200, 60, 70),
            _trace("slow", 500, 240, 70, 80),
        ],
        target_frames=5400,
    )

    assert report["accepted_runs"] == 2
    assert report["render"]["measured_frames_per_second"]["p50"] == 11.25
    assert report["render"]["p05_frames_per_second"] == 10.125
    assert (
        report["render"]["ideal_gpu_equivalents"][
            "minimum_satisfying_both"
        ]
        == 21
    )
    assert (
        report["render"]["planning_gpu_equivalents"][
            "at_90_percent_scaling_efficiency"
        ]
        == 24
    )
    assert (
        report["render"]["planning_gpu_equivalents"][
            "at_80_percent_scaling_efficiency"
        ]
        == 27
    )
    assert (
        report["roles"]["jukebox_preprocessing"]["required_speedup"][
            "p50_budget"
        ]
        == 27.5
    )
    assert (
        report["roles"]["lodge_generation"]["parallelism"].startswith(
            "A single fixed-seed"
        )
    )
