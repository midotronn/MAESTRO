"""Synthetic validation gates for comparison-ready metrics."""

from scripts.validate_comparison_metrics import validate_metric_controls


def test_comparison_metric_controls_pass():
    report = validate_metric_controls()

    assert report["passed"], report
    assert all(check["pass"] for check in report["checks"].values())
    controls = report["checks"]["section_structure"]["controls"]
    assert set(controls) == {
        "identical",
        "unrelated",
        "shifted",
        "mirrored",
        "retrograded",
        "frozen",
        "jittered",
    }
