"""Project-page content checks for method fidelity."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_abstract_matches_current_composition_and_editing_paths():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

    assert "chroma, MFCC, energy, and downbeats" in html
    assert "deterministic monotone levers" in html
    assert "genuinely new choreography invoke window regeneration" in html
    assert "final crossfaded window" in html
    assert "undoable checkpoint" in html
    assert "structure agent reads" not in html


def test_method_diagrams_cover_current_pipeline_contracts():
    framework = (ROOT / "docs" / "static" / "images" / "framework.svg").read_text(encoding="utf-8")
    editing = (ROOT / "docs" / "static" / "images" / "editing_loop.svg").read_text(encoding="utf-8")

    for label in ("Music analysis", "Storyboard Agent", "Plan-aware realization",
                  "Inertialized assembly", "LODGE", "EDGE"):
        assert label in framework
    for label in ("Planner", "Routed executor", "Final-window adaptation", "Verify",
                  "Checkpointed result", "monotone levers", "new motion: LODGE / EDGE"):
        assert label in editing
