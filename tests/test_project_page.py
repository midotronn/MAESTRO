"""Project-page content checks for method fidelity."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_abstract_matches_current_composition_and_editing_paths():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    text = " ".join(html.split())

    assert "chroma, MFCC, energy, and downbeats" in text
    assert "deterministic monotone levers" in text
    assert "licensed motion bank" in text
    assert "add a clap here" in text
    assert "genuinely new choreography invoke window regeneration" in text
    assert "final spliced window" in text
    assert "undoable checkpoint" in text
    assert "best safe result or unchanged original" in text
    assert "verifies every goal before saving" not in text
    assert "structure agent reads" not in text


def test_method_diagrams_cover_current_pipeline_contracts():
    framework = (ROOT / "docs" / "static" / "images" / "framework.svg").read_text(encoding="utf-8")
    editing = (ROOT / "docs" / "static" / "images" / "editing_loop.svg").read_text(encoding="utf-8")

    for label in ("Music analysis", "Storyboard Agent", "Plan-aware realization",
                  "Inertialized assembly", "LODGE", "EDGE", "named-motion cues",
                  "bank cue +"):
        assert label in framework
    for label in ("Planner", "Routed executor", "Final-window adaptation", "Verify",
                  "Checkpointed result", "monotone levers", "audited motion bank",
                  "new motion: LODGE / EDGE", "best safe result", "bounded refine"):
        assert label in editing
    assert "all goals met" not in editing
    assert "unknown named actions fail visibly" not in editing


def test_project_page_has_maestro_favicon():
    html = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    icon = ROOT / "docs" / "static" / "images" / "favicon.svg"

    assert 'rel="icon"' in html
    assert "favicon.svg" in html
    assert icon.is_file()
    assert 'aria-label="MAESTRO"' in icon.read_text(encoding="utf-8")
