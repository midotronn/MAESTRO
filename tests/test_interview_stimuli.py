from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STIMULI = ROOT / "experiments" / "user_study" / "stimuli"


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_stimulus_manifest_has_six_verified_high_energy_phrase_windows():
    selection = json.loads((STIMULI / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((STIMULI / "manifest.json").read_text(encoding="utf-8"))

    assert (
        selection["protocol"]
        == "maestro-expert-study-v15-llm-gpt4o-k10-front-facing-16s"
    )
    assert selection["selection_type"] == "capability-focused expert-elicitation"
    assert selection["generation_evidence"] == {
        "storyboard_model": "gpt-4o",
        "storyboard_required": True,
        "fallback_allowed": False,
        "all_storyboards_llm_authored": True,
        "best_of_k_requested": 10,
        "lodge_candidates_completed_per_song": 10,
        "edge_candidates_completed_per_song": 10,
    }
    assert selection["source_layout"]["lanes"] == ["LODGE", "EDGE", "MAESTRO"]
    assert selection["output"]["duration_seconds"] == 16
    assert [item["id"] for item in selection["excerpts"]] == [
        f"EX{index:02d}" for index in range(1, 7)
    ]
    assert [item["id"] for item in manifest["excerpts"]] == [
        f"EX{index:02d}" for index in range(1, 7)
    ]
    assert manifest["protocol"] == selection["protocol"]
    assert manifest["output"] == selection["output"]
    assert {
        song: sum(item["source_song"] == song for item in selection["excerpts"])
        for song in selection["sources"]
    } == {
        "Can't Stop the Feeling!": 0,
        "Dynamite": 2,
        "Levitating": 0,
        "Uptown Funk": 4,
    }
    assert {
        song
        for song, source in selection["sources"].items()
        if source["selected_for_long_study"]
    } == {"Dynamite", "Uptown Funk"}

    for source in selection["sources"].values():
        report_path = (
            STIMULI
            / "story-reports"
            / f"fd_{source['sid']}_STORY_bestofk.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["best_of_k"] == 10
        assert report["storyboard_model"] == "gpt-4o"
        assert report["storyboard_required"] is True
        assert report["storyboard"]["used_fallback"] is False
        assert report["backbone_selection"]["lodge"]["requested"] == 10
        assert report["backbone_selection"]["lodge"]["completed"] == 10
        assert report["backbone_selection"]["edge"]["requested"] == 10
        assert report["backbone_selection"]["edge"]["completed"] == 10

    for excerpt in manifest["excerpts"]:
        source = ROOT / excerpt["source_video"]
        assert excerpt["duration_seconds"] == 16
        assert excerpt["frames"] == 480
        assert excerpt["diagnostics"]["quality_margin"] > 0
        assert excerpt["diagnostics"]["beat_margin"] > 0
        assert excerpt["diagnostics"]["energetic_score"] >= 0.5
        assert excerpt["diagnostics"]["min_pose_difference_degrees"] > 4
        assert excerpt["diagnostics"]["max_near_identical_ratio"] < 0.85
        assert excerpt["source_sha256"] == _sha256(source)
        assignment = next(
            comparison
            for comparison in manifest["fixed_sequence"]
            if comparison["excerpt"] == excerpt["id"]
        )
        assert set(excerpt["permutations"]) == {
            "".join(str(lane) for lane in assignment["lanes"])
        }
        for permutation in excerpt["permutations"].values():
            output = STIMULI / permutation["output_video"]
            assert output.stat().st_size > 500_000
            with output.open("rb") as stream:
                assert stream.read(12).find(b"ftyp") >= 0
            assert permutation["output_sha256"] == _sha256(output)


def test_blind_player_uses_one_fixed_balanced_sequence():
    assignment_path = STIMULI / "player" / "assignments.json"
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    config = json.loads((STIMULI / "player" / "config.json").read_text(encoding="utf-8"))
    valid_excerpts = {f"EX{index:02d}" for index in range(1, 7)}
    sequence = assignments["sequence"]

    assert (
        assignments["protocol"]
        == "maestro-expert-study-v15-llm-gpt4o-k10-front-facing-16s"
    )
    assert config["protocol"] == assignments["protocol"]
    assert config["excerpt_duration_seconds"] == 16
    assert sequence == [
        {"excerpt": "EX01", "lanes": [0, 2, 1]},
        {"excerpt": "EX02", "lanes": [1, 2, 0]},
        {"excerpt": "EX03", "lanes": [0, 1, 2]},
        {"excerpt": "EX04", "lanes": [2, 0, 1]},
        {"excerpt": "EX05", "lanes": [1, 0, 2]},
        {"excerpt": "EX06", "lanes": [2, 1, 0]},
    ]
    assert "LODGE" not in assignment_path.read_text(encoding="utf-8")
    assert "EDGE" not in assignment_path.read_text(encoding="utf-8")
    for forbidden in (
        "phase",
        "pilot",
        "participant",
        "assignment_code",
        "guided_sequence",
        "open_ended_start",
    ):
        assert forbidden not in assignment_path.read_text(encoding="utf-8").lower()

    assert len(sequence) == 6
    for comparison in sequence:
        assert comparison["excerpt"] in valid_excerpts
        assert sorted(comparison["lanes"]) == [0, 1, 2]

    for position in range(3):
        counts = [
            sum(comparison["lanes"][position] == lane for comparison in sequence)
            for lane in range(3)
        ]
        assert counts == [2, 2, 2]


def test_blind_player_has_no_phase_or_participant_state():
    player = STIMULI / "player"
    page = (player / "index.html").read_text(encoding="utf-8")
    app = (player / "app.js").read_text(encoding="utf-8")

    assert 'id="comparisonSelect"' in page
    assert 'id="phase"' not in page
    assert 'id="participant"' not in page
    assert 'id="triplet"' not in page
    assert "Pilot" not in page
    assert "localStorage" not in app
    assert "phase=" not in app
    assert "participant=" not in app
    assert "triplet=" not in app
    assert 'max="16"' in page
    assert "0:00 / 0:16" in page
