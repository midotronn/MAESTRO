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


def test_frozen_stimulus_manifest_has_twelve_verified_positive_windows():
    selection = json.loads((STIMULI / "selection.json").read_text(encoding="utf-8"))
    manifest = json.loads((STIMULI / "manifest.json").read_text(encoding="utf-8"))

    assert selection["protocol"] == "maestro-expert-study-v14-llm-gpt4o-k10-front-facing"
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
    assert selection["output"]["duration_seconds"] == 6
    assert [item["id"] for item in selection["excerpts"]] == [
        f"EX{index:02d}" for index in range(1, 13)
    ]
    assert [item["id"] for item in manifest["excerpts"]] == [
        f"EX{index:02d}" for index in range(1, 13)
    ]
    assert {
        song: sum(item["source_song"] == song for item in selection["excerpts"])
        for song in selection["sources"]
    } == {
        "Can't Stop the Feeling!": 3,
        "Uptown Funk": 3,
        "Levitating": 3,
        "Dynamite": 3,
    }

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
        assert excerpt["duration_seconds"] == 6
        assert excerpt["frames"] == 180
        assert excerpt["diagnostics"]["quality_margin"] > 0
        assert excerpt["diagnostics"]["beat_margin"] > 0
        assert excerpt["diagnostics"]["min_pose_difference_degrees"] > 0
        assert excerpt["source_sha256"] == _sha256(source)
        assert set(excerpt["permutations"]) == {
            "012", "021", "102", "120", "201", "210"
        }
        for permutation in excerpt["permutations"].values():
            output = STIMULI / permutation["output_video"]
            assert output.stat().st_size > 250_000
            with output.open("rb") as stream:
                assert stream.read(12).find(b"ftyp") >= 0
            assert permutation["output_sha256"] == _sha256(output)


def test_blind_player_uses_one_fixed_balanced_sequence():
    assignment_path = STIMULI / "player" / "assignments.json"
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    config = json.loads((STIMULI / "player" / "config.json").read_text(encoding="utf-8"))
    valid_excerpts = {f"EX{index:02d}" for index in range(1, 13)}
    sequence = assignments["sequence"]

    assert assignments["protocol"] == "maestro-expert-study-v14-llm-gpt4o-k10-front-facing"
    assert config["protocol"] == assignments["protocol"]
    assert config["excerpt_duration_seconds"] == 6
    assert sequence == [
        {"excerpt": "EX03", "lanes": [0, 2, 1]},
        {"excerpt": "EX04", "lanes": [1, 2, 0]},
        {"excerpt": "EX08", "lanes": [0, 1, 2]},
        {"excerpt": "EX06", "lanes": [2, 0, 1]},
        {"excerpt": "EX09", "lanes": [1, 0, 2]},
        {"excerpt": "EX02", "lanes": [2, 1, 0]},
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
