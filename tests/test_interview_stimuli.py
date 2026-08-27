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

    assert selection["protocol"] == "maestro-expert-study-v13-approved-pop-front-facing"
    assert selection["selection_type"] == "capability-focused expert-elicitation"
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


def test_blind_player_assignments_cover_only_valid_lane_permutations():
    assignment_path = STIMULI / "player" / "assignments.json"
    assignments = json.loads(assignment_path.read_text(encoding="utf-8"))
    config = json.loads((STIMULI / "player" / "config.json").read_text(encoding="utf-8"))
    valid_excerpts = {f"EX{index:02d}" for index in range(1, 13)}

    assert assignments["protocol"] == "maestro-expert-study-v13-approved-pop-front-facing"
    assert config["protocol"] == assignments["protocol"]
    assert config["excerpt_duration_seconds"] == 6
    assert len(assignments["phases"]["pilot"]) == 5
    assert len(assignments["phases"]["main"]) == 18
    assert "LODGE" not in assignment_path.read_text(encoding="utf-8")
    assert "EDGE" not in assignment_path.read_text(encoding="utf-8")

    for phase in assignments["phases"].values():
        for participant in phase:
            assert len(participant["triplets"]) == 6
            for triplet in participant["triplets"]:
                assert triplet["excerpt"] in valid_excerpts
                assert sorted(triplet["lanes"]) == [0, 1, 2]

    for participant in assignments["phases"]["main"]:
        for position in range(3):
            counts = [
                sum(triplet["lanes"][position] == lane for triplet in participant["triplets"])
                for lane in range(3)
            ]
            assert counts == [2, 2, 2]
