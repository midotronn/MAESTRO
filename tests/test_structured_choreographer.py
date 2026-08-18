"""Structured whole-song planning and realization of curated common motions."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np

from agentlodge.agent import storyboard as SB
from agentlodge.audio.structure import MusicStructure, Section
from agentlodge.dance import story as ST
from agentlodge.editor.motion_bank import default_motion_bank
from agentlodge.editor.window_edit import MockWindowGenerator


def _identity_motion(n_frames: int) -> np.ndarray:
    rotation = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        (n_frames, 22),
    )
    translation = np.zeros((n_frames, 3), dtype=np.float32)
    translation[:, 1] = np.linspace(0.0, 0.2, n_frames, dtype=np.float32)
    contacts = np.ones((n_frames, 4), dtype=np.float32)
    return np.concatenate([translation, rotation, contacts], axis=1)


def _structure(labels: list[int], roles: list[str] | None = None,
               section_frames: int = 210) -> MusicStructure:
    roles = roles or ["verse"] * len(labels)
    sections = []
    first_by_label: dict[int, int] = {}
    for idx, (label, role) in enumerate(zip(labels, roles)):
        a, b = idx * section_frames, (idx + 1) * section_frames
        sections.append(Section(
            start_frame=a,
            end_frame=b,
            start_sec=a / 30.0,
            end_sec=b / 30.0,
            label=label,
            role=role,
            energy=float(idx / max(1, len(labels) - 1)),
            repeat_of=first_by_label.get(label),
        ))
        first_by_label.setdefault(label, idx)
    total = section_frames * len(labels)
    return MusicStructure(
        sections=sections,
        energy_curve=np.linspace(0.0, 1.0, total, dtype=np.float32),
        recurrence=np.eye(len(labels), dtype=np.float32),
        climax_index=max(0, len(labels) - 1),
        tempo=120.0,
        total_frames=total,
    )


def test_prompt_exposes_exact_motion_bank_catalog():
    bank = default_motion_bank()
    structure = _structure([0], ["intro"])
    metadata = SimpleNamespace(duration_seconds=7.0, bpm=120.0)

    prompt = SB._build_prompt(structure, metadata, None)

    assert len(bank.specs) == 19
    assert prompt.count("category=") == 19
    for spec in bank.specs:
        assert f"- {spec.id}:" in prompt


def test_author_storyboard_uses_llm_catalog_and_completes_empty_cues(monkeypatch):
    structure = _structure(
        [0, 1, 1],
        ["intro", "chorus", "chorus"],
    )
    payload = {
        "arc": "build to recurring chorus",
        "reasoning": "mock LLM plan",
        "plans": [
            {
                "section_index": idx,
                "role": section.role,
                "target_intensity": section.energy,
                "vocabulary": "flowing_smooth",
                "generator_bias": "auto",
                "reuse_of": 1 if idx == 2 else None,
                "variation": {},
                "common_motions": [],
            }
            for idx, section in enumerate(structure.sections)
        ],
    }
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[
                SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))
            ])

    class Client:
        def __init__(self, api_key=None):
            captured["api_key"] = api_key
            self.chat = SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    board = SB.author_storyboard(
        structure,
        SimpleNamespace(duration_seconds=21.0, bpm=120.0),
        None,
        "test-key",
        chat_model="test-model",
    )

    assert not board.used_fallback
    assert captured["api_key"] == "test-key"
    assert captured["model"] == "test-model"
    assert captured["max_tokens"] == 2400
    assert "Available common motions" in captured["messages"][0]["content"]
    assert len(board.plans[1].common_motions) == 1
    assert board.plans[2].common_motions == []


def test_all_nineteen_motion_ids_parse_and_realize_through_story_assembly():
    bank = default_motion_bank()
    specs = list(bank.specs)
    structure = _structure(
        list(range(len(specs))),
        ["verse"] * len(specs),
        section_frames=210,
    )
    raw_plans = []
    for idx, spec in enumerate(specs):
        raw_plans.append({
            "section_index": idx,
            "role": "verse",
            "target_intensity": 0.5,
            "vocabulary": "flowing_smooth",
            "generator_bias": "lodge",
            "reuse_of": None,
            "variation": {},
            "common_motions": [{
                "motion_id": spec.id,
                "position": 0.5,
                "anchor": spec.default_anchor,
                "intensity": 0.65,
                "direction": "auto" if spec.directions else None,
                "mirror": False,
                "repeats": 1,
                "motif": f"motif_{idx}",
                "rationale": "catalog realization test",
            }],
        })
    board = SB._parse_response(
        json.dumps({"arc": "catalog", "reasoning": "test", "plans": raw_plans}),
        structure,
    )
    assert [plan.common_motions[0].motion_id for plan in board.plans] == [
        spec.id for spec in specs
    ]

    base = MockWindowGenerator().generate(
        "edge", 0, structure.total_frames, 4, energy=0.5, beats=None
    )
    decisions = ST.select_sources(
        base,
        base.copy(),
        structure,
        board,
        music_beat_frames=np.arange(0, structure.total_frames, 15),
        motion_bank=bank,
    )
    assembled = ST.assemble_story(decisions)

    assert assembled.shape == base.shape
    assert [decision["common_motion_ids"] for decision in decisions] == [
        [spec.id] for spec in specs
    ]
    assert all(
        decision["common_motions"][0]["status"] == "applied"
        for decision in decisions
    )
    json.dumps([
        {key: value for key, value in decision.items() if key != "clip"}
        for decision in decisions
    ])


def test_common_motion_coercion_rejects_inventions_and_bounds_parameters():
    structure = _structure([0], ["chorus"], section_frames=500)
    plan = SB._coerce_plan({
        "common_motions": [
            {"motion_id": "moonwalk_teleport"},
            {
                "motion_id": "clap_single",
                "position": -3,
                "anchor": "somewhere",
                "intensity": 8,
                "direction": "upward",
                "repeats": 99,
            },
        ],
    }, 0, structure)

    assert len(plan.common_motions) == 1
    cue = plan.common_motions[0]
    assert cue.motion_id == "clap_single"
    assert cue.position == 0.05
    assert cue.anchor == "beat"
    assert cue.intensity == 1.0
    assert cue.direction == "auto"
    assert cue.repeats == 4


def test_rule_fallback_is_sparse_and_repeat_sections_inherit_motifs():
    structure = _structure(
        [0, 1, 0, 1, 2],
        ["intro", "verse", "chorus", "chorus", "outro"],
    )
    board = SB._rule_based_storyboard(structure, motif_reuse=True)
    valid_ids = {spec.id for spec in default_motion_bank().specs}

    assert board.used_fallback
    assert board.plans[2].reuse_of == 0
    assert board.plans[3].reuse_of == 1
    assert board.plans[2].common_motions == []
    assert board.plans[3].common_motions == []
    assert all(len(plan.common_motions) <= 1 for plan in board.plans)
    assert {
        cue.motion_id
        for plan in board.plans
        for cue in plan.common_motions
    } <= valid_ids


def test_llm_plan_with_no_catalog_use_gets_one_reusable_signature():
    structure = _structure(
        [0, 1, 1],
        ["intro", "chorus", "chorus"],
    )
    board = SB.Storyboard(
        arc="build",
        reasoning="LLM omitted cues",
        plans=[
            SB.SectionPlan(0, "intro", 0.2, "grounded_minimal", "lodge"),
            SB.SectionPlan(1, "chorus", 0.8, "explosive_fast", "edge"),
            SB.SectionPlan(2, "chorus", 0.9, "explosive_fast", "edge", reuse_of=1),
        ],
    )

    completed = SB._ensure_common_motion_coverage(board, structure)

    assert len(completed.plans[1].common_motions) == 1
    assert completed.plans[2].common_motions == []
    assert "returned no common motions" in completed.reasoning


def test_reused_section_inherits_common_motion_without_applying_it_twice():
    structure = _structure([0, 0], ["chorus", "chorus"])
    cue = SB.CommonMotionCue(
        motion_id="wave",
        position=0.5,
        anchor="beat",
        intensity=0.65,
        motif="chorus_wave",
    )
    extra = SB.CommonMotionCue(
        motion_id="clap_single",
        position=0.5,
        anchor="beat",
        intensity=0.7,
        motif="overlapping_accent",
    )
    board = SB.Storyboard(
        arc="repeat",
        plans=[
            SB.SectionPlan(0, "chorus", 0.0, "flowing_smooth", "lodge",
                           common_motions=[cue]),
            SB.SectionPlan(1, "chorus", 1.0, "flowing_smooth", "auto", reuse_of=0,
                           common_motions=[extra]),
        ],
    )
    base = _identity_motion(structure.total_frames)
    decisions = ST.select_sources(
        base,
        base.copy(),
        structure,
        board,
        music_beat_frames=np.arange(0, structure.total_frames, 15),
    )

    assert decisions[0]["common_motion_ids"] == ["wave"]
    assert decisions[1]["source"] == "reuse:0"
    assert decisions[1]["inherited_common_motion_ids"] == ["wave"]
    assert decisions[1]["common_motion_ids"] == ["wave"]
    assert decisions[1]["common_motions"][0]["status"] == "skipped"
    assert "inherited common motion" in decisions[1]["common_motions"][0]["detail"]


def test_fresh_source_reapplies_structural_motion_motif():
    structure = _structure([0, 0], ["chorus", "chorus"])
    cue = SB.CommonMotionCue(
        motion_id="wave",
        position=0.5,
        anchor="beat",
        intensity=0.65,
        motif="chorus_wave",
    )
    board = SB.Storyboard(
        arc="repeat with fresh source",
        plans=[
            SB.SectionPlan(0, "chorus", 0.0, "flowing_smooth", "lodge",
                           common_motions=[cue]),
            SB.SectionPlan(1, "chorus", 0.0, "flowing_smooth", "edge", reuse_of=0),
        ],
    )
    base = _identity_motion(structure.total_frames)
    decisions = ST.select_sources(
        base,
        base.copy(),
        structure,
        board,
        music_beat_frames=np.arange(0, structure.total_frames, 15),
    )

    assert decisions[1]["source"] == "edge"
    assert decisions[1]["inherited_common_motion_ids"] == []
    assert decisions[1]["recalled_common_motion_ids"] == ["wave"]
    assert decisions[1]["common_motion_ids"] == ["wave"]
    assert sum(
        report["status"] == "applied" for report in decisions[1]["common_motions"]
    ) == 1


def test_legacy_storyboard_plan_serializes_empty_common_motion_list():
    structure = _structure([0], ["intro"])
    board = SB._parse_response(json.dumps({
        "arc": "legacy",
        "plans": [{
            "section_index": 0,
            "role": "intro",
            "target_intensity": 0.1,
            "vocabulary": "grounded_minimal",
            "generator_bias": "lodge",
            "reuse_of": None,
            "variation": {},
        }],
    }), structure)

    assert board.plans[0].common_motions == []
    assert board.to_dict()["plans"][0]["common_motions"] == []
