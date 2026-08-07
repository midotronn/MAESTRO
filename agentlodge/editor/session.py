"""Interactive edit session: immutable song assets + a checkpointed, editable dance.

An :class:`EditSession` owns the immutable :class:`SongAssets` (beats, optional structure/backbone
material for real regeneration) and a :class:`CheckpointStore` whose ``head`` is the currently-shown
dance. Editing calls :func:`agentlodge.editor.agent_edit.run_agent_edit` (the LLM-planned, verified,
self-refining agent) on the head motion and commits the result as a new checkpoint; undo / redo /
restore / branch operate on the store. The whole thing is GPU-free and, with a
:class:`MockWindowGenerator`, fully unit-testable; a real pod-backed generator drops in via the
``generator`` argument without other changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agentlodge.editor.checkpoints import Checkpoint, CheckpointStore
from agentlodge.editor.agent_edit import run_agent_edit
from agentlodge.editor.window_edit import (
    WindowEditResult,
    WindowGenerator,
    window_metrics,
)

_ASSETS = "assets.json"


def _edit_label(instruction: str, a: int, b: int) -> str:
    """A concise checkpoint label from the user's instruction + window (in seconds)."""
    text = " ".join(str(instruction).split())
    if len(text) > 42:
        text = text[:41] + "\u2026"
    return f"{text} [{a // 30}-{b // 30}s]"


@dataclass
class SongAssets:
    """Immutable per-song context. Heavy backbone fields are optional (needed only for real regen)."""

    sid: str
    beats: np.ndarray | None = None
    fps: int = 30
    beat_strengths: np.ndarray | None = None
    wav_path: str | None = None
    metadata: dict = field(default_factory=dict)
    # optional material for real windowed regeneration (Phase 2 pod worker)
    structure: object | None = None
    storyboard: object | None = None
    lodge_z: np.ndarray | None = None
    edge_z: np.ndarray | None = None
    lodge_features: np.ndarray | None = None
    edge_slices: object | None = None

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.beats is not None:
            np.save(directory / "beats.npy", np.asarray(self.beats, dtype=np.float32))
        if self.beat_strengths is not None:
            if self.beats is None or len(self.beat_strengths) != len(self.beats):
                raise ValueError("beat_strengths must contain one value for every beat")
            np.save(
                directory / "beat_strengths.npy",
                np.asarray(self.beat_strengths, dtype=np.float32),
            )
        (directory / _ASSETS).write_text(json.dumps({
            "sid": self.sid, "fps": self.fps, "wav_path": self.wav_path,
            "metadata": self.metadata, "has_beats": self.beats is not None,
            "has_beat_strengths": self.beat_strengths is not None,
        }, indent=2))

    @classmethod
    def load(cls, directory: Path) -> "SongAssets":
        directory = Path(directory)
        d = json.loads((directory / _ASSETS).read_text())
        beats = None
        bp = directory / "beats.npy"
        if d.get("has_beats") and bp.exists():
            beats = np.load(bp).astype(np.float32)
        beat_strengths = None
        sp = directory / "beat_strengths.npy"
        if d.get("has_beat_strengths") and sp.exists():
            beat_strengths = np.load(sp).astype(np.float32)
        return cls(sid=d["sid"], beats=beats, fps=int(d.get("fps", 30)),
                   beat_strengths=beat_strengths,
                   wav_path=d.get("wav_path"), metadata=dict(d.get("metadata") or {}))


class EditSession:
    """A checkpointed, natural-language-editable dance for one song."""

    def __init__(self, assets: SongAssets, base_motion: np.ndarray | None = None,
                 generator: WindowGenerator | None = None, *, directory: str | Path | None = None,
                 k: int = 6, max_cycles: int = 3, blend_frames: int = 15,
                 api_key: str | None = None, _store: CheckpointStore | None = None):
        self.assets = assets
        self.generator = generator
        self.k = int(k)
        self.max_cycles = int(max_cycles)
        self.blend_frames = int(blend_frames)
        self.api_key = api_key
        self.dir = Path(directory).resolve() if directory is not None else None
        self.store = _store if _store is not None else CheckpointStore(
            self.dir / "checkpoints" if self.dir is not None else None)
        if len(self.store) == 0:
            if base_motion is None:
                raise ValueError("base_motion is required to seed a new session")
            self.store.commit(np.asarray(base_motion, dtype=np.float32), edit=None,
                              metrics=self._whole_metrics(base_motion), label="original")
        if self.dir is not None:
            self.assets.save(self.dir)

    # ------------------------------------------------------------------ helpers
    def _whole_metrics(self, motion: np.ndarray) -> dict:
        return window_metrics(np.asarray(motion), self.assets.beats)

    def current_motion(self) -> np.ndarray:
        return self.store.current_motion()

    def current(self) -> Checkpoint | None:
        return self.store.current()

    def _record(self, res: WindowEditResult, a: int, b: int, instruction: str) -> dict:
        return {
            "window": [int(a), int(b)], "instruction": instruction, "ok": res.ok,
            "backbone": res.backbone, "chosen_seed": res.chosen_seed,
            "metrics_before": res.metrics_before, "metrics_after": res.metrics_after,
            "feedback": res.feedback, "agent_summary": res.agent_summary, "log": res.log,
            **res.goal.to_dict(),
        }

    # ------------------------------------------------------------------ editing
    def edit(self, a: int, b: int, instruction: str, *, k: int | None = None,
             max_cycles: int | None = None, progress_cb=None) -> WindowEditResult:
        """Apply an NL window edit to the current dance and commit it as a new checkpoint."""
        res = run_agent_edit(
            self.current_motion(), a, b, instruction, self.generator,
            beats=self.assets.beats, beat_strengths=self.assets.beat_strengths,
            api_key=self.api_key,
            k=k or self.k, max_cycles=max_cycles or self.max_cycles,
            blend_frames=self.blend_frames, progress_cb=progress_cb,
        )
        metrics = dict(self._whole_metrics(res.motion))
        metrics["window_after"] = res.metrics_after
        self.store.commit(res.motion, edit=self._record(res, a, b, instruction),
                          metrics=metrics, label=_edit_label(instruction, a, b))
        return res

    def edit_from(self, from_id: str, a: int, b: int, instruction: str, *, k: int | None = None,
                  max_cycles: int | None = None, progress_cb=None) -> WindowEditResult:
        """Branch: apply an edit to an ARBITRARY earlier checkpoint, forking history."""
        base = self.store.motion(from_id)
        res = run_agent_edit(
            base, a, b, instruction, self.generator, beats=self.assets.beats,
            beat_strengths=self.assets.beat_strengths, api_key=self.api_key,
            k=k or self.k, max_cycles=max_cycles or self.max_cycles,
            blend_frames=self.blend_frames, progress_cb=progress_cb,
        )
        metrics = dict(self._whole_metrics(res.motion))
        metrics["window_after"] = res.metrics_after
        self.store.branch(from_id, res.motion, edit=self._record(res, a, b, instruction),
                          metrics=metrics, label=_edit_label(instruction, a, b) + " (branch)")
        return res

    # ------------------------------------------------------------------ history
    def undo(self) -> Checkpoint | None:
        return self.store.undo()

    def redo(self) -> Checkpoint | None:
        return self.store.redo()

    def restore(self, ckpt_id: str) -> Checkpoint:
        return self.store.restore(ckpt_id)

    def timeline(self) -> list[dict]:
        return self.store.timeline()

    def diff(self, id_a: str, id_b: str) -> dict:
        return self.store.diff(id_a, id_b)

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, directory: str | Path, generator: WindowGenerator | None = None, *,
             k: int = 6, max_cycles: int = 3, blend_frames: int = 15,
             api_key: str | None = None) -> "EditSession":
        """Reload a persisted session (assets + checkpoint tree) from ``directory``."""
        directory = Path(directory).resolve()
        assets = SongAssets.load(directory)
        store = CheckpointStore.load(directory / "checkpoints")
        return cls(assets, generator=generator, directory=directory, k=k, max_cycles=max_cycles,
                   blend_frames=blend_frames, api_key=api_key, _store=store)
