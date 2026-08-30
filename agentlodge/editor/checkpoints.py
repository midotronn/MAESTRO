"""Append-only checkpoint DAG for the interactive editor (AgentBanana-style undoable history).

Every accepted edit produces an immutable :class:`Checkpoint` whose ``parent_id`` points at the
state it was derived from, so the full edit history is a tree (a DAG with single parents). The
store tracks a ``head`` (the currently-shown state) and supports:

* ``commit``  -- add a child of ``head`` and move ``head`` onto it (linear edit),
* ``undo`` / ``redo`` -- walk ``head`` to its parent / back down the branch just undone,
* ``restore`` -- jump ``head`` to ANY checkpoint (rollback to an arbitrary point),
* ``branch``  -- commit a child of an arbitrary (possibly older) checkpoint, forking history,
* ``diff`` / ``timeline`` -- inspect the tree.

Motion snapshots are the source of truth for each state. With a backing directory they are saved
as ``.npy`` next to a ``manifest.json`` describing the tree + head, so a session survives a restart
(``CheckpointStore.load``). Without one the store is fully in-memory (used by unit tests).
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_MANIFEST = "manifest.json"


@dataclass
class Checkpoint:
    """One immutable edited state. ``motion`` is not stored here (see the store's snapshots)."""

    id: str
    parent_id: str | None
    ts: float
    label: str
    edit: dict | None = None          # the WindowEdit that produced this state (None for root)
    metrics: dict = field(default_factory=dict)
    n_frames: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "ts": self.ts,
            "label": self.label,
            "edit": self.edit,
            "metrics": self.metrics,
            "n_frames": self.n_frames,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(
            id=str(d["id"]),
            parent_id=d.get("parent_id"),
            ts=float(d.get("ts", 0.0)),
            label=str(d.get("label", "")),
            edit=d.get("edit"),
            metrics=dict(d.get("metrics") or {}),
            n_frames=int(d.get("n_frames", 0)),
        )


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class CheckpointStore:
    """A tree of :class:`Checkpoint` states with ``head``, undo/redo, restore and branch."""

    def __init__(self, directory: str | Path | None = None):
        self.dir = Path(directory).resolve() if directory is not None else None
        if self.dir is not None:
            self.dir.mkdir(parents=True, exist_ok=True)
        self._ckpts: dict[str, Checkpoint] = {}
        self._motions: dict[str, np.ndarray] = {}   # in-memory snapshots (always populated)
        self._children: dict[str, list[str]] = {}
        self.head: str | None = None
        self._redo: list[str] = []                  # checkpoints undone, available to redo

    # ------------------------------------------------------------------ internals
    def _snapshot_path(self, ckpt_id: str) -> Path | None:
        return None if self.dir is None else self.dir / f"motion_{ckpt_id}.npy"

    def _store_motion(self, ckpt_id: str, motion: np.ndarray) -> None:
        arr = np.ascontiguousarray(np.asarray(motion, dtype=np.float32))
        self._motions[ckpt_id] = arr
        p = self._snapshot_path(ckpt_id)
        if p is not None:
            np.save(p, arr)

    def _add(self, ckpt: Checkpoint, motion: np.ndarray) -> Checkpoint:
        self._ckpts[ckpt.id] = ckpt
        self._children.setdefault(ckpt.id, [])
        if ckpt.parent_id is not None:
            self._children.setdefault(ckpt.parent_id, []).append(ckpt.id)
        self._store_motion(ckpt.id, motion)
        return ckpt

    # ------------------------------------------------------------------ queries
    def __len__(self) -> int:
        return len(self._ckpts)

    def __contains__(self, ckpt_id: str) -> bool:
        return ckpt_id in self._ckpts

    def get(self, ckpt_id: str) -> Checkpoint:
        return self._ckpts[ckpt_id]

    def current(self) -> Checkpoint | None:
        return self._ckpts.get(self.head) if self.head is not None else None

    def motion(self, ckpt_id: str) -> np.ndarray:
        if ckpt_id in self._motions:
            return self._motions[ckpt_id]
        p = self._snapshot_path(ckpt_id)
        if p is not None and p.exists():
            arr = np.load(p).astype(np.float32)
            self._motions[ckpt_id] = arr
            return arr
        raise KeyError(f"no motion snapshot for checkpoint {ckpt_id!r}")

    def current_motion(self) -> np.ndarray:
        if self.head is None:
            raise RuntimeError("checkpoint store is empty")
        return self.motion(self.head)

    def children(self, ckpt_id: str) -> list[str]:
        return list(self._children.get(ckpt_id, []))

    def ancestry(self, ckpt_id: str | None = None) -> list[str]:
        """Root-first list of checkpoint ids from the root down to ``ckpt_id`` (default head)."""
        cur = ckpt_id if ckpt_id is not None else self.head
        chain: list[str] = []
        while cur is not None:
            chain.append(cur)
            cur = self._ckpts[cur].parent_id
        return list(reversed(chain))

    def timeline(self) -> list[dict]:
        """All checkpoints (ts order) with lineage metadata for a UI history tree."""
        out = []
        head_lineage = set(self.ancestry()) if self.head is not None else set()
        for c in sorted(self._ckpts.values(), key=lambda c: c.ts):
            d = c.to_dict()
            lineage = self.ancestry(c.id)
            d["is_head"] = c.id == self.head
            d["children"] = self.children(c.id)
            d["lineage"] = lineage
            d["depth"] = len(lineage) - 1
            d["is_ancestor_of_head"] = c.id in head_lineage
            d["is_branch"] = (
                c.parent_id is not None
                and len(self._children.get(c.parent_id, ())) > 1
            )
            out.append(d)
        return out

    def can_undo(self) -> bool:
        return self.head is not None and self._ckpts[self.head].parent_id is not None

    def can_redo(self) -> bool:
        return bool(self._redo) and self._redo[-1] in self._children.get(self.head or "", [])

    # ------------------------------------------------------------------ mutations
    def commit(self, motion: np.ndarray, *, edit: dict | None = None, metrics: dict | None = None,
               label: str = "", parent_id: str | None = "__head__") -> Checkpoint:
        """Add a checkpoint as a child of ``parent_id`` (default: current head) and move head to it.

        Committing after an ``undo`` starts a new branch (the redo stack is cleared).
        """
        parent = self.head if parent_id == "__head__" else parent_id
        if parent is not None and parent not in self._ckpts:
            raise KeyError(f"unknown parent checkpoint {parent!r}")
        ckpt = Checkpoint(
            id=_new_id(), parent_id=parent, ts=time.time(), label=label,
            edit=edit, metrics=dict(metrics or {}), n_frames=int(np.asarray(motion).shape[0]),
        )
        self._add(ckpt, motion)
        self.head = ckpt.id
        self._redo.clear()
        self._save_manifest()
        return ckpt

    def branch(self, from_id: str, motion: np.ndarray, *, edit: dict | None = None,
               metrics: dict | None = None, label: str = "") -> Checkpoint:
        """Fork: commit a new child of ``from_id`` regardless of the current head."""
        if from_id not in self._ckpts:
            raise KeyError(f"unknown checkpoint {from_id!r}")
        return self.commit(motion, edit=edit, metrics=metrics, label=label, parent_id=from_id)

    def undo(self) -> Checkpoint | None:
        """Move head to its parent. Returns the new head checkpoint, or ``None`` if at the root."""
        if not self.can_undo():
            return None
        cur = self.head
        self.head = self._ckpts[cur].parent_id
        self._redo.append(cur)
        self._save_manifest()
        return self.current()

    def redo(self) -> Checkpoint | None:
        """Re-apply the most recently undone edit (if head is still its parent)."""
        if not self.can_redo():
            return None
        nxt = self._redo.pop()
        self.head = nxt
        self._save_manifest()
        return self.current()

    def restore(self, ckpt_id: str) -> Checkpoint:
        """Roll back (or forward) to an arbitrary checkpoint by id. Clears the redo stack."""
        if ckpt_id not in self._ckpts:
            raise KeyError(f"unknown checkpoint {ckpt_id!r}")
        self.head = ckpt_id
        self._redo.clear()
        self._save_manifest()
        return self.current()

    # ------------------------------------------------------------------ diff
    def diff(self, id_a: str, id_b: str) -> dict:
        """Compare two checkpoints: frame-count delta, metric deltas, and changed-frame span."""
        a, b = self._ckpts[id_a], self._ckpts[id_b]
        ma, mb = self.motion(id_a), self.motion(id_b)
        metric_keys = set(a.metrics) | set(b.metrics)
        metric_delta = {
            k: round(float(b.metrics.get(k, 0.0)) - float(a.metrics.get(k, 0.0)), 5)
            for k in sorted(metric_keys)
            if isinstance(a.metrics.get(k, 0.0), (int, float))
            and isinstance(b.metrics.get(k, 0.0), (int, float))
        }
        changed = None
        if ma.shape == mb.shape and ma.size:
            per_frame = np.abs(ma - mb).sum(axis=1)
            nz = np.flatnonzero(per_frame > 1e-6)
            if nz.size:
                changed = [int(nz[0]), int(nz[-1]) + 1]
        return {
            "from": id_a,
            "to": id_b,
            "n_frames_from": a.n_frames,
            "n_frames_to": b.n_frames,
            "metric_delta": metric_delta,
            "changed_frame_span": changed,
        }

    # ------------------------------------------------------------------ persistence
    def _save_manifest(self) -> None:
        if self.dir is None:
            return
        manifest = {
            "head": self.head,
            "redo": self._redo,
            "checkpoints": [c.to_dict() for c in self._ckpts.values()],
        }
        (self.dir / _MANIFEST).write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "CheckpointStore":
        """Reload a persisted store (its manifest + motion snapshots) from ``directory``."""
        store = cls(directory)
        path = store.dir / _MANIFEST
        if not path.exists():
            return store
        manifest = json.loads(path.read_text())
        for d in manifest.get("checkpoints", []):
            ckpt = Checkpoint.from_dict(d)
            store._ckpts[ckpt.id] = ckpt
            store._children.setdefault(ckpt.id, [])
            if ckpt.parent_id is not None:
                store._children.setdefault(ckpt.parent_id, []).append(ckpt.id)
        store.head = manifest.get("head")
        store._redo = list(manifest.get("redo") or [])
        return store
