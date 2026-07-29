"""Backbone-backed window generators for the interactive editor (Phase 2).

Re-running a diffusion backbone per candidate per refine-cycle would make interactive editing far
too slow (minutes of GPU each). Instead we front-load the GPU **once** into a *candidate bank*: for
a song we sample K seeded full takes from each backbone (LODGE, EDGE), convert them into the same
Z-up 139 space as the assembled dance, and cache them as ``.npy`` files. A window edit then becomes
a fast, GPU-free selection: for window ``[a, b)`` the :class:`BankWindowGenerator` returns each
cached take's window slice as a candidate, and the edit loop's instruction-shaped reward picks the
best (EDGE slices win "more energetic", LODGE slices win "calmer", the tightest wins "more on
beat"), then splices it in. This realizes "query LODGE/EDGE multiple times and pick per the
instruction" as best-of-K over real backbone samples, while staying responsive and usable even when
the pod is down (the bank lives locally once pulled).

:class:`ResilientWindowGenerator` wraps a primary generator with a fallback (e.g. a
:class:`~agentlodge.editor.window_edit.MockWindowGenerator`) so a missing/unreachable bank never
breaks the UI -- it degrades to the offline stand-in and flags it.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_BANK_RE = re.compile(r"bank_(?P<sid>.+)_(?P<backbone>lodge|edge)_seed(?P<seed>\d+)\.npy$")


def _window_slice(take: np.ndarray, a: int, b: int) -> np.ndarray:
    """Return ``take[a:b]``, edge-padding if the take is shorter than ``b`` (never raises)."""
    L = int(take.shape[0])
    a2, b2 = max(0, min(a, L)), max(0, min(b, L))
    win = take[a2:b2]
    need = (b - a) - win.shape[0]
    if need > 0 and win.shape[0] > 0:
        win = np.concatenate([win, np.repeat(win[-1:], need, axis=0)], axis=0)
    elif win.shape[0] == 0:
        return np.zeros((0, take.shape[1]), dtype=np.float32)
    return np.ascontiguousarray(win, dtype=np.float32)


class BankWindowGenerator:
    """Serve window candidates from a cached bank of seeded LODGE/EDGE full takes (Z-up 139)."""

    def __init__(self, bank: dict[str, list[np.ndarray]], *, fallback=None):
        self.bank = {k: [np.asarray(t, dtype=np.float32) for t in v]
                     for k, v in bank.items() if v}
        self.fallback = fallback

    def n_takes(self, backbone: str) -> int:
        return len(self.bank.get(backbone, []))

    @property
    def backbones(self) -> list[str]:
        return [k for k, v in self.bank.items() if v]

    def generate(self, backbone: str, a: int, b: int, seed: int, *,
                 energy: float = 0.5, beats=None, context=None) -> np.ndarray | None:
        takes = self.bank.get(backbone) or []
        if not takes:
            if self.fallback is not None:
                return self.fallback.generate(backbone, a, b, seed, energy=energy,
                                              beats=beats, context=context)
            return None
        take = takes[int(seed) % len(takes)]      # cycle through available seeds
        return _window_slice(take, int(a), int(b))

    @classmethod
    def from_dir(cls, directory: str | Path, sid: str | None = None, *, fallback=None
                 ) -> "BankWindowGenerator":
        """Load a bank from ``bank_<sid>_<backbone>_seed<n>.npy`` files in ``directory``."""
        directory = Path(directory)
        bank: dict[str, list[tuple[int, np.ndarray]]] = {"lodge": [], "edge": []}
        for p in sorted(directory.glob("bank_*.npy")):
            m = _BANK_RE.search(p.name)
            if not m or (sid is not None and m.group("sid") != sid):
                continue
            bank[m.group("backbone")].append((int(m.group("seed")), np.load(p).astype(np.float32)))
        ordered = {k: [t for _, t in sorted(v, key=lambda x: x[0])] for k, v in bank.items()}
        logger.info("Loaded window bank for %s: %d LODGE, %d EDGE takes",
                    sid, len(ordered["lodge"]), len(ordered["edge"]))
        return cls(ordered, fallback=fallback)


class ResilientWindowGenerator:
    """Try a primary :class:`WindowGenerator`; on failure/empty, fall back (keeps the UI usable)."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.used_fallback = False

    def generate(self, backbone: str, a: int, b: int, seed: int, *,
                 energy: float = 0.5, beats=None, context=None) -> np.ndarray | None:
        try:
            out = self.primary.generate(backbone, a, b, seed, energy=energy,
                                        beats=beats, context=context)
            if out is not None and np.asarray(out).shape[0] >= 2:
                return out
        except Exception as exc:  # noqa: BLE001 - fall back rather than break the session
            logger.warning("primary window generator failed (%s); using fallback", exc)
        self.used_fallback = True
        return self.fallback.generate(backbone, a, b, seed, energy=energy,
                                      beats=beats, context=context)
