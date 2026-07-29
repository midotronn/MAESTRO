"""Interactive editable-dance framework (Phase 1 core).

Turns AgentLODGE from a one-shot generator into an interactive editor: a user selects a time
window of the rendered dance and gives a natural-language instruction ("more energetic", "more on
beat", "calmer"); an AgentBanana-style agent realizes it by re-querying LODGE/EDGE over just that
window in a bounded propose -> generate(K) -> splice -> verify -> refine cycle, preserving the rest
of the dance exactly, and records an undoable checkpoint.

This package is the local, GPU-free, unit-testable core:

* :mod:`agentlodge.editor.checkpoints` -- append-only checkpoint DAG (undo/redo/restore/branch).
* :mod:`agentlodge.editor.window_edit` -- the windowed regen loop + ``WindowGenerator`` seam
  (``MockWindowGenerator`` for offline tests; a real pod-backed generator lands in Phase 2).
* :mod:`agentlodge.editor.session` -- ``EditSession`` tying assets + state + checkpoints together.

The heavy backbones are reached only through the ``WindowGenerator`` protocol, so everything here
runs and is tested without torch or a GPU.
"""

from __future__ import annotations

from agentlodge.editor.checkpoints import Checkpoint, CheckpointStore
from agentlodge.editor.remote_generator import BankWindowGenerator, ResilientWindowGenerator
from agentlodge.editor.session import EditSession, SongAssets
from agentlodge.editor.window_edit import (
    EditGoal,
    MockWindowGenerator,
    WindowEditResult,
    WindowGenerator,
    apply_window_edit,
    goal_reward,
    parse_window_instruction,
    reward_weights_for,
    splice_window,
)

__all__ = [
    "BankWindowGenerator",
    "Checkpoint",
    "CheckpointStore",
    "EditGoal",
    "EditSession",
    "MockWindowGenerator",
    "ResilientWindowGenerator",
    "SongAssets",
    "WindowEditResult",
    "WindowGenerator",
    "apply_window_edit",
    "goal_reward",
    "parse_window_instruction",
    "reward_weights_for",
    "splice_window",
]
