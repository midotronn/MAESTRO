"""Unit tests for the checkpoint DAG (undo/redo/restore/branch/persistence) and EditSession."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentlodge.editor.checkpoints import CheckpointStore
from agentlodge.editor.session import EditSession, SongAssets
from agentlodge.editor.window_edit import MockWindowGenerator


def _m(val: float, n: int = 20) -> np.ndarray:
    return np.full((n, 139), float(val), dtype=np.float32)


# --------------------------------------------------------------------------- store basics
def test_commit_moves_head_and_links_parent():
    s = CheckpointStore()
    c0 = s.commit(_m(0), label="root")
    c1 = s.commit(_m(1), label="e1")
    assert s.head == c1.id
    assert c1.parent_id == c0.id
    assert np.array_equal(s.current_motion(), _m(1))
    assert s.children(c0.id) == [c1.id]


def test_undo_redo():
    s = CheckpointStore()
    s.commit(_m(0))
    c1 = s.commit(_m(1))
    c2 = s.commit(_m(2))
    assert np.array_equal(s.current_motion(), _m(2))
    back = s.undo()
    assert back.id == c1.id and np.array_equal(s.current_motion(), _m(1))
    fwd = s.redo()
    assert fwd.id == c2.id and np.array_equal(s.current_motion(), _m(2))


def test_undo_at_root_returns_none():
    s = CheckpointStore()
    s.commit(_m(0))
    assert s.undo() is None or s.can_undo() is False


def test_restore_jumps_to_arbitrary_checkpoint():
    s = CheckpointStore()
    c0 = s.commit(_m(0))
    s.commit(_m(1))
    s.commit(_m(2))
    s.restore(c0.id)
    assert s.head == c0.id and np.array_equal(s.current_motion(), _m(0))
    assert s.can_redo() is False


def test_branch_forks_history():
    s = CheckpointStore()
    c0 = s.commit(_m(0))
    c1 = s.commit(_m(1))
    # branch off the ROOT, not c1
    cb = s.branch(c0.id, _m(9))
    assert cb.parent_id == c0.id
    assert set(s.children(c0.id)) == {c1.id, cb.id}
    assert s.head == cb.id and np.array_equal(s.current_motion(), _m(9))


def test_commit_after_undo_starts_new_branch():
    s = CheckpointStore()
    c0 = s.commit(_m(0))
    c1 = s.commit(_m(1))
    s.undo()                       # head back to c0
    c2 = s.commit(_m(2))           # new branch off c0
    assert c2.parent_id == c0.id
    assert set(s.children(c0.id)) == {c1.id, c2.id}
    assert s.can_redo() is False   # redo stack cleared by the new commit


def test_ancestry_and_timeline():
    s = CheckpointStore()
    c0 = s.commit(_m(0))
    c1 = s.commit(_m(1))
    c2 = s.commit(_m(2))
    assert s.ancestry(c2.id) == [c0.id, c1.id, c2.id]
    tl = s.timeline()
    assert len(tl) == 3 and any(e["is_head"] for e in tl)


def test_diff_reports_changed_span_and_metric_delta():
    s = CheckpointStore()
    a = _m(0, 40)
    b = a.copy()
    b[10:20] += 1.0
    ca = s.commit(a, metrics={"bas": 0.4})
    cb = s.commit(b, metrics={"bas": 0.6})
    d = s.diff(ca.id, cb.id)
    assert d["changed_frame_span"] == [10, 20]
    assert d["metric_delta"]["bas"] == 0.2


# --------------------------------------------------------------------------- persistence
def test_store_persistence_roundtrip(tmp_path):
    d = tmp_path / "ck"
    s = CheckpointStore(d)
    s.commit(_m(0))
    c1 = s.commit(_m(1))
    s.undo()
    s2 = CheckpointStore.load(d)
    assert s2.head == s.head
    assert len(s2) == 2
    assert np.array_equal(s2.current_motion(), _m(0))
    assert c1.id in s2
    assert np.array_equal(s2.motion(c1.id), _m(1))


# --------------------------------------------------------------------------- session
def _session(tmp_path=None, n=180):
    gen = MockWindowGenerator()
    base = gen.generate("edge", 0, n, 0, energy=0.5, beats=None)
    beats = np.arange(0, n, 15).astype(float)
    assets = SongAssets(sid="unit", beats=beats, fps=30)
    return EditSession(assets, base, gen, directory=str(tmp_path) if tmp_path else None,
                       k=5, max_cycles=2), base


def test_session_edit_commits_a_checkpoint():
    sess, base = _session()
    assert len(sess.store) == 1
    res = sess.edit(60, 120, "make this calmer")
    assert len(sess.store) == 2
    assert np.array_equal(sess.current_motion(), res.motion)
    # outside the window matches the original
    assert np.array_equal(sess.current_motion()[:60], base[:60])
    assert np.array_equal(sess.current_motion()[120:], base[120:])


def test_session_undo_restores_previous_motion():
    sess, base = _session()
    sess.edit(60, 120, "make this more energetic")
    edited = sess.current_motion().copy()
    sess.undo()
    assert np.array_equal(sess.current_motion(), base)
    sess.redo()
    assert np.array_equal(sess.current_motion(), edited)


def test_session_branch_edit_from_older_checkpoint():
    sess, base = _session()
    root_id = sess.store.ancestry()[0]
    sess.edit(60, 120, "make this calmer")
    res = sess.edit_from(root_id, 60, 120, "make this more energetic")
    # branched off root -> root now has two children
    assert len(sess.store.children(root_id)) == 2
    assert np.array_equal(sess.current_motion(), res.motion)


def test_session_persistence_roundtrip(tmp_path):
    sess, _ = _session(tmp_path)
    sess.edit(60, 120, "tighten to the beat")
    head_motion = sess.current_motion().copy()
    head_id = sess.store.head
    sess2 = EditSession.load(tmp_path, MockWindowGenerator())
    assert sess2.store.head == head_id
    assert np.array_equal(sess2.current_motion(), head_motion)
    assert sess2.assets.sid == "unit"
    assert sess2.assets.beats is not None and len(sess2.assets.beats) > 0
    # the reloaded session is still editable
    r = sess2.edit(30, 90, "make this calmer")
    assert r.motion.shape == head_motion.shape
