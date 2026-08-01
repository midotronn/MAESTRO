"""Integration tests for the FastAPI editor backend (REST + WebSocket) over a temp song folder."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from agentlodge.editor.window_edit import MockWindowGenerator  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import server.app as A
    # point the server at a temp media/sessions tree with one synthetic song
    media = tmp_path / "media" / "sng"
    media.mkdir(parents=True)
    motion = MockWindowGenerator().generate("edge", 0, 300, 0, energy=0.5, beats=None)
    np.save(media / "base_motion.npy", motion)
    np.save(media / "beats.npy", np.arange(0, 300, 15).astype(np.float32))
    (media / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # tiny stub
    monkeypatch.setattr(A, "MEDIA", tmp_path / "media")
    monkeypatch.setattr(A, "SESSIONS", tmp_path / "sessions")
    A._sessions.clear()
    return TestClient(A.app)


def test_lists_songs_and_opens_session(client):
    songs = client.get("/api/songs").json()["songs"]
    assert any(s["sid"] == "sng" for s in songs)
    st = client.post("/api/session/sng").json()
    assert st["n_frames"] == 300 and st["duration"] == 10.0 and st["n_beats"] == 20
    assert st["head"] and len(st["timeline"]) == 1


def test_edit_commits_and_moves_metrics(client):
    client.post("/api/session/sng")
    r = client.post("/api/session/sng/edit",
                    json={"a_sec": 3, "b_sec": 6, "instruction": "make this calmer"}).json()
    res = r["result"]
    assert res["ok"] and res["goal"]["objective"] == "agent"
    assert res["log"] and res["log"][0]["tool"] == "energy"      # agent chose the energy tool
    assert res["agent_summary"]
    assert res["metrics_after"]["energy"] < res["metrics_before"]["energy"]
    assert len(r["state"]["timeline"]) == 2 and r["state"]["can_undo"]


def test_undo_redo_restore_endpoints(client):
    client.post("/api/session/sng")
    st0 = client.post("/api/session/sng").json()
    root = st0["head"]
    client.post("/api/session/sng/edit",
                json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    u = client.post("/api/session/sng/undo").json()
    assert u["head"] == root and u["can_redo"]
    rd = client.post("/api/session/sng/redo").json()
    assert rd["head"] != root
    rs = client.post("/api/session/sng/restore", json={"ckpt_id": root}).json()
    assert rs["head"] == root


def test_websocket_streams_progress_then_final(client):
    client.post("/api/session/sng")
    with client.websocket_connect("/api/session/sng/edit_ws") as ws:
        ws.send_json({"a_sec": 3, "b_sec": 6, "instruction": "tighten to the beat"})
        n_prog, final = 0, None
        while True:
            ev = ws.receive_json()
            if ev["type"] == "progress":
                n_prog += 1
            elif ev["type"] == "final":
                final = ev
                break
            elif ev["type"] == "error":
                pytest.fail(f"ws error: {ev}")
    assert n_prog >= 1
    assert final["result"]["goal"]["objective"] == "agent"
    assert final["result"]["log"][0]["tool"] == "beat_align"    # agent snapped to the beat
    assert final["state"]["head"]


def test_unknown_song_404(client):
    assert client.post("/api/session/nope").status_code == 404


def test_compare_requires_an_edit_first(client):
    client.post("/api/session/sng")
    # no edit yet -> nothing to compare
    assert client.post("/api/session/sng/compare").status_code == 400


def test_compare_after_edit_renders_before_and_after(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None):
        captured["sid"] = sid
        captured["before"] = np.asarray(before)
        captured["after"] = np.asarray(after)
        captured["metrics"] = metrics
        R._cset(sid, status="rendering", progress=10)

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit",
                json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    j = client.post("/api/session/sng/compare").json()
    assert captured["sid"] == "sng"
    # window 3-6s at 30 fps -> 90 frames for BOTH the pre-edit and current clips
    assert captured["before"].shape[0] == 90 and captured["after"].shape[0] == 90
    # the before/after clips differ (the edit changed the window)
    assert not np.allclose(captured["before"], captured["after"])
    m = captured["metrics"]
    assert set(("before", "after", "window", "window_sec")) <= set(m)
    assert "bas" in m["before"] and "bas" in m["after"]
    assert m["window_sec"] == [3.0, 6.0]
    assert j["status"] in ("queued", "rendering", "idle")
    assert client.get("/api/session/sng/compare").json()["status"] in ("queued", "rendering", "idle")
