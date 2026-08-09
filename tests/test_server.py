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
    monkeypatch.delenv("AGENTLODGE_LIVE", raising=False)
    monkeypatch.delenv("MAESTRO_REQUIRE_MOTION_AUDIT", raising=False)
    # point the server at a temp media/sessions tree with one synthetic song
    media = tmp_path / "media" / "sng"
    media.mkdir(parents=True)
    motion = MockWindowGenerator().generate("edge", 0, 300, 0, energy=0.5, beats=None)
    np.save(media / "base_motion.npy", motion)
    beats = np.arange(0, 300, 15).astype(np.float32)
    strengths = np.full(len(beats), 0.1, dtype=np.float32)
    strengths[8] = 1.0
    np.save(media / "beats.npy", beats)
    np.save(media / "beat_strengths.npy", strengths)
    (media / "preview.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")  # tiny stub
    monkeypatch.setattr(A, "MEDIA", tmp_path / "media")
    monkeypatch.setattr(A, "SESSIONS", tmp_path / "sessions")
    A._sessions.clear()
    return TestClient(A.app)


def test_live_editor_enables_the_blocking_motion_audit(monkeypatch):
    import server.app as A

    monkeypatch.delenv("AGENTLODGE_LIVE", raising=False)
    monkeypatch.delenv("MAESTRO_REQUIRE_MOTION_AUDIT", raising=False)
    assert not A._motion_audit_required()
    monkeypatch.setenv("AGENTLODGE_LIVE", "1")
    assert A._motion_audit_required()


def test_lists_songs_and_opens_session(client):
    songs = client.get("/api/songs").json()["songs"]
    assert any(s["sid"] == "sng" for s in songs)
    st = client.post("/api/session/sng").json()
    assert st["n_frames"] == 300 and st["duration"] == 10.0 and st["n_beats"] == 20
    assert st["head"] and len(st["timeline"]) == 1


def test_lists_named_motions_from_the_shared_manifest(client):
    data = client.get("/api/motions").json()
    assert data["version"] == "1.1.2"
    assert len(data["motions"]) == 20
    clap = next(m for m in data["motions"] if m["id"] == "clap_single")
    assert clap["name"] == "Single clap"
    assert "clap" in clap["aliases"]
    assert clap["default_anchor"] == "beat"
    assert clap["default_direction"] == "auto"
    assert clap["directions"] == ["forward", "left", "right"]
    assert clap["minimum_seconds"] > 0
    assert clap["source"] and clap["license"] and clap["attribution"]


def test_named_beat_action_uses_the_strongest_window_beat(client):
    client.post("/api/session/sng")
    result = client.post(
        "/api/session/sng/edit",
        json={"a_sec": 0, "b_sec": 6, "instruction": "add a clap here"},
    ).json()["result"]
    report = next(step["motion_bank"] for step in result["log"] if "motion_bank" in step)
    assert report["anchor"] == "beat"
    assert report["event_frame"] == 120
    assert result["ok"]


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


def test_editor_review_actions_explain_the_user_flow():
    """The UI should expose review tasks, not ambiguous renderer implementation details."""
    from server.app import STATIC

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    js = (STATIC / "app.js").read_text(encoding="utf-8")

    assert "Review edit" in html
    assert "Before and after with music" in html
    assert "Render full dance" in html
    assert "Slower, for the final review" in html
    assert 'rel="icon"' in html
    assert (STATIC / "favicon.svg").is_file()
    assert 'id="renderBtn"' not in html
    assert "startFullRender" in js
    assert 'startRender("window")' not in js
    assert 'id="motionSuggestions"' in html
    assert 'id="motionPicker"' in html
    assert "20 supported common motions" in html
    assert "/api/motions" in js
    assert "clap to the right" in js
    assert "insertion is unavailable until it has its own visual audit" in js
    assert "follows the dance flow" in js

    tour = js[js.index("const TOUR_STEPS"):js.index("let tourIdx")]
    assert "\\u2014" not in tour and "—" not in tour
    assert 'el: "motionPicker"' in tour
    assert "slightly exaggerated so they read clearly" in tour
    assert "strongest beat in the selected window" in tour


def test_basic_auth_middleware_guards_when_env_set():
    """With MAESTRO_AUTH_* set, requests without/with-wrong credentials get 401; correct creds pass."""
    import importlib
    import server.app as A
    A = importlib.reload(A)  # re-evaluate module-level auth wiring with the env in place
    from fastapi.testclient import TestClient
    import base64
    try:
        import os
        os.environ["MAESTRO_AUTH_USER"] = "maestro"
        os.environ["MAESTRO_AUTH_PASS"] = "s3cret"
        A2 = importlib.reload(A)
        c = TestClient(A2.app)
        assert c.get("/project/").status_code in (401, 404)  # 401 when guarded (docs may be absent -> still 401 first)
        assert c.get("/api/songs").status_code == 401
        good = base64.b64encode(b"maestro:s3cret").decode()
        assert c.get("/api/songs", headers={"Authorization": f"Basic {good}"}).status_code == 200
        bad = base64.b64encode(b"maestro:wrong").decode()
        assert c.get("/api/songs", headers={"Authorization": f"Basic {bad}"}).status_code == 401
    finally:
        import os
        os.environ.pop("MAESTRO_AUTH_USER", None)
        os.environ.pop("MAESTRO_AUTH_PASS", None)
        importlib.reload(A)  # restore the open (unguarded) app for other tests


def test_compare_requires_an_edit_first(client):
    client.post("/api/session/sng")
    # no edit yet -> nothing to compare
    assert client.post("/api/session/sng/compare").status_code == 400


def test_compare_after_edit_renders_before_and_after(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None,
                   audio_wav=None, audio_start=0.0, audio_dur=0.0):
        captured["sid"] = sid
        captured["before"] = np.asarray(before)
        captured["after"] = np.asarray(after)
        captured["metrics"] = metrics
        captured["audio"] = (audio_wav, audio_start, audio_dur)
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
    assert set(("before", "after", "window", "window_sec", "before_id", "before_label")) <= set(m)
    assert "bas" in m["before"] and "bas" in m["after"]
    assert m["window_sec"] == [3.0, 6.0]
    # audio is requested for the window (start = a_sec, dur = window length)
    assert captured["audio"][1] == 3.0 and abs(captured["audio"][2] - 3.0) < 1e-6
    assert j["status"] in ("queued", "rendering", "idle")
    assert client.get("/api/session/sng/compare").json()["status"] in ("queued", "rendering", "idle")


def test_reset_clears_edit_history(client):
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit", json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    st = client.post("/api/session/sng/edit",
                     json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"}).json()["state"]
    assert len(st["timeline"]) >= 3 and st["can_undo"]
    r = client.post("/api/session/sng/reset").json()
    assert len(r["timeline"]) == 1                       # only the original root remains
    assert r["timeline"][0]["label"] == "original"
    assert not r["can_undo"] and not r["can_redo"]
    # a follow-up compare now fails (no edit to compare) -> the history really is empty
    assert client.post("/api/session/sng/compare").status_code == 400


def test_compare_from_id_selects_chosen_prior_version(client, monkeypatch):
    import server.rendering as R
    captured: dict = {}

    def fake_start(sid, before, after, media_dir, *, metrics=None,
                   audio_wav=None, audio_start=0.0, audio_dur=0.0):
        captured["before"] = np.asarray(before)
        captured["metrics"] = metrics
        R._cset(sid, status="rendering", progress=10)

    monkeypatch.setattr(R, "start_compare_render", fake_start)
    client.post("/api/session/sng")
    client.post("/api/session/sng/edit", json={"a_sec": 3, "b_sec": 6, "instruction": "more energetic"})
    st = client.post("/api/session/sng/edit",
                     json={"a_sec": 3, "b_sec": 6, "instruction": "calmer"}).json()["state"]
    tl = st["timeline"]
    root = next(c for c in tl if c.get("parent_id") is None)   # the pre-any-edit base state
    client.post("/api/session/sng/compare", json={"from_id": root["id"]})
    # comparing against the ROOT -> "before" is the untouched base window (chosen version honoured)
    assert captured["metrics"]["before_id"] == root["id"]
    assert captured["metrics"].get("before_label") is not None
    assert captured["before"].shape[0] == 90
