"""FastAPI backend for the AgentLODGE interactive dance editor (Phase 3).

Serves the web editor and exposes the edit session over REST + WebSocket:

* ``GET  /``                               -> the editor web UI
* ``GET  /api/songs``                      -> available songs (a ``server/media/<sid>/`` each)
* ``POST /api/session/{sid}``              -> create/load the session; returns duration, beats,
                                              checkpoint timeline, current metrics, preview URL
* ``GET  /api/session/{sid}/media/{name}`` -> stream the current preview video
* ``POST /api/session/{sid}/edit``         -> run one NL window edit (blocking) + commit
* ``WS   /api/session/{sid}/edit_ws``      -> run an edit while streaming live cycle progress
* ``POST /api/session/{sid}/undo|redo``    -> walk history
* ``POST /api/session/{sid}/restore``      -> roll back/forward to a checkpoint
* ``GET  /api/session/{sid}/timeline``     -> checkpoint tree

A song folder ``server/media/<sid>/`` holds ``base_motion.npy`` (Z-up 139), ``beats.npy`` (30 FPS
frame indices) and ``preview.mp4``. If a ``bank/`` subfolder with ``bank_<sid>_<bb>_seed<n>.npy``
exists the real backbone candidate bank is used (wrapped in a resilient fallback); otherwise the
offline :class:`MockWindowGenerator` drives edits so the UI is fully usable without a GPU.

Run:  uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import shutil
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agentlodge.editor.remote_generator import (
    BankWindowGenerator,
    LiveWindowGenerator,
    ResilientWindowGenerator,
)
from agentlodge.editor.session import EditSession, SongAssets
from agentlodge.editor.window_edit import MockWindowGenerator, _window_beats, window_metrics
from server import processing
from server import rendering

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"
STATIC = HERE / "static"
SESSIONS = HERE / "sessions"
FPS = 30

app = FastAPI(title="MAESTRO Interactive Editor")
_sessions: dict[str, EditSession] = {}


class BasicAuthMiddleware:
    """Env-gated HTTP Basic Auth over the whole app (HTTP + WebSocket).

    Enabled only when ``MAESTRO_AUTH_USER`` and ``MAESTRO_AUTH_PASS`` are set, so local dev stays
    open while a publicly-hosted instance (e.g. on the RunPod proxy) is protected. Checks the
    ``Authorization`` header directly so it also guards WebSocket upgrades, which
    ``@app.middleware('http')`` would miss. The browser caches the credentials from the first prompt
    and replays them on same-origin subresource + WS requests.
    """

    def __init__(self, app, user: str, password: str):
        self.app = app
        self.user = user
        self.password = password

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers") or [])
            if not self._authorized(headers.get(b"authorization", b"").decode()):
                if scope["type"] == "http":
                    await send({"type": "http.response.start", "status": 401, "headers": [
                        (b"www-authenticate", b'Basic realm="MAESTRO"'),
                        (b"content-type", b"text/plain; charset=utf-8")]})
                    await send({"type": "http.response.body", "body": b"Authentication required"})
                else:
                    await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)

    def _authorized(self, header: str) -> bool:
        if not header.startswith("Basic "):
            return False
        try:
            user, _, pw = base64.b64decode(header[6:]).decode().partition(":")
        except Exception:  # noqa: BLE001 - malformed header -> unauthorized
            return False
        return (secrets.compare_digest(user, self.user)
                and secrets.compare_digest(pw, self.password))


_AUTH_USER = os.environ.get("MAESTRO_AUTH_USER")
_AUTH_PASS = os.environ.get("MAESTRO_AUTH_PASS")
if _AUTH_USER and _AUTH_PASS:
    app.add_middleware(BasicAuthMiddleware, user=_AUTH_USER, password=_AUTH_PASS)


@app.on_event("startup")
def _prewarm() -> None:
    # warm the pod's torch/scipy page cache so the first render's FK is fast (best-effort, async)
    try:
        rendering.prewarm_pod()
    except Exception:  # noqa: BLE001
        pass


# --------------------------------------------------------------------------- session loading
def _song_dir(sid: str) -> Path:
    d = (MEDIA / sid).resolve()
    if not d.is_dir() or MEDIA not in d.parents:
        raise HTTPException(404, f"unknown song {sid!r}")
    return d


def _bank_or_mock(sid: str, d: Path):
    """The offline generator: real candidate bank if present (mock-backed), else pure mock."""
    bank_dir = d / "bank"
    fallback = MockWindowGenerator()
    if bank_dir.is_dir() and any(bank_dir.glob(f"bank_{sid}_*.npy")):
        bank = BankWindowGenerator.from_dir(bank_dir, sid, fallback=fallback)
        if bank.backbones:
            return ResilientWindowGenerator(bank, fallback), "bank"
    return fallback, "mock"


def _live_enabled() -> bool:
    return os.environ.get("AGENTLODGE_LIVE", "").strip().lower() in ("1", "true", "yes", "on")


def _make_generator(sid: str, d: Path):
    """Pick the window generator for a song.

    Live pod mode (``AGENTLODGE_LIVE=1`` + ``AGENTLODGE_POD_HOST`` set) queries LODGE/EDGE on the GPU
    pod on demand for an unbounded search, falling back to the local candidate bank (then the offline
    mock) whenever the pod can't produce a take -- so the UI never breaks. Otherwise we use the local
    bank if present, else the mock.
    """
    offline, offline_kind = _bank_or_mock(sid, d)
    if _live_enabled() and processing.pod_config().host:
        provider = processing.PodTakeProvider(sid, d / "bank")
        return LiveWindowGenerator(provider, fallback=offline), "live"
    return offline, offline_kind


def _live_edit_budget() -> tuple[int, int]:
    """(k, max_cycles) for live sessions -- small, because each new seed is minutes of GPU."""
    k = int(os.environ.get("AGENTLODGE_LIVE_K", "2"))
    cycles = int(os.environ.get("AGENTLODGE_LIVE_CYCLES", "2"))
    return max(1, k), max(1, cycles)


def _load_session(sid: str) -> EditSession:
    if sid in _sessions:
        return _sessions[sid]
    d = _song_dir(sid)
    motion_p, beats_p = d / "base_motion.npy", d / "beats.npy"
    if not motion_p.exists():
        raise HTTPException(404, f"song {sid!r} has no base_motion.npy")
    motion = np.load(motion_p).astype(np.float32)
    beats = np.load(beats_p).astype(np.float32) if beats_p.exists() else None
    generator, gkind = _make_generator(sid, d)
    assets = SongAssets(sid=sid, beats=beats, fps=FPS)
    api_key = os.environ.get("OPENAI_API_KEY") or None    # enables the LLM edit agent when present
    sess_dir = SESSIONS / sid
    if (sess_dir / "checkpoints" / "manifest.json").exists():
        sess = EditSession.load(sess_dir, generator, api_key=api_key)
    else:
        sess = EditSession(assets, motion, generator, directory=str(sess_dir), api_key=api_key)
    sess.generator_kind = gkind          # type: ignore[attr-defined]
    sess.agent_llm = bool(api_key)       # type: ignore[attr-defined]
    if gkind == "live":                  # each new seed is real GPU: search fewer, deeper
        sess.k, sess.max_cycles = _live_edit_budget()
    _sessions[sid] = sess
    return sess


def _session_state(sid: str, sess: EditSession) -> dict:
    head = sess.current()
    n = int(sess.current_motion().shape[0])
    return {
        "sid": sid,
        "fps": FPS,
        "n_frames": n,
        "duration": round(n / FPS, 2),
        "n_beats": int(len(sess.assets.beats)) if sess.assets.beats is not None else 0,
        "generator": getattr(sess, "generator_kind", "mock"),
        "agent_llm": bool(getattr(sess, "agent_llm", False)),
        "head": head.id if head else None,
        "metrics": head.metrics if head else {},
        "timeline": sess.timeline(),
        "preview_url": f"/api/session/{sid}/media/preview.mp4",
        "can_undo": sess.store.can_undo(),
        "can_redo": sess.store.can_redo(),
    }


# --------------------------------------------------------------------------- request models
class EditBody(BaseModel):
    a_sec: float
    b_sec: float
    instruction: str
    k: int | None = None
    max_cycles: int | None = None
    from_id: str | None = None       # branch from an older checkpoint


class RestoreBody(BaseModel):
    ckpt_id: str


class RenderBody(BaseModel):
    scope: str = "window"        # "window" (fast, silent) | "full" (whole song + music)
    a_sec: float | None = None
    b_sec: float | None = None


class CompareBody(BaseModel):
    from_id: str | None = None   # which prior checkpoint to use as "before" (default: head's parent)


# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    # Cache-bust app.js / style.css by file mtime so a redeploy is picked up on a normal refresh
    # (the assets are served without Cache-Control, so browsers were pinning the old versions).
    def _v(name: str) -> str:
        try:
            return str(int((STATIC / name).stat().st_mtime))
        except OSError:
            return "1"
    html = html.replace("/static/app.js", f"/static/app.js?v={_v('app.js')}")
    html = html.replace("/static/style.css", f"/static/style.css?v={_v('style.css')}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


def _display_name(d: Path) -> str:
    meta = d / "meta.json"
    if meta.exists():
        try:
            return str(json.loads(meta.read_text()).get("name") or d.name)
        except Exception:
            pass
    return d.name


@app.get("/api/songs")
def songs() -> dict:
    if not MEDIA.is_dir():
        return {"songs": []}
    out = []
    for d in sorted(MEDIA.iterdir()):
        if d.is_dir() and (d / "base_motion.npy").exists():
            out.append({"sid": d.name, "name": _display_name(d), "has_bank": (d / "bank").is_dir()})
    return {"songs": out}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    """Accept an audio file, then process it into a dance on the GPU pod (background job)."""
    name = file.filename or "song"
    if not any(name.lower().endswith(e) for e in (".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "please upload an audio file (.wav/.mp3/.m4a/.flac/.ogg)")
    sid = processing.slugify(name)
    d = (MEDIA / sid)
    d.mkdir(parents=True, exist_ok=True)
    src = d / ("source" + Path(name).suffix.lower())
    with open(src, "wb") as f:
        shutil.copyfileobj(file.file, f)
    wav = _ensure_wav(src)
    processing.start_processing(sid, wav, d, Path(name).stem)
    return {"sid": sid, "name": Path(name).stem, "status": "queued"}


def _ensure_wav(src: Path) -> Path:
    """Return a .wav for the upload (transcode non-wav with imageio-ffmpeg if needed)."""
    if src.suffix.lower() == ".wav":
        return src
    out = src.with_suffix(".wav")
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        import subprocess
        subprocess.run([ff, "-y", "-i", str(src), "-ar", "22050", "-ac", "1", str(out)],
                       capture_output=True, timeout=120)
        if out.exists():
            return out
    except Exception:
        pass
    return src


@app.get("/api/jobs/{sid}")
def job_status(sid: str) -> dict:
    return processing.get_job(sid)


@app.post("/api/session/{sid}")
def open_session(sid: str) -> dict:
    return _session_state(sid, _load_session(sid))


@app.get("/api/session/{sid}/media/{name}")
def media(sid: str, name: str) -> FileResponse:
    d = _song_dir(sid)
    p = (d / name).resolve()
    if d not in p.parents or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


def _clamp_window(sess: EditSession, a_sec: float, b_sec: float) -> tuple[int, int]:
    n = int(sess.current_motion().shape[0])
    a = int(max(0, min(round(a_sec * FPS), n - 2)))
    b = int(max(a + 1, min(round(b_sec * FPS), n)))
    return a, b


@app.post("/api/session/{sid}/edit")
def edit(sid: str, body: EditBody) -> dict:
    sess = _load_session(sid)
    a, b = _clamp_window(sess, body.a_sec, body.b_sec)
    if body.from_id:
        res = sess.edit_from(body.from_id, a, b, body.instruction,
                             k=body.k, max_cycles=body.max_cycles)
    else:
        res = sess.edit(a, b, body.instruction, k=body.k, max_cycles=body.max_cycles)
    return {"result": res.summary(), "cycles": res.cycles, "state": _session_state(sid, sess)}


@app.post("/api/session/{sid}/render")
def render(sid: str, body: RenderBody) -> dict:
    """Render the CURRENT edited motion on the pod's Blender; poll for status.

    scope="window" renders the section that was just edited (fast); an explicit a_sec/b_sec overrides
    it; scope="full" renders the whole dance with music. Falls back to full if there is no edit yet.
    """
    sess = _load_session(sid)
    motion = sess.current_motion()
    a = b = None
    scope = body.scope
    if scope == "window":
        if body.a_sec is not None and body.b_sec is not None and body.b_sec > body.a_sec + 0.1:
            a, b = _clamp_window(sess, body.a_sec, body.b_sec)
        else:
            head = sess.current()                      # default to the last edit's window
            win = (getattr(head, "edit", None) or {}).get("window") if head else None
            if win and len(win) == 2:
                a, b = int(win[0]), int(win[1])
            else:
                scope = "full"                         # nothing edited yet -> render the whole dance
    rendering.start_render(sid, motion, _song_dir(sid), scope=scope, a=a, b=b)
    return rendering.get_render_job(sid)


@app.get("/api/session/{sid}/render")
def render_status(sid: str) -> dict:
    return rendering.get_render_job(sid)


def _song_wav(sid: str) -> str | None:
    """Best-effort path to the song's wav for muxing window audio (editor is co-located on the pod)."""
    ws = os.environ.get("WORKSPACE", "/workspace")
    for p in (Path(ws) / "LODGE" / "data" / "finedance" / "music_wav" / f"{sid}.wav",
              _song_dir(sid) / f"{sid}.wav"):
        if p.exists():
            return str(p)
    return None


@app.post("/api/session/{sid}/compare")
def compare(sid: str, body: CompareBody | None = None) -> dict:
    """Render the edited window as two synced clips: the CURRENT state (after) vs a chosen PRIOR
    version (before). ``from_id`` selects which prior checkpoint to compare against (default: the
    head's parent, i.e. the state right before the last edit). The window is the head's edit window,
    and the window's music is muxed in so the comparison plays with sound.
    """
    sess = _load_session(sid)
    head = sess.current()
    if head is None or not head.edit or not head.parent_id:
        raise HTTPException(400, "make an edit first, then compare before and after.")
    win = (head.edit or {}).get("window")
    if not win or len(win) != 2:
        raise HTTPException(400, "the last edit has no window to compare.")
    from_id = body.from_id if body else None
    before_id = from_id if (from_id and from_id in sess.store and from_id != head.id) \
        else head.parent_id
    after = sess.current_motion()
    before = sess.store.motion(before_id)
    n = min(int(before.shape[0]), int(after.shape[0]))
    a = int(max(0, min(int(win[0]), n - 2)))
    b = int(max(a + 1, min(int(win[1]), n)))
    wb = _window_beats(sess.assets.beats, a, b) if sess.assets.beats is not None else None
    before_ck = sess.store.get(before_id)
    metrics = {
        "before": window_metrics(before[a:b], wb),
        "after": window_metrics(after[a:b], wb),
        "window": [a, b],
        "window_sec": [round(a / FPS, 2), round(b / FPS, 2)],
        "before_id": before_id,
        "before_label": (before_ck.label or "original") if before_ck else "original",
    }
    rendering.start_compare_render(
        sid, before[a:b], after[a:b], _song_dir(sid), metrics=metrics,
        audio_wav=_song_wav(sid), audio_start=a / FPS, audio_dur=(b - a) / FPS)
    return rendering.get_compare_job(sid)


@app.get("/api/session/{sid}/compare")
def compare_status(sid: str) -> dict:
    return rendering.get_compare_job(sid)


@app.post("/api/session/{sid}/undo")
def undo(sid: str) -> dict:
    sess = _load_session(sid)
    sess.undo()
    return _session_state(sid, sess)


@app.post("/api/session/{sid}/redo")
def redo(sid: str) -> dict:
    sess = _load_session(sid)
    sess.redo()
    return _session_state(sid, sess)


@app.post("/api/session/{sid}/restore")
def restore(sid: str, body: RestoreBody) -> dict:
    sess = _load_session(sid)
    sess.restore(body.ckpt_id)
    return _session_state(sid, sess)


@app.get("/api/session/{sid}/timeline")
def timeline(sid: str) -> dict:
    sess = _load_session(sid)
    return {"timeline": sess.timeline(), "head": sess.store.head}


@app.websocket("/api/session/{sid}/edit_ws")
async def edit_ws(ws: WebSocket, sid: str) -> None:
    """Run an edit in a worker thread while streaming live cycle-progress events to the client."""
    await ws.accept()
    try:
        req = await ws.receive_json()
        sess = _load_session(sid)
        a, b = _clamp_window(sess, float(req["a_sec"]), float(req["b_sec"]))
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def progress_cb(ev: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "progress", **ev})

        async def run_edit():
            def _do():
                if req.get("from_id"):
                    return sess.edit_from(req["from_id"], a, b, req["instruction"],
                                          k=req.get("k"), max_cycles=req.get("max_cycles"),
                                          progress_cb=progress_cb)
                return sess.edit(a, b, req["instruction"], k=req.get("k"),
                                 max_cycles=req.get("max_cycles"), progress_cb=progress_cb)
            res = await loop.run_in_executor(None, _do)
            await queue.put({"type": "final", "result": res.summary(), "cycles": res.cycles,
                             "state": _session_state(sid, sess)})

        task = asyncio.create_task(run_edit())
        while True:
            ev = await queue.get()
            await ws.send_json(ev)
            if ev.get("type") == "final":
                break
        await task
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


if STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

DOCS = HERE.parent / "docs"
if DOCS.is_dir():
    app.mount("/project", StaticFiles(directory=str(DOCS), html=True), name="project")
