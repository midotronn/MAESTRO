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
frame indices), optional ``beat_strengths.npy`` (one onset-salience value per beat), and
``preview.mp4``. If a ``bank/`` subfolder with ``bank_<sid>_<bb>_seed<n>.npy`` exists the real
backbone candidate bank is used (wrapped in a resilient fallback); otherwise the offline
:class:`MockWindowGenerator` drives edits so the UI is fully usable without a GPU.

Run:  uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import secrets
import shutil
import time
from pathlib import Path

import numpy as np
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agentlodge.editor.remote_generator import (
    BankWindowGenerator,
    LiveWindowGenerator,
    ResilientWindowGenerator,
)
from agentlodge.editor.motion_audit import validate_audit_receipt
from agentlodge.editor.motion_bank import default_motion_bank
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
_PLANNER_STATUS: dict[str, object] = {
    "configured": False,
    "verified": False,
    "model": os.environ.get("AGENTLODGE_PLANNER_MODEL", "gpt-4o"),
    "message": "planner verification has not run",
}


@app.middleware("http")
async def _capture_request_start(request: Request, call_next):
    request.state.maestro_received_at = time.time()
    return await call_next(request)


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
def _enforce_motion_audit() -> None:
    """A live editor may not start with stale or incomplete visual evidence."""
    if _unaudited_research_enabled():
        logging.getLogger("uvicorn.error").warning(
            "motion audit gate BYPASSED for unaudited research mode"
        )
        return
    if not _motion_audit_required():
        return
    receipt = validate_audit_receipt()
    logging.getLogger("uvicorn.error").info(
        "motion audit gate passed (%s, %d cases)",
        receipt["motion_fingerprint"][:12],
        len(receipt["cases"]),
    )


@app.on_event("startup")
def _prewarm() -> None:
    # warm the pod's torch/scipy page cache so the first render's FK is fast (best-effort, async)
    try:
        rendering.prewarm_pod()
    except Exception:  # noqa: BLE001
        pass
    try:
        from server import warm_render
        warm_render.ensure_configured_pool(wait_ready=0)
    except Exception:  # noqa: BLE001 - cold rendering remains available
        pass


@app.on_event("startup")
def _report_planner() -> None:
    """Say at startup which planner the editor will use, and why.

    Without a key the editor silently degrades to keyword planning, and the only sign is a small
    tag on an edit result reading "offline planner". That is easy to miss and easier to
    misdiagnose -- it looks like the edit went wrong rather than like the server was launched
    without a credential. Launching is exactly when this is worth knowing.
    """
    # Deliberately uvicorn's own logger rather than a module one: uvicorn configures logging for
    # its loggers only, so a __name__ logger emits nothing under the way this server is actually
    # run -- verified by grepping the pod's log and finding zero lines. Do not "tidy" this back.
    logger = logging.getLogger("uvicorn.error")
    if _openai_api_key():
        logger.info("LLM edit planner enabled (OpenAI credential present)")
    else:
        logger.warning(
            "LLM edit planner DISABLED: no OPENAI_API_KEY or readable OAI_KEY_FILE. "
            "Edits will use the offline keyword planner and results will be tagged "
            "'offline planner' in the UI."
        )


def _probe_llm_planner(api_key: str) -> str:
    from agentlodge.editor.agent_edit import plan_edit

    plan = plan_edit(
        "make the selected window more energetic",
        {"energy": 0.5, "bas": 0.5, "jerk": 0.5, "foot": 0.5},
        0.0,
        5.0,
        api_key=api_key,
    )
    if plan.planner != "llm":
        raise RuntimeError(plan.planner_note or f"planner returned {plan.planner!r}")
    return plan.planner_note or "AI agent (LLM reasoning)"


@app.on_event("startup")
def _verify_planner() -> None:
    global _PLANNER_STATUS

    key = _openai_api_key()
    model = os.environ.get("AGENTLODGE_PLANNER_MODEL", "gpt-4o")
    verify = os.environ.get("AGENTLODGE_VERIFY_LLM_PLANNER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    required = os.environ.get("AGENTLODGE_REQUIRE_LLM_PLANNER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not key:
        _PLANNER_STATUS = {
            "configured": False,
            "verified": False,
            "model": model,
            "message": "no OpenAI credential is configured",
        }
        if required:
            raise RuntimeError("LLM planner is required but no OpenAI credential is configured")
        return
    if not verify:
        _PLANNER_STATUS = {
            "configured": True,
            "verified": False,
            "model": model,
            "message": "credential present; live verification disabled",
        }
        return
    try:
        message = _probe_llm_planner(key)
    except Exception as exc:
        _PLANNER_STATUS = {
            "configured": True,
            "verified": False,
            "model": model,
            "message": f"live verification failed: {type(exc).__name__}",
        }
        if required:
            raise RuntimeError("LLM planner live verification failed") from exc
        return
    _PLANNER_STATUS = {
        "configured": True,
        "verified": True,
        "model": model,
        "message": message,
    }


# --------------------------------------------------------------------------- session loading
def _openai_api_key() -> str | None:
    value = os.environ.get("OPENAI_API_KEY", "").strip()
    if value:
        return value
    configured = os.environ.get("OAI_KEY_FILE", "").strip()
    candidates = (
        [Path(configured).expanduser()]
        if configured
        else [
            Path.home() / ".oai_key",
            Path(os.environ.get("WORKSPACE", "/workspace")) / ".oai_key",
        ]
    )
    for key_file in candidates:
        try:
            value = key_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


@app.get("/api/planner/status")
def planner_status() -> dict:
    return dict(_PLANNER_STATUS)


def _song_dir(sid: str) -> Path:
    media_root = MEDIA.resolve()
    d = (media_root / sid).resolve()
    if not d.is_dir() or media_root not in d.parents:
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


def _unaudited_research_enabled() -> bool:
    return os.environ.get("MAESTRO_ALLOW_UNAUDITED_RESEARCH", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _motion_audit_required() -> bool:
    if _unaudited_research_enabled():
        return False
    explicit = os.environ.get("MAESTRO_REQUIRE_MOTION_AUDIT", "").strip().lower()
    return _live_enabled() or explicit in ("1", "true", "yes", "on")


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
    motion_p = d / "base_motion.npy"
    beats_p = d / "beats.npy"
    strengths_p = d / "beat_strengths.npy"
    if not motion_p.exists():
        raise HTTPException(404, f"song {sid!r} has no base_motion.npy")
    from agentlodge.dance.format import to_editor139

    motion = to_editor139(np.load(motion_p))
    beats = np.load(beats_p).astype(np.float32) if beats_p.exists() else None
    beat_strengths = (
        np.load(strengths_p).astype(np.float32) if strengths_p.exists() else None
    )
    if beat_strengths is not None and (
        beats is None or beat_strengths.size != beats.size
    ):
        raise HTTPException(
            500,
            f"song {sid!r} has {beat_strengths.size} beat strengths for "
            f"{0 if beats is None else beats.size} beats",
        )
    generator, gkind = _make_generator(sid, d)
    assets = SongAssets(
        sid=sid, beats=beats, fps=FPS, beat_strengths=beat_strengths,
    )
    api_key = _openai_api_key()                           # enables the LLM edit agent when present
    sess_dir = SESSIONS / sid
    if (sess_dir / "checkpoints" / "manifest.json").exists():
        sess = EditSession.load(sess_dir, generator, api_key=api_key)
        if beat_strengths is not None:
            sess.assets.beat_strengths = beat_strengths
            sess.assets.save(sess_dir)
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


class BrowserTimingBody(BaseModel):
    request_id: str
    browser_started_at_ms: float
    browser_completed_at_ms: float
    browser_upload_seconds: float
    browser_total_seconds: float
    source_bytes: int = 0


# --------------------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    # Cache-bust editor assets by file mtime so a redeploy is picked up on a normal refresh
    # (the assets are served without Cache-Control, so browsers were pinning the old versions).
    def _v(name: str) -> str:
        try:
            return str(int((STATIC / name).stat().st_mtime))
        except OSError:
            return "1"
    html = html.replace("/static/app.js", f"/static/app.js?v={_v('app.js')}")
    html = html.replace(
        "/static/compare_highlight.js",
        f"/static/compare_highlight.js?v={_v('compare_highlight.js')}",
    )
    html = html.replace("/static/style.css", f"/static/style.css?v={_v('style.css')}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


def _song_metadata(d: Path) -> dict:
    meta = d / "meta.json"
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logging.getLogger("uvicorn.error").warning(
                "Ignoring invalid song metadata %s: %s",
                meta,
                exc,
            )
    return {}


@app.get("/api/songs")
def songs() -> dict:
    if not MEDIA.is_dir():
        return {"songs": []}
    interview_mode = os.environ.get("MAESTRO_INTERVIEW_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    out = []
    for d in sorted(MEDIA.iterdir()):
        if d.is_dir() and (d / "base_motion.npy").exists():
            metadata = _song_metadata(d)
            if interview_mode and metadata.get("interview") is not True:
                continue
            try:
                order = float(metadata.get("order", 1000))
            except (TypeError, ValueError):
                order = 1000
            out.append(
                {
                    "sid": d.name,
                    "name": str(metadata.get("name") or d.name),
                    "has_bank": (d / "bank").is_dir(),
                    "_order": order,
                }
            )
    out.sort(key=lambda song: (song["_order"], song["name"].casefold(), song["sid"]))
    for song in out:
        song.pop("_order", None)
    return {"songs": out}


@app.get("/api/motions")
def motions() -> dict:
    bank = default_motion_bank()
    return {"version": bank.version, "motions": bank.list_public()}


@app.post("/api/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    request_id: str | None = Form(None),
    client_started_at_ms: float | None = Form(None),
) -> dict:
    """Accept an audio file, then process it into a dance on the GPU pod (background job)."""
    upload_received_at = time.time()
    name = file.filename or "song"
    if not any(name.lower().endswith(e) for e in (".wav", ".mp3", ".m4a", ".flac", ".ogg")):
        raise HTTPException(400, "please upload an audio file (.wav/.mp3/.m4a/.flac/.ogg)")
    sid = processing.slugify(name)
    trace_id = request_id or request.headers.get("x-maestro-request-id") or secrets.token_hex(12)
    d = (MEDIA / sid)
    d.mkdir(parents=True, exist_ok=True)
    src = d / ("source" + Path(name).suffix.lower())
    digest = hashlib.sha256()
    source_bytes = 0
    with open(src, "wb") as f:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            digest.update(chunk)
            source_bytes += len(chunk)
    wav = _ensure_wav(src)
    audio_ready_at = time.time()
    processing.start_processing(
        sid,
        wav,
        d,
        Path(name).stem,
        request_id=trace_id,
        request_received_at=getattr(
            request.state,
            "maestro_received_at",
            upload_received_at,
        ),
        upload_received_at=upload_received_at,
        audio_ready_at=audio_ready_at,
        client_started_at_ms=client_started_at_ms,
        source_bytes=source_bytes,
        source_sha256=digest.hexdigest(),
    )
    return {
        "sid": sid,
        "name": Path(name).stem,
        "status": "queued",
        "request_id": trace_id,
        "source_bytes": source_bytes,
    }


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


@app.post("/api/jobs/{sid}/browser-timing")
def browser_timing(sid: str, body: BrowserTimingBody) -> dict:
    try:
        payload = body.model_dump() if hasattr(body, "model_dump") else body.dict()
        timing = processing.record_browser_timing(sid, payload)
    except KeyError:
        raise HTTPException(404, "unknown upload job") from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"ok": True, "timing": timing}


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
    rendering.start_render(
        sid,
        motion,
        _song_dir(sid),
        scope=scope,
        a=a,
        b=b,
        audio_wav=_song_wav(sid) if scope == "full" else None,
    )
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


@app.post("/api/session/{sid}/reset")
def reset_session(sid: str) -> dict:
    """Clear the edit history: drop every checkpoint and start over from the original dance."""
    _sessions.pop(sid, None)                              # evict the cached session
    sess_dir = SESSIONS / sid
    if sess_dir.exists():
        shutil.rmtree(sess_dir, ignore_errors=True)      # remove the persisted checkpoint tree
    sess = _load_session(sid)                            # rebuild a fresh session (root only)
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

STUDY = HERE.parent / "experiments" / "user_study" / "stimuli" / "player"
if STUDY.is_dir():
    app.mount("/study", StaticFiles(directory=str(STUDY), html=True), name="study")

DOCS = HERE.parent / "docs"
if DOCS.is_dir():
    app.mount("/project", StaticFiles(directory=str(DOCS), html=True), name="project")
