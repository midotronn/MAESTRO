"""Song upload processing: dispatch the full generation pipeline to the GPU pod.

Generating a dance for an arbitrary song is the expensive pipeline (Jukebox feature extraction +
two diffusion backbones + render), which only runs on the GPU pod. This module accepts an uploaded
audio file, then in a background thread copies it to the configured pod, runs
``scripts/process_song.sh`` there, and pulls the results (``base_motion.npy`` / ``beats.npy`` /
``preview.mp4`` / the candidate ``bank/``) into ``server/media/<sid>/`` so the new song appears in
the editor. Job status is tracked in-memory and polled by the UI.

The pod connection is configured via env (same vars as scripts/pod.ps1):
    AGENTLODGE_POD_HOST, AGENTLODGE_POD_PORT, AGENTLODGE_POD_KEY, AGENTLODGE_POD_WS,
    AGENTLODGE_POD_USER, AGENTLODGE_BANK_K
If no pod is configured or reachable, the job ends with a clear, actionable error (the editor stays
usable for the songs that are already processed).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()
REPO = Path(__file__).resolve().parents[1]


@dataclass
class PodConfig:
    host: str | None = None
    port: str = "22"
    key: str = os.path.expanduser("~/.ssh/id_ed25519")
    ws: str = "/workspace"
    user: str = "root"
    bank_k: str = "4"

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def pod_config() -> PodConfig:
    return PodConfig(
        host=os.environ.get("AGENTLODGE_POD_HOST"),
        port=os.environ.get("AGENTLODGE_POD_PORT", "22"),
        key=os.environ.get("AGENTLODGE_POD_KEY", os.path.expanduser("~/.ssh/id_ed25519")),
        ws=os.environ.get("AGENTLODGE_POD_WS", "/workspace"),
        user=os.environ.get("AGENTLODGE_POD_USER", "root"),
        bank_k=os.environ.get("AGENTLODGE_BANK_K", "4"),
    )


def slugify(name: str) -> str:
    stem = Path(name).stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")[:24] or "song"
    return f"{slug}_{int(time.time()) % 100000}"


# --------------------------------------------------------------------------- job state
def get_job(sid: str) -> dict:
    with _LOCK:
        return dict(_JOBS.get(sid, {"status": "unknown", "message": "no such job", "progress": 0}))


def _set(sid: str, **kw) -> None:
    with _LOCK:
        _JOBS.setdefault(sid, {}).update(kw)


def _co_located(cfg: PodConfig) -> bool:
    return os.name != "nt" and cfg.host is not None and cfg.host.lower() in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _pod_repo(cfg: PodConfig) -> str:
    configured = os.environ.get("AGENTLODGE_POD_REPO", "").strip()
    if configured:
        return configured.rstrip("/")
    if _co_located(cfg):
        return str(REPO)
    return f"{cfg.ws}/AgentLODGE"


# --------------------------------------------------------------------------- ssh helpers
def _ssh(cfg: PodConfig, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", f"ConnectTimeout=15",
         "-p", cfg.port, "-i", cfg.key, cfg.target, cmd],
        capture_output=True, text=True, timeout=timeout)


def _scp_to(cfg: PodConfig, local: str, remote: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["scp", "-P", cfg.port, "-i", cfg.key, local, f"{cfg.target}:{remote}"],
                          capture_output=True, text=True, timeout=timeout)


def _scp_from(cfg: PodConfig, remote: str, local: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(["scp", "-P", cfg.port, "-i", cfg.key, f"{cfg.target}:{remote}", local],
                          capture_output=True, text=True, timeout=timeout)


def _run_pod(cfg: PodConfig, cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run directly in hosted mode; use SSH only when the GPU pod is remote."""
    if _co_located(cfg):
        return subprocess.run(
            ["bash", "-lc", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    return _ssh(cfg, cmd, timeout=timeout)


# --------------------------------------------------------------------------- live take provider
class PodTakeProvider:
    """Fetch a full LODGE/EDGE take for ``(backbone, seed)`` from the GPU pod (live pod mode).

    Wired as the ``take_provider`` of
    :class:`~agentlodge.editor.remote_generator.LiveWindowGenerator`. Each call returns a full Z-up
    139 take (or ``None`` on any failure, so the generator can fall back). Results are cached in the
    song's local ``bank/`` directory using the candidate-bank naming, so repeated seeds are served
    from disk and the persistent bank grows as a side effect of live editing.

    The pod-side generator (``scripts/gen_take.py``) is uploaded lazily on first use.
    """

    def __init__(self, sid: str, bank_dir: Path, cfg: PodConfig | None = None, *,
                 gen_timeout: int = 60 * 20):
        self.sid = sid
        self.bank_dir = Path(bank_dir)
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or pod_config()
        self.gen_timeout = int(gen_timeout)
        self._script_shipped = False

    # -- helpers ------------------------------------------------------------
    def _local_path(self, backbone: str, seed: int, window=None) -> Path:
        if window is not None:
            return self.bank_dir / f"bank_{self.sid}_{backbone}_seed{seed}_w{window[0]}_{window[1]}.npy"
        return self.bank_dir / f"bank_{self.sid}_{backbone}_seed{seed}.npy"

    def _ensure_script(self) -> bool:
        if self._script_shipped:
            return True
        ws = self.cfg.ws
        if _ssh(self.cfg, f"mkdir -p {ws}/AgentLODGE/scripts", timeout=30).returncode != 0:
            return False
        r = _scp_to(self.cfg, str(REPO / "scripts" / "gen_take.py"),
                    f"{ws}/AgentLODGE/scripts/gen_take.py")
        if r.returncode != 0:
            return False
        _ssh(self.cfg, f"sed -i 's/\\r$//' {ws}/AgentLODGE/scripts/gen_take.py", timeout=30)
        self._script_shipped = True
        return True

    # -- take_provider callable --------------------------------------------
    def __call__(self, backbone: str, seed: int, a: int | None = None, b: int | None = None):
        import numpy as np

        window = (int(a), int(b)) if (a is not None and b is not None) else None
        local = self._local_path(backbone, int(seed), window)
        # a full-song seed already on disk (the base bank) also serves as the source when no window
        if local.exists():                                   # already have it (bank or prior live pull)
            try:
                return np.load(local).astype(np.float32)
            except Exception:                                # noqa: BLE001 - corrupt cache: regenerate
                local.unlink(missing_ok=True)

        # Warm LODGE daemon fast path: when the editor is co-located on the pod and the song is
        # preprocessed for generation, a persistent daemon (models preloaded) serves the window in
        # seconds instead of reloading the checkpoints per seed. EDGE / any failure returns None here
        # and falls through to the per-call gen_take path below.
        if window is not None:
            try:
                from server import warm_gen
                wp = warm_gen.warm_generate(self.sid, backbone, int(seed), window[0], window[1],
                                            timeout=self.gen_timeout)
                if wp is not None:
                    arr = np.load(wp).astype(np.float32)
                    try:
                        np.save(local, arr)                  # cache into the song bank too
                    except Exception:                        # noqa: BLE001
                        pass
                    return arr
            except Exception:                                # noqa: BLE001 - never break the edit
                pass

        if not self.cfg.host:
            return None
        try:
            if not self._ensure_script():
                return None
            ws = self.cfg.ws
            win_args = f" {window[0]} {window[1]}" if window is not None else ""
            run = _ssh(self.cfg,
                       f"cd {ws}/AgentLODGE && WORKSPACE={ws} "
                       f"{_venv_python(self.cfg)} scripts/gen_take.py {self.sid} {backbone} "
                       f"{int(seed)}{win_args}",
                       timeout=self.gen_timeout)
            remote = _parse_take_path(run.stdout)
            if remote is None:
                return None
            if _scp_from(self.cfg, remote, str(local), timeout=300).returncode != 0:
                return None
            return np.load(local).astype(np.float32)
        except subprocess.TimeoutExpired:
            return None
        except Exception:                                    # noqa: BLE001 - never break the edit
            return None


def _venv_python(cfg: PodConfig) -> str:
    """Prefer the configured generation interpreter, else the active pod checkout's venv."""
    return os.environ.get(
        "AGENTLODGE_POD_PYTHON",
        f"{_pod_repo(cfg)}/.venv/bin/python",
    )


def _parse_take_path(stdout: str) -> str | None:
    """Extract the take path from ``TAKE_DONE <path> <n>`` / ``TAKE_CACHED <path>`` output."""
    for line in reversed((stdout or "").splitlines()):
        line = line.strip()
        if line.startswith(("TAKE_DONE ", "TAKE_CACHED ")):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return None


def _launch_background_bank(
    cfg: PodConfig,
    sid: str,
    out: str,
    pod_repo: str,
    pod_python: str,
) -> bool:
    """Start seeds 1..K-1 only after the initial preview is ready."""
    try:
        bank_k = max(1, int(cfg.bank_k))
    except (TypeError, ValueError):
        bank_k = 4
    if bank_k <= 1:
        return False

    done = f"{out}/bank.done"
    failed = f"{out}/bank.failed"
    log = f"{out}/bank.log"
    bank_script = f"{pod_repo}/scripts/build_window_bank.py"
    bank_pattern = f"bank_{sid}_*.npy"
    expected_files = bank_k * 2
    inner = (
        f"cd {shlex.quote(cfg.ws)} && "
        f"WORKSPACE={shlex.quote(cfg.ws)} "
        f"AGENTLODGE_ROOT={shlex.quote(pod_repo)} "
        f"AGENTLODGE_BANK_K={bank_k} "
        f"{shlex.quote(pod_python)} {shlex.quote(bank_script)} {shlex.quote(sid)} && "
        f"test \"$(find {shlex.quote(cfg.ws)} -maxdepth 1 -type f "
        f"-name {shlex.quote(bank_pattern)} | wc -l)\" -ge {expected_files} && "
        f"find {shlex.quote(cfg.ws)} -maxdepth 1 -type f "
        f"-name {shlex.quote(bank_pattern)} "
        f"-exec cp -f {{}} {shlex.quote(out)} \\; && "
        f"touch {shlex.quote(done)}"
    )
    wrapped = f"{{ {inner}; }} || {{ touch {shlex.quote(failed)}; exit 1; }}"
    command = (
        f"rm -f {shlex.quote(done)} {shlex.quote(failed)}; "
        f"setsid bash -c {shlex.quote(wrapped)} "
        f"> {shlex.quote(log)} 2>&1 < /dev/null & echo BANK_PID=$!"
    )
    run = _run_pod(cfg, command, timeout=60)
    return run.returncode == 0 and "BANK_PID=" in run.stdout


def _start_bank_sync(sid: str, cfg: PodConfig, out: str, bank_dir: Path) -> None:
    """Copy the deferred bank into the editor when its detached build completes."""

    def sync() -> None:
        timeout = int(os.environ.get("AGENTLODGE_BANK_TIMEOUT", str(3 * 60 * 60)))
        deadline = time.monotonic() + max(60, timeout)
        done = f"{out}/bank.done"
        failed = f"{out}/bank.failed"
        log = f"{out}/bank.log"
        while time.monotonic() < deadline:
            if _co_located(cfg):
                state = (
                    "DONE"
                    if Path(done).exists()
                    else "FAILED"
                    if Path(failed).exists()
                    else ""
                )
            else:
                check = _ssh(
                    cfg,
                    f"test -f {shlex.quote(done)} && echo DONE || "
                    f"(test -f {shlex.quote(failed)} && echo FAILED || true)",
                    timeout=30,
                )
                state = check.stdout.strip()
            if state == "DONE":
                bank_dir.mkdir(parents=True, exist_ok=True)
                try:
                    expected_files = max(1, int(cfg.bank_k)) * 2
                except (TypeError, ValueError):
                    expected_files = 8
                if _co_located(cfg):
                    for source in Path(out).glob(f"bank_{sid}_*.npy"):
                        shutil.copy2(source, bank_dir / source.name)
                    copied = len(list(bank_dir.glob(f"bank_{sid}_*.npy"))) >= expected_files
                else:
                    pull = _scp_from(
                        cfg,
                        f"{out}/bank_{sid}_*.npy",
                        str(bank_dir) + "/",
                        timeout=600,
                    )
                    copied = (
                        pull.returncode == 0
                        and len(list(bank_dir.glob(f"bank_{sid}_*.npy"))) >= expected_files
                    )
                _set(
                    sid,
                    bank_status="ready" if copied else "error",
                    bank_error="" if copied else "generated bank could not be copied",
                )
                return
            if state == "FAILED":
                detail = ""
                if not _co_located(cfg):
                    tail = _ssh(cfg, f"tail -n 20 {shlex.quote(log)}", timeout=30)
                    detail = (tail.stderr or tail.stdout).strip()
                _set(
                    sid,
                    bank_status="error",
                    bank_error=detail or "background bank failed",
                )
                return
            time.sleep(15)
        _set(sid, bank_status="error", bank_error="background bank timed out")

    threading.Thread(target=sync, name=f"bank-sync-{sid}", daemon=True).start()



# --------------------------------------------------------------------------- pipeline
def start_processing(sid: str, wav_path: Path, media_dir: Path, display_name: str) -> None:
    _set(sid, status="queued", message="queued", progress=5, name=display_name)
    threading.Thread(target=_process, args=(sid, wav_path, media_dir, display_name), daemon=True).start()


def _process(sid: str, wav_path: Path, media_dir: Path, display_name: str) -> None:
    cfg = pod_config()
    if not cfg.host:
        _set(sid, status="error", progress=0,
             message="No GPU pod configured. Set AGENTLODGE_POD_HOST/PORT/KEY (see POD_SETUP.md / "
                     "scripts/pod.ps1), start the pod, then re-upload.")
        return
    pod_repo = _pod_repo(cfg)
    hosted = _co_located(cfg)
    try:
        _set(sid, status="processing", progress=8, message="checking the GPU pod…")
        probe = _run_pod(cfg, "echo ok", timeout=25)
        if probe.returncode != 0 or "ok" not in probe.stdout:
            _set(sid, status="error", progress=0,
                 message=f"Can't reach the GPU pod at {cfg.host}:{cfg.port}. Start it "
                         f"(scripts/pod.ps1 setup), then re-upload.")
            return

        ws = cfg.ws
        _set(sid, progress=14, message="uploading audio to the pod…")
        remote_wav_dir = f"{ws}/LODGE/data/finedance/music_wav"
        remote_wav = f"{remote_wav_dir}/{sid}.wav"
        remote_scripts = f"{pod_repo}/scripts"
        _run_pod(
            cfg,
            f"mkdir -p {shlex.quote(remote_wav_dir)} {shlex.quote(remote_scripts)}",
        )
        if hosted:
            Path(remote_wav_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, remote_wav)
        else:
            upload = _scp_to(cfg, str(wav_path), remote_wav)
            if upload.returncode != 0:
                _set(
                    sid,
                    status="error",
                    progress=0,
                    message=f"upload to pod failed: {upload.stderr[-200:]}",
                )
                return
            # A remote checkout may lag the server, so ship every script changed
            # by the optimized upload path.
            for script_name in (
                "preprocess_song.py",
                "make_song_bestofk.py",
                "build_window_bank.py",
                "render_one_ybot.sh",
                "process_song.sh",
            ):
                copied = _scp_to(
                    cfg,
                    str(REPO / "scripts" / script_name),
                    f"{remote_scripts}/{script_name}",
                )
                if copied.returncode != 0:
                    _set(
                        sid,
                        status="error",
                        progress=0,
                        message=f"could not upload {script_name}: {copied.stderr[-160:]}",
                    )
                    return

        _set(sid, progress=22, message="generating the dance on the pod (several minutes)…")
        pod_python = _venv_python(cfg)
        pipeline_scripts = " ".join(
            shlex.quote(f"{remote_scripts}/{name}")
            for name in (
                "preprocess_song.py",
                "make_song_bestofk.py",
                "build_window_bank.py",
                "render_one_ybot.sh",
                "process_song.sh",
            )
        )
        run = _run_pod(
            cfg,
            f"cd {shlex.quote(pod_repo)} && "
            f"sed -i 's/\\r$//' {pipeline_scripts} && "
            f"WORKSPACE={shlex.quote(ws)} "
            f"AGENTLODGE_ROOT={shlex.quote(pod_repo)} "
            f"AL_PY={shlex.quote(pod_python)} "
            f"AGENTLODGE_BANK_K={shlex.quote(str(cfg.bank_k))} "
            f"AGENTLODGE_SKIP_RENDER={'1' if hosted else '0'} "
            f"bash {shlex.quote(remote_scripts + '/process_song.sh')} {shlex.quote(sid)}",
            timeout=int(
                os.environ.get("AGENTLODGE_UPLOAD_GENERATION_TIMEOUT", str(60 * 60))
            ),
        )
        if run.returncode != 0 or f"PROCESS_{sid}_DONE" not in run.stdout:
            tail = (run.stdout[-400:] + "\n" + run.stderr[-400:]).strip()
            _set(sid, status="error", progress=0, message=f"pod processing failed: {tail[-300:]}")
            return

        _set(sid, progress=82, message="downloading the generated dance…")
        media_dir.mkdir(parents=True, exist_ok=True)
        bank_dir = media_dir / "bank"
        bank_dir.mkdir(exist_ok=True)
        out = f"{ws}/upload_{sid}"
        for name, dst in [("base_motion.npy", media_dir / "base_motion.npy"),
                          ("beats.npy", media_dir / "beats.npy"),
                          ("beat_strengths.npy", media_dir / "beat_strengths.npy")]:
            source = f"{out}/{name}"
            if hosted:
                shutil.copy2(source, dst)
            else:
                fetched = _scp_from(cfg, source, str(dst))
                if fetched.returncode != 0:
                    _set(
                        sid,
                        status="error",
                        progress=0,
                        message=f"could not fetch {name}: {fetched.stderr[-160:]}",
                    )
                    return
        if hosted:
            for source in Path(out).glob(f"bank_{sid}_*.npy"):
                shutil.copy2(source, bank_dir / source.name)
        else:
            _scp_from(cfg, f"{out}/bank_{sid}_*.npy", str(bank_dir) + "/")

        if hosted:
            import numpy as np

            from server import rendering

            _set(sid, progress=86, message="rendering the full-quality preview…")
            motion = np.load(media_dir / "base_motion.npy")
            if not rendering._render_warm_local(
                sid,
                motion,
                media_dir,
                "full",
                audio_wav=str(wav_path),
            ):
                _set(
                    sid,
                    status="error",
                    progress=0,
                    message="full-quality warm render failed",
                )
                return
            shutil.copy2(media_dir / "edited.mp4", media_dir / "preview.mp4")
        else:
            preview = _scp_from(cfg, f"{out}/preview.mp4", str(media_dir / "preview.mp4"))
            if preview.returncode != 0:
                _set(
                    sid,
                    status="error",
                    progress=0,
                    message=f"could not fetch preview.mp4: {preview.stderr[-160:]}",
                )
                return

        # persist a friendly display name
        (media_dir / "meta.json").write_text(
            json.dumps({"name": display_name}, ensure_ascii=False),
            encoding="utf-8",
        )
        bank_started = _launch_background_bank(
            cfg,
            sid,
            out,
            pod_repo,
            pod_python,
        )
        _set(
            sid,
            status="done",
            progress=100,
            message="ready",
            bank_status="building" if bank_started else "ready",
        )
        if bank_started:
            _start_bank_sync(sid, cfg, out, bank_dir)
    except subprocess.TimeoutExpired:
        _set(sid, status="error", progress=0, message="timed out talking to the pod (is it still up?).")
    except Exception as exc:  # noqa: BLE001
        _set(sid, status="error", progress=0, message=f"processing error: {exc}")
