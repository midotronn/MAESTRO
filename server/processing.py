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

import os
import re
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
    try:
        _set(sid, status="processing", progress=8, message="checking the GPU pod…")
        probe = _ssh(cfg, "echo ok", timeout=25)
        if probe.returncode != 0 or "ok" not in probe.stdout:
            _set(sid, status="error", progress=0,
                 message=f"Can't reach the GPU pod at {cfg.host}:{cfg.port}. Start it "
                         f"(scripts/pod.ps1 setup), then re-upload.")
            return

        ws = cfg.ws
        _set(sid, progress=14, message="uploading audio to the pod…")
        _ssh(cfg, f"mkdir -p {ws}/LODGE/data/finedance/music_wav")
        r = _scp_to(cfg, str(wav_path), f"{ws}/LODGE/data/finedance/music_wav/{sid}.wav")
        if r.returncode != 0:
            _set(sid, status="error", progress=0, message=f"upload to pod failed: {r.stderr[-200:]}")
            return
        # ship the pipeline script
        _ssh(cfg, f"mkdir -p {ws}/AgentLODGE/scripts")
        _scp_to(cfg, str(REPO / "scripts" / "process_song.sh"), f"{ws}/AgentLODGE/scripts/process_song.sh")
        _scp_to(cfg, str(REPO / "scripts" / "build_window_bank.py"), f"{ws}/AgentLODGE/scripts/build_window_bank.py")

        _set(sid, progress=22, message="generating the dance on the pod (several minutes)…")
        run = _ssh(
            cfg,
            f"cd {ws}/AgentLODGE && sed -i 's/\\r$//' scripts/process_song.sh scripts/build_window_bank.py && "
            f"WORKSPACE={ws} AGENTLODGE_BANK_K={cfg.bank_k} bash scripts/process_song.sh {sid}",
            timeout=60 * 40)
        if run.returncode != 0 or f"PROCESS_{sid}_DONE" not in run.stdout:
            tail = (run.stdout[-400:] + "\n" + run.stderr[-400:]).strip()
            _set(sid, status="error", progress=0, message=f"pod processing failed: {tail[-300:]}")
            return

        _set(sid, progress=82, message="downloading the generated dance…")
        media_dir.mkdir(parents=True, exist_ok=True)
        (media_dir / "bank").mkdir(exist_ok=True)
        out = f"{ws}/upload_{sid}"
        for name, dst in [("base_motion.npy", media_dir / "base_motion.npy"),
                          ("beats.npy", media_dir / "beats.npy"),
                          ("preview.mp4", media_dir / "preview.mp4")]:
            g = _scp_from(cfg, f"{out}/{name}", str(dst))
            if g.returncode != 0:
                _set(sid, status="error", progress=0, message=f"could not fetch {name}: {g.stderr[-160:]}")
                return
        _scp_from(cfg, f"{out}/bank_{sid}_*.npy", str(media_dir / "bank") + "/")
        # persist a friendly display name
        (media_dir / "meta.json").write_text(f'{{"name": "{display_name}"}}')
        _set(sid, status="done", progress=100, message="ready")
    except subprocess.TimeoutExpired:
        _set(sid, status="error", progress=0, message="timed out talking to the pod (is it still up?).")
    except Exception as exc:  # noqa: BLE001
        _set(sid, status="error", progress=0, message=f"processing error: {exc}")
