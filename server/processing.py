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
import math
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

_PIPELINE_STAGE_SPECS = {
    "assets": (22, 25, 120.0, "checking generation model assets"),
    "preprocess": (25, 40, 240.0, "extracting music features"),
    "generation": (40, 68, 360.0, "generating LODGE and EDGE motion"),
    "polish": (68, 73, 90.0, "assembling and polishing the choreography"),
    "seed_bank": (73, 76, 30.0, "building the initial editing bank"),
    "beats": (76, 79, 30.0, "tracking beats"),
    "staging": (79, 80, 10.0, "staging generated assets"),
    "preview": (80, 82, 300.0, "rendering the initial preview"),
}
_PIPELINE_PROGRESS_RE = re.compile(
    r"^MAESTRO_PROGRESS\s+([a-z0-9_-]+)\s+(\d{1,3})\s*(.*)$"
)
_PIPELINE_SUBPROGRESS_RE = re.compile(
    r"^MAESTRO_SUBPROGRESS\s+([a-z0-9_-]+)\s+(\d{1,3})\s*(.*)$"
)
_PIPELINE_TIMING_RE = re.compile(
    r"^MAESTRO_TIMING\s+([a-z0-9_-]+)\s+(start|end)\s+(\d+)\s*(.*)$"
)


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
def _public_job(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if not key.startswith("_") and key != "trace_path"
    }


def _close_active_stage(job: dict, ended_at: float) -> None:
    timeline = job.setdefault("stage_timeline", [])
    if not timeline or timeline[-1].get("ended_at") is not None:
        return
    timeline[-1]["ended_at"] = ended_at
    timeline[-1]["duration_seconds"] = round(
        max(0.0, ended_at - float(timeline[-1]["started_at"])),
        3,
    )


def _persist_job(trace_path: str | None, job: dict) -> None:
    if not trace_path:
        return
    try:
        path = Path(trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = _public_job(job)
        started = payload.get("started")
        if started:
            ended = payload.get("finished") or time.time()
            payload["elapsed"] = round(max(0.0, float(ended) - float(started)), 3)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        pass


def get_job(sid: str) -> dict:
    with _LOCK:
        raw = dict(
            _JOBS.get(
                sid,
                {"status": "unknown", "message": "no such job", "progress": 0},
            )
        )
    job = _public_job(raw)
    started = job.get("started")
    if started:
        ended = job.get("finished") or time.time()
        job["elapsed"] = round(max(0.0, float(ended) - float(started)), 1)
    return job


def _set(sid: str, **kw) -> None:
    now = time.time()
    kw["updated"] = now
    if kw.get("status") in {"done", "error"}:
        kw.setdefault("finished", now)
    snapshot = None
    trace_path = None
    with _LOCK:
        job = _JOBS.setdefault(sid, {})
        supplied_timeline = kw.pop("stage_timeline", None)
        if supplied_timeline is not None:
            job["stage_timeline"] = list(supplied_timeline)
        stage = kw.get("stage")
        if stage and stage != job.get("_active_stage"):
            _close_active_stage(job, now)
            job["_active_stage"] = stage
            job.setdefault("stage_timeline", []).append(
                {
                    "stage": stage,
                    "started_at": now,
                    "ended_at": None,
                    "duration_seconds": None,
                }
            )
        job.update(kw)
        if kw.get("status") in {"done", "error"}:
            _close_active_stage(job, float(job["finished"]))
            trace_path = job.get("trace_path")
            snapshot = dict(job)
    if snapshot is not None:
        _persist_job(trace_path, snapshot)


def record_browser_timing(sid: str, timing: dict) -> dict:
    """Attach the browser-observed upload and total latency to a completed job trace."""
    with _LOCK:
        job = _JOBS.get(sid)
        if job is None:
            raise KeyError(sid)
        request_id = str(timing.get("request_id") or "")
        expected = str(job.get("request_id") or "")
        if expected and request_id != expected:
            raise ValueError("request_id does not match the upload job")
        normalized = {
            "request_id": request_id or expected,
            "browser_started_at_ms": float(timing["browser_started_at_ms"]),
            "browser_completed_at_ms": float(timing["browser_completed_at_ms"]),
            "browser_upload_seconds": round(
                max(0.0, float(timing["browser_upload_seconds"])),
                3,
            ),
            "browser_total_seconds": round(
                max(0.0, float(timing["browser_total_seconds"])),
                3,
            ),
            "source_bytes": int(timing.get("source_bytes") or job.get("source_bytes") or 0),
        }
        job["browser_timing"] = normalized
        job["browser_upload_seconds"] = normalized["browser_upload_seconds"]
        job["browser_total_seconds"] = normalized["browser_total_seconds"]
        job["updated"] = time.time()
        trace_path = job.get("trace_path")
        snapshot = dict(job)
    _persist_job(trace_path, snapshot)
    return normalized


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


def _read_tail(path: Path, max_bytes: int = 20000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _pipeline_event(log_tail: str) -> dict | None:
    """Return the most recent structured progress event emitted by the pod pipeline."""
    latest = None
    for raw_line in log_tail.splitlines():
        line = raw_line.strip()
        match = _PIPELINE_PROGRESS_RE.match(line)
        if match:
            stage, progress, message = match.groups()
            latest = {
                "stage": stage,
                "progress": max(0, min(100, int(progress))),
                "message": message.strip(),
                "subprogress": None,
            }
            continue
        match = _PIPELINE_SUBPROGRESS_RE.match(line)
        if match:
            stage, progress, message = match.groups()
            latest = {
                "stage": stage,
                "progress": None,
                "message": message.strip(),
                "subprogress": max(0, min(100, int(progress))),
            }
    return latest


def _pipeline_timing_report(text: str) -> dict:
    """Parse timing markers emitted by the remote shell and Python pipeline stages."""
    events = []
    active: dict[str, list[int]] = {}
    stages: dict[str, dict] = {}
    for raw_line in text.splitlines():
        match = _PIPELINE_TIMING_RE.match(raw_line.strip())
        if not match:
            continue
        stage, state, timestamp_ms, detail = match.groups()
        timestamp = int(timestamp_ms)
        events.append(
            {
                "stage": stage,
                "state": state,
                "timestamp_ms": timestamp,
                "detail": detail.strip(),
            }
        )
        if state == "start":
            active.setdefault(stage, []).append(timestamp)
            continue
        starts = active.get(stage) or []
        if not starts:
            continue
        started = starts.pop()
        duration = max(0.0, (timestamp - started) / 1000.0)
        summary = stages.setdefault(
            stage,
            {
                "attempts": 0,
                "duration_seconds": 0.0,
                "first_started_at_ms": started,
                "last_ended_at_ms": timestamp,
            },
        )
        summary["attempts"] += 1
        summary["duration_seconds"] = round(
            float(summary["duration_seconds"]) + duration,
            3,
        )
        summary["first_started_at_ms"] = min(
            int(summary["first_started_at_ms"]),
            started,
        )
        summary["last_ended_at_ms"] = max(
            int(summary["last_ended_at_ms"]),
            timestamp,
        )
    events.sort(key=lambda event: (event["timestamp_ms"], event["state"] != "start"))
    return {"events": events, "stages": stages}


def _read_pod_text(cfg: PodConfig, path: str, max_bytes: int = 200000) -> str:
    if _co_located(cfg):
        return _read_tail(Path(path), max_bytes=max_bytes)
    result = _ssh(
        cfg,
        f"tail -c {int(max_bytes)} {shlex.quote(path)} 2>/dev/null || true",
        timeout=30,
    )
    return result.stdout if result.returncode == 0 else ""


def _pipeline_state(
    cfg: PodConfig,
    *,
    done_path: str,
    failed_path: str,
    log_path: str,
) -> tuple[str, str]:
    if _co_located(cfg):
        state = (
            "done"
            if Path(done_path).exists()
            else "failed"
            if Path(failed_path).exists()
            else "running"
        )
        return state, _read_tail(Path(log_path))

    check = _ssh(
        cfg,
        f"state=running; "
        f"test -f {shlex.quote(done_path)} && state=done; "
        f"test -f {shlex.quote(failed_path)} && state=failed; "
        f"printf '__MAESTRO_STATE__=%s\\n' \"$state\"; "
        f"tail -c 20000 {shlex.quote(log_path)} 2>/dev/null || true",
        timeout=30,
    )
    if check.returncode != 0:
        return "running", ""
    lines = (check.stdout or "").splitlines()
    state = "running"
    tail_lines = []
    for line in lines:
        if line.startswith("__MAESTRO_STATE__="):
            state = line.split("=", 1)[1].strip()
        else:
            tail_lines.append(line)
    return state, "\n".join(tail_lines)


def _run_pipeline(
    cfg: PodConfig,
    command: str,
    *,
    sid: str,
    out: str,
    timeout: int,
) -> subprocess.CompletedProcess:
    """Launch the long pod pipeline detached, then turn its log markers into live job progress."""
    done_path = f"{out}/process.done"
    failed_path = f"{out}/process.failed"
    log_path = f"{out}/process.log"
    timing_path = f"{out}/timings.tsv"
    inner = (
        f"{command}; rc=$?; "
        f"if [ \"$rc\" -eq 0 ]; then touch {shlex.quote(done_path)}; "
        f"else touch {shlex.quote(failed_path)}; fi; exit \"$rc\""
    )
    launch_command = (
        f"mkdir -p {shlex.quote(out)}; "
        f"rm -f {shlex.quote(done_path)} {shlex.quote(failed_path)} {shlex.quote(log_path)}; "
        f"setsid bash -c {shlex.quote(inner)} "
        f"> {shlex.quote(log_path)} 2>&1 < /dev/null & echo PROCESS_PID=$!"
    )
    launch = _run_pod(cfg, launch_command, timeout=60)
    direct_output = (launch.stdout or "") + "\n" + (launch.stderr or "")
    if f"PROCESS_{sid}_DONE" in direct_output:
        timing_report = _pipeline_timing_report(
            _read_pod_text(cfg, timing_path)
        )
        if timing_report["events"]:
            _set(sid, remote_pipeline_timings=timing_report)
        return launch
    if launch.returncode != 0 or "PROCESS_PID=" not in (launch.stdout or ""):
        return launch

    deadline = time.monotonic() + max(60, timeout)
    current_stage = "assets"
    stage_started = time.monotonic()
    last_tail = ""
    last_progress = 22
    while time.monotonic() < deadline:
        try:
            state, log_tail = _pipeline_state(
                cfg,
                done_path=done_path,
                failed_path=failed_path,
                log_path=log_path,
            )
        except (OSError, subprocess.SubprocessError):
            state, log_tail = "running", ""
        if log_tail:
            last_tail = log_tail
        event = _pipeline_event(last_tail)
        now = time.monotonic()
        if event and event["stage"] in _PIPELINE_STAGE_SPECS:
            if event["stage"] != current_stage:
                current_stage = event["stage"]
                stage_started = now
        start, end, expected_seconds, default_message = _PIPELINE_STAGE_SPECS[current_stage]
        elapsed = max(0.0, now - stage_started)
        estimated_fraction = min(0.97, 1.0 - math.exp(-elapsed / expected_seconds))
        estimated = start + int((end - start) * estimated_fraction)
        explicit = start
        message = default_message
        if event and event["stage"] == current_stage:
            if event["progress"] is not None:
                explicit = event["progress"]
            elif event["subprogress"] is not None:
                explicit = start + round(
                    (end - start) * event["subprogress"] / 100.0
                )
            message = event["message"] or message
        progress = max(last_progress, start, min(end, max(explicit, estimated)))
        last_progress = progress
        stage_progress = (
            100
            if end <= start
            else round(100 * (progress - start) / (end - start))
        )
        _set(
            sid,
            status="processing",
            stage=current_stage,
            stage_progress=stage_progress,
            progress=progress,
            message=message,
        )
        if state == "done":
            timing_report = _pipeline_timing_report(
                _read_pod_text(cfg, timing_path)
            )
            if timing_report["events"]:
                _set(sid, remote_pipeline_timings=timing_report)
            return subprocess.CompletedProcess(
                launch.args,
                0,
                stdout=last_tail,
                stderr="",
            )
        if state == "failed":
            timing_report = _pipeline_timing_report(
                _read_pod_text(cfg, timing_path)
            )
            if timing_report["events"]:
                _set(sid, remote_pipeline_timings=timing_report)
            return subprocess.CompletedProcess(
                launch.args,
                1,
                stdout=last_tail,
                stderr="pod pipeline failed",
            )
        time.sleep(2)
    raise subprocess.TimeoutExpired(command, timeout, output=last_tail)


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
    distributed_enabled = os.environ.get(
        "AGENTLODGE_DISTRIBUTED",
        "",
    ).strip().lower() in {"1", "true", "yes", "on"}
    distributed_capabilities = {
        item.strip().lower()
        for item in os.environ.get(
            "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
            "",
        ).split(",")
        if item.strip()
    }
    use_resident_worker = distributed_enabled and (
        "dance.generate" in distributed_capabilities
        or "dance" in distributed_capabilities
    )
    bank_client = (
        "dispatch_bank_generation.py"
        if use_resident_worker
        else "build_window_bank.py"
    )
    bank_script = f"{pod_repo}/scripts/{bank_client}"
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
        try:
            bank_k = max(1, int(cfg.bank_k))
        except (TypeError, ValueError):
            bank_k = 4
        expected_files = bank_k * 2
        deferred_files = max(1, expected_files - 2)
        while time.monotonic() < deadline:
            detail = ""
            try:
                if _co_located(cfg):
                    state = (
                        "DONE"
                        if Path(done).exists()
                        else "FAILED"
                        if Path(failed).exists()
                        else ""
                    )
                    count = len(list(Path(cfg.ws).glob(f"bank_{sid}_*.npy")))
                    detail = _read_tail(Path(log), max_bytes=4000)
                else:
                    check = _ssh(
                        cfg,
                        f"state=RUNNING; "
                        f"test -f {shlex.quote(done)} && state=DONE; "
                        f"test -f {shlex.quote(failed)} && state=FAILED; "
                        f"printf '__STATE__=%s\\n' \"$state\"; "
                        f"printf '__COUNT__='; "
                        f"find {shlex.quote(cfg.ws)} -maxdepth 1 -type f "
                        f"-name {shlex.quote(f'bank_{sid}_*.npy')} | wc -l; "
                        f"tail -n 8 {shlex.quote(log)} 2>/dev/null || true",
                        timeout=30,
                    )
                    state = ""
                    count = 2
                    detail_lines = []
                    for line in (check.stdout or "").splitlines():
                        if line.startswith("__STATE__="):
                            state = line.split("=", 1)[1].strip()
                        elif line.startswith("__COUNT__="):
                            try:
                                count = int(line.split("=", 1)[1].strip())
                            except ValueError:
                                count = 2
                        else:
                            detail_lines.append(line)
                    detail = "\n".join(detail_lines)
            except (OSError, subprocess.SubprocessError):
                state, count = "", 2
            generated = max(0, count - 2)
            bank_progress = min(99, round(100 * generated / deferred_files))
            current_line = next(
                (
                    line.strip()
                    for line in reversed(detail.splitlines())
                    if "generating seed" in line or "saved bank_" in line
                ),
                "",
            )
            _set(
                sid,
                bank_status="building",
                bank_progress=bank_progress,
                bank_message=current_line or "generating additional editing alternatives",
            )
            if state == "DONE":
                bank_dir.mkdir(parents=True, exist_ok=True)
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
                    bank_progress=100 if copied else bank_progress,
                    bank_message=(
                        "editing alternatives ready"
                        if copied
                        else "generated bank could not be copied"
                    ),
                    bank_error="" if copied else "generated bank could not be copied",
                )
                return
            if state == "FAILED":
                _set(
                    sid,
                    bank_status="error",
                    bank_progress=bank_progress,
                    bank_message="background editing bank failed",
                    bank_error=detail or "background bank failed",
                )
                return
            time.sleep(5)
        _set(
            sid,
            bank_status="error",
            bank_message="background editing bank timed out",
            bank_error="background bank timed out",
        )

    threading.Thread(target=sync, name=f"bank-sync-{sid}", daemon=True).start()



# --------------------------------------------------------------------------- pipeline
def start_processing(
    sid: str,
    wav_path: Path,
    media_dir: Path,
    display_name: str,
    *,
    request_id: str | None = None,
    request_received_at: float | None = None,
    upload_received_at: float | None = None,
    audio_ready_at: float | None = None,
    client_started_at_ms: float | None = None,
    source_bytes: int = 0,
    source_sha256: str = "",
) -> None:
    processing_started = time.time()
    request_received = float(request_received_at or processing_started)
    upload_received = max(
        request_received,
        float(upload_received_at or processing_started),
    )
    audio_ready = max(upload_received, float(audio_ready_at or processing_started))
    initial_timeline = [
        {
            "stage": "browser_upload",
            "started_at": request_received,
            "ended_at": upload_received,
            "duration_seconds": round(upload_received - request_received, 3),
        },
        {
            "stage": "audio_prepare",
            "started_at": upload_received,
            "ended_at": audio_ready,
            "duration_seconds": round(audio_ready - upload_received, 3),
        },
    ]
    _set(
        sid,
        status="queued",
        stage="queued",
        stage_progress=0,
        message="queued",
        progress=5,
        name=display_name,
        request_id=request_id or sid,
        started=request_received,
        processing_started=processing_started,
        client_started_at_ms=client_started_at_ms,
        source_bytes=max(0, int(source_bytes)),
        source_sha256=str(source_sha256),
        service_state=os.environ.get("AGENTLODGE_SERVICE_STATE", "unknown"),
        stage_timeline=initial_timeline,
        trace_path=str(media_dir / "performance_trace.json"),
        bank_status="pending",
        bank_progress=0,
        bank_message="",
    )
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
        _set(
            sid,
            status="processing",
            stage="pod",
            stage_progress=0,
            progress=8,
            message="checking the GPU pod…",
        )
        probe = _run_pod(cfg, "echo ok", timeout=25)
        if probe.returncode != 0 or "ok" not in probe.stdout:
            _set(sid, status="error", progress=0,
                 message=f"Can't reach the GPU pod at {cfg.host}:{cfg.port}. Start it "
                         f"(scripts/pod.ps1 setup), then re-upload.")
            return

        ws = cfg.ws
        _set(
            sid,
            stage="audio",
            stage_progress=0,
            progress=14,
            message="uploading audio to the pod…",
        )
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
                "dispatch_song_generation.py",
                "dispatch_bank_generation.py",
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

        _set(
            sid,
            stage="assets",
            stage_progress=0,
            progress=22,
            message="starting the generation pipeline…",
        )
        pod_python = _venv_python(cfg)
        pipeline_scripts = " ".join(
            shlex.quote(f"{remote_scripts}/{name}")
            for name in (
                "preprocess_song.py",
                "make_song_bestofk.py",
                "dispatch_song_generation.py",
                "dispatch_bank_generation.py",
                "build_window_bank.py",
                "render_one_ybot.sh",
                "process_song.sh",
            )
        )
        out = f"{ws}/upload_{sid}"
        distributed_env = " ".join(
            f"{name}={shlex.quote(value)}"
            for name in (
                "AGENTLODGE_DISTRIBUTED",
                "AGENTLODGE_DISTRIBUTED_CAPABILITIES",
                "AGENTLODGE_WORKER_REGISTRY",
                "AGENTLODGE_WORKERS_JSON",
                "AGENTLODGE_SHARED_ROOT",
                "AGENTLODGE_DISTRIBUTED_TMP",
                "AGENTLODGE_WORKER_HEARTBEAT_MAX_AGE",
                "AGENTLODGE_JUKEBOX_TIMEOUT",
                "AGENTLODGE_GENERATION_TIMEOUT",
            )
            if (value := os.environ.get(name, "").strip())
        )
        pipeline_command = (
            f"cd {shlex.quote(pod_repo)} && "
            f"sed -i 's/\\r$//' {pipeline_scripts} && "
            f"{distributed_env + ' ' if distributed_env else ''}"
            f"WORKSPACE={shlex.quote(ws)} "
            f"AGENTLODGE_ROOT={shlex.quote(pod_repo)} "
            f"AL_PY={shlex.quote(pod_python)} "
            f"MAESTRO_TIMING_FILE={shlex.quote(out + '/timings.tsv')} "
            f"AGENTLODGE_BANK_K={shlex.quote(str(cfg.bank_k))} "
            f"AGENTLODGE_SKIP_RENDER={'1' if hosted else '0'} "
            f"bash {shlex.quote(remote_scripts + '/process_song.sh')} {shlex.quote(sid)}"
        )
        run = _run_pipeline(
            cfg,
            pipeline_command,
            sid=sid,
            out=out,
            timeout=int(
                os.environ.get("AGENTLODGE_UPLOAD_GENERATION_TIMEOUT", str(60 * 60))
            ),
        )
        if run.returncode != 0 or f"PROCESS_{sid}_DONE" not in run.stdout:
            tail = (run.stdout[-400:] + "\n" + run.stderr[-400:]).strip()
            _set(sid, status="error", progress=0, message=f"pod processing failed: {tail[-300:]}")
            return

        _set(
            sid,
            stage="transfer",
            stage_progress=0,
            progress=82,
            message="collecting the generated dance…",
        )
        media_dir.mkdir(parents=True, exist_ok=True)
        bank_dir = media_dir / "bank"
        bank_dir.mkdir(exist_ok=True)
        assets = [
            ("base_motion.npy", media_dir / "base_motion.npy"),
            ("beats.npy", media_dir / "beats.npy"),
            ("beat_strengths.npy", media_dir / "beat_strengths.npy"),
        ]
        for index, (name, dst) in enumerate(assets, start=1):
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
            _set(
                sid,
                progress=82 + round(3 * index / len(assets)),
                stage_progress=round(75 * index / len(assets)),
                message=f"collected {index}/{len(assets)} generated assets",
            )
        if hosted:
            for source in Path(out).glob(f"bank_{sid}_*.npy"):
                shutil.copy2(source, bank_dir / source.name)
        else:
            _scp_from(cfg, f"{out}/bank_{sid}_*.npy", str(bank_dir) + "/")
        _set(sid, progress=86, stage_progress=100, message="generated assets are ready")

        if hosted:
            import numpy as np

            from server import rendering

            _set(
                sid,
                stage="render",
                stage_progress=0,
                progress=86,
                message="starting the full-quality preview render…",
            )
            motion = np.load(media_dir / "base_motion.npy")
            render_result: dict[str, object] = {"ok": False}

            def render_preview() -> None:
                try:
                    render_result["ok"] = rendering._render_warm_local(
                        sid,
                        motion,
                        media_dir,
                        "full",
                        audio_wav=str(wav_path),
                    )
                except Exception as exc:  # noqa: BLE001
                    render_result["error"] = str(exc)

            render_thread = threading.Thread(
                target=render_preview,
                name=f"upload-preview-{sid}",
                daemon=True,
            )
            render_thread.start()
            while render_thread.is_alive():
                render_job = rendering.get_render_job(sid)
                render_progress = max(
                    0,
                    min(100, int(render_job.get("progress") or 0)),
                )
                normalized = (
                    max(0.0, min(1.0, (render_progress - 24) / 72.0))
                    if render_progress >= 24
                    else render_progress / 100.0
                )
                mapped = 86 + round(12 * normalized)
                _set(
                    sid,
                    progress=mapped,
                    stage_progress=round(100 * normalized),
                    rendered_frames=int(render_job.get("rendered_frames") or 0),
                    render_frames=int(render_job.get("frames") or len(motion)),
                    render_workers=int(render_job.get("workers") or 0),
                    gpu_endpoints=list(render_job.get("worker_ids") or []),
                    message=render_job.get("message")
                    or "rendering the full-quality preview",
                )
                render_thread.join(timeout=1)
            render_job = rendering.get_render_job(sid)
            _set(
                sid,
                rendered_frames=int(render_job.get("rendered_frames") or 0),
                render_frames=int(render_job.get("frames") or len(motion)),
                render_workers=int(render_job.get("workers") or 0),
                gpu_endpoints=list(render_job.get("worker_ids") or []),
                render_source_frame_hashes=list(
                    render_job.get("render_source_frame_hashes") or []
                ),
                render_shard_sha256=list(
                    render_job.get("render_shard_sha256") or []
                ),
            )
            if not render_result.get("ok"):
                _set(
                    sid,
                    status="error",
                    progress=0,
                    message=(
                        "full-quality warm render failed: "
                        + str(render_result.get("error") or "unknown render error")
                    ),
                )
                return
            shutil.copy2(media_dir / "edited.mp4", media_dir / "preview.mp4")
            _set(sid, progress=98, stage_progress=100, message="preview render complete")
        else:
            _set(
                sid,
                stage="transfer",
                stage_progress=80,
                progress=86,
                message="downloading the rendered preview…",
            )
            preview = _scp_from(cfg, f"{out}/preview.mp4", str(media_dir / "preview.mp4"))
            if preview.returncode != 0:
                _set(
                    sid,
                    status="error",
                    progress=0,
                    message=f"could not fetch preview.mp4: {preview.stderr[-160:]}",
                )
                return
            _set(sid, progress=98, stage_progress=100, message="preview downloaded")

        # persist a friendly display name
        _set(
            sid,
            stage="finalize",
            stage_progress=50,
            progress=99,
            message="finalizing the song in the editor…",
        )
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
            stage="ready",
            stage_progress=100,
            progress=100,
            message="ready",
            bank_status="building" if bank_started else "ready",
            bank_progress=0 if bank_started else 100,
            bank_message=(
                "generating additional editing alternatives"
                if bank_started
                else "editing alternatives ready"
            ),
        )
        if bank_started:
            _start_bank_sync(sid, cfg, out, bank_dir)
    except subprocess.TimeoutExpired:
        _set(sid, status="error", progress=0, message="timed out talking to the pod (is it still up?).")
    except Exception as exc:  # noqa: BLE001
        _set(sid, status="error", progress=0, message=f"processing error: {exc}")
