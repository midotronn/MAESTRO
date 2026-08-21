import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _bash_executable() -> str | None:
    found = shutil.which("bash")
    if found:
        return found
    candidate = Path(r"C:\Program Files\Git\bin\bash.exe")
    return str(candidate) if candidate.is_file() else None


def _bash_path(bash: str, path: Path) -> str:
    if os.name != "nt":
        return str(path)
    result = subprocess.run(
        [bash, "-lc", f"cygpath -u {shlex.quote(str(path))}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _run_gpu_guard(
    tmp_path: Path,
    *,
    gpu_indices: tuple[int, ...],
    requested: str | None,
    validate_shim: bool,
    capability: str = "render.frames",
    inherited_visibility: bool = False,
) -> subprocess.CompletedProcess[str]:
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir(exist_ok=True)
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$FAKE_GPU_INDICES\"\n",
        encoding="utf-8",
        newline="\n",
    )
    nvidia_smi.chmod(0o755)
    helper = _bash_path(
        bash,
        ROOT / "scripts" / "render_worker_env.sh",
    )
    fake_bin_path = _bash_path(bash, fake_bin)
    override = (
        ""
        if validate_shim
        else "agentlodge_validate_selector_shim() { return 0; }\n"
    )
    command = (
        "set -euo pipefail\n"
        f"PATH={shlex.quote(fake_bin_path)}:/usr/bin:/bin:$PATH\n"
        f"source {shlex.quote(helper)}\n"
        f"{override}"
        f"agentlodge_configure_gpu {shlex.quote(capability)} /workspace\n"
        "printf '%s|%s|%s|%s|%s\\n' "
        '"${AGENTLODGE_RESOLVED_GPU_INDEX:-}" '
        '"${AGENTLODGE_RENDER_MULTI_GPU:-}" '
        '"${CUDA_VISIBLE_DEVICES:-}" '
        '"${NVIDIA_VISIBLE_DEVICES:-}" '
        '"${LD_PRELOAD:-}"\n'
    )
    environment = os.environ.copy()
    environment["FAKE_GPU_INDICES"] = "\n".join(
        str(value) for value in gpu_indices
    )
    environment.pop("AGENTLODGE_GPU_INDEX", None)
    if inherited_visibility:
        environment["CUDA_VISIBLE_DEVICES"] = "inherited-cuda"
        environment["NVIDIA_VISIBLE_DEVICES"] = "inherited-nvidia"
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop("NVIDIA_VISIBLE_DEVICES", None)
    environment.pop("LD_PRELOAD", None)
    environment.pop("AGENTLODGE_RENDER_MULTI_GPU", None)
    if requested is not None:
        environment["AGENTLODGE_GPU_INDEX"] = requested
    return subprocess.run(
        [bash, "-c", command],
        capture_output=True,
        text=True,
        env=environment,
    )


def test_selector_source_uses_cuda_index_not_enumeration_order():
    source = (
        ROOT / "scripts" / "egl_cuda_device_selector.c"
    ).read_text(encoding="utf-8")

    assert "EGL_CUDA_DEVICE_NV" in source
    assert 'getenv("AGENTLODGE_GPU_INDEX")' in source
    assert "cuda_index == (EGLAttrib)wanted" in source
    assert "AGENTLODGE_EGL_DEVICE_INDEX" not in source
    assert "eglGetPlatformDisplayEXT" in source
    assert "agentlodge_egl_selector_version = 2" in source
    assert "agentlodge_egl_selector_build_id" in source
    assert "AGENTLODGE_EGL_ATTESTATION_PATH" in source
    assert "rename(temporary, path)" in source
    dlsym_body = source.split("void *dlsym(void *handle", maxsplit=1)[1]
    assert "name == NULL" not in dlsym_body
    assert not list((ROOT / "scripts").glob("*.so"))


def test_gpu_guard_allows_valid_multigpu_and_scopes_preload(tmp_path):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0, 1),
        requested="0",
        validate_shim=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0|1|||"


def test_multigpu_render_keeps_all_cuda_devices_visible(tmp_path):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0, 1),
        requested="1",
        validate_shim=False,
        inherited_visibility=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1|1|||"


@pytest.mark.parametrize(
    "capability",
    ("jukebox.extract", "lodge.generate", "edge.generate"),
)
def test_cuda_workers_retain_requested_device_isolation(
    tmp_path,
    capability,
):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0, 1),
        requested="1",
        validate_shim=False,
        capability=capability,
        inherited_visibility=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1||1|inherited-nvidia|"


@pytest.mark.parametrize("requested", [None, "2", "invalid"])
def test_gpu_guard_rejects_missing_or_invalid_multigpu_index(
    tmp_path,
    requested,
):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0, 1),
        requested=requested,
        validate_shim=False,
    )

    assert result.returncode != 0
    assert "AGENTLODGE_GPU_INDEX" in result.stderr


def test_gpu_guard_rejects_missing_selector_shim(tmp_path):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0, 1),
        requested="0",
        validate_shim=True,
    )

    assert result.returncode != 0
    assert "selector shim is missing" in result.stderr


def test_gpu_guard_preserves_one_gpu_without_selector(tmp_path):
    result = _run_gpu_guard(
        tmp_path,
        gpu_indices=(0,),
        requested=None,
        validate_shim=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0||||"


def test_render_path_helper_is_unique_per_worker(tmp_path):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    helper = _bash_path(
        bash,
        ROOT / "scripts" / "render_worker_env.sh",
    )
    local_base = _bash_path(bash, tmp_path / "local")
    outputs = []
    for worker_id in ("render-g0-d0", "render-g0-d1"):
        command = (
            "set -euo pipefail\n"
            f"source {shlex.quote(helper)}\n"
            f"AGENTLODGE_RENDER_LOCAL_BASE={shlex.quote(local_base)}\n"
            "export AGENTLODGE_RENDER_LOCAL_BASE\n"
            f"agentlodge_configure_render_paths {shlex.quote(worker_id)}\n"
            "printf '%s|%s|%s\\n' "
            '"$AGENTLODGE_RENDER_LOCAL_ROOT" '
            '"$AGENTLODGE_RENDER_DAEMON_ROOT" '
            '"$AGENTLODGE_WORKER_TMP"\n'
        )
        environment = os.environ.copy()
        for name in (
            "AGENTLODGE_RENDER_LOCAL_ROOT",
            "AGENTLODGE_RENDER_DAEMON_ROOT",
            "AGENTLODGE_WORKER_TMP",
            "AGENTLODGE_HTTP_WORKER_SCRATCH",
        ):
            environment.pop(name, None)
        result = subprocess.run(
            [bash, "-c", command],
            capture_output=True,
            text=True,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.strip())

    assert outputs[0] != outputs[1]
    assert "render-g0-d0" in outputs[0]
    assert "render-g0-d1" in outputs[1]


def test_warm_render_uses_configured_unique_daemon_roots(
    tmp_path,
    monkeypatch,
):
    import server.warm_render as warm_render

    first = tmp_path / "worker-a" / "daemon"
    second = tmp_path / "worker-b" / "daemon"
    monkeypatch.setenv("AGENTLODGE_RENDER_DAEMON_ROOT", str(first))
    assert warm_render._dir(0) == first.resolve() / "d0"
    monkeypatch.setenv("AGENTLODGE_RENDER_DAEMON_ROOT", str(second))
    assert warm_render._dir(0) == second.resolve() / "d0"
    assert warm_render._dir(0) != first.resolve() / "d0"


def test_selector_preload_is_blender_subprocess_only(
    tmp_path,
    monkeypatch,
):
    import server.warm_render as warm_render

    class FakeProcess:
        pid = 24680

    captured = {}
    selector = tmp_path / "selector.so"
    selector.write_bytes(b"selector")
    daemon_root = tmp_path / "daemon"
    monkeypatch.setenv("AGENTLODGE_RENDER_MULTI_GPU", "1")
    monkeypatch.setenv("AGENTLODGE_GPU_INDEX", "1")
    monkeypatch.setenv("AGENTLODGE_EGL_SELECTOR_SHIM", str(selector))
    monkeypatch.setenv("AGENTLODGE_RENDER_DAEMON_ROOT", str(daemon_root))
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setattr(warm_render, "_blender", lambda: tmp_path / "blender")
    monkeypatch.setattr(warm_render, "_scene", lambda: tmp_path / "scene.blend")
    monkeypatch.setattr(warm_render, "_ybot", lambda: tmp_path / "ybot.fbx")
    monkeypatch.setattr(
        warm_render,
        "_daemon_script",
        lambda: tmp_path / "daemon.py",
    )

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr(warm_render.subprocess, "Popen", fake_popen)
    warm_render._start_daemon(0, width=320, height=240, samples=2)

    assert "LD_PRELOAD" not in os.environ
    assert captured["env"]["LD_PRELOAD"] == str(selector.resolve())
    assert captured["env"]["AGENTLODGE_EGL_ATTESTATION_PATH"].endswith(
        "egl-selector.attestation.json"
    )
    assert captured["env"]["AGENTLODGE_GPU_INDEX"] == "1"
    assert captured["command"][0] == str(tmp_path / "blender")


def test_warm_render_rejects_invalid_multigpu_selector_config(
    tmp_path,
    monkeypatch,
):
    import server.warm_render as warm_render

    selector = tmp_path / "selector.so"
    selector.write_bytes(b"selector")
    monkeypatch.setenv("AGENTLODGE_RENDER_MULTI_GPU", "1")
    monkeypatch.setenv("AGENTLODGE_EGL_SELECTOR_SHIM", str(selector))
    monkeypatch.setenv("AGENTLODGE_GPU_INDEX", "not-an-index")
    with pytest.raises(RuntimeError, match="AGENTLODGE_GPU_INDEX"):
        warm_render._daemon_environment()

    monkeypatch.setenv("AGENTLODGE_GPU_INDEX", "0")
    monkeypatch.setenv(
        "AGENTLODGE_EGL_SELECTOR_SHIM",
        str(tmp_path / "missing.so"),
    )
    with pytest.raises(RuntimeError, match="selector shim is missing"):
        warm_render._daemon_environment()


def test_one_gpu_daemon_environment_does_not_require_selector(monkeypatch):
    import server.warm_render as warm_render

    monkeypatch.delenv("AGENTLODGE_RENDER_MULTI_GPU", raising=False)
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.delenv("AGENTLODGE_EGL_SELECTOR_SHIM", raising=False)

    environment = warm_render._daemon_environment()

    assert "LD_PRELOAD" not in environment
    assert (
        environment["__EGL_VENDOR_LIBRARY_FILENAMES"]
        == "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )


def test_renderer_provenance_includes_root_motion_code():
    import server.warm_render as warm_render

    assert "render_root_motion.py" in {
        path.name for path in warm_render._renderer_files()
    }
    daemon_source = (
        ROOT / "scripts" / "blender_daemon.py"
    ).read_text(encoding="utf-8")
    assert "render_root_motion_sha256" in daemon_source
    assert 'RENDER_CONTRACT_VERSION = "render.frames-ffv1-v3"' in daemon_source


def test_daemon_attestation_matches_gpu_scene_renderer_and_quality(
    tmp_path,
    monkeypatch,
):
    import json

    import server.warm_render as warm_render

    provenance = {
        "render_contract_version": warm_render.RENDER_CONTRACT_VERSION,
        "daemon_protocol_version": warm_render.PROTOCOL_VERSION,
        "scene": {"blend_sha256": "1" * 64, "ybot_sha256": "2" * 64},
        "renderer": {
            "blender_version": "4.2.3 LTS",
            "blender_daemon_sha256": "3" * 64,
            "blender_render_ybot_sha256": "4" * 64,
            "blender_studio_sha256": "5" * 64,
            "render_root_motion_sha256": "6" * 64,
        },
        "selector": None,
    }
    d = tmp_path / "d0"
    d.mkdir()
    (d / "daemon.pid").write_text("1234")
    attestation = {
        "schema_version": 1,
        "pid": 1234,
        **provenance,
        "quality": {
            "width": 1080,
            "height": 1080,
            "samples": 96,
            "engine": "eevee",
            "denoise": 1,
            "frame_format": "tga",
        },
        "gpu": {
            "cuda_index": 0,
            "uuid": "GPU-test-0",
            "pci_bus_id": "00000000:01:00.0",
            "selection_mode": "single-visible-gpu",
        },
    }
    (d / "daemon.attestation.json").write_text(json.dumps(attestation))
    monkeypatch.setattr(warm_render, "render_provenance", lambda: provenance)
    monkeypatch.setattr(warm_render, "_selector_identity", lambda: None)
    monkeypatch.setenv("AGENTLODGE_RESOLVED_GPU_INDEX", "0")

    assert warm_render._attestation_matches(
        d,
        width=1080,
        height=1080,
        samples=96,
        engine="eevee",
        denoise=1,
        frame_format="tga",
    )
    attestation["gpu"]["cuda_index"] = 1
    (d / "daemon.attestation.json").write_text(json.dumps(attestation))
    assert not warm_render._attestation_matches(
        d,
        width=1080,
        height=1080,
        samples=96,
        engine="eevee",
        denoise=1,
        frame_format="tga",
    )


def test_warm_pool_restarts_attestation_mismatch(tmp_path, monkeypatch):
    import server.warm_render as warm_render

    process = {"alive": True}
    calls = []
    d = tmp_path / "d0"
    d.mkdir()
    (d / "daemon.ready").write_text(str(warm_render.PROTOCOL_VERSION))
    monkeypatch.setattr(warm_render, "DAEMON_ROOT", tmp_path)
    monkeypatch.setattr(warm_render, "POOL_SIZE", 1)
    monkeypatch.setattr(warm_render, "on_pod", lambda: True)
    monkeypatch.setattr(warm_render, "_pid_alive", lambda _d: process["alive"])
    monkeypatch.setattr(warm_render, "_alive", lambda _d: process["alive"])
    monkeypatch.setattr(warm_render, "render_provenance", lambda: {})
    monkeypatch.setattr(
        warm_render,
        "_attestation_matches",
        lambda _d, **_kwargs: False,
    )

    def stop(_d):
        calls.append("stop")
        process["alive"] = False

    def start(_index, **_kwargs):
        calls.append("start")

    monkeypatch.setattr(warm_render, "_stop_daemon", stop)
    monkeypatch.setattr(warm_render, "_start_daemon", start)
    warm_render.ensure_pool(
        width=1080,
        height=1080,
        samples=96,
        engine="eevee",
        denoise=1,
        frame_format="tga",
    )

    assert calls == ["stop", "start"]


def test_configured_pool_preserves_filament_export_quality(monkeypatch):
    import server.warm_render as warm_render

    captured = {}

    def fake_ensure_pool(**kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setenv("AGENTLODGE_FULL_RENDER_BACKEND", "filament")
    monkeypatch.setenv("AGENTLODGE_RENDER_FULL_W", "1080")
    monkeypatch.setenv("AGENTLODGE_RENDER_FULL_H", "1080")
    monkeypatch.setenv("AGENTLODGE_RENDER_FULL_SAMPLES", "96")
    monkeypatch.setenv("AGENTLODGE_RENDER_ENGINE", "eevee")
    monkeypatch.setenv("AGENTLODGE_RENDER_DENOISE", "1")
    monkeypatch.setattr(warm_render, "ensure_pool", fake_ensure_pool)

    assert warm_render.ensure_configured_pool(wait_ready=7) == 1
    assert captured == {
        "width": 1080,
        "height": 1080,
        "samples": 96,
        "engine": "eevee",
        "denoise": 1,
        "frame_format": "tga",
        "wait_ready": 7,
    }


def test_render_paths_default_to_isolated_tmp(tmp_path):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    helper = _bash_path(bash, ROOT / "scripts" / "render_worker_env.sh")
    environment = os.environ.copy()
    for name in (
        "AGENTLODGE_RENDER_LOCAL_ROOT",
        "AGENTLODGE_RENDER_LOCAL_BASE",
        "AGENTLODGE_RENDER_USE_SHM",
        "AGENTLODGE_SHM_RESERVATION_FILE",
    ):
        environment.pop(name, None)
    result = subprocess.run(
        [
            bash,
            "-c",
            (
                "set -euo pipefail\n"
                f"source {shlex.quote(helper)}\n"
                "agentlodge_configure_render_paths render-g0-d0\n"
                "printf '%s|%s\\n' "
                '"$AGENTLODGE_RENDER_LOCAL_ROOT" '
                '"${AGENTLODGE_SHM_RESERVATION_FILE:-}"\n'
            ),
        ],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/agentlodge-render-render-g0-d0|"


def test_shm_opt_in_reservation_fails_closed_on_overcommit(tmp_path):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    helper = _bash_path(bash, ROOT / "scripts" / "render_worker_env.sh")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "flock").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    (fake_bin / "df").write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fake 1000 0 1000 0%% /shm\\n'\n",
        encoding="utf-8",
        newline="\n",
    )
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    shm_root = tmp_path / "shm"
    ledger = shm_root / ".agentlodge-reservations"
    ledger.mkdir(parents=True)
    command = (
        "set -euo pipefail\n"
        f"PATH={shlex.quote(_bash_path(bash, fake_bin))}:/usr/bin:/bin:$PATH\n"
        f"source {shlex.quote(helper)}\n"
        f"AGENTLODGE_SHM_ROOT={shlex.quote(_bash_path(bash, shm_root))}\n"
        "AGENTLODGE_RENDER_USE_SHM=1\n"
        "AGENTLODGE_SHM_RESERVATION_BYTES=200000\n"
        "AGENTLODGE_SHM_HEADROOM_BYTES=0\n"
        "AGENTLODGE_SHM_MAX_PERCENT=50\n"
        "export AGENTLODGE_SHM_ROOT AGENTLODGE_RENDER_USE_SHM "
        "AGENTLODGE_SHM_RESERVATION_BYTES AGENTLODGE_SHM_HEADROOM_BYTES "
        "AGENTLODGE_SHM_MAX_PERCENT\n"
        'printf "%s 400000 existing\\n" "$$" > '
        '"$AGENTLODGE_SHM_ROOT/.agentlodge-reservations/existing.reservation"\n'
        "agentlodge_configure_render_paths render-g0-d0\n"
    )
    result = subprocess.run(
        [bash, "-c", command],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode != 0
    assert "reservations would exceed" in result.stderr


def test_shm_opt_in_accounts_for_reserved_free_space(tmp_path):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    helper = _bash_path(bash, ROOT / "scripts" / "render_worker_env.sh")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    (fake_bin / "flock").write_text(
        "#!/usr/bin/env bash\nexit 0\n",
        encoding="utf-8",
        newline="\n",
    )
    (fake_bin / "df").write_text(
        "#!/usr/bin/env bash\n"
        "printf 'Filesystem 1024-blocks Used Available Capacity Mounted on\\n'\n"
        "printf 'fake 2000 1000 1000 50%% /shm\\n'\n",
        encoding="utf-8",
        newline="\n",
    )
    for executable in fake_bin.iterdir():
        executable.chmod(0o755)
    shm_root = tmp_path / "shm"
    ledger = shm_root / ".agentlodge-reservations"
    ledger.mkdir(parents=True)
    command = (
        "set -euo pipefail\n"
        f"PATH={shlex.quote(_bash_path(bash, fake_bin))}:/usr/bin:/bin:$PATH\n"
        f"source {shlex.quote(helper)}\n"
        f"AGENTLODGE_SHM_ROOT={shlex.quote(_bash_path(bash, shm_root))}\n"
        "AGENTLODGE_RENDER_USE_SHM=1\n"
        "AGENTLODGE_SHM_RESERVATION_BYTES=300000\n"
        "AGENTLODGE_SHM_HEADROOM_BYTES=200000\n"
        "AGENTLODGE_SHM_MAX_PERCENT=90\n"
        "export AGENTLODGE_SHM_ROOT AGENTLODGE_RENDER_USE_SHM "
        "AGENTLODGE_SHM_RESERVATION_BYTES AGENTLODGE_SHM_HEADROOM_BYTES "
        "AGENTLODGE_SHM_MAX_PERCENT\n"
        'printf "%s 600000 existing\\n" "$$" > '
        '"$AGENTLODGE_SHM_ROOT/.agentlodge-reservations/existing.reservation"\n'
        "agentlodge_configure_render_paths render-g0-d0\n"
    )
    result = subprocess.run(
        [bash, "-c", command],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )

    assert result.returncode != 0
    assert "aggregate reservations plus headroom" in result.stderr


def test_setup_and_launcher_wire_selector_without_global_preload():
    launcher = (
        ROOT / "scripts" / "start_runpod_worker.sh"
    ).read_text(encoding="utf-8")
    setup = (ROOT / "scripts" / "setup_pod.sh").read_text(encoding="utf-8")
    setup_gen = (
        ROOT / "scripts" / "setup_gen_pod.sh"
    ).read_text(encoding="utf-8")
    bootstrap = (
        ROOT / "scripts" / "runpod_bootstrap.sh"
    ).read_text(encoding="utf-8")
    pod_helper = (ROOT / "scripts" / "pod.ps1").read_text(encoding="utf-8")

    assert "render_worker_env.sh" in launcher
    assert "agentlodge_configure_render_paths" in launcher
    assert "export LD_PRELOAD" not in launcher
    assert "build_egl_selector.sh" in setup
    assert "build_egl_selector.sh" in setup_gen
    assert "build_egl_selector.sh" in bootstrap
    assert "egl_cuda_device_selector.c" in pod_helper


def test_selector_builds_when_linux_gcc_is_available(tmp_path):
    bash = _bash_executable()
    if bash is None:
        pytest.skip("bash is not available")
    probe = subprocess.run(
        [bash, "-lc", "command -v gcc >/dev/null && command -v nm >/dev/null"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("Linux gcc/nm toolchain is not available")
    build_script = _bash_path(
        bash,
        ROOT / "scripts" / "build_egl_selector.sh",
    )
    root = _bash_path(bash, ROOT)
    output = _bash_path(bash, tmp_path / "selector.so")
    result = subprocess.run(
        [bash, build_script, output],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "WORKSPACE": root,
            "AGENTLODGE_ROOT": root,
        },
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "selector.so").is_file()
    assert (tmp_path / "selector.so.build.json").is_file()
