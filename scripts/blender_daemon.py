"""Persistent (warm) Blender render daemon for the Y-Bot character.

Run INSIDE Blender with the cached scene preloaded, so the ~8s Blender startup + FBX import + studio
build is paid ONCE and every subsequent render skips it::

    __EGL_VENDOR_LIBRARY_FILENAMES=<egl.json> blender -b <scene.blend> -noaudio \
        -P blender_daemon.py -- --ybot <ybot.fbx> --requests-dir /workspace/blend_daemon \
        --width 448 --height 448 --samples 8

The daemon watches ``<requests-dir>`` for ``*.req`` files (JSON), each describing one render::

    {"id": "abc", "poses": "/path/poses.npz", "frames_dir": "/path/frames",
     "width": 448, "height": 448, "samples": 8, "color": "0.5,0.5,0.52", "yaw": 0,
     "rig_metrics": "/optional/path/rig_metrics.npz"}

For each request it renders the poses to ``frames_dir`` (reusing the warm scene) and writes
``<id>.done`` (or ``<id>.fail`` with the error). A client then muxes the frames with ffmpeg. The
daemon writes ``daemon.ready`` once the scene is up and touches ``daemon.hb`` each poll so a client
can detect a live daemon.
"""

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace

import bpy  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render_ybot as ybot  # noqa: E402
import blender_studio as studio  # noqa: E402

PROTOCOL_VERSION = 6
RENDER_CONTRACT_VERSION = "render.frames-ffv1-v3"
SELECTOR_VERSION = 2


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--ybot", required=True)
    p.add_argument("--requests-dir", required=True)
    p.add_argument("--width", type=int, default=448)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--engine", default="eevee")
    p.add_argument("--denoise", type=int, default=1)
    p.add_argument("--frame-format", default="png")
    p.add_argument("--color", default="0.5,0.5,0.52")
    p.add_argument("--idle-exit", type=int, default=3600,
                   help="Exit after this many seconds with no requests (0 = never).")
    return p.parse_args(argv)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    target = Path(path)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_inventory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU attestation failed: {result.stderr[-400:]}")
    inventory = []
    for line in result.stdout.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            raise RuntimeError(f"invalid nvidia-smi GPU identity row: {line!r}")
        inventory.append(
            {
                "cuda_index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2],
            }
        )
    if not inventory:
        raise RuntimeError("nvidia-smi reported no GPUs for daemon attestation")
    return inventory


def _selector_identity():
    multi_gpu = os.environ.get("AGENTLODGE_RENDER_MULTI_GPU", "").strip().lower()
    if multi_gpu not in {"1", "true", "yes", "on"}:
        return None
    attestation_path = os.environ.get("AGENTLODGE_EGL_ATTESTATION_PATH", "").strip()
    selector_path = os.environ.get("AGENTLODGE_EGL_SELECTOR_SHIM", "").strip()
    requested = os.environ.get("AGENTLODGE_GPU_INDEX", "").strip()
    if not attestation_path or not selector_path or not requested.isdigit():
        raise RuntimeError("multi-GPU daemon is missing selector attestation configuration")
    try:
        selector_attestation = json.loads(
            Path(attestation_path).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("EGL selector did not publish a valid attestation") from exc
    requested_index = int(requested)
    expected_build_id = "sha256:" + _file_sha256(
        Path(__file__).with_name("egl_cuda_device_selector.c")
    )
    expected = {
        "schema_version": 1,
        "selector_version": SELECTOR_VERSION,
        "pid": os.getpid(),
        "requested_cuda_index": requested_index,
        "selected_cuda_index": requested_index,
    }
    for key, value in expected.items():
        if selector_attestation.get(key) != value:
            raise RuntimeError(
                f"EGL selector attestation mismatch for {key}: "
                f"{selector_attestation.get(key)!r} != {value!r}"
            )
    if selector_attestation.get("selector_build_id") != expected_build_id:
        raise RuntimeError("EGL selector build identity does not match repository source")
    selector = Path(selector_path).resolve()
    if not selector.is_file():
        raise RuntimeError(f"EGL selector binary is missing: {selector}")
    return {
        "version": SELECTOR_VERSION,
        "build_id": expected_build_id,
        "binary_sha256": _file_sha256(selector),
        "egl_device_index": int(selector_attestation["egl_device_index"]),
        "requested_cuda_index": requested_index,
        "selected_cuda_index": requested_index,
    }


def _daemon_attestation(cfg, requests_dir):
    selector = _selector_identity()
    inventory = _gpu_inventory()
    if selector is None:
        if len(inventory) != 1:
            raise RuntimeError(
                "render daemon sees multiple GPUs without the EGL selector shim"
            )
        selected_index = inventory[0]["cuda_index"]
        selection_mode = "single-visible-gpu"
    else:
        selected_index = selector["selected_cuda_index"]
        selection_mode = "egl-cuda-device-nv"
    matches = [
        item for item in inventory if item["cuda_index"] == selected_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"selected CUDA index {selected_index} does not identify one physical GPU"
        )
    scripts_dir = Path(__file__).resolve().parent
    scene_path = Path(str(bpy.data.filepath or "")).resolve()
    if not scene_path.is_file():
        raise RuntimeError("Blender daemon scene file is unavailable for attestation")
    ybot_path = Path(cfg.ybot).resolve()
    payload = {
        "schema_version": 1,
        "render_contract_version": RENDER_CONTRACT_VERSION,
        "daemon_protocol_version": PROTOCOL_VERSION,
        "pid": os.getpid(),
        "scene": {
            "blend_sha256": _file_sha256(scene_path),
            "ybot_sha256": _file_sha256(ybot_path),
        },
        "renderer": {
            "blender_version": str(bpy.app.version_string),
            "blender_daemon_sha256": _file_sha256(Path(__file__).resolve()),
            "blender_render_ybot_sha256": _file_sha256(
                scripts_dir / "blender_render_ybot.py"
            ),
            "blender_studio_sha256": _file_sha256(
                scripts_dir / "blender_studio.py"
            ),
            "render_root_motion_sha256": _file_sha256(
                scripts_dir / "render_root_motion.py"
            ),
        },
        "quality": {
            "width": int(cfg.width),
            "height": int(cfg.height),
            "samples": int(cfg.samples),
            "engine": str(cfg.engine).lower(),
            "denoise": int(cfg.denoise),
            "frame_format": str(cfg.frame_format).lower().lstrip("."),
        },
        "gpu": {
            **matches[0],
            "selection_mode": selection_mode,
        },
        "selector": selector,
    }
    _atomic_json(Path(requests_dir) / "daemon.attestation.json", payload)
    return payload


def _ensure_scene(cfg, color):
    """Make sure the rig + studio are loaded (they are when opened with the cached .blend); build
    them once if the daemon was started without a scene."""
    if not any(o.type == "ARMATURE" for o in bpy.data.objects):
        arm = ybot.import_ybot(cfg.ybot)
        ybot.style_robot(arm, color)
        ybot.normalise_scale(arm)
        arm.rotation_mode = "QUATERNION"
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"
        ybot.setup_studio()
    studio.configure_render(cfg.width, cfg.height, cfg.samples,
                            engine=cfg.engine, denoise=bool(cfg.denoise))


def _warm_renderer(cfg, requests_dir):
    """Compile the EEVEE pipeline before advertising readiness so the first user render is fast."""
    warmup = os.path.join(requests_dir, ".warmup.png")
    studio.configure_render(64, 64, 1, engine=cfg.engine, denoise=False)
    bpy.context.scene.render.filepath = warmup
    bpy.ops.render.render(write_still=True)
    try:
        os.remove(warmup)
    except OSError:
        pass
    studio.configure_render(
        cfg.width,
        cfg.height,
        cfg.samples,
        engine=cfg.engine,
        denoise=bool(cfg.denoise),
    )
    print("BLENDER_DAEMON_WARM", flush=True)


def _render_request(req, cfg):
    color = tuple(float(c) for c in str(req.get("color", cfg.color)).split(","))[:3]
    args = SimpleNamespace(
        poses=req["poses"], frames_dir=req["frames_dir"], ybot=cfg.ybot,
        width=int(req.get("width", cfg.width)), height=int(req.get("height", cfg.height)),
        samples=int(req.get("samples", cfg.samples)), engine=req.get("engine", cfg.engine),
        denoise=int(req.get("denoise", cfg.denoise)), color=req.get("color", cfg.color),
        align_x=0.0, yaw=float(req.get("yaw", 0.0)), stride=int(req.get("stride", 1)),
        frame_start=int(req.get("frame_start", 0)),
        frame_end=int(req.get("frame_end", -1)),
        keep_existing_frames=not bool(req.get("clear_frames", True)),
        frame_format=req.get("frame_format", "png"),
        build_scene="", force_align=False, fk_npz=req.get("fk_npz", ""),
        fast=bool(req.get("fast", True)),
        foot_grounding=bool(req.get("foot_grounding", False)),
        lock_root=bool(req.get("lock_root", False)),
        fixed_camera=bool(req.get("fixed_camera", False)),
        rig_metrics=req.get("rig_metrics", ""),
        projection_only=bool(req.get("projection_only", False)),
        batch_render=bool(req.get("batch_render", False)),
        video_path=req.get("video_path", ""),
        export_glb=req.get("export_glb", ""),
    )
    os.makedirs(args.frames_dir, exist_ok=True)
    ybot.render_take(args, color)


def main():
    cfg = _args()
    rdir = cfg.requests_dir
    os.makedirs(rdir, exist_ok=True)
    color = tuple(float(c) for c in cfg.color.split(","))[:3]
    _ensure_scene(cfg, color)
    _warm_renderer(cfg, rdir)
    attestation = _daemon_attestation(cfg, rdir)
    with open(os.path.join(rdir, "daemon.ready"), "w") as ready:
        ready.write(str(PROTOCOL_VERSION))
    print(
        "BLENDER_DAEMON_READY "
        f"cuda={attestation['gpu']['cuda_index']} "
        f"uuid={attestation['gpu']['uuid']}",
        flush=True,
    )

    # Heartbeat on a background thread so it keeps ticking even while a long render holds the main
    # thread; a client uses the heartbeat freshness to tell a live (busy) daemon from a dead one.
    _stop = threading.Event()

    def _beat():
        while not _stop.is_set():
            try:
                with open(os.path.join(rdir, "daemon.hb"), "w") as f:
                    f.write(str(int(time.time())))
            except Exception:  # noqa: BLE001
                pass
            _stop.wait(3)

    threading.Thread(target=_beat, daemon=True).start()

    last_activity = time.time()
    while True:
        reqs = sorted(glob.glob(os.path.join(rdir, "*.req")))
        for reqpath in reqs:
            rid = os.path.splitext(os.path.basename(reqpath))[0]
            done = os.path.join(rdir, rid + ".done")
            fail = os.path.join(rdir, rid + ".fail")
            try:
                with open(reqpath) as f:
                    req = json.load(f)
                os.remove(reqpath)                      # claim the request
            except Exception:  # noqa: BLE001 - half-written file: retry next poll
                continue
            t0 = time.time()
            try:
                _render_request(req, cfg)
                with open(done, "w") as f:
                    f.write(f"{time.time() - t0:.2f}")
                print(f"DAEMON_RENDERED {rid} in {time.time() - t0:.1f}s", flush=True)
            except Exception:  # noqa: BLE001 - never let one bad request kill the daemon
                with open(fail, "w") as f:
                    f.write(traceback.format_exc()[-1500:])
                print(f"DAEMON_FAILED {rid}", flush=True)
            last_activity = time.time()
        if cfg.idle_exit and (time.time() - last_activity) > cfg.idle_exit:
            print("BLENDER_DAEMON_IDLE_EXIT", flush=True)
            _stop.set()
            return
        time.sleep(0.05)


if __name__ == "__main__":
    main()
