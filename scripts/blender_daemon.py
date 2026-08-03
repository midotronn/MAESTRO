"""Persistent (warm) Blender render daemon for the Y-Bot character.

Run INSIDE Blender with the cached scene preloaded, so the ~8s Blender startup + FBX import + studio
build is paid ONCE and every subsequent render skips it::

    __EGL_VENDOR_LIBRARY_FILENAMES=<egl.json> blender -b <scene.blend> -noaudio \
        -P blender_daemon.py -- --ybot <ybot.fbx> --requests-dir /workspace/blend_daemon \
        --width 448 --height 448 --samples 8

The daemon watches ``<requests-dir>`` for ``*.req`` files (JSON), each describing one render::

    {"id": "abc", "poses": "/path/poses.npz", "frames_dir": "/path/frames",
     "width": 448, "height": 448, "samples": 8, "color": "0.5,0.5,0.52", "yaw": 0}

For each request it renders the poses to ``frames_dir`` (reusing the warm scene) and writes
``<id>.done`` (or ``<id>.fail`` with the error). A client then muxes the frames with ffmpeg. The
daemon writes ``daemon.ready`` once the scene is up and touches ``daemon.hb`` each poll so a client
can detect a live daemon.
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import bpy  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_render_ybot as ybot  # noqa: E402
import blender_studio as studio  # noqa: E402


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
    p.add_argument("--color", default="0.5,0.5,0.52")
    p.add_argument("--idle-exit", type=int, default=3600,
                   help="Exit after this many seconds with no requests (0 = never).")
    return p.parse_args(argv)


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


def _render_request(req, cfg):
    color = tuple(float(c) for c in str(req.get("color", cfg.color)).split(","))[:3]
    args = SimpleNamespace(
        poses=req["poses"], frames_dir=req["frames_dir"], ybot=cfg.ybot,
        width=int(req.get("width", cfg.width)), height=int(req.get("height", cfg.height)),
        samples=int(req.get("samples", cfg.samples)), engine=req.get("engine", cfg.engine),
        denoise=int(req.get("denoise", cfg.denoise)), color=req.get("color", cfg.color),
        align_x=0.0, yaw=float(req.get("yaw", 0.0)), stride=int(req.get("stride", 1)),
        build_scene="", force_align=False, fk_npz=req.get("fk_npz", ""),
        fast=bool(req.get("fast", True)),
    )
    os.makedirs(args.frames_dir, exist_ok=True)
    ybot.render_take(args, color)


def main():
    cfg = _args()
    rdir = cfg.requests_dir
    os.makedirs(rdir, exist_ok=True)
    color = tuple(float(c) for c in cfg.color.split(","))[:3]
    _ensure_scene(cfg, color)
    open(os.path.join(rdir, "daemon.ready"), "w").close()
    print("BLENDER_DAEMON_READY", flush=True)

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
        time.sleep(0.2)


if __name__ == "__main__":
    main()
