#!/usr/bin/env python3
"""One-off patches for upstream EDGE on RunPod (PyTorch 2.6 weights_only default).

PyTorch 2.6 changed ``torch.load``'s default to ``weights_only=True``. EDGE's
checkpoint stores a ``dataset.preprocess.Normalizer`` object, so it must be loaded
with ``weights_only=False``.
"""
from __future__ import annotations

import os
from pathlib import Path

EDGE = Path(os.environ.get("EDGE_CODE_PATH", "/workspace/EDGE"))

edge_py = EDGE / "EDGE.py"
if edge_py.exists():
    body = edge_py.read_text()
    needle = "checkpoint_path, map_location=self.accelerator.device"
    patched = needle + ", weights_only=False"
    if needle in body and patched not in body:
        edge_py.write_text(body.replace(needle, patched))
        print("patched EDGE.py torch.load weights_only")
    else:
        print("EDGE.py already patched or pattern missing")
else:
    print(f"EDGE.py not found at {edge_py}")

# render_sample (long/stitch branch): upstream calls skeleton_render unconditionally, and its
# `if sound:` path loads audio from name[0] even when render=False; the fk_out pkl name also indexes
# name[0]. With headless motion-only inference there are no wav slices (name is empty), so both crash
# with IndexError. Guard the render behind `render` and make the pkl name robust to an empty name, so
# EDGE inference is fully wav-independent (motion comes from the cached jukebox features).
diff_py = EDGE / "model" / "diffusion.py"
if diff_py.exists():
    body = diff_py.read_text()
    old = (
        "            skeleton_render(\n"
        "                full_pose[0],\n"
        "                epoch=f\"{epoch}\",\n"
        "                out=render_out,\n"
        "                name=name,\n"
        "                sound=sound,\n"
        "                stitch=True,\n"
        "                sound_folder=sound_folder,\n"
        "                render=render\n"
        "            )\n"
        "            if fk_out is not None:\n"
        "                outname = f'{epoch}_{\"_\".join(os.path.splitext(os.path.basename(name[0]))[0].split(\"_\")[:-1])}.pkl'\n"
    )
    new = (
        "            if render:\n"
        "                skeleton_render(\n"
        "                    full_pose[0],\n"
        "                    epoch=f\"{epoch}\",\n"
        "                    out=render_out,\n"
        "                    name=name,\n"
        "                    sound=sound,\n"
        "                    stitch=True,\n"
        "                    sound_folder=sound_folder,\n"
        "                    render=render\n"
        "                )\n"
        "            if fk_out is not None:\n"
        "                _basenm = os.path.basename(name[0]) if name else \"take_0.wav\"\n"
        "                outname = f'{epoch}_{\"_\".join(os.path.splitext(_basenm)[0].split(\"_\")[:-1]) or \"0\"}.pkl'\n"
    )
    if new.split("\n")[0] + "\n" in body and "_basenm = os.path.basename" in body:
        print("diffusion.py render_sample already patched")
    elif old in body:
        diff_py.write_text(body.replace(old, new))
        print("patched diffusion.py render_sample (render-guard + wav-independent fk_out name)")
    else:
        print("diffusion.py render_sample pattern missing (upstream changed?)")
else:
    print(f"diffusion.py not found at {diff_py}")
