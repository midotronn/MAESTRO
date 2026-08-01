"""3D rendering of a generated dance with pyrender.

Two modes, chosen automatically:

* **SMPL-X mesh**: if a SMPL-X body model (``SMPLX_NEUTRAL.npz``) is available, the
  139-dim motion is turned into a full SMPL-X body mesh per frame and rendered as a solid
  character (like the EDGE / LODGE / AIST++ visualisations). The body model is licence
  gated (https://smpl-x.is.tue.mpg.de), so it must be supplied by the user.
* **Articulated body (fallback)**: when no body model is present, a shaded 3D figure is
  built directly from the forward-kinematics joints (spheres at joints + tapered capsules
  along the bones). No gated assets required, so this always works.

Both paths render off-screen through OSMesa with directional lighting, soft shadows, a
ground plane and an auto-framed camera, then pipe frames to ffmpeg and mux the audio.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = os.environ.get("PYRENDER_GL", "egl")

import numpy as np

# pyrender still references a few aliases removed in NumPy 2.0; restore them so it imports
# and renders under modern NumPy.
for _alias, _val in (("infty", np.inf), ("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _val)

import pyrender
import trimesh

from agentlodge.config import FPS
from agentlodge.video.stick_figure import BODY_PARENTS, joints_from_motion139

logger = logging.getLogger(__name__)

# Larger spheres for the structural joints (pelvis, spine, neck, head), smaller elsewhere.
_BIG_JOINTS = {0, 3, 6, 9, 12, 15}
BODY_COLOR = np.array([0.40, 0.55, 0.85, 1.0])
FLOOR_COLOR = np.array([0.85, 0.85, 0.88, 1.0])


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera-to-world pose matrix looking from ``eye`` at ``target`` (OpenGL convention)."""
    eye, target, up = map(lambda v: np.asarray(v, dtype=np.float64), (eye, target, up))
    forward = eye - target
    forward /= np.linalg.norm(forward) + 1e-9
    right = np.cross(up, forward)
    right /= np.linalg.norm(right) + 1e-9
    true_up = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _capsule(p0: np.ndarray, p1: np.ndarray, radius: float) -> trimesh.Trimesh | None:
    seg = np.linalg.norm(p1 - p0)
    if seg < 1e-4:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, segment=[p0, p1], sections=12)
    return cyl


def _mannequin_frame(joints: np.ndarray) -> trimesh.Trimesh:
    """Build one 3D articulated figure (spheres + capsules) from (22, 3) joints."""
    parts: list[trimesh.Trimesh] = []
    for j, xyz in enumerate(joints):
        r = 0.055 if j in _BIG_JOINTS else 0.035
        s = trimesh.creation.uv_sphere(radius=r, count=[10, 10])
        s.apply_translation(xyz)
        parts.append(s)
    for i, parent in enumerate(BODY_PARENTS):
        if parent < 0:
            continue
        cap = _capsule(joints[i], joints[parent], radius=0.028)
        if cap is not None:
            parts.append(cap)
    mesh = trimesh.util.concatenate(parts)
    mesh.visual.vertex_colors = (BODY_COLOR * 255).astype(np.uint8)
    return mesh


def _smplx_vertices(motion139: np.ndarray, model_path: Path, lodge_code_path: Path):
    """Per-frame SMPL-X vertices (L, V, 3) and faces from a 139-dim motion + body model.

    139 layout: root_translation(3) + 22-joint 6D rotation(132) + contact(4).
    """
    import torch
    from agentlodge.dance.format import to_native_finedance139
    from agentlodge.env_paths import lodge_import_paths, use_code_paths

    native = to_native_finedance139(np.asarray(motion139, dtype=np.float32))
    trans = native[:, 4:7]
    rot6d = native[:, 7:139].reshape(-1, 22, 6)

    with use_code_paths(*lodge_import_paths(lodge_code_path)):
        from dld.data.render_joints.smplfk import ax_from_6v
        poses = ax_from_6v(torch.from_numpy(rot6d).float()).reshape(-1, 22, 3).numpy()

    import smplx as smplx_pkg

    model = smplx_pkg.create(
        str(model_path), model_type="smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, batch_size=1,
    )
    length = poses.shape[0]
    verts = np.empty((length, model.get_num_verts(), 3), dtype=np.float32)
    with torch.no_grad():
        for i in range(length):
            out = model(
                global_orient=torch.from_numpy(poses[i, 0:1]).float().reshape(1, 3),
                body_pose=torch.from_numpy(poses[i, 1:22]).float().reshape(1, 63),
                transl=torch.from_numpy(trans[i]).float().reshape(1, 3),
            )
            verts[i] = out.vertices[0].numpy()
    faces = model.faces.astype(np.int64)
    # SMPL-X is Y-up; match the joints' Z-up convention used for framing.
    verts = verts[..., [0, 2, 1]]
    return verts, faces


def _frame_camera(points: np.ndarray, img_aspect: float):
    """Return (eye, target, up, floor_z) framing the whole motion (points: (L, N, 3), z-up)."""
    flat = points.reshape(-1, 3)
    cx = 0.5 * (flat[:, 0].min() + flat[:, 0].max())
    cy = 0.5 * (flat[:, 1].min() + flat[:, 1].max())
    floor_z = float(np.percentile(points[:, :, 2].min(axis=1), 5))
    top_z = float(flat[:, 2].max())
    center = np.array([cx, cy, 0.5 * (floor_z + top_z)])
    span = max(
        float(flat[:, 0].max() - flat[:, 0].min()),
        float(flat[:, 1].max() - flat[:, 1].min()),
        float(top_z - floor_z), 1.0,
    )
    dist = span * 2.2
    eye = center + np.array([dist * 0.9, -dist * 1.4, dist * 0.55])
    up = np.array([0.0, 0.0, 1.0])
    return eye, center, up, floor_z


def _floor(center_xy, floor_z, size=8.0) -> trimesh.Trimesh:
    plane = trimesh.creation.box(extents=[size, size, 0.02])
    plane.apply_translation([center_xy[0], center_xy[1], floor_z - 0.01])
    plane.visual.vertex_colors = (FLOOR_COLOR * 255).astype(np.uint8)
    return plane


def render_dance_video(
    motion_npy: Path,
    output_mp4: Path,
    *,
    lodge_code_path: Path,
    audio_path: Path | None = None,
    smplx_model_path: Path | None = None,
    img_size: tuple[int, int] = (720, 720),
    fps: int = FPS,
) -> Path:
    """Render a (L, 139) motion .npy to a shaded 3D mp4 (SMPL-X mesh or articulated body)."""
    output_mp4 = output_mp4.resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    motion = np.load(motion_npy)

    joints = joints_from_motion139(motion, lodge_code_path=lodge_code_path)  # (L,22,3) z-up
    use_mesh = smplx_model_path is not None and Path(smplx_model_path).exists()

    if use_mesh:
        logger.info("Rendering SMPL-X mesh from %s", smplx_model_path)
        verts, faces = _smplx_vertices(motion, Path(smplx_model_path), lodge_code_path)
        frame_points = verts
    else:
        logger.info("No SMPL-X model supplied; rendering articulated 3D body from joints")
        verts, faces = None, None
        frame_points = joints

    w, h = img_size
    eye, target, up, floor_z = _frame_camera(frame_points, w / h)
    cam_pose = _look_at(eye, target, up)

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=[0.35, 0.35, 0.35])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=w / h)
    scene.add(camera, pose=cam_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    light_pose = _look_at(target + np.array([2.0, -2.0, 5.0]), target, up)
    scene.add(light, pose=light_pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
              pose=_look_at(target + np.array([-3.0, 2.0, 3.0]), target, up))

    floor_center = (target[0], target[1])
    scene.add(pyrender.Mesh.from_trimesh(_floor(floor_center, floor_z), smooth=False))

    renderer = pyrender.OffscreenRenderer(w, h)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")

    tmp_video = output_mp4.with_suffix(".noaudio.mp4")
    cmd = [
        ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(tmp_video),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    node = None
    length = frame_points.shape[0]
    try:
        for i in range(length):
            if use_mesh:
                mesh = trimesh.Trimesh(vertices=verts[i], faces=faces, process=False)
                mesh.visual.vertex_colors = (BODY_COLOR * 255).astype(np.uint8)
                pm = pyrender.Mesh.from_trimesh(mesh, smooth=True)
            else:
                pm = pyrender.Mesh.from_trimesh(_mannequin_frame(joints[i]), smooth=False)
            if node is not None:
                scene.remove_node(node)
            node = scene.add(pm)
            color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            proc.stdin.write(np.ascontiguousarray(color[:, :, :3]).tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
        renderer.delete()

    if audio_path is not None and Path(audio_path).exists():
        cmd = [
            ffmpeg, "-loglevel", "error", "-y", "-i", str(tmp_video),
            "-i", str(audio_path), "-shortest", "-c:v", "copy",
            "-c:a", "aac", "-q:a", "4", str(output_mp4),
        ]
        subprocess.run(cmd, check=True)
        tmp_video.unlink(missing_ok=True)
    else:
        tmp_video.replace(output_mp4)

    logger.info("Saved 3D dance video to %s", output_mp4)
    return output_mp4
