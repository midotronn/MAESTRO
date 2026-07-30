#!/usr/bin/env python3
"""Compute posed SMPL-X vertices for a dance and render them in Blender.

Blender's bundled Python cannot import torch / the smplx package, so this orchestrator
(run from the AgentLODGE venv) does the heavy lifting:

1. turn a (L, 139) motion array into per-frame SMPL-X body vertices, faces and per-loop
   UVs using the licensed SMPL-X body model;
2. save them to a compact ``.npz``;
3. invoke Blender headless with ``scripts/blender_render_smplx.py`` to render a shaded,
   textured mp4 (camera framed to the motion, sun + fill lights, ground with shadows);
4. mux the input audio.

The SMPL-X body model is licence gated (https://smpl-x.is.tue.mpg.de) and must be
supplied by the user; nothing here is bundled.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def compute_smplx_meshes(
    motion139: np.ndarray, model_dir: Path, lodge_code_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return (verts (L, V, 3) native y-up, faces (F, 3)) for a 139-dim motion."""
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = smplx_pkg.create(
        str(model_dir), model_type="smplx", gender="neutral",
        use_pca=False, flat_hand_mean=True, batch_size=1,
    ).to(device)
    length = poses.shape[0]
    poses_t = torch.from_numpy(poses).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)
    verts = np.empty((length, model.get_num_verts(), 3), dtype=np.float32)
    batch = 256
    with torch.no_grad():
        for s in range(0, length, batch):
            e = min(s + batch, length)
            b = e - s
            z3 = torch.zeros(b, 3, device=device)
            z45 = torch.zeros(b, 45, device=device)
            z10 = torch.zeros(b, 10, device=device)
            # Match LODGE's exact SMPL-X call: explicit flat hands / neutral face / zero
            # shape so nothing falls back to a non-flat default that would bend the limbs.
            out = model(
                betas=z10,
                global_orient=poses_t[s:e, 0].reshape(b, 3),
                body_pose=poses_t[s:e, 1:22].reshape(b, 63),
                transl=trans_t[s:e].reshape(b, 3),
                jaw_pose=z3, leye_pose=z3, reye_pose=z3,
                left_hand_pose=z45, right_hand_pose=z45,
                expression=z10,
            )
            verts[s:e] = out.vertices.detach().cpu().numpy()
    return verts, model.faces.astype(np.int32)


def compute_smpl_poses(
    motion139: np.ndarray, lodge_code_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (poses (L, 24, 3) SMPL axis-angle, trans (L, 3), fk_joints (L, 22, 3)).

    Joints 0-21 come from the motion's rot6d; the two SMPL hand joints (22, 23) are left
    at zero. ``fk_joints`` are the ground-truth SMPL joint positions from forward kinematics
    and are used to fix the Y-Bot's global orientation (Kabsch), so the robot stands
    upright and balanced exactly like the reference skeleton.
    """
    import torch

    from agentlodge.dance.format import to_native_finedance139
    from agentlodge.env_paths import lodge_import_paths, use_code_paths

    native = to_native_finedance139(np.asarray(motion139, dtype=np.float32))
    trans = native[:, 4:7].astype(np.float32)
    rot6d = native[:, 7:139].reshape(-1, 22, 6)

    jpath = Path(lodge_code_path) / "data" / "smplx_neu_J_1.npy"
    with use_code_paths(*lodge_import_paths(lodge_code_path)):
        from dld.data.render_joints.smplfk import SMPLX_Skeleton, ax_from_6v

        ax = ax_from_6v(torch.from_numpy(rot6d).float()).reshape(-1, 22, 3).numpy()
        fk_joints = np.zeros((ax.shape[0], 22, 3), dtype=np.float32)
        if jpath.exists():
            poses66 = torch.from_numpy(ax.reshape(-1, 66)).float()
            fk = SMPLX_Skeleton(device="cpu", batch=1, Jpath=str(jpath))
            fk_joints = (
                fk.forward(poses66, torch.from_numpy(trans).float())
                .detach()
                .cpu()
                .numpy()[:, :22]
                .astype(np.float32)
            )

    length = ax.shape[0]
    poses = np.zeros((length, 24, 3), dtype=np.float32)
    poses[:, :22] = ax
    fk_joints = _orient_joints_zup(fk_joints)
    return poses, trans, fk_joints


def _orient_joints_zup(joints: np.ndarray) -> np.ndarray:
    """Reorient FK joints (L, 22, 3) so the human's vertical axis maps to +Z, using a proper
    (det=+1) rotation. The FK output frame is data-dependent (LODGE/FineDance is Y-up,
    EDGE-derived motion is Z-up); the Y-Bot renderer Kabsch-aligns to these joints, so a Y-up
    target would lay the robot on the floor. Normalising to Z-up here makes the robot stand
    regardless of the source model.
    """
    if joints.size == 0:
        return joints
    # Detect the body's vertical ANATOMICALLY (head-15 above feet-7,8), averaged over frames.
    # This is robust to wide-arm / dynamic poses (unlike a bounding-box extent argmax, which can
    # pick the arm-span axis) and is invariant under rotation about the vertical, so facing
    # normalisation stays consistent.
    hf = (joints[:, 15, :] - joints[:, [7, 8], :].mean(axis=1)).mean(axis=0)
    up = int(np.argmax(np.abs(hf)))
    positive = hf[up] >= 0
    if up == 1:  # Y vertical -> Z. +Y via Rx(+90):(x,y,z)->(x,-z,y); -Y via Rx(-90):(x,y,z)->(x,z,-y)
        if positive:
            joints = np.stack([joints[..., 0], -joints[..., 2], joints[..., 1]], axis=-1)
        else:
            joints = np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)
    elif up == 0:  # X vertical -> Z. +X via Ry(+90):(x,y,z)->(-z,y,x); -X via Ry(-90):(x,y,z)->(z,y,-x)
        if positive:
            joints = np.stack([-joints[..., 2], joints[..., 1], joints[..., 0]], axis=-1)
        else:
            joints = np.stack([joints[..., 2], joints[..., 1], -joints[..., 0]], axis=-1)
    elif not positive:  # already Z vertical but pointing -Z -> flip via Rx(180):(x,y,z)->(x,-y,-z)
        joints = np.stack([joints[..., 0], -joints[..., 1], -joints[..., 2]], axis=-1)
    return joints.astype(np.float32)


def _ground_verts(verts: np.ndarray, fps: int = 30) -> np.ndarray:
    """Rest the dancer on the floor: subtract a lightly smoothed per-frame lowest-vertex
    height so the planted foot sits at z=0 (instead of the whole clip floating above a
    single global-minimum floor). Light smoothing keeps brief hops without popping.
    """
    foot_min = verts[:, :, 2].min(axis=1)  # (L,)
    if foot_min.shape[0] >= 3:
        k = min(5, foot_min.shape[0] | 1)
        pad = k // 2
        padded = np.pad(foot_min, (pad, pad), mode="edge")
        foot_min = np.convolve(padded, np.ones(k) / k, mode="valid")[: verts.shape[0]]
    out = verts.copy()
    out[:, :, 2] -= foot_min[:, None]
    return out


def _orient_verts_zup(verts: np.ndarray) -> np.ndarray:
    """Rotate SMPL-X verts (L, V, 3) so the body's vertical axis maps to +Z.

    The FK output orientation is data dependent (LODGE vs EDGE-derived motion), so detect
    the vertical axis as the one with the largest median per-frame extent and map it to Z
    with a proper (det=+1) rotation, keeping face winding/normals valid.
    """
    ext = np.median(verts.max(axis=1) - verts.min(axis=1), axis=0)
    up = int(np.argmax(ext))
    if up == 2:
        return verts
    if up == 1:  # Y-up -> Z-up via Rx(+90): (x, y, z) -> (x, -z, y)
        out = np.stack([verts[..., 0], -verts[..., 2], verts[..., 1]], axis=-1)
    else:  # X-up -> Z-up via Ry(+90): (x, y, z) -> (-z, y, x)
        out = np.stack([-verts[..., 2], verts[..., 1], verts[..., 0]], axis=-1)
    return out.astype(np.float32)


def render_blender_dance(
    motion_npy: Path,
    output_mp4: Path,
    *,
    lodge_code_path: Path,
    smplx_model_dir: Path,
    blender_bin: Path,
    blender_script: Path,
    audio_path: Path | None = None,
    uv_npz: Path | None = None,
    texture_png: Path | None = None,
    img_size: tuple[int, int] = (960, 960),
    samples: int = 64,
    fps: int = 30,
) -> Path:
    output_mp4 = Path(output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    motion = np.load(motion_npy)

    logger.info("Computing SMPL-X meshes for %d frames...", motion.shape[0])
    verts, faces = compute_smplx_meshes(motion, smplx_model_dir, lodge_code_path)
    verts = _orient_verts_zup(verts)
    verts = _ground_verts(verts)

    uv = None
    if uv_npz is not None and Path(uv_npz).exists():
        uv_data = np.load(uv_npz, allow_pickle=True)
        key = "uv_coordinates" if "uv_coordinates" in uv_data.files else uv_data.files[0]
        uv = uv_data[key].astype(np.float32)

    work = Path(tempfile.mkdtemp(prefix="blender_smplx_"))
    mesh_npz = work / "meshes.npz"
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = {"verts": verts, "faces": faces}
    if uv is not None:
        save_kwargs["uv"] = uv
    np.savez(mesh_npz, **save_kwargs)

    cmd = [
        str(blender_bin), "-b", "-noaudio", "-P", str(blender_script), "--",
        "--meshes", str(mesh_npz),
        "--frames-dir", str(frames_dir),
        "--width", str(img_size[0]), "--height", str(img_size[1]),
        "--samples", str(samples),
    ]
    if texture_png is not None and Path(texture_png).exists():
        cmd += ["--texture", str(texture_png)]
    logger.info("Rendering in Blender (%dx%d, %d samples)...", *img_size, samples)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender render failed (exit {result.returncode}).\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"Blender produced no frames in {frames_dir}\n{result.stdout[-1500:]}")
    logger.info("Blender rendered %d frames; encoding video...", len(frames))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    silent = work / "silent.mp4"
    subprocess.run([
        ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(silent),
    ], check=True)

    if audio_path is not None and Path(audio_path).exists():
        subprocess.run([
            ffmpeg, "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio_path),
            "-shortest", "-c:v", "copy", "-c:a", "aac", "-q:a", "4", str(output_mp4),
        ], check=True)
    else:
        shutil.copy(silent, output_mp4)

    shutil.rmtree(work, ignore_errors=True)
    logger.info("Saved Blender dance video to %s", output_mp4)
    return output_mp4


def _encode_video(frames_dir: Path, work: Path, output_mp4: Path,
                  audio_path: Path | None, fps: int, render_stdout: str = "") -> Path:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError(f"Blender produced no frames in {frames_dir}\n{render_stdout[-1500:]}")
    logger.info("Blender rendered %d frames; encoding video...", len(frames))
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH")
    silent = work / "silent.mp4"
    subprocess.run([
        ffmpeg, "-loglevel", "error", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(silent),
    ], check=True)
    if audio_path is not None and Path(audio_path).exists():
        subprocess.run([
            ffmpeg, "-loglevel", "error", "-y", "-i", str(silent), "-i", str(audio_path),
            "-shortest", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(output_mp4),
        ], check=True)
    else:
        shutil.copy(silent, output_mp4)
    logger.info("Saved Blender dance video to %s", output_mp4)
    return output_mp4


def render_ybot_dance(
    motion_npy: Path,
    output_mp4: Path,
    *,
    lodge_code_path: Path,
    blender_bin: Path,
    blender_script: Path,
    ybot_fbx: Path,
    audio_path: Path | None = None,
    color: str = "0.5,0.5,0.52",
    align_x: float = 0.0,
    img_size: tuple[int, int] = (720, 720),
    samples: int = 32,
    engine: str = "eevee",
    denoise: bool = True,
    fps: int = 30,
) -> Path:
    """Render a dance as EDGE's Mixamo Y-Bot robot by posing ybot.fbx's SMPL armature."""
    output_mp4 = Path(output_mp4).resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    motion = np.load(motion_npy)

    logger.info("Computing SMPL poses for %d frames...", motion.shape[0])
    poses, trans, fk_joints = compute_smpl_poses(motion, lodge_code_path)

    work = Path(tempfile.mkdtemp(prefix="blender_ybot_"))
    poses_npz = work / "poses.npz"
    frames_dir = work / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    np.savez(poses_npz, poses=poses, trans=trans, fk_joints=fk_joints)

    cmd = [
        str(blender_bin), "-b", "-noaudio", "-P", str(blender_script), "--",
        "--poses", str(poses_npz),
        "--ybot", str(ybot_fbx),
        "--frames-dir", str(frames_dir),
        "--width", str(img_size[0]), "--height", str(img_size[1]),
        "--samples", str(samples),
        "--engine", str(engine),
        "--denoise", "1" if denoise else "0",
        "--color", color,
        "--align-x", str(align_x),
    ]
    logger.info("Rendering Y-Bot in Blender (%dx%d, %d samples, %s)...",
                img_size[0], img_size[1], samples, engine)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Blender Y-Bot render failed (exit {result.returncode}).\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    out = _encode_video(frames_dir, work, output_mp4, audio_path, fps, result.stdout)
    shutil.rmtree(work, ignore_errors=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentlodge-root", required=True)
    parser.add_argument("--motion-npy", required=True)
    parser.add_argument("--output-mp4", required=True)
    parser.add_argument("--lodge-code-path", required=True)
    parser.add_argument("--smplx-model-dir", default="")
    parser.add_argument("--blender-bin", required=True)
    parser.add_argument("--blender-script", required=True)
    parser.add_argument("--character", choices=["smplx", "ybot"], default="smplx")
    parser.add_argument("--ybot-fbx", default="")
    parser.add_argument("--color", default="0.5,0.5,0.52")
    parser.add_argument("--align-x", type=float, default=0.0)
    parser.add_argument("--audio", default="")
    parser.add_argument("--uv-npz", default="")
    parser.add_argument("--texture", default="")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--engine", default="eevee", choices=["eevee", "cycles"])
    parser.add_argument("--denoise", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.path.insert(0, str(Path(args.agentlodge_root).resolve()))

    if args.character == "ybot":
        out = render_ybot_dance(
            Path(args.motion_npy).resolve(),
            Path(args.output_mp4).resolve(),
            lodge_code_path=Path(args.lodge_code_path).resolve(),
            blender_bin=Path(args.blender_bin).resolve(),
            blender_script=Path(args.blender_script).resolve(),
            ybot_fbx=Path(args.ybot_fbx).resolve(),
            audio_path=Path(args.audio).resolve() if args.audio else None,
            color=args.color,
            align_x=args.align_x,
            img_size=(args.width, args.height),
            samples=args.samples,
            engine=args.engine,
            denoise=bool(args.denoise),
        )
        print(f"Saved Blender dance video to {out} ({out.stat().st_size} bytes)")
        return 0

    out = render_blender_dance(
        Path(args.motion_npy).resolve(),
        Path(args.output_mp4).resolve(),
        lodge_code_path=Path(args.lodge_code_path).resolve(),
        smplx_model_dir=Path(args.smplx_model_dir).resolve(),
        blender_bin=Path(args.blender_bin).resolve(),
        blender_script=Path(args.blender_script).resolve(),
        audio_path=Path(args.audio).resolve() if args.audio else None,
        uv_npz=Path(args.uv_npz).resolve() if args.uv_npz else None,
        texture_png=Path(args.texture).resolve() if args.texture else None,
        img_size=(args.width, args.height),
        samples=args.samples,
    )
    print(f"Saved Blender dance video to {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
