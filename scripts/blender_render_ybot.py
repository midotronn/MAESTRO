"""Blender headless renderer for the EDGE Mixamo Y-Bot robot character.

EDGE's ``ybot.fbx`` is the segmented Y-Bot robot mesh skinned to an SMPL-named armature
(bones ``m_avg_Pelvis`` ... one per SMPL joint). EDGE drives it by writing SMPL axis-angle
rotations onto those bones via the Autodesk FBX SDK; we reproduce that entirely in Blender
so no FBX SDK is required: import the FBX and pose the ``m_avg_*`` bones from SMPL poses.

Run inside Blender::

    blender -b -noaudio -P blender_render_ybot.py -- \
        --poses poses.npz --ybot ybot.fbx --frames-dir out/ \
        --width 720 --height 720 --samples 32 [--color 0.5,0.5,0.5] [--align-x -90]

``poses.npz`` holds ``poses`` (L, J, 3) SMPL axis-angle (J>=22; hand joints may be zero)
and optional ``fk_joints`` (L, 22, 3). When FK is available, its root trajectory is preserved
and the camera follows it; ``--lock-root`` retains the legacy centred preview.
"""

import argparse
import glob
import math
import os
import sys

import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blender_studio as studio  # noqa: E402
from render_root_motion import centered_follow_locations, prepare_root_motion  # noqa: E402

# SMPL joint order (matches EDGE SMPL-to-FBX/SmplObject.py and ax_from_6v output).
JOINT_NAMES = [
    "m_avg_Pelvis", "m_avg_L_Hip", "m_avg_R_Hip", "m_avg_Spine1",
    "m_avg_L_Knee", "m_avg_R_Knee", "m_avg_Spine2", "m_avg_L_Ankle",
    "m_avg_R_Ankle", "m_avg_Spine3", "m_avg_L_Foot", "m_avg_R_Foot",
    "m_avg_Neck", "m_avg_L_Collar", "m_avg_R_Collar", "m_avg_Head",
    "m_avg_L_Shoulder", "m_avg_R_Shoulder", "m_avg_L_Elbow", "m_avg_R_Elbow",
    "m_avg_L_Wrist", "m_avg_R_Wrist", "m_avg_L_Hand", "m_avg_R_Hand",
]
FOOT_BONES = ["m_avg_L_Foot", "m_avg_R_Foot", "m_avg_L_Foot_end", "m_avg_R_Foot_end"]
TARGET_HEIGHT = 1.7  # metres; normalise the robot so the shared studio framing fits.


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--poses", default="")
    p.add_argument("--ybot", required=True)
    p.add_argument("--frames-dir", default="")
    p.add_argument("--width", type=int, default=720)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--engine", default="eevee", choices=["eevee", "cycles"],
                   help="eevee = fast high-quality raster; cycles = path-traced GPU (photoreal)")
    p.add_argument("--denoise", type=int, default=1, help="Cycles denoiser on (1) / off (0)")
    p.add_argument("--color", default="0.5,0.5,0.52",
                   help="Robot base colour r,g,b in 0-1")
    p.add_argument("--align-x", type=float, default=0.0,
                   help="Degrees about X aligning SMPL local rotations to the armature frame")
    p.add_argument("--yaw", type=float, default=0.0,
                   help="Extra degrees about Z to face the dancer toward the camera")
    p.add_argument("--stride", type=int, default=1,
                   help="Render every Nth frame (validation only; alignment still uses all)")
    p.add_argument("--build-scene", default="",
                   help="Import the rig + studio, save this .blend and exit (skips the per-render "
                        "FBX import when render opens the cached scene).")
    p.add_argument("--force-align", action="store_true",
                   help="Skip alignment auto-detect; use exactly --align-x about X (0 = identity)")
    p.add_argument("--fk-npz", default="",
                   help="Optional npz with ground-truth FK 'joints' (L,22,3) to fix the "
                        "global orientation via Kabsch; else read 'fk_joints' from --poses")
    p.add_argument("--fast", action="store_true",
                   help="Fast preview: ground on the cached foot meshes only + one fewer depsgraph "
                        "update per frame (used by the warm compare daemon).")
    p.add_argument("--lock-root", action="store_true",
                   help="Legacy preview mode: pin the pelvis horizontally and ground the feet "
                        "every frame instead of preserving the FK root trajectory.")
    p.add_argument("--fixed-camera", action="store_true",
                   help="Preserve root motion but keep the studio camera and key light fixed. "
                        "Use for visual audits where camera following would hide travel.")
    return p.parse_args(argv)


def foot_meshes(arm, meshes, poses, L, n_body, apply_pose):
    """The subset of robot meshes that reach the floor (the feet/soles), found once over a few
    sample frames. Per-frame grounding then scans ~8 meshes instead of ~100 with the same result."""
    sample = list(range(0, L, max(1, L // 6)))[:6] or [0]
    contrib: set[str] = set()
    for i in sample:
        apply_pose(i)
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        zmins = []
        for obj in meshes:
            ev = obj.evaluated_get(dg)
            me = ev.data
            n = len(me.vertices)
            if n == 0:
                zmins.append(float("inf"))
                continue
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get("co", co)
            co = co.reshape(-1, 3)
            M = np.array(ev.matrix_world)
            wz = co @ M[:3, :3].T[:, 2] + M[2, 3]
            zmins.append(float(wz.min()))
        gmin = min(zmins)
        for obj, zm in zip(meshes, zmins):
            if zm <= gmin + 0.06:               # within 6cm of the contact -> a foot-region mesh
                contrib.add(obj.name)
    subset = [o for o in meshes if o.name in contrib]
    return subset or meshes


def kabsch_rotation(P, Q):
    """Rotation R (3x3) best aligning centred P onto centred Q (scale-invariant)."""
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def axis_angle_to_matrix(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-8:
        return Matrix.Identity(3)
    axis = Vector((float(v[0]) / ang, float(v[1]) / ang, float(v[2]) / ang))
    return Quaternion(axis, ang).to_matrix()


def import_ybot(path):
    bpy.ops.import_scene.fbx(filepath=path)
    arms = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if not arms:
        raise RuntimeError("No armature found in ybot.fbx")
    return arms[0]


def style_robot(arm, color):
    """Grey metallic material on every robot segment; hide non-robot helper meshes.
    Returns the list of visible robot mesh objects (used for accurate grounding)."""
    mat = studio.make_material("Ybot", "", color, metallic=0.85, roughness=0.4)
    robot_meshes = []
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if obj.name.startswith("Alpha_"):
            obj.data.materials.clear()
            obj.data.materials.append(mat)
            for poly in obj.data.polygons:
                poly.use_smooth = True
            robot_meshes.append(obj)
        else:
            obj.hide_render = True
    return robot_meshes


def lowest_mesh_z(depsgraph, meshes):
    """World-space minimum z over the posed/skinned robot mesh vertices.

    Grounding on the foot *bone* leaves the foot *mesh* (the sole) poking through the
    floor, so we ground on the actual deformed surface instead. ``foreach_get`` + numpy
    keeps this cheap even across the ~100 Y-Bot segments."""
    mz = float("inf")
    for obj in meshes:
        ev = obj.evaluated_get(depsgraph)
        me = ev.data
        n = len(me.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        M = np.array(ev.matrix_world)
        world_z = co @ M[:3, :3].T[:, 2] + M[2, 3]
        zmin = float(world_z.min())
        if zmin < mz:
            mz = zmin
    return mz


def rest_rotations(arm):
    """Per-driven-bone armature-space rest rotation quaternion (from bone.matrix_local)."""
    rest = {}
    for name in JOINT_NAMES:
        bone = arm.data.bones.get(name)
        if bone is not None:
            rest[name] = bone.matrix_local.to_3x3().to_quaternion()
    return rest


def normalise_scale(arm):
    """Scale the whole rig so the rest robot is ~TARGET_HEIGHT tall."""
    bpy.context.view_layer.update()
    zs = []
    for bone in arm.data.bones:
        for pt in (bone.head_local, bone.tail_local):
            zs.append((arm.matrix_world @ Vector(pt)).z)
    height = max(zs) - min(zs)
    if height > 1e-6:
        s = TARGET_HEIGHT / height
        arm.scale = (s, s, s)
    bpy.context.view_layer.update()


def setup_studio():
    """Studio: dancer centred at origin, floor at z=0, static follow-camera + lights. Pose-independent,
    so it can be baked into the cached scene once."""
    body_size = TARGET_HEIGHT
    studio.setup_world_and_ground(np.array([-1.5, -1.5, 0.0]), np.array([1.5, 1.5, body_size]))
    spot = studio.setup_lights(0.0, 0.0, body_size)
    centroids = np.array([[0.0, 0.0, 0.55 * body_size]])       # static camera -> one centroid suffices
    cam, target, offset, target_z, follow_xy = studio.setup_follow_camera(centroids, body_size, 0.0)
    spot_front, spot_high = studio.attach_follow_spot(spot, target, 0.0, body_size, offset)
    target.location = (0.0, 0.0, target_z)
    cam.location = (offset[0], offset[1], target_z + offset[2])
    spot.location = (0.0, spot_front, spot_high)


def build_scene(args, color):
    """Import the Y-Bot + build the studio and SAVE it as a .blend, then exit. A later render can open
    this scene (rig + material + scale + lights + camera baked) and skip the ~10s FBX import."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = import_ybot(args.ybot)
    style_robot(arm, color)
    normalise_scale(arm)
    arm.rotation_mode = "QUATERNION"
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
    setup_studio()
    studio.configure_render(args.width, args.height, args.samples,
                            engine=args.engine, denoise=bool(args.denoise))
    bpy.ops.wm.save_as_mainfile(filepath=args.build_scene)
    print(f"YBOT_SCENE_BUILT {args.build_scene}")


def render_take(args, color):
    """Render poses.npz (``args.poses``) to ``args.frames_dir``. Reuses a preloaded scene (a cached
    .blend or a warm daemon that already imported the rig + studio) when present; otherwise it does
    the one-time FBX import + studio setup. Factored out of ``main`` so a persistent daemon can call
    it per request and skip Blender's ~8s startup + scene load."""
    data = np.load(args.poses)
    poses = data["poses"].astype(np.float32)  # (L, J, 3)
    L = poses.shape[0]
    contacts = None
    if "contacts" in data.files:
        contacts = np.asarray(data["contacts"], dtype=np.float32)
        if contacts.ndim != 2 or contacts.shape[0] != L:
            raise ValueError(
                f"contacts must have shape ({L}, channels), got {contacts.shape}"
            )
        if not np.isfinite(contacts).all():
            raise ValueError("contacts contain non-finite values")

    # Reuse the cached scene if this Blender was opened with one (rig + studio already loaded), else
    # do the full one-time setup (import FBX, style, scale, studio).
    preloaded = any(o.type == "ARMATURE" for o in bpy.data.objects)
    if preloaded:
        arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
        robot_meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Alpha_")]
        print("YBOT_SCENE cached scene reused (skipped FBX import)")
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        arm = import_ybot(args.ybot)
        robot_meshes = style_robot(arm, color)
        normalise_scale(arm)

    # Keep the FBX import's Y-up -> Z-up object rotation: it is what actually stands the
    # Y-up SMPL body upright in Blender's Z-up world. (Resetting it to identity tips the
    # whole dance onto the floor, which per-bone alignment cannot fully undo.)
    arm.rotation_mode = "QUATERNION"
    import_rot = arm.rotation_quaternion.copy()
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"

    rest = rest_rotations(arm)
    n_body = min(22, poses.shape[1])  # SMPL joints 0..21 (hands 22,23 stay at rest)

    def whead(name):
        return arm.matrix_world @ arm.pose.bones[name].head

    def wtail(name):
        return arm.matrix_world @ arm.pose.bones[name].tail

    def apply_pose(i):
        # DIRECT mapping (EDGE-style): each SMPL joint's local rotation is the bone's local
        # pose rotation. Verified against ground-truth FK joints (Procrustes RMSE ~0.03,
        # vs ~0.2+ and a tipped-over body for the old per-bone conjugation).
        for j in range(n_body):
            name = JOINT_NAMES[j]
            if name not in rest:
                continue
            arm.pose.bones[name].rotation_quaternion = axis_angle_to_matrix(
                poses[i, j]
            ).to_quaternion()

    def body_joint_positions():
        return np.array([[*whead(JOINT_NAMES[j])] for j in range(n_body)], dtype=np.float64)

    # Global orientation: the direct-posed body matches the ground-truth FK skeleton up to a
    # single constant rotation (the data-frame vs armature-frame difference). Recover it once
    # by Kabsch-aligning the posed bone joints to the FK joints over sampled frames, so the
    # dance stands upright and balanced every frame (no per-frame heuristic).
    fk = None
    if args.fk_npz and os.path.exists(args.fk_npz):
        fk = np.load(args.fk_npz)["joints"].astype(np.float64)
    elif "fk_joints" in data.files:
        fk = data["fk_joints"].astype(np.float64)
    if fk is not None:
        if fk.ndim != 3 or fk.shape[0] != L or fk.shape[1] < n_body or fk.shape[2] != 3:
            raise ValueError(
                f"FK joints must have shape ({L}, >={n_body}, 3), got {fk.shape}"
            )
        if not np.isfinite(fk).all():
            raise ValueError("FK joints contain non-finite values")
        if float(np.abs(fk).max()) < 1e-6:
            fk = None  # all-zero FK (smplx_neu_J absent) -> keep the legacy locked preview

    if fk is not None:
        sample = list(range(0, L, max(1, L // 12)))[:12]
        Ps, Qs = [], []
        for i in sample:
            apply_pose(i)
            bpy.context.view_layer.update()
            P = body_joint_positions()             # (n_body,3) world, import_rot applied
            Q = fk[i, :n_body].astype(np.float64)
            Ps.append(P - P.mean(0))               # per-frame centre (body translates)
            Qs.append(Q - Q.mean(0))
        Pc = np.concatenate(Ps)
        Qc = np.concatenate(Qs)
        R = kabsch_rotation(Pc, Qc)                # R @ Pc ~= Qc (aligns to FK frame)
        resid = float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum(1).mean()))
        world_rot = Matrix(R.tolist()).to_quaternion() @ import_rot
        print(f"YBOT_ALIGN kabsch-fk residual={resid:.4f}")
    else:
        world_rot = import_rot
        print("YBOT_ALIGN no FK joints; keeping import rotation")

    if abs(args.yaw) > 1e-6:  # optional facing tweak about the vertical axis
        world_rot = Quaternion(Vector((0.0, 0.0, 1.0)), math.radians(args.yaw)) @ world_rot
    arm.rotation_quaternion = world_rot
    bpy.context.view_layer.update()

    def pose_frame(i):
        apply_pose(i)

    # Studio (world/ground/lights/camera): baked in the cached scene; only build it when not preloaded.
    if not preloaded:
        setup_studio()
    # Render settings are cheap -> always (re)set so any resolution/samples work with a cached scene.
    studio.configure_render(args.width, args.height, args.samples,
                            engine=args.engine, denoise=bool(args.denoise))

    scene = bpy.context.scene
    frames_dir = args.frames_dir.rstrip("/")
    os.makedirs(frames_dir, exist_ok=True)
    # Clear frames from any previous render in this dir. The warm daemon reuses the same
    # _cmp_*_frames dirs and ffmpeg muxes frame_%05d.png contiguously, so leftover trailing frames
    # from an earlier, LONGER window would be appended to the end of this clip (a "glitch" tail).
    for _old in glob.glob(f"{frames_dir}/frame_*.png"):
        try:
            os.remove(_old)
        except OSError:
            pass
    stride = max(1, args.stride)
    fast = bool(getattr(args, "fast", False))
    # The per-frame bottleneck is the foot-grounding vertex scan (~100 meshes), not the render. In
    # fast mode, scan only the cached foot meshes and fold the horizontal centring + vertical
    # grounding into a single location set (2 depsgraph updates/frame instead of 3). The horizontal
    # shift does not change z, so the grounded result is identical to the full-scan path.
    ground_meshes = foot_meshes(arm, robot_meshes, poses, L, n_body, apply_pose) if fast else robot_meshes
    preserve_root = fk is not None and not bool(getattr(args, "lock_root", False))
    root_path = None
    follow_xy = None
    root_ground_offset = 0.0
    contact_ground_offsets = None
    cam = bpy.data.objects.get("Cam")
    target = bpy.data.objects.get("Target")
    spot = bpy.data.objects.get("Spot")
    cam_offset = spot_offset = None
    if cam is not None and target is not None:
        cam_home, target_home, spot_home = centered_follow_locations(
            np.array(cam.location),
            np.array(target.location),
            None if spot is None else np.array(spot.location),
        )
        target.location = tuple(target_home)
        cam.location = tuple(cam_home)
        if spot is not None and spot_home is not None:
            spot.location = tuple(spot_home)
        cam_offset = cam.location.copy() - target.location
        if spot is not None:
            spot_offset = spot.location.copy() - target.location

    if preserve_root:
        root_plan = prepare_root_motion(fk, yaw_degrees=float(args.yaw))
        root_path = root_plan.root_path
        follow_xy = root_plan.follow_xy
        grounded = (
            np.any(contacts > 0.5, axis=1)
            if contacts is not None
            else np.zeros(L, dtype=bool)
        )
        calibration_frames = (
            np.flatnonzero(grounded)
            if grounded.any()
            else root_plan.calibration_frames
        )
        measured_offsets = np.full(L, np.nan, dtype=np.float64)
        mesh_floor = float("inf")
        for frame in calibration_frames:
            frame = int(frame)
            apply_pose(frame)
            arm.location = (0.0, 0.0, 0.0)
            bpy.context.view_layer.update()
            pelvis = whead("m_avg_Pelvis")
            target_root = root_path[frame]
            arm.location = (
                float(target_root[0] - pelvis.x),
                float(target_root[1] - pelvis.y),
                float(target_root[2] - pelvis.z),
            )
            bpy.context.view_layer.update()
            frame_floor = lowest_mesh_z(
                bpy.context.evaluated_depsgraph_get(), ground_meshes
            )
            if frame_floor == float("inf"):
                continue
            if grounded.any():
                measured_offsets[frame] = -frame_floor
            else:
                mesh_floor = min(mesh_floor, frame_floor)
        valid_offsets = np.flatnonzero(np.isfinite(measured_offsets))
        if valid_offsets.size:
            contact_ground_offsets = np.interp(
                np.arange(L, dtype=np.float64),
                valid_offsets.astype(np.float64),
                measured_offsets[valid_offsets],
            )
            print(
                "YBOT_GROUND contact-aware "
                f"frames={valid_offsets.size} "
                f"offset=[{contact_ground_offsets.min():.4f},"
                f"{contact_ground_offsets.max():.4f}]"
            )
        elif mesh_floor != float("inf"):
            root_ground_offset = -mesh_floor

    # Rate-limit the floor offset so per-frame foot-height jitter (the lowest vertex hopping between
    # feet/segments) does not bob the WHOLE body up and down. Real height changes are gradual and
    # still track; only single-frame pops are clamped. Tunable via YBOT_GROUND_MAX_DZ (metres/frame).
    ground_max_dz = float(os.environ.get("YBOT_GROUND_MAX_DZ", "0.02"))
    prev_gz = None
    for i in range(0, L, stride):
        pose_frame(i)
        arm.location = (0.0, 0.0, 0.0)
        bpy.context.view_layer.update()
        pelvis = whead("m_avg_Pelvis")
        if preserve_root:
            target_root = root_path[i]
            ground_offset = (
                root_ground_offset
                if contact_ground_offsets is None
                else float(contact_ground_offsets[i])
            )
            arm.location = (
                float(target_root[0] - pelvis.x),
                float(target_root[1] - pelvis.y),
                float(target_root[2] - pelvis.z + ground_offset),
            )
            bpy.context.view_layer.update()
            if (
                not bool(getattr(args, "fixed_camera", False))
                and target is not None and cam is not None and cam_offset is not None
            ):
                tx, ty = map(float, follow_xy[i])
                target.location = (tx, ty, target.location.z)
                cam.location = (
                    tx + cam_offset.x,
                    ty + cam_offset.y,
                    target.location.z + cam_offset.z,
                )
                if spot is not None and spot_offset is not None:
                    spot.location = (
                        tx + spot_offset.x,
                        ty + spot_offset.y,
                        target.location.z + spot_offset.z,
                    )
        else:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            mz = lowest_mesh_z(depsgraph, ground_meshes)
            if fast:
                raw_gz = 0.0 if mz == float("inf") else -mz
                gz = raw_gz if prev_gz is None else min(
                    prev_gz + ground_max_dz,
                    max(prev_gz - ground_max_dz, raw_gz),
                )
                prev_gz = gz
                arm.location = (-pelvis.x, -pelvis.y, gz)
                bpy.context.view_layer.update()
            else:
                arm.location = (-pelvis.x, -pelvis.y, 0.0)
                bpy.context.view_layer.update()
                depsgraph = bpy.context.evaluated_depsgraph_get()
                mz = lowest_mesh_z(depsgraph, robot_meshes)
                raw_gz = 0.0 if mz == float("inf") else -mz
                gz = raw_gz if prev_gz is None else min(
                    prev_gz + ground_max_dz,
                    max(prev_gz - ground_max_dz, raw_gz),
                )
                prev_gz = gz
                arm.location = (arm.location.x, arm.location.y, gz)
                bpy.context.view_layer.update()
        scene.render.filepath = f"{frames_dir}/frame_{i:05d}.png"
        bpy.ops.render.render(write_still=True)
    print(f"BLENDER_RENDERED {L} frames -> {frames_dir}")


def main():
    args = parse_args()
    try:
        color = tuple(float(c) for c in args.color.split(","))[:3]
    except Exception:
        color = (0.5, 0.5, 0.52)

    if args.build_scene:
        build_scene(args, color)
        return
    if not args.poses or not args.frames_dir:
        raise SystemExit("--poses and --frames-dir are required to render")
    render_take(args, color)


if __name__ == "__main__":
    main()
