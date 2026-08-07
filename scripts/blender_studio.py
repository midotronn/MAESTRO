"""Shared Blender studio helpers for AgentLODGE dance renders.

Both ``blender_render_smplx.py`` (smooth SMPL-X body) and ``blender_render_ybot.py``
(EDGE Mixamo Y-Bot robot) run headless inside Blender and import these helpers to build
an identical EDGE-style studio: a seamless cyclorama backdrop, a following key spotlight
with vignette falloff, a metallic body material, a gliding follow camera, and EEVEE-Next
render settings.

Blender runs each ``-P`` script with its directory on ``sys.path`` only if the script
inserts it, so callers do ``sys.path.insert(0, dirname(__file__))`` before importing this.
"""

import math

import bpy  # type: ignore
import numpy as np

from render_root_motion import smooth_path


def make_material(name, texture_path, color, metallic=0.9, roughness=0.35):
    """Metallic Principled-BSDF body material (EDGE-style). ``color`` is an (r,g,b)."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if texture_path:
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(texture_path)
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def setup_world_and_ground(bmin, bmax):
    """Studio cyclorama: a seamless curved backdrop (floor blends up into a back wall) lit
    to fade from a bright pool into darker grey, with a dim world so the spotlight reads.
    Returns (cx, cy, floor_z)."""
    world = bpy.data.worlds.new("W")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.045, 0.045, 0.05, 1.0)
        bg.inputs[1].default_value = 1.0
    bpy.context.scene.world = world

    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    floor_z = float(bmin[2])

    width, front, back, height, radius = 60.0, 22.0, 14.0, 22.0, 6.0
    # Profile in (y, z), front -> back along the floor, then a quarter-circle up into a wall.
    prof = [(front, 0.0), (-back + radius, 0.0)]
    ccy = -back + radius
    for a in range(1, 13):
        ang = (math.pi / 2) * (a / 12)
        prof.append((ccy - radius * math.sin(ang), radius * (1 - math.cos(ang))))
    prof.append((-back, height))

    x0, x1 = cx - width / 2, cx + width / 2
    verts, faces = [], []
    n = len(prof)
    for (py, pz) in prof:
        verts.append((x0, cy + py, floor_z + pz))
        verts.append((x1, cy + py, floor_z + pz))
    for i in range(n - 1):
        a0, a1 = 2 * i, 2 * i + 1
        b0, b1 = 2 * (i + 1), 2 * (i + 1) + 1
        faces.append((a0, b0, b1, a1))

    mesh = bpy.data.meshes.new("cyclo")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True
    cyc = bpy.data.objects.new("Cyclorama", mesh)
    bpy.context.collection.objects.link(cyc)
    gmat = bpy.data.materials.new("Studio")
    gmat.use_nodes = True
    gb = gmat.node_tree.nodes.get("Principled BSDF")
    if gb:
        gb.inputs["Base Color"].default_value = (0.55, 0.55, 0.57, 1.0)
        gb.inputs["Roughness"].default_value = 0.85
    cyc.data.materials.append(gmat)
    return cx, cy, floor_z


def setup_lights(cx, cy, top_z):
    """A key spot from high-front makes the bright floor pool + vignette; a soft area fill
    keeps the character readable. Returns the spot so it can follow the dancer."""
    spot = bpy.data.objects.new("Spot", bpy.data.lights.new("Spot", "SPOT"))
    spot.data.energy = 6000.0
    spot.data.spot_size = math.radians(60.0)
    spot.data.spot_blend = 0.5
    spot.data.shadow_soft_size = 1.2
    bpy.context.collection.objects.link(spot)

    fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "AREA"))
    fill.data.energy = 120.0
    fill.data.size = 12.0
    fill.location = (cx - 4.0, cy - 6.0, top_z + 2.0)
    bpy.context.collection.objects.link(fill)
    return spot


def setup_follow_camera(centroids, body_size, floor_z):
    """Camera that follows the dancer, keeping them large and centered.

    Framing is sized to the body (not the whole trajectory), and the look-at target
    tracks the smoothed horizontal centroid at a fixed mid-body height, so the dancer
    stays big and steady as it travels across the floor. Returns
    (cam, target, offset, target_z, follow_xy).
    """
    dist = max(body_size, 1.0) * 2.6
    target_z = floor_z + 0.55 * body_size
    offset = np.array([dist * 0.35, -dist, body_size * 0.30])

    cam_data = bpy.data.cameras.new("Cam")
    cam_data.lens = 50.0
    cam = bpy.data.objects.new("Cam", cam_data)
    bpy.context.collection.objects.link(cam)
    target = bpy.data.objects.new("Target", None)
    bpy.context.collection.objects.link(target)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    follow_xy = smooth_path(np.asarray(centroids)[:, :2])
    return cam, target, offset, target_z, follow_xy


def attach_follow_spot(spot, target, floor_z, body_size, offset):
    """Point the key spot at the follow target and return (spot_front, spot_high) so the
    caller can ride it high-front of the dancer per frame (bright pool follows the body)."""
    tc = spot.constraints.new("TRACK_TO")
    tc.target = target
    tc.track_axis = "TRACK_NEGATIVE_Z"
    tc.up_axis = "UP_Y"
    front_sign = -1.0 if offset[1] < 0 else 1.0
    return front_sign * body_size * 2.2, float(floor_z) + body_size * 4.0


def _enable_cycles_gpu(scene):
    """Enable Cycles GPU (OptiX preferred, then CUDA/HIP/ONEAPI). Returns True if a GPU is on."""
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        scene.cycles.device = "CPU"
        return False
    for dtype in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = dtype
            prefs.get_devices()
        except Exception:
            continue
        gpus = [d for d in prefs.devices if d.type == dtype]
        if gpus:
            for d in prefs.devices:
                d.use = (d.type == dtype)  # GPU only (skip CPU so it isn't the bottleneck)
            scene.cycles.device = "GPU"
            print(f"CYCLES_DEVICE {dtype} x{len(gpus)}")
            return True
    scene.cycles.device = "CPU"
    print("CYCLES_DEVICE CPU (no GPU found)")
    return False


def configure_render(width, height, samples, engine="eevee", denoise=True):
    """Configure the render engine. ``engine`` is 'eevee' (fast, high-quality raster) or 'cycles'
    (path-traced GPU, photoreal). Adds AgX tonemapping + full-res output for a crisp result."""
    scene = bpy.context.scene
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    # nicer, higher-contrast tone mapping (Blender 4.x default is AgX; fall back to Filmic).
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        try:
            scene.view_settings.view_transform = "Filmic"
        except Exception:
            pass

    if str(engine).lower() == "cycles":
        scene.render.engine = "CYCLES"
        _enable_cycles_gpu(scene)
        cy = scene.cycles
        cy.samples = int(samples)
        try:
            cy.use_adaptive_sampling = True
            cy.adaptive_threshold = 0.01
            cy.adaptive_min_samples = max(16, int(samples) // 8)
        except Exception:
            pass
        cy.use_denoising = bool(denoise)
        for dn in ("OPTIX", "OPENIMAGEDENOISE"):
            try:
                cy.denoiser = dn
                break
            except Exception:
                continue
        try:
            cy.max_bounces = 8
            cy.diffuse_bounces = 4
            cy.glossy_bounces = 6
            cy.transmission_bounces = 8
            scene.render.use_persistent_data = True  # reuse BVH across frames -> much faster
        except Exception:
            pass
    else:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
        ee = scene.eevee
        try:
            ee.taa_render_samples = int(samples)
            ee.use_shadows = True
            ee.use_raytracing = True          # screen-space GI + reflections (metallic robot)
            try:
                ee.ray_tracing_options.resolution_scale = "1:1"
            except Exception:
                pass
            for attr, val in (("use_gtao", True), ("use_bloom", False),
                              ("shadow_ray_count", 2), ("shadow_step_count", 6)):
                try:
                    setattr(ee, attr, val)
                except Exception:
                    pass
        except Exception:
            pass
