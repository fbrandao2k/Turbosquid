"""Scene preparation: import the model, normalize it, set up render, lights and ground."""

import fnmatch
import math
import os
from dataclasses import dataclass, field

import bpy
import numpy as np
from mathutils import Matrix, Vector

RENDERABLE_TYPES = {"MESH", "CURVE", "SURFACE", "META", "FONT", "CURVES", "POINTCLOUD"}


@dataclass
class Model:
    """Everything the shot renderers need to know about the imported model."""

    pivot: bpy.types.Object  # empty that normalizes the model; rotate it for the turntable
    objects: list  # visible renderable objects (original materials)
    points: np.ndarray  # (N, 3) vertex positions in normalized world space
    median_edge: float  # median edge length in normalized world space
    scale: float  # normalization scale factor (original units -> normalized)
    lights: dict = field(default_factory=dict)
    camera: bpy.types.Object = None
    ground: bpy.types.Object = None


# ---------------------------------------------------------------- import ----

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _patch_fbx_light_import():
    """Blender 5.x's FBX importer still sets light.cycles.cast_shadow, which Cycles
    removed, so any FBX containing a light fails to import. Give it a dummy property."""
    probe = bpy.data.lights.new("ts_probe", "POINT")
    try:
        settings = type(getattr(probe, "cycles", None))  # not exposed in bpy.types
        if hasattr(settings, "bl_rna") and "cast_shadow" not in settings.bl_rna.properties:
            settings.cast_shadow = bpy.props.BoolProperty()
    finally:
        bpy.data.lights.remove(probe)


def import_model(path):
    """Import any supported model file and return the newly created objects."""
    ext = os.path.splitext(path)[1].lower()
    before = set(bpy.data.objects)
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        _patch_fbx_light_import()
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    elif ext in (".usd", ".usda", ".usdc", ".usdz"):
        bpy.ops.wm.usd_import(filepath=path)
    elif ext == ".blend":
        with bpy.data.libraries.load(path, link=False) as (src, dst):
            dst.objects = list(src.objects)
        for obj in dst.objects:
            if obj is not None:
                bpy.context.scene.collection.objects.link(obj)
    else:
        raise ValueError(f"Unsupported model format: {ext}")
    new_objects = [o for o in bpy.data.objects if o not in before]
    if not new_objects:
        raise RuntimeError(f"Nothing was imported from {path}")
    return new_objects


def _is_excluded(obj, patterns):
    return any(fnmatch.fnmatchcase(obj.name, p) for p in patterns)


def _world_vertices_and_edges(objects):
    """Return (points, edge_lengths) of the evaluated objects in world space."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_points, all_edges = [], []
    for obj in objects:
        obj_eval = obj.evaluated_get(depsgraph)
        try:
            mesh = obj_eval.to_mesh()
        except RuntimeError:
            continue
        if mesh is None or len(mesh.vertices) == 0:
            obj_eval.to_mesh_clear()
            continue
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        mw = np.array(obj_eval.matrix_world)
        co = co @ mw[:3, :3].T + mw[:3, 3]
        ev = np.empty(len(mesh.edges) * 2, dtype=np.int64)
        mesh.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        if len(mesh.loops):
            # Ignore loose vertices/edges (scan debris) that render as nothing.
            loop_verts = np.empty(len(mesh.loops), dtype=np.int64)
            loop_edges = np.empty(len(mesh.loops), dtype=np.int64)
            mesh.loops.foreach_get("vertex_index", loop_verts)
            mesh.loops.foreach_get("edge_index", loop_edges)
            loose = len(mesh.vertices) - len(np.unique(loop_verts))
            if loose:
                print(f"[tspreview] note: {obj.name} has {loose} loose vertices (ignored for framing)")
            ev = ev[np.unique(loop_edges)]
            co = co[np.unique(loop_verts)]
        all_points.append(co)
        if len(ev):
            if len(ev) > 200_000:
                ev = ev[np.random.default_rng(0).choice(len(ev), 200_000, replace=False)]
            verts = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
            mesh.vertices.foreach_get("co", verts)
            verts = verts.reshape(-1, 3) @ mw[:3, :3].T
            all_edges.append(np.linalg.norm(verts[ev[:, 0]] - verts[ev[:, 1]], axis=1))
        obj_eval.to_mesh_clear()
    if not all_points:
        raise RuntimeError("The model has no renderable geometry")
    edges = np.concatenate(all_edges) if all_edges else np.array([0.0])
    return np.concatenate(all_points), edges


def prepare_model(path, exclude_patterns):
    """Import the model, hide excluded objects and normalize it.

    The model is parented to a pivot empty so that it sits on Z=0, is centered
    on the Z axis and has a bounding radius of 1. Rotating the pivot around Z
    spins the model for the turntable.
    """
    imported = import_model(path)
    for obj in imported:
        if obj.type == "LIGHT":  # lights saved in the file would fight the preview rig
            obj.hide_render = True
            print(f"[tspreview] ignoring light from file: {obj.name}")
        elif _is_excluded(obj, exclude_patterns):
            obj.hide_render = True
            obj.hide_viewport = True
            print(f"[tspreview] excluded object: {obj.name}")
    visible = [o for o in imported if o.type in RENDERABLE_TYPES and not o.hide_render]
    if not visible:
        raise RuntimeError("No visible renderable objects after exclusions")
    for obj in visible:
        print(f"[tspreview] object: {obj.name} ({obj.type})")

    bpy.context.view_layer.update()
    points, edge_lengths = _world_vertices_and_edges(visible)
    lo, hi = points.min(axis=0), points.max(axis=0)
    center = (lo + hi) / 2
    radius = float(np.linalg.norm(points - center, axis=1).max()) or 1.0
    scale = 1.0 / radius
    offset = np.array([center[0], center[1], lo[2]])

    pivot = bpy.data.objects.new("TS_Pivot", None)
    bpy.context.scene.collection.objects.link(pivot)
    pivot.scale = (scale, scale, scale)
    to_origin = Matrix.Translation(Vector(-offset))
    for obj in imported:
        if obj.parent is None:
            basis = obj.matrix_world.copy()
            obj.parent = pivot
            obj.matrix_parent_inverse = to_origin
            obj.matrix_basis = basis
    bpy.context.view_layer.update()

    points = (points - offset) * scale
    median_edge = float(np.median(edge_lengths)) * scale
    return Model(pivot=pivot, objects=visible, points=points, median_edge=median_edge, scale=scale)


# ---------------------------------------------------------------- render ----

def setup_device(preferred):
    """Enable GPU rendering in Cycles, falling back to CPU."""
    scene = bpy.context.scene
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    order = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"] if preferred == "AUTO" else [preferred]
    for backend in order:
        if backend == "CPU":
            break
        try:
            cprefs.compute_device_type = backend
        except TypeError:
            continue
        cprefs.refresh_devices()
        gpus = [d for d in cprefs.devices if d.type == backend]
        if not gpus:
            continue
        for dev in cprefs.devices:
            dev.use = dev.type == backend
        scene.cycles.device = "GPU"
        print(f"[tspreview] rendering on {backend}: {', '.join(d.name for d in gpus)}")
        return backend
    scene.cycles.device = "CPU"
    print("[tspreview] rendering on CPU")
    return "CPU"


def setup_render(cfg):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    setup_device(cfg.get("device", "AUTO"))
    cycles = scene.cycles
    cycles.samples = int(cfg.get("samples", 128))
    cycles.use_adaptive_sampling = True
    cycles.adaptive_threshold = 0.01
    cycles.use_denoising = True
    cycles.denoiser = "OPENIMAGEDENOISE"
    if hasattr(cycles, "denoising_use_gpu"):
        cycles.denoising_use_gpu = True

    render = scene.render
    render.film_transparent = True
    render.use_persistent_data = True
    render.dither_intensity = 0.0
    render.resolution_percentage = 100
    settings = render.image_settings
    settings.file_format = "PNG"
    settings.color_mode = "RGBA"
    settings.color_depth = "8"
    settings.compression = 15

    scene.display_settings.display_device = "sRGB"
    view = scene.view_settings
    view.view_transform = cfg.get("view_transform", "AgX")
    try:
        view.look = cfg.get("look", "None")
    except TypeError:
        print(f"[tspreview] look '{cfg.get('look')}' not available, using None")
        view.look = "None"
    view.exposure = float(cfg.get("exposure", 0.0))
    view.gamma = 1.0


def setup_world(light_cfg):
    """Uniform ambient light for diffuse/shadows, studio HDRI only in reflections.

    A directional HDRI would otherwise cast a second hard shadow on the ground.
    """
    scene = bpy.context.scene
    world = bpy.data.worlds.new("TS_World")
    scene.world = world
    if not world.use_nodes:
        world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    ambient = nodes.new("ShaderNodeBackground")
    ambient.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    ambient.inputs["Strength"].default_value = float(light_cfg.get("ambient_strength", 0.0))
    reflection = nodes.new("ShaderNodeBackground")
    reflection.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1.0)
    reflection.inputs["Strength"].default_value = float(light_cfg.get("hdri_strength", 0.8))

    hdri_name = light_cfg.get("hdri")
    hdri_dir = bpy.utils.system_resource("DATAFILES", path="studiolights/world")
    hdri_path = hdri_name if hdri_name and os.path.isabs(hdri_name) else (
        os.path.join(hdri_dir, hdri_name) if hdri_name and hdri_dir else None)
    if hdri_path and os.path.isfile(hdri_path):
        env = nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri_path, check_existing=True)
        links.new(env.outputs["Color"], reflection.inputs["Color"])

    light_path = nodes.new("ShaderNodeLightPath")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(light_path.outputs["Is Glossy Ray"], mix.inputs["Fac"])
    links.new(ambient.outputs["Background"], mix.inputs[1])
    links.new(reflection.outputs["Background"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])


DEFAULT_LIGHTS = {
    "key": {"strength": 3.0, "azimuth": -40, "elevation": 50, "softness_deg": 20, "shadow": True},
    "fill": {"strength": 1.2, "azimuth": 55, "elevation": 20, "softness_deg": 45, "shadow": False},
    "rim": {"strength": 2.5, "azimuth": 160, "elevation": 35, "softness_deg": 20, "shadow": False},
}


def setup_lights(model, light_cfg):
    """Sun lights; only shadow-casting ones show up on the ground shadow catcher."""
    model.lights = {}
    for name, spec in light_cfg.get("lights", DEFAULT_LIGHTS).items():
        data = bpy.data.lights.new(f"TS_{name}", "SUN")
        data.energy = float(spec.get("strength", 1.0))
        data.angle = math.radians(float(spec.get("softness_deg", 20)))
        data.use_shadow = bool(spec.get("shadow", False))
        obj = bpy.data.objects.new(f"TS_{name}", data)
        obj.rotation_mode = "QUATERNION"
        bpy.context.scene.collection.objects.link(obj)
        model.lights[name] = (obj, spec)


def aim_lights(model, cam_rotation):
    """Place the light rig relative to the camera's azimuth so every shot is lit alike.

    Each light's azimuth is an offset from the camera (negative = camera left),
    its elevation is absolute, so shadows always fall behind the model.
    """
    to_camera = cam_rotation.to_matrix().col[2]
    cam_azimuth = math.degrees(math.atan2(to_camera.x, -to_camera.y))
    for obj, spec in model.lights.values():
        az = math.radians(cam_azimuth + float(spec.get("azimuth", 0)))
        el = math.radians(float(spec.get("elevation", 45)))
        from_dir = Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))
        obj.rotation_quaternion = (-from_dir).to_track_quat("-Z", "Y")


def setup_camera(model, lens_mm):
    data = bpy.data.cameras.new("TS_Camera")
    data.lens = lens_mm
    data.sensor_width = 36.0
    data.sensor_fit = "AUTO"
    cam = bpy.data.objects.new("TS_Camera", data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.rotation_mode = "QUATERNION"
    model.camera = cam


def setup_ground(model, enabled):
    """Invisible shadow catcher at the model's base (Z=0)."""
    if not enabled:
        return
    bpy.ops.mesh.primitive_plane_add(size=200.0, location=(0.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "TS_ShadowCatcher"
    ground.is_shadow_catcher = True
    # Pure diffuse: a glossy catcher would also "catch" noisy reflection occlusion.
    mat = bpy.data.materials.new("TS_Ground")
    if not mat.use_nodes:
        mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    output = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    ground.data.materials.append(mat)
    model.ground = ground


def make_material(name, grey, roughness):
    mat = bpy.data.materials.new(name)
    if not mat.use_nodes:
        mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (grey, grey, grey, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    return mat
