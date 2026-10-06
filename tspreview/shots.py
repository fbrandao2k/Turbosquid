"""Renderers for each TurboSquid preview image type."""

import math
import os
import time

import bpy
import numpy as np

from . import framing, imaging
from . import scene as ts_scene


def _image_format(cfg):
    fmt = str(cfg.get("image_format", "PNG")).upper()
    return "JPEG" if fmt in ("JPG", "JPEG") else "PNG"


class Renderer:
    def __init__(self, model, cfg, out_dir):
        self.model = model
        self.cfg = cfg
        self.out_dir = out_dir
        self.tmp_dir = os.path.join(out_dir, "_tmp")
        self.written = []

    # ------------------------------------------------------------ helpers --

    def _place_camera(self, points, rotation, resolution, fill, lens=None):
        lens = float(lens or self.cfg.get("lens_mm", 50))
        cam = self.model.camera
        cam.data.lens = lens
        location, distance = framing.fit_camera(points, rotation, lens, resolution[0], resolution[1], fill)
        cam.location = location
        cam.rotation_quaternion = rotation
        cam.data.clip_start = max(distance * 0.002, 1e-4)
        cam.data.clip_end = distance + 100.0
        ts_scene.aim_lights(self.model, rotation)
        return distance

    def _render(self, stem, resolution, samples, greyscale=False, subdir="", file_format=None):
        scene = bpy.context.scene
        scene.render.resolution_x, scene.render.resolution_y = int(resolution[0]), int(resolution[1])
        scene.cycles.samples = int(samples)
        tmp_path = os.path.join(self.tmp_dir, stem + ".png")
        scene.render.filepath = tmp_path
        start = time.time()
        bpy.ops.render.render(write_still=True)

        rgba = imaging.load_rgba(tmp_path)
        if self.model.ground is not None:
            rgba = imaging.clean_shadow_alpha(rgba, float(self.cfg.get("shadow_noise_floor", 0.02)))
        rgb = imaging.composite_on_color(rgba, self.cfg["background_rgb"])
        if greyscale:
            rgb = imaging.to_greyscale(rgb)
        fmt = file_format or _image_format(self.cfg)
        out_path = imaging.output_path(os.path.join(self.out_dir, subdir), stem, fmt)
        imaging.save_rgb(rgb, out_path, fmt, self.cfg.get("jpeg_quality", 95))
        if self.cfg.get("keep_transparent"):
            raw_dir = os.path.join(self.out_dir, "transparent", subdir)
            os.makedirs(raw_dir, exist_ok=True)
            os.replace(tmp_path, os.path.join(raw_dir, stem + ".png"))
        else:
            os.remove(tmp_path)
        print(f"[tspreview] wrote {out_path}  ({time.time() - start:.1f}s)")
        self.written.append(out_path)

    def _shot_rotation_and_points(self, shot, defaults):
        base_dir = framing.view_direction(shot.get("azimuth", 0), shot.get("elevation", 20))
        points = self.model.points
        if shot.get("detail"):
            fraction = float(shot.get("detail_fraction", defaults.get("detail_fraction", 0.3)))
            points, base_dir = framing.detail_view(points, base_dir, shot["detail"], fraction)
        return framing.camera_rotation(base_dir), points

    def cleanup(self):
        if os.path.isdir(self.tmp_dir) and not os.listdir(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    # ------------------------------------------------------------- shots ---

    def search(self):
        """1200x1200 hero render on the exact RGB 247 background (always PNG)."""
        scfg = self.cfg["search"]
        res = scfg["resolution"]
        rotation = framing.camera_rotation(framing.view_direction(scfg["azimuth"], scfg["elevation"]))
        self._place_camera(self.model.points, rotation, res, scfg.get("fill", 0.92))
        self._render(f"search_{res[0]}x{res[1]}", res, self.cfg["samples"], file_format="PNG")

    def product(self):
        pcfg = self.cfg["product"]
        res = pcfg["resolution"]
        for i, shot in enumerate(pcfg["shots"], start=1):
            rotation, points = self._shot_rotation_and_points(shot, pcfg)
            self._place_camera(points, rotation, res, shot.get("fill", pcfg.get("fill", 0.85)), shot.get("lens"))
            self._render(f"product_{i:02d}_{shot.get('name', 'shot')}", res, self.cfg["samples"])

    def wireframe(self):
        """Greyscale clay render with the mesh edges drawn as geometry."""
        wcfg = self.cfg["wireframe"]
        res = wcfg["resolution"]
        surface = ts_scene.make_material("TS_Clay", float(wcfg.get("surface_grey", 0.72)), 0.6)
        wire = ts_scene.make_material("TS_Wire", float(wcfg.get("wire_grey", 0.03)), 0.9)

        duplicates = []
        for obj in self.model.objects:
            if obj.type != "MESH":
                continue
            dup = obj.copy()
            dup.data = obj.data.copy()
            bpy.context.scene.collection.objects.link(dup)
            for slot in dup.material_slots:
                slot.link = "DATA"
            dup.data.materials.clear()
            dup.data.materials.append(surface)
            dup.data.materials.append(wire)
            dup.data.polygons.foreach_set("material_index", np.zeros(len(dup.data.polygons), dtype=np.int32))
            mod = dup.modifiers.new("TS_Wireframe", "WIREFRAME")
            mod.use_replace = False
            mod.use_even_offset = True
            mod.use_relative_offset = False
            mod.use_boundary = True
            mod.offset = 0.0
            mod.material_offset = 1
            duplicates.append(dup)
            obj.hide_render = True
        bpy.context.view_layer.update()

        try:
            for i, shot in enumerate(wcfg["shots"], start=1):
                rotation, points = self._shot_rotation_and_points(shot, wcfg)
                lens = float(shot.get("lens", self.cfg.get("lens_mm", 50)))
                distance = self._place_camera(points, rotation, res, shot.get("fill", wcfg.get("fill", 0.85)), lens)
                tan_x, _ = framing.half_fov_tangents(lens, 36.0, res[0], res[1])
                pixel_size = 2.0 * distance * tan_x / res[0]
                thickness = float(wcfg.get("wire_width_px", 1.2)) * pixel_size
                thickness = min(thickness, float(wcfg.get("max_wire_fraction_of_edge", 0.2)) * self.model.median_edge)
                for dup in duplicates:
                    world_scale = abs(dup.matrix_world.to_3x3().determinant()) ** (1.0 / 3.0) or 1.0
                    dup.modifiers["TS_Wireframe"].thickness = thickness / world_scale
                self._render(f"wireframe_{i:02d}_{shot.get('name', 'shot')}", res, self.cfg["samples"], greyscale=True)
        finally:
            for dup in duplicates:
                mesh = dup.data
                bpy.data.objects.remove(dup)
                bpy.data.meshes.remove(mesh)
            for obj in self.model.objects:
                obj.hide_render = False

    def turntable(self):
        """12-36 frames, model spinning counter-clockwise (seen from above)."""
        tcfg = self.cfg["turntable"]
        res = tcfg["resolution"]
        frames = int(tcfg.get("frames", 24))
        if not 12 <= frames <= 36:
            raise ValueError(f"TurboSquid turntables need 12-36 frames, got {frames}")
        rotation = framing.camera_rotation(framing.view_direction(tcfg.get("azimuth", 0), tcfg.get("elevation", 20)))
        angles = [2.0 * math.pi * k / frames for k in range(frames)]
        points = _turntable_sample(self.model.points, frames)
        swept = np.concatenate([framing.rotate_z(points, a) for a in angles])
        self._place_camera(swept, rotation, res, tcfg.get("fill", 0.85))

        pivot = self.model.pivot
        try:
            for k, angle in enumerate(angles, start=1):
                # Positive rotation about +Z is counter-clockwise viewed from above.
                pivot.rotation_euler = (0.0, 0.0, angle)
                self._render(f"turntable_{k:02d}", res, tcfg.get("samples", self.cfg["samples"]), subdir="turntable")
        finally:
            pivot.rotation_euler = (0.0, 0.0, 0.0)


def _turntable_sample(points, frames, budget=4_000_000):
    """Keep the points that matter for framing a spinning model within a budget."""
    if len(points) * frames <= budget:
        return points
    rng = np.random.default_rng(0)
    keep = [points[rng.choice(len(points), 20_000, replace=False)]]
    # Outermost points (largest distance from the spin axis) per height band.
    radius = np.hypot(points[:, 0], points[:, 1])
    bands = np.clip((points[:, 2] / max(points[:, 2].max(), 1e-9) * 128).astype(int), 0, 127)
    order = np.lexsort((-radius, bands))
    sorted_bands = bands[order]
    starts = np.searchsorted(sorted_bands, np.arange(128))
    for b, s in enumerate(starts):
        e = starts[b + 1] if b + 1 < 128 else len(order)
        keep.append(points[order[s:min(e, s + 64)]])
    return np.concatenate(keep)
