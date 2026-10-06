"""800x800 UV layout image: the diffuse texture overlaid by the 0-1 UV wireframe."""

import os
import re
from collections import defaultdict

import bpy
import numpy as np

from . import imaging

SUPERSAMPLE = 3


def _upstream_image_node(socket, depth=0, seen=None):
    """First Image Texture node feeding into `socket` (depth-first, input order)."""
    seen = seen if seen is not None else set()
    if depth > 12:
        return None
    for link in socket.links:
        node = link.from_node
        if node.name in seen:
            continue
        seen.add(node.name)
        if node.type == "TEX_IMAGE" and node.image is not None:
            return node
        for inp in node.inputs:
            found = _upstream_image_node(inp, depth + 1, seen)
            if found:
                return found
    return None


def find_diffuse(material):
    """Return (image, uv_map_name_or_None) of the material's base color texture."""
    tree = material.node_tree if material else None
    if tree is None:
        return None, None
    node = None
    for shader in tree.nodes:
        if shader.type == "BSDF_PRINCIPLED":
            node = _upstream_image_node(shader.inputs["Base Color"])
        elif shader.type in ("BSDF_DIFFUSE", "EMISSION"):
            node = _upstream_image_node(shader.inputs["Color"])
        if node:
            break
    if node is None:  # fall back to any color (non-data) texture
        node = next((n for n in tree.nodes if n.type == "TEX_IMAGE" and n.image is not None
                     and not n.image.colorspace_settings.is_data), None)
    if node is None:
        return None, None
    uv_name = None
    vec = node.inputs.get("Vector")
    if vec is not None and vec.links and vec.links[0].from_node.type == "UVMAP":
        uv_name = vec.links[0].from_node.uv_map or None
    return node.image, uv_name


def _uv_layer(mesh, name):
    if name and name in mesh.uv_layers:
        return mesh.uv_layers[name]
    for layer in mesh.uv_layers:
        if layer.active_render:
            return layer
    return mesh.uv_layers.active


def _uv_edges(mesh, uv_name, material_index=None):
    """(E, 2, 2) array of UV-space edges of the faces using `material_index`."""
    layer = _uv_layer(mesh, uv_name)
    if layer is None or len(mesh.polygons) == 0:
        return np.zeros((0, 2, 2))
    n_loops = len(mesh.loops)
    uv = np.empty(n_loops * 2, dtype=np.float64)
    layer.uv.foreach_get("vector", uv)
    uv = uv.reshape(-1, 2)

    n_polys = len(mesh.polygons)
    starts = np.empty(n_polys, dtype=np.int64)
    totals = np.empty(n_polys, dtype=np.int64)
    mesh.polygons.foreach_get("loop_start", starts)
    mesh.polygons.foreach_get("loop_total", totals)
    if material_index is not None:
        mat_idx = np.empty(n_polys, dtype=np.int64)
        mesh.polygons.foreach_get("material_index", mat_idx)
        keep = mat_idx == material_index
        starts, totals = starts[keep], totals[keep]
    if len(starts) == 0:
        return np.zeros((0, 2, 2))

    loop_start = np.repeat(starts, totals)
    loop_total = np.repeat(totals, totals)
    offset = np.arange(totals.sum()) - np.repeat(np.cumsum(totals) - totals, totals)
    a = loop_start + offset
    b = loop_start + (offset + 1) % loop_total
    p, q = uv[a], uv[b]
    # Each interior edge appears twice (once per face); drop the duplicates.
    swap = ((p[:, 0] > q[:, 0]) | ((p[:, 0] == q[:, 0]) & (p[:, 1] > q[:, 1])))[:, None]
    lo, hi = np.where(swap, q, p), np.where(swap, p, q)
    _, unique = np.unique(np.round(np.hstack([lo, hi]), 6), axis=0, return_index=True)
    return np.stack([lo[unique], hi[unique]], axis=1)


def _rasterize(edges, size, width_px):
    """Anti-aliased coverage mask (size, size) of the UV edges, bottom row first."""
    big = size * SUPERSAMPLE
    mask = np.zeros((big, big), dtype=bool)
    if len(edges):
        p0, p1 = edges[:, 0] * big, edges[:, 1] * big
        lengths = np.linalg.norm(p1 - p0, axis=1)
        steps = np.ceil(lengths / 0.5).astype(np.int64) + 1
        idx = np.repeat(np.arange(len(edges)), steps)
        local = np.arange(steps.sum()) - np.repeat(np.cumsum(steps) - steps, steps)
        t = local / np.maximum(steps[idx] - 1, 1)
        pts = p0[idx] + (p1 - p0)[idx] * t[:, None]
        ix, iy = np.floor(pts[:, 0]).astype(np.int64), np.floor(pts[:, 1]).astype(np.int64)
        inside = (ix >= 0) & (ix < big) & (iy >= 0) & (iy < big)
        mask[iy[inside], ix[inside]] = True
    radius = max(int(round(width_px * SUPERSAMPLE / 2.0)) - 1, 0)
    for _ in range(radius):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE).mean(axis=(1, 3)).astype(np.float32)


def _texture_rgb(image, size):
    """The image resized to (size, size) as display-space RGB, bottom row first."""
    copy = image.copy()
    try:
        copy.scale(size, size)
        buf = np.empty(size * size * 4, dtype=np.float32)
        copy.pixels.foreach_get(buf)
        rgb = buf.reshape(size, size, 4)[..., :3]
        if copy.is_float:  # float buffers are scene-linear; encode to sRGB
            rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(np.clip(rgb, 0, None), 1 / 2.4) - 0.055)
        return np.clip(rgb, 0.0, 1.0)
    finally:
        bpy.data.images.remove(copy)


def _line_rgb(setting, texture):
    """Line color: an RGB list, "black", "white", or "auto" (contrast per pixel)."""
    if isinstance(setting, (list, tuple)):
        return np.asarray(setting[:3], dtype=np.float32) / (255.0 if max(setting[:3]) > 1 else 1.0)
    if setting == "black":
        return np.zeros(3, dtype=np.float32)
    if setting == "white":
        return np.ones(3, dtype=np.float32)
    luminance = texture @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.where(luminance[..., None] > 0.5, 0.0, 1.0).astype(np.float32)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "material"


def render_uv_images(objects, cfg, out_dir):
    """Write one UV image per diffuse texture; the most used one is uv_800x800."""
    ucfg = cfg["uv"]
    size = int(ucfg.get("resolution", 800))
    fmt = "JPEG" if str(cfg.get("image_format", "PNG")).upper() in ("JPG", "JPEG") else "PNG"

    groups = defaultdict(list)  # image -> [(mesh, uv_name, material_index, face_count)]
    names = {}
    untextured = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        if not mesh.uv_layers:
            continue
        mat_idx = np.empty(len(mesh.polygons), dtype=np.int64)
        mesh.polygons.foreach_get("material_index", mat_idx)
        for i, slot in enumerate(obj.material_slots):
            count = int((mat_idx == i).sum())
            if count == 0:
                continue
            image, uv_name = find_diffuse(slot.material)
            if image is None:
                untextured.append((mesh, None, i, count))
                continue
            groups[image.name].append((mesh, uv_name, i, count))
            names.setdefault(image.name, slot.material.name)
        if not obj.material_slots:
            untextured.append((mesh, None, None, len(mesh.polygons)))

    written = []
    if not groups:
        if not untextured:
            print("[tspreview] UV image skipped: the model has no UV coordinates")
            return written
        print("[tspreview] no diffuse texture found; drawing UVs on a grey background")
        groups = {None: untextured}

    ranked = sorted(groups.items(), key=lambda kv: -sum(entry[3] for entry in kv[1]))
    for rank, (image_name, entries) in enumerate(ranked):
        if image_name is None:
            texture = np.full((size, size, 3), 0.5, dtype=np.float32)
        else:
            texture = _texture_rgb(bpy.data.images[image_name], size)
        edges = np.concatenate([_uv_edges(mesh, uv_name, idx) for mesh, uv_name, idx, _ in entries])
        coverage = _rasterize(edges, size, float(ucfg.get("line_width_px", 1.0)))
        alpha = (coverage * float(ucfg.get("line_opacity", 0.9)))[..., None]
        line = _line_rgb(ucfg.get("line_color", "auto"), texture)
        rgb = texture * (1.0 - alpha) + line * alpha

        stem = f"uv_{size}x{size}" if rank == 0 else f"uv_{size}x{size}_{_safe(names.get(image_name, 'texture'))}"
        path = imaging.output_path(out_dir, stem, fmt)
        imaging.save_rgb(rgb, path, fmt, cfg.get("jpeg_quality", 95))
        print(f"[tspreview] wrote {path}  ({len(edges)} UV edges, texture: {image_name})")
        written.append(path)
    return written
