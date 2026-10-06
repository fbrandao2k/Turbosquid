"""Polygon / vertex statistics, executed inside Blender.

    blender -b --factory-startup --python-exit-code 1 -P tspreview/blender_stats.py -- \
        model1.glb model2.fbx --exclude "Cube*"

Use model_stats.py instead of calling this directly. Prints one
"[tsstats] {json}" line per model.
"""

import argparse
import fnmatch
import json
import os
import sys

import bpy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tspreview import scene as ts_scene  # noqa: E402

SUBDIVISION_MODIFIERS = {"SUBSURF", "MULTIRES"}
COUNT_KEYS = ("polygons", "triangles", "tris", "quads", "ngons", "vertices", "vertices_in_file",
              "edges", "loose_vertices", "loose_edges")


def mesh_stats(mesh):
    n_verts, n_edges, n_polys = len(mesh.vertices), len(mesh.edges), len(mesh.polygons)
    sides = np.empty(n_polys, dtype=np.int64)
    mesh.polygons.foreach_get("loop_total", sides)

    co = np.empty(n_verts * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    welded = 0
    if n_verts:
        # Formats like glTF split vertices along UV seams / hard edges; merging
        # identical positions gives the geometric vertex count a DCC app shows.
        tol = max(float(np.ptp(co, axis=0).max()) * 1e-6, 1e-9)
        welded = len(np.unique(np.round(co / tol).astype(np.int64), axis=0))

    loop_verts = np.empty(len(mesh.loops), dtype=np.int64)
    loop_edges = np.empty(len(mesh.loops), dtype=np.int64)
    mesh.loops.foreach_get("vertex_index", loop_verts)
    mesh.loops.foreach_get("edge_index", loop_edges)

    return {
        "polygons": n_polys,
        "triangles": int((sides - 2).sum()),
        "tris": int((sides == 3).sum()),
        "quads": int((sides == 4).sum()),
        "ngons": int((sides > 4).sum()),
        "vertices": welded,
        "vertices_in_file": n_verts,
        "edges": n_edges,
        "loose_vertices": n_verts - len(np.unique(loop_verts)),
        "loose_edges": n_edges - len(np.unique(loop_edges)),
    }


def geometry_type(total):
    if total["polygons"] == 0:
        return "No polygons"
    if total["ngons"]:
        return "Polygonal Ngons used"
    if total["quads"] and not total["tris"]:
        return "Polygonal Quads only"
    if total["tris"] and not total["quads"]:
        return "Polygonal Tris only"
    return "Polygonal Quads/Tris"


def model_stats(path, exclude):
    ts_scene.reset_scene()
    objects = ts_scene.import_model(path)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for obj in sorted(objects, key=lambda o: o.name):
        if obj.type not in ts_scene.RENDERABLE_TYPES:
            continue
        if any(fnmatch.fnmatchcase(obj.name, p) for p in exclude):
            continue
        if obj.type == "MESH":
            stats = mesh_stats(obj.data)  # base mesh, before modifiers
        else:  # curves, text, ...: count the mesh they produce
            obj_eval = obj.evaluated_get(depsgraph)
            mesh = obj_eval.to_mesh()
            stats = mesh_stats(mesh)
            obj_eval.to_mesh_clear()
        stats["name"] = obj.name
        stats["subdivision"] = any(m.type in SUBDIVISION_MODIFIERS for m in obj.modifiers)
        rows.append(stats)

    total = {k: sum(r[k] for r in rows) for k in COUNT_KEYS}
    return {
        "file": path,
        "objects": rows,
        "total": total,
        "geometry_type": geometry_type(total) if rows else "No mesh objects",
        "triangulated_format": os.path.splitext(path)[1].lower() in (".glb", ".gltf"),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="tsstats")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("models", nargs="+")
    args = parser.parse_args(argv)
    for path in args.models:
        try:
            result = model_stats(path, args.exclude)
        except Exception as exc:  # report and continue with the next model
            lines = [l.strip() for l in str(exc).splitlines() if l.strip()] or [type(exc).__name__]
            reason = next((l for l in reversed(lines) if "Error" in l), lines[-1])
            result = {"file": path, "error": reason}
        # Leading newline: Blender's own error output may not end with one.
        print("\n[tsstats] " + json.dumps(result), flush=True)


main()
