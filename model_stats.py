#!/usr/bin/env python3
"""Report polygon and vertex counts of 3D models (for TurboSquid's product specs).

Examples:
    python model_stats.py "D:/Models/Wizard_Wand/ExportedFinalModels/harryPotterWand8.glb"
    python model_stats.py "D:/Models/Wizard_Wand/ExportedFinalModels"          # every model in the folder
    python model_stats.py model.fbx --exclude "Helper*" --json stats.json

Counts follow TurboSquid's guidance: polygons are faces as modeled (a quad
counts as one polygon), measured on the base mesh without subdivision, over
all parts of the model. Vertices are geometric vertices (identical positions
merged); "in file" is the raw count stored in the file, which for glTF is
higher because vertices are split along UV seams and hard edges.
"""

import argparse
import json
import os
import subprocess
import sys

from make_previews import collect_models, find_blender

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
BLENDER_SCRIPT = os.path.join(REPO_DIR, "tspreview", "blender_stats.py")
BATCH_SIZE = 40  # models per Blender session (keeps the command line short)

COLUMNS = [
    ("Object", "name"),
    ("Polygons", "polygons"),
    ("Triangles", "triangles"),
    ("Vertices", "vertices"),
    ("Verts in file", "vertices_in_file"),
    ("Quads", "quads"),
    ("Tris", "tris"),
    ("N-gons", "ngons"),
    ("Loose verts", "loose_vertices"),
]


def run_blender(blender, models, exclude, verbose):
    cmd = [blender, "-b", "--factory-startup", "--python-exit-code", "1", "-P", BLENDER_SCRIPT, "--"]
    cmd += [os.path.abspath(m) for m in models]
    if exclude:
        cmd += ["--exclude", *exclude]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    found = {}
    for line in proc.stdout:
        # Per-model import errors come back inside the JSON result.
        marker = line.find("[tsstats] ")
        if marker >= 0:
            result = json.loads(line[marker + len("[tsstats] "):])
            found[os.path.normcase(result["file"])] = result
        elif verbose:
            print(line, end="", flush=True)
    proc.wait()
    missing = {"error": "Blender returned no result (rerun with --verbose for details)"}
    return [found.get(os.path.normcase(os.path.abspath(m)), dict(missing, file=m)) for m in models]


def fmt(value):
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else str(value)


def print_table(rows):
    widths = [max(len(title), *(len(fmt(r.get(key, ""))) for r in rows)) for title, key in COLUMNS]
    line = "  ".join(title.ljust(w) if i == 0 else title.rjust(w) for i, ((title, _), w) in enumerate(zip(COLUMNS, widths)))
    print("  " + line)
    print("  " + "-" * len(line))
    for r in rows:
        cells = [fmt(r.get(key, "")) for _, key in COLUMNS]
        print("  " + "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))))


def print_report(result, show_objects):
    print(f"\n{os.path.basename(result['file'])}")
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    total = dict(result["total"], name="TOTAL")
    rows = (result["objects"] if show_objects and len(result["objects"]) > 1 else []) + [total]
    print_table(rows)
    print(f"\n  Geometry type : {result['geometry_type']}")
    print(f"  For TurboSquid: Polygons = {total['polygons']:,}   Vertices = {total['vertices']:,}")

    if result["triangulated_format"]:
        print("  note: glTF/GLB always stores triangles. If the source model is built from quads,"
              " count the .fbx/.blend instead to report the quad count.")
    loose = [o for o in result["objects"] if o["loose_vertices"] or o["loose_edges"]]
    for o in loose:
        print(f"  warning: {o['name']} has {o['loose_vertices']:,} loose vertices and"
              f" {o['loose_edges']:,} loose edges (not part of any face; CheckMate may flag these)")
    subdiv = [o["name"] for o in result["objects"] if o["subdivision"]]
    if subdiv:
        print(f"  note: subdivision modifiers on {', '.join(subdiv)}; counts are for the base mesh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="model file, or a folder of models")
    parser.add_argument("--pattern", default="*", help="file pattern(s) when input is a folder, ';'-separated")
    parser.add_argument("--exclude", nargs="+", metavar="NAME", help="object names/globs to leave out")
    parser.add_argument("--summary", action="store_true", help="only show totals, not every object")
    parser.add_argument("--json", metavar="FILE", help="also write the full results to a JSON file")
    parser.add_argument("--blender", help="path to blender executable")
    parser.add_argument("-v", "--verbose", action="store_true", help="show all Blender output")
    args = parser.parse_args()

    blender = find_blender(args.blender)
    models = collect_models(args.input, args.pattern)
    print(f"Counting {len(models)} model(s) with {blender}")

    results = []
    for i in range(0, len(models), BATCH_SIZE):
        results += run_blender(blender, models[i:i + BATCH_SIZE], args.exclude, args.verbose)
    for result in results:
        print_report(result, show_objects=not args.summary)

    ok = [r for r in results if "error" not in r]
    if len(ok) > 1:
        print("\nAll models")
        print_table([dict(r["total"], name=os.path.basename(r["file"])) for r in ok])

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json}")
    if len(ok) < len(models):
        sys.exit(1)


if __name__ == "__main__":
    main()
