#!/usr/bin/env python3
"""Generate TurboSquid preview images for a 3D model using Blender (headless).

Examples:
    python make_previews.py "D:/Models/Wizard_Wand/ExportedFinalModels/harryPotterWand8.glb"
    python make_previews.py "D:/Models/Wizard_Wand/ExportedFinalModels" --pattern "*.glb"
    python make_previews.py model.glb --only search,uv --exclude "Cube*" --frames 36

Outputs (per model, in <output>/<model name>/):
    search_1200x1200.png          1200x1200 on RGB 247,247,247
    product_XX_<angle>.png        1920x1080 renders from several angles
    wireframe_XX_<angle>.png      1920x1080 greyscale topology renders
    turntable/turntable_XX.png    1920x1080 frames, counter-clockwise
    uv_800x800.png                diffuse texture with the 0-1 UV layout
"""

import argparse
import copy
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(REPO_DIR, "config", "default.json")
BLENDER_SCRIPT = os.path.join(REPO_DIR, "tspreview", "blender_main.py")
MODEL_EXTENSIONS = (".glb", ".gltf", ".fbx", ".obj", ".stl", ".ply", ".usd", ".usda", ".usdc", ".usdz", ".blend")
OUTPUT_TYPES = ("search", "product", "wireframe", "turntable", "uv")


def find_blender(explicit=None):
    candidates = [explicit, os.environ.get("BLENDER_PATH"), shutil.which("blender")]
    installed = glob.glob(r"C:\Program Files\Blender Foundation\Blender *\blender.exe")
    installed += glob.glob(r"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe")
    installed += glob.glob("/Applications/Blender.app/Contents/MacOS/Blender")

    def version_key(path):
        match = re.search(r"Blender (\d+)\.(\d+)", path)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)

    candidates += sorted(installed, key=version_key, reverse=True)
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    sys.exit("Blender not found. Install Blender 4.2+ or pass --blender / set BLENDER_PATH.")


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def build_config(args):
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = json.load(f)
    if args.config:
        with open(args.config, encoding="utf-8") as f:
            cfg = deep_merge(cfg, json.load(f))
    if args.only:
        wanted = [o.strip().lower() for o in args.only.split(",") if o.strip()]
        unknown = set(wanted) - set(OUTPUT_TYPES)
        if unknown:
            sys.exit(f"Unknown --only value(s): {', '.join(sorted(unknown))}. Choose from {', '.join(OUTPUT_TYPES)}")
        cfg["outputs"] = wanted
    if args.exclude:
        cfg["exclude_objects"] = list(cfg.get("exclude_objects", [])) + args.exclude
    if args.samples:
        cfg["samples"] = args.samples
        cfg["turntable"]["samples"] = max(16, args.samples // 2)
    if args.frames:
        if not 12 <= args.frames <= 36:
            sys.exit("--frames must be between 12 and 36 (TurboSquid requirement)")
        cfg["turntable"]["frames"] = args.frames
    if args.device:
        cfg["device"] = args.device.upper()
    if args.format:
        cfg["image_format"] = "JPEG" if args.format.lower() in ("jpg", "jpeg") else "PNG"
    if args.no_shadow:
        cfg["shadow"] = False
    if args.keep_transparent:
        cfg["keep_transparent"] = True
    return cfg


def collect_models(path, pattern):
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        sys.exit(f"Input not found: {path}")
    patterns = [p.strip() for p in pattern.split(";") if p.strip()]
    models = sorted(
        os.path.join(path, name) for name in os.listdir(path)
        if name.lower().endswith(MODEL_EXTENSIONS) and any(fnmatch.fnmatch(name.lower(), p.lower()) for p in patterns)
    )
    if not models:
        sys.exit(f"No models matching '{pattern}' in {path}")
    return models


def render_model(blender, model_path, out_dir, cfg, verbose=False):
    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, "render_settings.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    cmd = [
        blender, "-b", "--factory-startup", "--python-exit-code", "1",
        "-P", BLENDER_SCRIPT, "--",
        "--input", os.path.abspath(model_path), "--output", os.path.abspath(out_dir), "--config", config_path,
    ]
    print(f"\n=== {os.path.basename(model_path)} -> {out_dir}")
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in proc.stdout:
        # Blender prints a progress line per render tile/sample; keep ours and errors.
        if verbose or line.startswith("[tspreview]") or "Error" in line or "Traceback" in line or line.startswith("  "):
            print(line, end="", flush=True)
    code = proc.wait()
    status = "OK" if code == 0 else f"FAILED (exit {code})"
    print(f"=== {os.path.basename(model_path)}: {status} in {time.time() - started:.0f}s")
    return code == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="model file, or a folder of models")
    parser.add_argument("-o", "--output", help="output root (default: <model folder>/TurboSquid_Previews)")
    parser.add_argument("--pattern", default="*.glb", help="file pattern(s) when input is a folder, ';'-separated")
    parser.add_argument("--config", help="JSON file overriding config/default.json")
    parser.add_argument("--only", help=f"comma list of outputs: {','.join(OUTPUT_TYPES)}")
    parser.add_argument("--exclude", nargs="+", metavar="NAME", help="object names/globs to hide, e.g. \"Cube*\"")
    parser.add_argument("--samples", type=int, help="Cycles samples for stills (turntable uses half)")
    parser.add_argument("--frames", type=int, help="turntable frame count (12-36)")
    parser.add_argument("--device", choices=["auto", "optix", "cuda", "hip", "oneapi", "metal", "cpu"])
    parser.add_argument("--format", choices=["png", "jpg"], help="image format (search image is always PNG)")
    parser.add_argument("--no-shadow", action="store_true", help="disable the ground contact shadow")
    parser.add_argument("--blender", help="path to blender executable")
    parser.add_argument("--keep-transparent", action="store_true", help="also save RGBA renders in transparent/")
    parser.add_argument("-v", "--verbose", action="store_true", help="show all Blender output")
    args = parser.parse_args()

    blender = find_blender(args.blender)
    print(f"Using Blender: {blender}")
    cfg = build_config(args)
    models = collect_models(args.input, args.pattern)

    failures = []
    for model_path in models:
        stem = os.path.splitext(os.path.basename(model_path))[0]
        root = args.output or os.path.join(os.path.dirname(os.path.abspath(model_path)), "TurboSquid_Previews")
        if not render_model(blender, model_path, os.path.join(root, stem), cfg, args.verbose):
            failures.append(model_path)

    if failures:
        print("\nFailed models:\n  " + "\n  ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
