"""Entry point executed inside Blender.

    blender -b --factory-startup --python-exit-code 1 -P tspreview/blender_main.py -- \
        --input model.glb --output out_dir --config render_settings.json

Use make_previews.py instead of calling this directly; it finds Blender and
builds the config for you.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tspreview import scene as ts_scene  # noqa: E402
from tspreview import shots, uv_overlay  # noqa: E402

ALL_OUTPUTS = ("search", "product", "wireframe", "turntable", "uv")


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="tspreview")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args(argv)


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)
    outputs = [o for o in cfg.get("outputs", ALL_OUTPUTS) if o in ALL_OUTPUTS]
    os.makedirs(args.output, exist_ok=True)
    started = time.time()

    ts_scene.reset_scene()
    model = ts_scene.prepare_model(args.input, cfg.get("exclude_objects", []))
    print(f"[tspreview] {len(model.points)} vertices, median edge {model.median_edge:.4f} (normalized units)")

    renderer = None
    if any(o != "uv" for o in outputs):
        ts_scene.setup_render(cfg)
        ts_scene.setup_world(cfg.get("lighting", {}))
        ts_scene.setup_lights(model, cfg.get("lighting", {}))
        ts_scene.setup_camera(model, float(cfg.get("lens_mm", 50)))
        ts_scene.setup_ground(model, bool(cfg.get("shadow", True)))
        renderer = shots.Renderer(model, cfg, args.output)

    written = []
    for name in outputs:
        print(f"[tspreview] === {name} ===")
        if name == "uv":
            written += uv_overlay.render_uv_images(model.objects, cfg, args.output)
        else:
            getattr(renderer, name)()
    if renderer:
        written += renderer.written
        renderer.cleanup()

    print(f"[tspreview] done: {len(written)} images in {time.time() - started:.0f}s -> {args.output}")


main()
