# TurboSquid Preview Generator

Automates the preview images TurboSquid asks for when publishing a model.
Point it at a `.glb` (or `.fbx`, `.obj`, `.blend`, ...) and it renders everything
headless in Blender (Cycles, GPU when available).

| TurboSquid requirement | Output file(s) |
|---|---|
| 1200x1200 render on RGB (247,247,247) background | `search_1200x1200.png` |
| Minimum five 1920x1080 renders from a variety of angles | `product_01_front_left.png` ... `product_09_detail_end_b.png` |
| Minimum one 1920x1080 greyscale image showing all topology | `wireframe_01_front_left.png`, `wireframe_02_back_right.png` |
| 1920x1080 turntable, 12-36 frames, counter-clockwise | `turntable/turntable_01.png` ... `turntable_24.png` |
| 800x800 diffuse texture overlaid by the 0-1 UV space | `uv_800x800.png` (+ one per extra texture) |

## Requirements

- **Blender 4.2 or newer** (developed and tested with Blender 5.1.1). Found
  automatically in `C:\Program Files\Blender Foundation\`; otherwise pass
  `--blender` or set `BLENDER_PATH`.
- **Python 3.8+** to run the launcher (standard library only, no `pip install`).
- An NVIDIA/AMD/Intel GPU is used automatically (OptiX > CUDA > HIP > oneAPI),
  falling back to CPU.

## Quick start

```powershell
python make_previews.py "D:\3D_Reconstruction\3D_Models\Wizard_Wand\ExportedFinalModels\harryPotterWand8.glb"
```

Images are written to `<model folder>\TurboSquid_Previews\<model name>\`.
A full run (1 search + 9 product + 2 wireframe + 24 turntable frames + UV)
takes about 5 minutes on an RTX 2060.

On Windows you can also drag a model file (or folder) onto `make_previews.bat`.

### More examples

```powershell
# Every .glb in a folder
python make_previews.py "D:\...\ExportedFinalModels" --pattern "*.glb"

# Only some outputs, custom output folder
python make_previews.py model.glb --only search,uv -o D:\Previews

# Hide helper objects (glob patterns on object names), 36-frame turntable
python make_previews.py model.glb --exclude "Cube*" "Helper_*" --frames 36

# Quick low-quality preview pass
python make_previews.py model.glb --samples 32 --frames 12

# Your own settings file (merged over config/default.json)
python make_previews.py model.glb --config my_settings.json
```

### Options

| Option | Meaning |
|---|---|
| `-o, --output DIR` | Output root (default `<model folder>/TurboSquid_Previews`) |
| `--pattern "*.glb;*.fbx"` | Files to pick when the input is a folder |
| `--only search,product,wireframe,turntable,uv` | Render a subset |
| `--exclude NAME ...` | Object names/globs to hide |
| `--samples N` | Cycles samples for stills (turntable uses half) |
| `--frames N` | Turntable frames, 12-36 |
| `--device auto/optix/cuda/hip/oneapi/metal/cpu` | Render device |
| `--format png/jpg` | Output format (the search image is always PNG so the 247 background stays exact) |
| `--no-shadow` | No ground contact shadow |
| `--keep-transparent` | Also keep the raw RGBA renders in `transparent/` |
| `--config FILE` | JSON overriding `config/default.json` |
| `--blender PATH` | Blender executable |
| `-v, --verbose` | Show all Blender output |

## Customizing (config/default.json)

Copy only the keys you want to change into your own JSON and pass it with
`--config`. The most useful ones:

- **`product.shots`**: list of camera angles. `azimuth` 0 = front (glTF/Blender
  front, camera on -Y), positive turns toward the model's right/+X; `elevation`
  is degrees above the ground. Add `"detail": "end_a"` / `"end_b"` for a
  close-up of either end of the model's longest axis. Optional per-shot `fill`
  (fraction of the frame the model spans) and `lens` (mm).
- **`search`**: angle and `fill` (0.92 = model spans 92% of the frame).
- **`wireframe`**: `surface_grey`, `wire_grey`, `wire_width_px`, `shots`.
- **`turntable`**: `frames`, `elevation`, `samples`.
- **`uv.line_color`**: `"auto"` (dark lines on light texels and light on dark),
  `"white"`, `"black"` or `[r, g, b]`; plus `line_opacity`, `line_width_px`.
- **`lighting.lights`**: key/fill/rim sun lights. `azimuth` is relative to
  the camera (negative = camera left), `elevation` is absolute, so every shot
  gets the same look. Only lights with `"shadow": true` cast the ground shadow.
- **`view_transform` / `look` / `exposure`**: color management
  (default AgX, Medium High Contrast).
- **`shadow` / `shadow_noise_floor`**: ground contact shadow on/off, and the
  alpha level (0.02 = 2%) below which faint shadow grain is cleared to the
  pure background color.

Example `my_settings.json`:

```json
{
  "exclude_objects": ["Cube*"],
  "turntable": {"frames": 36, "elevation": 15},
  "uv": {"line_color": "white"}
}
```

## How it works

1. `make_previews.py` (plain Python) merges the config with CLI flags, saves
   it as `render_settings.json` next to the images and starts Blender in
   background mode with `tspreview/blender_main.py`.
2. The model is imported and parented to a pivot that centers it on the Z
   axis, rests it on Z=0 and scales it to unit size. Loose vertices
   (scan debris) are ignored when framing.
3. For every shot the camera is placed with an exact perspective fit of the
   model's vertices, so the model is centered and fills the frame at any angle.
4. Cycles renders with a transparent background over an invisible shadow
   catcher. The result is alpha-composited onto the background color in numpy,
   so background pixels are exactly RGB 247,247,247.
5. Wireframes use temporary copies of the meshes with a clay material plus a
   Wireframe modifier (real quads/n-gons, not just triangles), with line
   thickness matched to the pixel size, then converted to greyscale.
6. The turntable rotates the pivot by +360/N degrees per frame (counter-
   clockwise seen from above) with a camera framed for the whole sweep. Frames
   are named `turntable_01`... as TurboSquid requires.
7. The UV image finds each material's base color texture, scales it to 800x800
   and rasterizes the UV edges on top (anti-aliased).

## Project layout

```
make_previews.py        launcher (finds Blender, CLI, batch processing)
make_previews.bat       drag-and-drop wrapper for Windows
config/default.json     all render settings and camera angles
tspreview/
  blender_main.py       entry point that runs inside Blender
  scene.py              import, normalization, render/world/lights/ground setup
  framing.py            camera directions and perspective fitting
  shots.py              search / product / wireframe / turntable renderers
  uv_overlay.py         UV layout image
  imaging.py            background compositing, greyscale, saving
```
