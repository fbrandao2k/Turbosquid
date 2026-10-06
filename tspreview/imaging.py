"""Pixel-level post-processing done with numpy inside Blender."""

import os

import bpy
import numpy as np


def load_rgba(path):
    """Load an 8-bit image file as a float array (H, W, 4), bottom row first."""
    img = bpy.data.images.load(path, check_existing=False)
    try:
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, 4)
    finally:
        bpy.data.images.remove(img)


def clean_shadow_alpha(rgba, floor=0.02):
    """Remove shadow-catcher noise from the alpha channel.

    Cycles does not denoise alpha, so the ground shadow has isolated outlier
    pixels and faint grain far from the model. Shadow pixels are black and
    semi-transparent: median-filter those, then lift the alpha floor so faint
    grain becomes exactly transparent while the soft falloff stays smooth.
    """
    alpha = rgba[..., 3]
    padded = np.pad(alpha, 1, mode="edge")
    h, w = alpha.shape
    neighbours = np.stack([padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)])
    median = np.median(neighbours, axis=0)
    shadow = (alpha < 0.98) & (rgba[..., :3].max(axis=-1) < 0.06)
    alpha = np.where(shadow, median, alpha)
    if floor > 0:
        alpha = np.clip((alpha - floor) / (1.0 - floor), 0.0, 1.0)
    out = rgba.copy()
    out[..., 3] = alpha
    return out


def composite_on_color(rgba, rgb255):
    """Alpha-over a straight-alpha display-space image onto a flat color."""
    bg = np.asarray(rgb255, dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    return rgba[..., :3] * alpha + bg * (1.0 - alpha)


def to_greyscale(rgb):
    lum = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return np.repeat(lum[..., None], 3, axis=2)


def save_rgb(rgb, path, file_format="PNG", quality=95):
    """Write an (H, W, 3) float array in [0, 1] (bottom row first) to disk."""
    h, w = rgb.shape[:2]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = bpy.data.images.new("ts_out", w, h, alpha=False, float_buffer=False)
    try:
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
        img.pixels.foreach_set(rgba.ravel())
        img.filepath_raw = path
        img.file_format = file_format
        img.save(filepath=path, quality=int(quality))
    finally:
        bpy.data.images.remove(img)


def output_path(directory, stem, file_format):
    ext = ".jpg" if file_format == "JPEG" else ".png"
    return os.path.join(directory, stem + ext)
