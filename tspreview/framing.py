"""Camera placement: view directions and tight perspective framing of a point set."""

import math

import numpy as np
from mathutils import Vector


def view_direction(azimuth_deg, elevation_deg):
    """Unit vector from the model toward the camera.

    Azimuth 0 looks at the model's front (camera on -Y, Blender/glTF front),
    positive azimuth moves the camera toward +X. Elevation is above the ground.
    """
    az = math.radians(azimuth_deg)
    el = math.radians(max(-89.0, min(89.0, elevation_deg)))
    return Vector((math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)))


def camera_rotation(to_camera):
    """Quaternion for a camera looking against `to_camera`, keeping +Z up."""
    return (-Vector(to_camera)).normalized().to_track_quat("-Z", "Y")


def half_fov_tangents(lens_mm, sensor_mm, res_x, res_y):
    """tan(half FOV) horizontally and vertically for sensor_fit AUTO."""
    if res_x >= res_y:
        tan_x = (sensor_mm / 2.0) / lens_mm
        tan_y = tan_x * res_y / res_x
    else:
        tan_y = (sensor_mm / 2.0) / lens_mm
        tan_x = tan_y * res_x / res_y
    return tan_x, tan_y


def fit_camera(points, rotation, lens_mm, res_x, res_y, fill, sensor_mm=36.0):
    """Closest camera position that shows every point, centered in frame.

    `fill` is the fraction of the frame (in the limiting axis) the points span.
    Returns (location, distance) where distance is measured along the view axis
    from the camera to the points' bounding center.
    """
    rot = rotation.to_matrix()
    right = np.array(rot.col[0])
    up = np.array(rot.col[1])
    forward = -np.array(rot.col[2])
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    rel = points - center
    x, y, z = rel @ right, rel @ up, rel @ forward
    tan_x, tan_y = half_fov_tangents(lens_mm, sensor_mm, res_x, res_y)
    tx, ty = tan_x * fill, tan_y * fill

    # For each axis, a point needs: |coord - shift| <= t * (z + D).
    # Solving for the optimal lateral shift gives a closed form for D.
    ax, bx = (x / tx - z).max(), (-x / tx - z).max()
    ay, by = (y / ty - z).max(), (-y / ty - z).max()
    dist_x, shift_x = (ax + bx) / 2.0, tx * (ax - bx) / 2.0
    dist_y, shift_y = (ay + by) / 2.0, ty * (ay - by) / 2.0
    distance = max(dist_x, dist_y, 1e-3)
    location = center - forward * distance + right * shift_x + up * shift_y
    return Vector(location), float(distance)


def principal_axis(points):
    """Longest principal axis of the point cloud (unit vector)."""
    sample = points
    if len(sample) > 100_000:
        sample = sample[np.random.default_rng(0).choice(len(sample), 100_000, replace=False)]
    centered = sample - sample.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def detail_view(points, base_dir, end, fraction):
    """Subset of points near one end of the longest axis and a view direction for it.

    `end` is "end_a" (positive end of the axis) or "end_b" (negative end).
    """
    axis = principal_axis(points)
    t = (points - points.mean(axis=0)) @ axis
    sign = 1.0 if end == "end_a" else -1.0
    t = t * sign
    cutoff = t.max() - fraction * (t.max() - t.min())
    subset = points[t >= cutoff]

    axis_v = Vector(axis) * sign
    base = Vector(base_dir).normalized()
    perp = base - axis_v * base.dot(axis_v)
    if perp.length < 0.2:  # base view looks down the axis; use a side view
        perp = axis_v.cross(Vector((0.0, 0.0, 1.0)))
        if perp.length < 0.2:
            perp = Vector((0.0, -1.0, 0.0))
    direction = (perp.normalized() + axis_v * 0.35).normalized()
    min_z = math.sin(math.radians(12))
    if direction.z < min_z:
        direction.z = min_z
        direction.normalize()
    return subset, direction


def rotate_z(points, angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return points @ rot.T
