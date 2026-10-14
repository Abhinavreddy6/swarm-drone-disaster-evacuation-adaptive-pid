"""
Shared path planning and trajectory smoothing utilities.
Converts grid search to continuous, smooth 3D trajectories.
"""

import numpy as np
from scipy.interpolate import CubicSpline

def cells_to_world(cells, resolution, origin=(0.0, 0.0), default_altitude=15.0):
    """
    Convert list of (row, col) grid coordinates to continuous (x, y, z) coordinates.
    row -> y, col -> x
    """
    coords = []
    for r, c in cells:
        x = origin[0] + c * resolution + resolution / 2.0
        y = origin[1] + r * resolution + resolution / 2.0
        coords.append(np.array([x, y, default_altitude], dtype=float))
    return coords

def path_length_world(path_world):
    """Calculate total continuous Euclidean distance along a path."""
    if len(path_world) < 2:
        return 0.0
    length = 0.0
    for i in range(len(path_world) - 1):
        length += float(np.linalg.norm(np.array(path_world[i+1]) - np.array(path_world[i])))
    return length

def calculate_smoothness(path_world):
    """
    Calculates path curvature smoothness metric (integral of squared second derivatives).
    Lower value indicates a smoother path.
    """
    if len(path_world) < 3:
        return 0.0
    pts = np.array(path_world)
    diffs = np.diff(pts, axis=0)
    curvatures = []
    for i in range(len(diffs) - 1):
        v1 = diffs[i]
        v2 = diffs[i+1]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 > 1e-4 and n2 > 1e-4:
            cos_theta = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            angle = np.arccos(cos_theta)
            curvatures.append(angle ** 2)
    return float(np.mean(curvatures)) if curvatures else 0.0

def smooth_path(path_world, num_samples=None, default_altitude=15.0):
    """
    Smooth discrete waypoints using Catmull-Rom / cubic spline interpolation
    into continuous real-valued coordinates.
    """
    if len(path_world) <= 2:
        return [np.array(p, dtype=float) for p in path_world]

    pts = np.array(path_world)
    # Ensure altitude dimension exists
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.full(len(pts), default_altitude)])

    # Remove duplicates
    diffs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.insert(diffs > 1e-3, 0, True)
    pts = pts[keep]

    if len(pts) <= 2:
        return [p for p in pts]

    # Parameterize along cumulative distance
    dists = np.zeros(len(pts))
    dists[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    total_dist = dists[-1]

    if total_dist < 1e-3:
        return [p for p in pts]

    if num_samples is None:
        num_samples = max(int(total_dist / 3.0), len(pts) * 3)

    u_query = np.linspace(0, total_dist, num_samples)
    try:
        cs_x = CubicSpline(dists, pts[:, 0], bc_type='natural')
        cs_y = CubicSpline(dists, pts[:, 1], bc_type='natural')
        cs_z = CubicSpline(dists, pts[:, 2], bc_type='natural')
        smoothed = np.column_stack([cs_x(u_query), cs_y(u_query), cs_z(u_query)])
        return [pt for pt in smoothed]
    except Exception:
        return [p for p in pts]
