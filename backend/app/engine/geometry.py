from __future__ import annotations

import math

import numpy as np

def normalize(v: np.ndarray) -> np.ndarray:
    # math.hypot beats np.linalg.norm by ~5x on 2-vectors: the numpy call
    # overhead dominates the arithmetic at this size, and this runs ~300k
    # times per simulated second.
    n = math.hypot(float(v[0]), float(v[1]))
    return v / n if n > 1e-9 else np.zeros(2)


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def perp_distance_unsigned(
    point: np.ndarray,
    line_origin: np.ndarray,
    line_dir: np.ndarray,  # must be a unit vector
) -> float:
    """Perpendicular distance from point to an infinite line defined by origin + direction."""
    return abs(float(np.cross(line_dir, point - line_origin)))


def project_onto_line(
    point: np.ndarray,
    line_origin: np.ndarray,
    line_dir: np.ndarray,  # must be a unit vector
) -> float:
    """Signed scalar projection of point onto line (distance along line_dir from origin)."""
    return float(np.dot(point - line_origin, line_dir))


def point_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float:
    """
    Distance from `point` to the *segment* (not the infinite line).

    Route scoring previously used an infinite-line perpendicular plus a binary
    filter that dropped anyone whose projection fell outside [0, length]. That
    made the risk discontinuous at the endpoints: an opponent standing exactly
    on the pass target -- the most dangerous defender there is -- was excluded
    entirely. Clamping the projection instead makes the falloff smooth.
    """
    ax, ay = float(seg_start[0]), float(seg_start[1])
    sx, sy = float(seg_end[0]) - ax, float(seg_end[1]) - ay
    px, py = float(point[0]) - ax, float(point[1]) - ay
    seg_len_sq = sx * sx + sy * sy
    if seg_len_sq < 1e-18:
        return math.hypot(px, py)
    t = (px * sx + py * sy) / seg_len_sq
    t = min(1.0, max(0.0, t))
    return math.hypot(px - sx * t, py - sy * t)
