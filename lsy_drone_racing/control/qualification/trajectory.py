"""Route-reference helpers for qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control.qualification.geometry import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    gate_axis_points,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_AVOID_DEPTH = 3
SECTOR_OBSTACLE_INDEX = (0, 0, 1, 3)
ROUTE_OVERRIDE_FILE = "logs/qualification_route_waypoints.json"
UNCHANGED_OVERRIDE_TOL = 1e-8


def build_route_points(
    route_idx: int,
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    extra: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Assemble the ordered control points for one target gate."""
    if route_idx == 0:
        _, p_exit = gate_axis_points(gate_pos[0], gate_rpy[0])
        points = [np.array([-1.5, 0.75, 0.05], dtype=np.float64)]
        if extra is not None:
            points.append(extra)
        points += [gate_pos[0], p_exit, np.array([1.25, 0.0, 1.0], dtype=np.float64)]
    elif route_idx == 1:
        p_enter, p_exit = gate_axis_points(gate_pos[1], gate_rpy[1])
        points = [gate_pos[0], np.array([1.25, 0.0, 1.0], dtype=np.float64)]
        if extra is not None:
            points.append(extra)
        points += [p_enter, gate_pos[1], p_exit, np.array([0.0, 1.0, 1.0], dtype=np.float64)]
    elif route_idx == 2:
        p_enter, p_exit = gate_axis_points(gate_pos[2], gate_rpy[2], 0.4, 0.3)
        points = [gate_pos[1]]
        if extra is not None:
            points.append(extra)
        points += [np.array([0.0, 0.25, 1.0], dtype=np.float64), p_enter, gate_pos[2], p_exit]
    elif route_idx == 3:
        _, p_exit = gate_axis_points(gate_pos[3], gate_rpy[3], 0.2, 0.4)
        points = [gate_pos[2], np.array([-0.5, -0.5, 0.8], dtype=np.float64)]
        if extra is not None:
            points.append(extra)
        points += [gate_pos[3], p_exit]
    else:
        raise ValueError(f"Route index {route_idx} out of range [0, 3]")
    return np.asarray(points, dtype=np.float64)


def build_route_groups(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
) -> tuple[NDArray[np.floating], ...]:
    """Return default control-point groups for all four target gates."""
    return tuple(build_route_points(route_idx, gate_pos, gate_rpy) for route_idx in range(4))


def flatten_route_groups(
    groups: tuple[NDArray[np.floating], ...],
) -> tuple[NDArray[np.floating], tuple[int, ...]]:
    """Flatten grouped control points and return the original group lengths."""
    lengths = tuple(int(len(group)) for group in groups)
    if not groups:
        return np.zeros((0, 3), dtype=np.float64), lengths
    return np.vstack(groups).astype(np.float64), lengths


def load_route_overrides(path: str | Path | None) -> tuple[NDArray[np.floating], ...] | None:
    """Load optional edited XY route points from the planner JSON."""
    if path is None:
        return None

    route_path = Path(path)
    if not route_path.exists():
        return None
    with open(route_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    points_xy = payload.get("qualification_route_points_xy")
    lengths = payload.get("qualification_route_lengths")
    if points_xy is None:
        points_xy = payload.get("waypoints_xy")
    if lengths is None:
        return None

    points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    lengths = tuple(int(length) for length in lengths)
    if sum(lengths) != len(points_xy) or any(length < 0 for length in lengths):
        return None

    nominal_groups = build_route_groups(DEFAULT_GATE_POS, DEFAULT_GATE_RPY)
    nominal_points, nominal_lengths = flatten_route_groups(nominal_groups)
    if lengths == nominal_lengths and len(points_xy) == len(nominal_points):
        unchanged = np.linalg.norm(points_xy - nominal_points[:, :2], axis=1)
        points_xy = points_xy.copy()
        points_xy[unchanged <= UNCHANGED_OVERRIDE_TOL] = np.nan

    groups = []
    offset = 0
    for length in lengths:
        groups.append(points_xy[offset : offset + length].copy())
        offset += length
    return tuple(groups)


def _with_override_xy(
    defaults: NDArray[np.floating],
    override_xy: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
    if override_xy is None:
        return defaults
    override_xy = np.asarray(override_xy, dtype=np.float64).reshape(-1, 2)
    if len(override_xy) == 0:
        return defaults
    if len(override_xy) == len(defaults):
        points = defaults.copy()
        finite = np.all(np.isfinite(override_xy), axis=1)
        points[finite, :2] = override_xy[finite]
        return points

    if len(override_xy) == 1:
        z_values = np.array([defaults[0, 2]], dtype=np.float64)
    else:
        distances = np.linalg.norm(np.diff(override_xy, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(np.maximum(distances, 1e-9))))
        alpha = arc / max(float(arc[-1]), 1e-9)
        z_values = defaults[0, 2] + alpha * (defaults[-1, 2] - defaults[0, 2])
    return np.column_stack((override_xy, z_values)).astype(np.float64)


def clearance_point(
    reference: CubicSpline,
    t_start: float,
    t_end: float,
    route_idx: int,
    obstacles: NDArray[np.floating],
    trigger: float = 0.15,
) -> NDArray[np.floating] | None:
    """Sample a reference and return one push-away point if needed."""
    obstacle = np.asarray(obstacles, dtype=np.float64)[SECTOR_OBSTACLE_INDEX[route_idx]]
    samples = reference(np.linspace(t_start, t_end, 40))
    diff = obstacle[:2] - samples[:, :2]
    dist = np.linalg.norm(diff, axis=1)
    closest = int(np.argmin(dist))
    if dist[closest] >= trigger:
        return None
    point = samples[closest]
    push = diff[closest] / (dist[closest] + 1e-6) * 0.2
    return np.array([point[0] - push[0], point[1] - push[1], point[2]], dtype=np.float64)


def build_reference_curve(
    route_idx: int,
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    obstacles: NDArray[np.floating],
    t_start: float,
    duration: float,
    extra: NDArray[np.floating] | None = None,
    depth: int = 0,
    route_overrides: tuple[NDArray[np.floating], ...] | None = None,
) -> tuple[CubicSpline, float]:
    """Build a reference curve and recursively inject one clearance point."""
    default_points = build_route_points(route_idx, gate_pos, gate_rpy, extra)
    override_xy = (
        route_overrides[route_idx]
        if route_overrides is not None and route_idx < len(route_overrides) and extra is None
        else None
    )
    waypoints = _with_override_xy(default_points, override_xy)
    t_end = float(t_start + duration)
    knots = np.linspace(t_start, t_end, len(waypoints))
    reference = CubicSpline(knots, waypoints)

    if depth < MAX_AVOID_DEPTH:
        avoid_point = clearance_point(reference, t_start, t_end, route_idx, obstacles)
        if avoid_point is not None:
            return build_reference_curve(
                route_idx,
                gate_pos,
                gate_rpy,
                obstacles,
                t_start,
                duration,
                avoid_point,
                depth + 1,
                route_overrides=route_overrides,
            )
    return reference, t_end
