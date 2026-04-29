"""Route-reference helpers for qualification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from lsy_drone_racing.control.qualificationv2.geometry import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    gate_axis_points,
)
from lsy_drone_racing.control.qualificationv2.speed_profile import (
    SectorSpeedProfile,
    sector_knots_from_speed_profile,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_AVOID_DEPTH = 3
SECTOR_OBSTACLE_INDEX = (0, 0, 1, 3)
ROUTE_OVERRIDE_FILE = "logs/qualification_route_waypoints.json"
UNCHANGED_OVERRIDE_TOL = 1e-8
GATE1_EXIT_OFFSET = 0.35
GATE1_LEFT_OFFSET_TOWARD_GATE0 = 0.02
GATE1_POLE_AVOIDANCE_POINT = np.array([1.35, -0.18, 1.0], dtype=np.float64)
GATE1_POLE_ARC_POINTS = np.array(
    [
        [1.08, -0.16, 0.86],
        [1.38, 0.12, 1.08],
    ],
    dtype=np.float64,
)
GATE0_LATERAL_OFFSET_AWAY_FROM_START = 0.035
GATE0_VERTICAL_OFFSET = 0.02
GATE2_ENTRY_OFFSET = 0.30
GATE2_MINIMAL_EXIT_OFFSET = 0.08
DEFAULT_CLEARANCE_TRIGGERS = (0.15, 0.15, 0.15, 0.17)
DEFAULT_CLEARANCE_MARGINS = (0.03, 0.03, 0.03, 0.02)
DEFAULT_CLEARANCE_PUSH_MAX = (0.12, 0.12, 0.12, 0.06)


@dataclass(frozen=True)
class RouteTuning:
    """Optional route geometry changes relative to the backup route."""

    gate0_lateral_offset: float = 0.0
    gate0_vertical_offset: float = 0.0
    gate1_exit_offset: float = 0.0
    gate1_left_offset_toward_gate0: float = 0.0
    clearance_triggers: tuple[float, float, float, float] = DEFAULT_CLEARANCE_TRIGGERS
    clearance_margins: tuple[float, float, float, float] = DEFAULT_CLEARANCE_MARGINS
    clearance_push_max: tuple[float, float, float, float] = DEFAULT_CLEARANCE_PUSH_MAX

    def clearance_trigger(self, route_idx: int) -> float:
        """Return obstacle clearance trigger for one route section."""
        return float(self.clearance_triggers[route_idx])

    def clearance_margin(self, route_idx: int) -> float:
        """Return extra target margin after a clearance point is inserted."""
        return float(self.clearance_margins[route_idx])

    def clearance_push_limit(self, route_idx: int) -> float:
        """Return max single correction distance for one clearance point."""
        return float(self.clearance_push_max[route_idx])


def _gate_horizontal_axis(
    gate_rpy: NDArray[np.floating],
    local_axis: NDArray[np.floating],
) -> NDArray[np.floating]:
    from scipy.spatial.transform import Rotation as R

    axis = R.from_euler("xyz", gate_rpy).apply(local_axis.astype(np.float64))
    axis[2] = 0.0
    return axis / (np.linalg.norm(axis) + 1e-9)


def _gate1_crossing_and_exit_points(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    route_tuning: RouteTuning,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    gate1_pos = np.asarray(gate_pos[1], dtype=np.float64)
    gate1_rpy = np.asarray(gate_rpy[1], dtype=np.float64)
    normal = _gate_horizontal_axis(gate1_rpy, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    left_axis = _gate_horizontal_axis(gate1_rpy, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    toward_gate0 = np.asarray(gate_pos[0], dtype=np.float64) - gate1_pos
    if float(np.dot(left_axis, toward_gate0)) < 0.0:
        left_axis = -left_axis
    lateral_offset = route_tuning.gate1_left_offset_toward_gate0 * left_axis
    crossing = gate1_pos + lateral_offset
    exit_point = gate1_pos + route_tuning.gate1_exit_offset * normal + lateral_offset
    return crossing, exit_point


def _gate0_biased_center(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    route_tuning: RouteTuning,
) -> NDArray[np.floating]:
    from scipy.spatial.transform import Rotation as R

    rotation = R.from_euler("xyz", np.asarray(gate_rpy[0], dtype=np.float64))
    local_offset = np.array(
        [
            0.0,
            route_tuning.gate0_lateral_offset,
            route_tuning.gate0_vertical_offset,
        ],
        dtype=np.float64,
    )
    return np.asarray(gate_pos[0], dtype=np.float64) + rotation.apply(local_offset)


def _gate_extra_exit_point(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    gate_idx: int,
    offset: float,
) -> NDArray[np.floating]:
    normal = _gate_horizontal_axis(
        np.asarray(gate_rpy[gate_idx], dtype=np.float64),
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    return np.asarray(gate_pos[gate_idx], dtype=np.float64) + float(offset) * normal


def build_route_points(
    route_idx: int,
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    extra: NDArray[np.floating] | None = None,
    route_tuning: RouteTuning | None = None,
) -> NDArray[np.floating]:
    """Assemble the ordered control points for one target gate."""
    route_tuning = route_tuning or RouteTuning()
    if route_idx == 0:
        gate0_target = _gate0_biased_center(gate_pos, gate_rpy, route_tuning)
        _, p_exit = gate_axis_points(gate0_target, gate_rpy[0])
        points = [np.array([-1.5, 0.75, 0.05], dtype=np.float64)]
        if extra is not None:
            points.append(extra)
        points += [gate0_target, p_exit]
    elif route_idx == 1:
        p_enter, p_exit = gate_axis_points(gate_pos[1], gate_rpy[1])
        gate0_start = _gate0_biased_center(gate_pos, gate_rpy, route_tuning)
        _, gate0_exit = gate_axis_points(gate0_start, gate_rpy[0])
        points = [gate0_exit, *GATE1_POLE_ARC_POINTS.copy()]
        if extra is not None:
            points.append(extra)
        if route_tuning.gate1_exit_offset > 0.0 or route_tuning.gate1_left_offset_toward_gate0 != 0.0:
            gate1_crossing, gate1_exit = _gate1_crossing_and_exit_points(
                gate_pos,
                gate_rpy,
                route_tuning,
            )
            p_enter = p_enter + (gate1_crossing - gate_pos[1])
            points += [
                p_enter,
                gate1_exit,
            ]
        else:
            points += [p_enter, gate_pos[1], p_exit]
    elif route_idx == 2:
        p_enter, p_exit = gate_axis_points(
            gate_pos[2],
            gate_rpy[2],
            GATE2_ENTRY_OFFSET,
            GATE2_MINIMAL_EXIT_OFFSET,
        )
        if route_tuning.gate1_exit_offset > 0.0:
            _, gate1_exit = _gate1_crossing_and_exit_points(
                gate_pos,
                gate_rpy,
                route_tuning,
            )
            points = [gate1_exit]
        else:
            points = [gate_pos[1]]
        if extra is not None:
            points.append(extra)
        points += [np.array([0.0, 0.25, 1.0], dtype=np.float64), p_enter, gate_pos[2], p_exit]
    elif route_idx == 3:
        _, p_exit = gate_axis_points(gate_pos[3], gate_rpy[3], 0.2, 0.4)
        gate2_start = _gate_extra_exit_point(
            gate_pos,
            gate_rpy,
            gate_idx=2,
            offset=GATE2_MINIMAL_EXIT_OFFSET,
        )
        points = [gate2_start, np.array([-0.55, -0.42, 0.85], dtype=np.float64)]
        if extra is not None:
            points.append(extra)
        points += [gate_pos[3], p_exit]
    else:
        raise ValueError(f"Route index {route_idx} out of range [0, 3]")
    return np.asarray(points, dtype=np.float64)


def build_route_groups(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    route_tuning: RouteTuning | None = None,
) -> tuple[NDArray[np.floating], ...]:
    """Return default control-point groups for all four target gates."""
    return tuple(
        build_route_points(route_idx, gate_pos, gate_rpy, route_tuning=route_tuning)
        for route_idx in range(4)
    )


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
    margin: float = 0.03,
    push_max: float = 0.12,
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
    push_distance = min(max(float(trigger + margin - dist[closest]), 0.0), float(push_max))
    push = diff[closest] / (dist[closest] + 1e-6) * push_distance
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
    speed_profile: SectorSpeedProfile | None = None,
    route_tuning: RouteTuning | None = None,
) -> tuple[CubicSpline, float]:
    """Build a reference curve and recursively inject one clearance point."""
    default_points = build_route_points(
        route_idx,
        gate_pos,
        gate_rpy,
        extra,
        route_tuning=route_tuning,
    )
    override_xy = (
        route_overrides[route_idx]
        if route_overrides is not None and route_idx < len(route_overrides) and extra is None
        else None
    )
    waypoints = _with_override_xy(default_points, override_xy)
    t_end = float(t_start + duration)
    knots = (
        np.linspace(t_start, t_end, len(waypoints))
        if speed_profile is None
        else sector_knots_from_speed_profile(t_start, t_end, waypoints, speed_profile)
    )
    reference = (
        CubicSpline(knots, waypoints)
        if route_idx in (0, 1)
        else PchipInterpolator(knots, waypoints, axis=0)
    )

    if depth < MAX_AVOID_DEPTH:
        tuning = route_tuning or RouteTuning()
        avoid_point = clearance_point(
            reference,
            t_start,
            t_end,
            route_idx,
            obstacles,
            trigger=tuning.clearance_trigger(route_idx),
            margin=tuning.clearance_margin(route_idx),
            push_max=tuning.clearance_push_limit(route_idx),
        )
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
                speed_profile=speed_profile,
                route_tuning=route_tuning,
            )
    return reference, t_end
