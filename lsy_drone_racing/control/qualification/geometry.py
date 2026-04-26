"""Geometry helpers for the qualification controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray

DEFAULT_GATE_RPY = np.array(
    [
        [0.0, 0.0, -0.78],
        [0.0, 0.0, 2.35],
        [0.0, 0.0, 3.14],
        [0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

DEFAULT_GATE_POS = np.array(
    [
        [0.5, 0.25, 0.7],
        [1.05, 0.75, 1.2],
        [-1.0, -0.25, 0.7],
        [0.0, -0.75, 1.2],
    ],
    dtype=np.float64,
)

DEFAULT_OBSTACLES = np.array(
    [
        [0.0, 0.75, 1.55],
        [1.0, 0.25, 1.55],
        [-1.5, -0.25, 1.55],
        [-0.5, -0.75, 1.55],
    ],
    dtype=np.float64,
)


def gate_axis_points(
    gate_pos: NDArray[np.floating],
    gate_rpy: NDArray[np.floating],
    r_in: float = 0.25,
    r_out: float = 0.3,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Return two points aligned with the gate's local x-axis, projected to horizontal."""
    axis = R.from_euler("xyz", gate_rpy).apply(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    axis[2] = 0.0
    axis /= np.linalg.norm(axis) + 1e-9
    return gate_pos - r_in * axis, gate_pos + r_out * axis


def normalize_gate_index(target_gate: object) -> int:
    """Normalize scalar or array-like target-gate observations to a Python int."""
    gate_array = np.asarray(target_gate)
    if gate_array.size != 1:
        raise ValueError(f"Expected scalar target_gate, got shape {gate_array.shape}")
    return int(gate_array.reshape(-1)[0])


def quat_z_column(quat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return the body z-axis from quaternion [qx, qy, qz, qw]."""
    qx, qy, qz, qw = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            2 * (qx * qz + qw * qy),
            2 * (qy * qz - qw * qx),
            1 - 2 * (qx * qx + qy * qy),
        ],
        dtype=np.float64,
    )


def euler_xyz_from_matrix(r_mat: NDArray[np.floating]) -> NDArray[np.floating]:
    """Extract intrinsic XYZ Euler angles from a rotation matrix."""
    sy = -r_mat[2, 0]
    cy = np.sqrt(r_mat[2, 1] ** 2 + r_mat[2, 2] ** 2)
    pitch = np.arctan2(sy, cy)
    if cy > 1e-6:
        roll = np.arctan2(r_mat[2, 1], r_mat[2, 2])
        yaw = np.arctan2(r_mat[1, 0], r_mat[0, 0])
    else:
        roll = np.arctan2(-r_mat[1, 2], r_mat[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float64)
