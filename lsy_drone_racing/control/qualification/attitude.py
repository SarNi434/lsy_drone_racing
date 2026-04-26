"""Attitude-control helper for qualification tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline


def tracking_command(
    reference: CubicSpline,
    pos: NDArray[np.floating],
    vel: NDArray[np.floating],
    quat: NDArray[np.floating],
    t_eval: float,
    *,
    i_error: NDArray[np.floating],
    freq: float,
    mass: float,
    gravity: float,
    kp: NDArray[np.floating],
    ki: NDArray[np.floating],
    kd: NDArray[np.floating],
    i_clamp: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Track the reference and return action plus new integral state."""
    ref_pos = reference(t_eval)
    ref_vel = reference.derivative()(t_eval)

    e_pos = ref_pos - pos
    e_vel = ref_vel - vel
    next_i_error = np.clip(i_error + e_pos / freq, -i_clamp, i_clamp)

    thrust_vec = kp * e_pos + ki * next_i_error + kd * e_vel
    thrust_vec[2] += mass * gravity

    z_body = R.from_quat(quat).as_matrix()[:, 2]
    thrust_cmd = float(thrust_vec.dot(z_body))

    z_des = thrust_vec / (np.linalg.norm(thrust_vec) + 1e-6)
    y_des = np.cross(z_des, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    y_des /= np.linalg.norm(y_des) + 1e-6
    x_des = np.cross(y_des, z_des)

    r_des = np.column_stack([x_des, y_des, z_des])
    attitude = R.from_matrix(r_des).as_euler("xyz", degrees=False)
    action = np.array([*attitude, thrust_cmd], dtype=np.float32)
    return action, next_i_error
