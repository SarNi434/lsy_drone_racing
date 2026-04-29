"""Attitude-control helper for qualification tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.qualificationv2.pid import PositionPid

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
    position_pid: PositionPid | None = None,
    dt: float | None = None,
    mass: float,
    gravity: float,
    i_error: NDArray[np.floating] | None = None,
    freq: float | None = None,
    kp: NDArray[np.floating] | None = None,
    ki: NDArray[np.floating] | None = None,
    kd: NDArray[np.floating] | None = None,
    i_clamp: NDArray[np.floating] | None = None,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Track the reference using the old attitude law.

    Supports the old loose PID arguments and the newer ``PositionPid`` module
    interface. New callers get only the action back; legacy callers keep the
    old ``(action, integral_error)`` return shape.
    """
    ref_pos = reference(t_eval)
    ref_vel = reference.derivative()(t_eval)
    ref_acc = reference.derivative(2)(t_eval)

    # CubicSpline a_ref on linspace knots is spiky; cap lateral feedforward
    # without clipping vertical acceleration needed for gate height changes.
    lateral_acc_norm = float(np.linalg.norm(ref_acc[:2]))
    if lateral_acc_norm > 16.0:
        ref_acc = ref_acc.copy()
        ref_acc[:2] *= 16.0 / lateral_acc_norm

    e_pos = ref_pos - pos
    e_vel = ref_vel - vel
    if position_pid is not None:
        if dt is None:
            raise TypeError("dt is required when position_pid is provided")
        pid_force = position_pid.update(e_pos, e_vel, dt)
        legacy_return = False
    else:
        if i_error is None or freq is None or kp is None or ki is None or kd is None or i_clamp is None:
            raise TypeError("legacy PID arguments are required when position_pid is not provided")
        next_i_error = np.clip(i_error + e_pos / freq, -i_clamp, i_clamp)
        pid_force = kp * e_pos + ki * next_i_error + kd * e_vel
        legacy_return = True

    thrust_vec = pid_force + 0.75 * mass * ref_acc
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
    if legacy_return:
        return action, next_i_error
    return action
