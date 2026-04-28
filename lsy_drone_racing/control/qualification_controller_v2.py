"""Qualification controller for the level-2 racing setup.

This is a loadable controller entrypoint. The implementation is split across
``control/qualificationv2`` helpers so this file contains exactly one Controller
subclass, which keeps ``load_controller`` happy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.qualificationv2 import attitudecopy as _attitude_helpers
from lsy_drone_racing.control.qualificationv2 import trajectory as _trajectory_helpers
from lsy_drone_racing.control.qualificationv2.geometry import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    DEFAULT_OBSTACLES,
    normalize_gate_index,
)
from lsy_drone_racing.control.qualificationv2.pid import PositionPid
from lsy_drone_racing.control.qualificationv2.tuning import gate1_offset_tuning
from lsy_drone_racing.control.qualificationv2.trajectory import (
    ROUTE_OVERRIDE_FILE,
    RouteTuning,
    load_route_overrides,
)

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray
    from scipy.interpolate import CubicSpline

_tracking_command = getattr(
    _attitude_helpers,
    "tracking_command",
    getattr(_attitude_helpers, "attitude_command", None),
)
if _tracking_command is None:
    raise ImportError("qualification attitude helper does not expose a tracking command")

_build_reference_curve = getattr(
    _trajectory_helpers,
    "build_reference_curve",
    getattr(_trajectory_helpers, "plan_sector_spline", None),
)
if _build_reference_curve is None:
    raise ImportError("qualification trajectory helper does not expose a reference builder")


class QualificationController(Controller):
    """Reactive reference-tracking controller for qualification runs."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialise controller state and tuning parameters."""
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._t_max = 25.0

        self.gate_rpy = DEFAULT_GATE_RPY.copy()
        self._ref_gate_pos = DEFAULT_GATE_POS.copy()
        self._obstacles = DEFAULT_OBSTACLES.copy()
        self._route_overrides = load_route_overrides(ROUTE_OVERRIDE_FILE)

        self.tuning = gate1_offset_tuning()
        self.leg_times = self.tuning.leg_times.copy()
        self.leg_speed_profiles = self.tuning.speed_profiles
        self.route_tuning = getattr(self.tuning, "route_tuning", RouteTuning())
        self.pid_gains_by_section = getattr(
            self.tuning,
            "pid_gains_by_section",
            (self.tuning.pid_gains,) * 4,
        )

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.mass = float(drone_params["mass"])
        self.g = 9.81
        self.position_pid = PositionPid(self.pid_gains_by_section[0])

        self._reference: CubicSpline | None = None
        self._reference_t_end = self._t_max
        self._tick = 0
        self._finished = False
        self._init = True
        self._active_leg = -1
        self._leg_start_t = np.zeros(4, dtype=np.float64)
        self._last_action = np.zeros(4, dtype=np.float32)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude + thrust command."""
        del info
        t = min(self._tick / self._freq, self._t_max)

        target_gate = normalize_gate_index(obs["target_gate"])
        if target_gate == -1:
            self._finished = True
            return self._last_action
        if t >= self._t_max:
            self._finished = True
        if target_gate != self._active_leg:
            self._set_section_pid(target_gate)

        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        quat = np.asarray(obs["quat"], dtype=np.float64)

        self.gate_rpy = R.from_quat(gate_quat).as_euler("xyz", degrees=False)
        dist_2d = float(np.linalg.norm(gate_pos[target_gate, :2] - pos[:2]))
        gate_shifted = (
            float(np.linalg.norm(self._ref_gate_pos[target_gate] - gate_pos[target_gate]))
            > 0.005
        )
        needs_replan = (
            self._init
            or self._active_leg != target_gate
            or (gate_shifted and dist_2d < 0.65)
        )

        if needs_replan:
            self._refresh_reference(t, target_gate, gate_pos)
            self._init = False
            self._ref_gate_pos = gate_pos.copy()

        t_eval = min(t, self._reference_t_end)
        action = _tracking_command(
            self._require_reference(),
            pos,
            vel,
            quat,
            t_eval,
            position_pid=self.position_pid,
            dt=1.0 / self._freq,
            mass=self.mass,
            gravity=self.g,
        )
        self._last_action = action
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the tick counter and report whether the controller is done."""
        del action, obs, reward, info
        self._tick += 1
        if terminated or truncated:
            self._finished = True
        return self._finished

    def episode_callback(self) -> None:
        """Reset all per-episode state."""
        self.reset()

    def reset(self) -> None:
        """Reset all per-episode state."""
        self._tick = 0
        self._init = True
        self._active_leg = -1
        self._finished = False
        self.position_pid.reset()
        self._last_action = np.zeros(4, dtype=np.float32)
        self._reference = None
        self._reference_t_end = self._t_max
        self._leg_start_t[:] = 0.0

    def episode_reset(self) -> None:
        """Reset all per-episode state."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Draw the current reference and immediate setpoint."""
        if self._reference is None:
            return
        leg = max(self._active_leg, 0)
        t_now = min(self._tick / self._freq, self._reference_t_end)
        setpoint = self._reference(t_now).reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
        t0 = self._leg_start_t[leg]
        t1 = t0 + self.leg_times[leg]
        trajectory = self._reference(np.linspace(t0, t1, 100))
        draw_line(sim, trajectory, rgba=(0.0, 1.0, 0.0, 1.0))

    def diagnostic(self) -> dict:
        """Expose compact state for the web dashboard."""
        return {
            "controller_phase": "FINISHED" if self._finished else "TRACKING",
            "target_gate": self._active_leg,
            "traj_local_time": self._tick / self._freq,
            "traj_total_time": self._reference_t_end,
            "plan_mode": "qualification_reference",
        }

    def _refresh_reference(
        self,
        t: float,
        target_gate: int,
        gate_pos: NDArray[np.floating],
    ) -> None:
        if self._active_leg != target_gate:
            self._leg_start_t[target_gate] = t
            self._active_leg = target_gate
        try:
            self._reference, self._reference_t_end = _build_reference_curve(
                target_gate,
                gate_pos,
                self.gate_rpy,
                self._obstacles,
                float(self._leg_start_t[target_gate]),
                float(self.leg_times[target_gate]),
                route_overrides=self._route_overrides,
                speed_profile=self.leg_speed_profiles[target_gate],
                route_tuning=self.route_tuning,
            )
        except TypeError as exc:
            if (
                "route_overrides" not in str(exc)
                and "speed_profile" not in str(exc)
                and "route_tuning" not in str(exc)
            ):
                raise
            self._reference, self._reference_t_end = _build_reference_curve(
                target_gate,
                gate_pos,
                self.gate_rpy,
                self._obstacles,
                float(self._leg_start_t[target_gate]),
                float(self.leg_times[target_gate]),
            )

    def _require_reference(self) -> CubicSpline:
        if self._reference is None:
            raise RuntimeError("QualificationController used before planning a reference")
        return self._reference

    def _set_section_pid(self, target_gate: int) -> None:
        gains = self.pid_gains_by_section[target_gate]
        if self.position_pid.gains is gains:
            return
        self.position_pid = PositionPid(gains)
