"""Qualification controller for the level-2 racing setup.

This is a loadable controller entrypoint. The implementation is split across
``control/qualification`` helpers so this file contains exactly one Controller
subclass, which keeps ``load_controller`` happy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.qualification import attitude as _attitude_helpers
from lsy_drone_racing.control.qualification import trajectory as _trajectory_helpers
from lsy_drone_racing.control.qualification.geometry import (
    DEFAULT_GATE_POS,
    DEFAULT_GATE_RPY,
    DEFAULT_OBSTACLES,
    normalize_gate_index,
)
from lsy_drone_racing.control.qualification.params import ReplanTuning, leg_times, pid_gains
from lsy_drone_racing.control.qualification.replanning import replan_decision
from lsy_drone_racing.control.qualification.trajectory import (
    ROUTE_OVERRIDE_FILE,
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

        self.gate_rpy = DEFAULT_GATE_RPY.copy()
        self._ref_gate_pos = DEFAULT_GATE_POS.copy()
        self._obstacles = DEFAULT_OBSTACLES.copy()
        self._route_overrides = load_route_overrides(ROUTE_OVERRIDE_FILE)

        self.leg_times = leg_times()

        drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.mass = float(drone_params["mass"])
        self.g = 9.81
        self.kp, self.ki, self.kd, self.i_clamp = pid_gains()
        self.i_err = np.zeros(3, dtype=np.float64)

        self._reference: CubicSpline | None = None
        self._tick = 0
        self._finished = False
        self._init = True
        self._active_leg = -1
        self._leg_start_t = np.zeros(4, dtype=np.float64)
        self._last_action = np.zeros(4, dtype=np.float32)
        self._last_replan_t = -np.inf
        self._replan_tuning = ReplanTuning()

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next attitude + thrust command."""
        del info
        t = self._tick / self._freq

        target_gate = normalize_gate_index(obs["target_gate"])
        if target_gate == -1:
            self._finished = True
            return self._last_action

        gate_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gate_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        obstacles = np.asarray(obs["obstacles_pos"], dtype=np.float64)
        pos = np.asarray(obs["pos"], dtype=np.float64)
        vel = np.asarray(obs["vel"], dtype=np.float64)
        quat = np.asarray(obs["quat"], dtype=np.float64)

        self.gate_rpy = R.from_quat(gate_quat).as_euler("xyz", degrees=False)
        obstacle_shift = float(np.max(np.linalg.norm(self._obstacles - obstacles, axis=1)))
        self._obstacles = obstacles.copy()

        try:
            replan = replan_decision(
                t=t,
                last_replan_t=self._last_replan_t,
                init=self._init,
                active_leg=self._active_leg,
                target_gate=target_gate,
                pos=pos,
                gate_pos=gate_pos,
                ref_gate_pos=self._ref_gate_pos,
                obstacle_shift=obstacle_shift,
                reference_is_expired=False,
                tuning=self._replan_tuning,
            )
        except TypeError as exc:
            # Compatibility path for environments still running the older
            # replan_decision signature without time/cooldown fields.
            if "unexpected keyword argument 't'" not in str(exc) and "last_replan_t" not in str(exc):
                raise
            replan = replan_decision(
                init=self._init,
                active_leg=self._active_leg,
                target_gate=target_gate,
                pos=pos,
                gate_pos=gate_pos,
                ref_gate_pos=self._ref_gate_pos,
                obstacle_shift=obstacle_shift,
                reference_is_expired=False,
                tuning=self._replan_tuning,
            )

        if replan.needs_replan:
            self._refresh_reference(
                t,
                target_gate,
                gate_pos,
                reset_leg_clock=replan.reference_expired,
                start_pos=pos if replan.reference_expired else None,
            )
            self._last_replan_t = t
            self._init = False
            self._ref_gate_pos = gate_pos.copy()

        # Keep evaluating at wall-clock t (MASF behavior); clamping to leg end can
        # freeze the setpoint before the gate is actually passed.
        t_eval = t
        action, self.i_err = _tracking_command(
            self._require_reference(),
            pos,
            vel,
            quat,
            t_eval,
            i_error=self.i_err,
            freq=self._freq,
            mass=self.mass,
            gravity=self.g,
            kp=self.kp,
            ki=self.ki,
            kd=self.kd,
            i_clamp=self.i_clamp,
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
        self.i_err[:] = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)
        self._reference = None
        self._leg_start_t[:] = 0.0
        self._last_replan_t = -np.inf

    def episode_reset(self) -> None:
        """Reset all per-episode state."""
        self.reset()

    def render_callback(self, sim: Sim) -> None:
        """Draw the current reference and immediate setpoint."""
        if self._reference is None:
            return
        leg = max(self._active_leg, 0)
        t_now = min(self._tick / self._freq, self._current_reference_end())
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
            "traj_total_time": self._current_reference_end(),
            "plan_mode": "qualification_reference",
        }

    def _refresh_reference(
        self,
        t: float,
        target_gate: int,
        gate_pos: NDArray[np.floating],
        reset_leg_clock: bool = False,
        start_pos: NDArray[np.floating] | None = None,
    ) -> None:
        if self._active_leg != target_gate or reset_leg_clock:
            self._leg_start_t[target_gate] = t
            self._active_leg = target_gate
        try:
            self._reference, _ = _build_reference_curve(
                target_gate,
                gate_pos,
                self.gate_rpy,
                self._obstacles,
                float(self._leg_start_t[target_gate]),
                float(self.leg_times[target_gate]),
                route_overrides=self._route_overrides,
                start_pos=start_pos,
            )
        except TypeError as exc:
            if "route_overrides" not in str(exc) and "start_pos" not in str(exc):
                raise
            self._reference, _ = _build_reference_curve(
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

    def _current_reference_end(self) -> float:
        if self._active_leg < 0:
            return 0.0
        return float(self._leg_start_t[self._active_leg] + self.leg_times[self._active_leg])
