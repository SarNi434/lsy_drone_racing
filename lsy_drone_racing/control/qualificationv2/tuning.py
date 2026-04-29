"""Central tuning values for the abgabe1 qualification controller.  """

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lsy_drone_racing.control.qualificationv2.pid import PositionPidGains
from lsy_drone_racing.control.qualificationv2.speed_profile import SectorSpeedProfile
from lsy_drone_racing.control.qualificationv2.trajectory import (
    GATE1_EXIT_OFFSET,
    GATE1_LEFT_OFFSET_TOWARD_GATE0,
    RouteTuning,
)


@dataclass(frozen=True)
class QualificationTuning:
    """All high-level controller tuning that should be easy to adjust."""

    leg_times: np.ndarray
    speed_profiles: tuple[SectorSpeedProfile, ...]
    pid_gains_by_section: tuple[PositionPidGains, PositionPidGains, PositionPidGains, PositionPidGains]
    route_tuning: RouteTuning

    @property
    def pid_gains(self) -> PositionPidGains:
        """Compatibility alias for code that expects one global PID table."""
        return self.pid_gains_by_section[0]


def _pid_gains(
    *,
    kp: tuple[float, float, float] = (0.57, 0.57, 1.55),
    ki: tuple[float, float, float] = (0.045, 0.045, 0.05),
    kd: tuple[float, float, float] = (0.50, 0.50, 0.50),
    i_clamp: tuple[float, float, float] = (1.5, 1.5, 0.4),
) -> PositionPidGains:
    return PositionPidGains.from_xyz(kp=kp, ki=ki, kd=kd, i_clamp=i_clamp)


def current_qualification_tuning() -> QualificationTuning:
    """Return the current hand-tuned qualification parameters."""
    base_leg_times = np.array([3.85, 2.5, 3.5, 2.25], dtype=np.float64)
    alpha = 0.84
    beta = np.array([0.65, 1.08, 0.81, 0.99], dtype=np.float64)

    return QualificationTuning(
        leg_times=base_leg_times * alpha * beta,
        speed_profiles=(
            SectorSpeedProfile(start=1.0, mid=1.0, end=1.0),
            SectorSpeedProfile(start=1.0, mid=1.0, end=1.0),
            SectorSpeedProfile(start=1.0, mid=1.0, end=1.0),
            SectorSpeedProfile(start=1.0, mid=1.0, end=1.0),
        ),
        pid_gains_by_section=(
            _pid_gains(kp=(0.60, 0.60, 1.65), ki=(0.05, 0.05, 0.05), kd=(0.35, 0.35, 0.50), i_clamp=(1.5, 1.5, 0.4)),
            _pid_gains(kp=(0.6, 0.6, 1.55), ki=(0.045, 0.045, 0.05), kd=(0.55, 0.55, 0.50), i_clamp=(1.5, 1.5, 0.4)),
            _pid_gains(kp=(0.65, 0.65, 1.55), ki=(0.045, 0.045, 0.05), kd=(0.45, 0.45, 0.50), i_clamp=(1.5, 1.5, 0.4)),
            _pid_gains(kp=(0.65, 0.65, 1.65), ki=(0.045, 0.045, 0.05), kd=(0.30, 0.30, 0.50), i_clamp=(1.5, 1.5, 0.4)),
        ),

        route_tuning=RouteTuning(),
    )


def gate1_offset_tuning() -> QualificationTuning:
    """Return current tuning plus the experimental gate-1 crossing/exit offsets."""
    tuning = current_qualification_tuning()
    return QualificationTuning(
        leg_times=tuning.leg_times,
        speed_profiles=tuning.speed_profiles,
        pid_gains_by_section=tuning.pid_gains_by_section,
        route_tuning=RouteTuning(
            gate1_exit_offset=GATE1_EXIT_OFFSET,
            gate1_left_offset_toward_gate0=GATE1_LEFT_OFFSET_TOWARD_GATE0,
            clearance_triggers=(0.15, 0.15, 0.15, 0.1),
            clearance_margins=(0.03, 0.03, 0.03, 0.02),
            clearance_push_max=(0.12, 0.12, 0.12, 0.06),
        ),
    )


def default_qualification_tuning() -> QualificationTuning:
    """Return the current safe default tuning."""
    return current_qualification_tuning()
