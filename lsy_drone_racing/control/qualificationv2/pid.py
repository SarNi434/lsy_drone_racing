"""PID helpers for qualification reference tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class PositionPidGains:
    """Axis-wise PID gains and integral clamp for position tracking."""

    kp: NDArray[np.floating]
    ki: NDArray[np.floating]
    kd: NDArray[np.floating]
    i_clamp: NDArray[np.floating]

    @classmethod
    def from_xyz(
        cls,
        *,
        kp: tuple[float, float, float],
        ki: tuple[float, float, float],
        kd: tuple[float, float, float],
        i_clamp: tuple[float, float, float],
    ) -> PositionPidGains:
        """Create gains from x/y/z tuples."""
        return cls(
            kp=np.asarray(kp, dtype=np.float64),
            ki=np.asarray(ki, dtype=np.float64),
            kd=np.asarray(kd, dtype=np.float64),
            i_clamp=np.asarray(i_clamp, dtype=np.float64),
        )


class PositionPid:
    """Stateful axis-wise PID controller for position errors."""

    def __init__(self, gains: PositionPidGains) -> None:
        self.gains = gains
        self.integral = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        """Clear accumulated integral error."""
        self.integral[:] = 0.0

    def update(
        self,
        pos_error: NDArray[np.floating],
        vel_error: NDArray[np.floating],
        dt: float,
    ) -> NDArray[np.floating]:
        """Return PID correction for position and velocity errors."""
        pos_error = np.asarray(pos_error, dtype=np.float64)
        vel_error = np.asarray(vel_error, dtype=np.float64)
        self.integral = np.clip(
            self.integral + pos_error * float(dt),
            -self.gains.i_clamp,
            self.gains.i_clamp,
        )
        return (
            self.gains.kp * pos_error
            + self.gains.ki * self.integral
            + self.gains.kd * vel_error
        )
