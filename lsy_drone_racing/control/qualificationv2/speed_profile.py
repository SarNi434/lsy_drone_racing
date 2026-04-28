"""Speed-profile timing helpers for qualification route splines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class SectorSpeedProfile:
    """Piecewise-linear speed multiplier profile over one sector.

    Values are relative to nominal speed. ``start`` applies just after the
    previous gate, ``mid`` applies around sector center, and ``end`` applies as
    the drone approaches the target gate.
    """

    start: float = 1.0
    mid: float = 1.0
    end: float = 1.0

    def multiplier(self, progress: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return speed multipliers for normalized progress values in [0, 1]."""
        progress = np.asarray(progress, dtype=np.float64)
        speeds = np.interp(
            np.clip(progress, 0.0, 1.0),
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
            np.array([self.start, self.mid, self.end], dtype=np.float64),
        )
        return np.maximum(speeds, 1e-3)


def sector_knots_from_speed_profile(
    t_start: float,
    t_end: float,
    waypoints: NDArray[np.floating],
    profile: SectorSpeedProfile,
) -> NDArray[np.floating]:
    """Allocate spline knot times from relative speed multipliers.

    The backup controller used equal time between route markers. This helper
    preserves that baseline when all multipliers are 1.0, then shortens or
    lengthens each marker interval according to the local speed multiplier.
    """
    waypoints = np.asarray(waypoints, dtype=np.float64)
    if waypoints.ndim != 2:
        raise ValueError("waypoints must be a 2D array")

    n_points = len(waypoints)
    if n_points <= 0:
        raise ValueError("waypoints must not be empty")
    if n_points == 1:
        return np.array([float(t_start)], dtype=np.float64)

    duration = float(t_end - t_start)
    if duration <= 0.0:
        raise ValueError("t_end must be greater than t_start")

    segment_centers = (np.arange(n_points - 1, dtype=np.float64) + 0.5) / (n_points - 1)
    raw_times = 1.0 / profile.multiplier(segment_centers)
    intervals = duration * raw_times / float(np.sum(raw_times))
    return np.concatenate(([float(t_start)], float(t_start) + np.cumsum(intervals)))
