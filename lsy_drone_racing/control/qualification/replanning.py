"""Reference refresh decisions for the qualification controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from lsy_drone_racing.control.qualification.params import ReplanTuning


@dataclass(frozen=True)
class ReplanDecision:
    """Outcome of the replan decision logic."""

    needs_replan: bool
    reference_expired: bool
    obstacle_shift: float
    max_gate_shift: float


def reference_expired(
    *,
    reference_exists: bool,
    t: float,
    reference_t_end: float,
    last_expired_replan_t: float,
    tuning: ReplanTuning,
) -> bool:
    """Return true when the current reference has gone stale long enough to rebuild."""
    return (
        reference_exists
        and t > reference_t_end + tuning.expiry_grace
        and t - last_expired_replan_t >= tuning.expiry_cooldown
    )


def replan_decision(
    *,
    t: float,
    last_replan_t: float,
    init: bool,
    active_leg: int,
    target_gate: int,
    pos: NDArray[np.floating],
    gate_pos: NDArray[np.floating],
    ref_gate_pos: NDArray[np.floating],
    obstacle_shift: float,
    reference_is_expired: bool,
    tuning: ReplanTuning,
) -> ReplanDecision:
    """Decide whether the reference should be rebuilt.

    Uses a stability-first policy:
    - always replan on init/leg transitions,
    - enforce a cooldown to avoid replan thrashing,
    - suppress replans very close to a gate unless geometry changed a lot.
    """
    dist_2d = float(np.linalg.norm(gate_pos[target_gate, :2] - pos[:2]))
    gate_shift = np.linalg.norm(ref_gate_pos - gate_pos, axis=1)
    target_gate_shifted = float(gate_shift[target_gate]) > tuning.gate_tol
    max_gate_shift = float(np.max(gate_shift))
    large_gate_shifted = max_gate_shift > tuning.large_gate_tol
    obstacle_shifted = obstacle_shift > tuning.obstacle_tol
    cooldown_active = (t - last_replan_t) < tuning.replan_cooldown
    near_gate_lock = dist_2d < tuning.near_gate_lock_dist

    if init or active_leg != target_gate:
        return ReplanDecision(
            needs_replan=True,
            reference_expired=False,
            obstacle_shift=obstacle_shift,
            max_gate_shift=max_gate_shift,
        )

    if cooldown_active:
        return ReplanDecision(
            needs_replan=False,
            reference_expired=False,
            obstacle_shift=obstacle_shift,
            max_gate_shift=max_gate_shift,
        )

    # Close to gate: avoid reshaping approach corridor unless geometry changed strongly.
    if near_gate_lock and not large_gate_shifted:
        return ReplanDecision(
            needs_replan=False,
            reference_expired=False,
            obstacle_shift=obstacle_shift,
            max_gate_shift=max_gate_shift,
        )

    hard_target_shift = target_gate_shifted and dist_2d < tuning.hard_gate_replan_dist
    needs_replan = (
        large_gate_shifted
        or obstacle_shifted
        or (target_gate_shifted and dist_2d < tuning.near_gate_dist)
        or (reference_is_expired and not near_gate_lock)
        or hard_target_shift
    )
    return ReplanDecision(
        needs_replan=needs_replan,
        reference_expired=bool(reference_is_expired and needs_replan),
        obstacle_shift=obstacle_shift,
        max_gate_shift=max_gate_shift,
    )
