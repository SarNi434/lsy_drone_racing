"""Tuning constants for the qualification controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplanTuning:
    """Thresholds controlling when the reference is rebuilt."""

    # Gate and map shift thresholds.
    gate_tol: float = 0.01
    large_gate_tol: float = 0.12
    obstacle_tol: float = 0.06
    near_gate_dist: float = 0.65
    near_gate_lock_dist: float = 0.38
    replan_cooldown: float = 0.35
    hard_gate_replan_dist: float = 0.20
    expiry_grace: float = 0.15
    expiry_cooldown: float = 0.8


def leg_times() -> np.ndarray:
    """Return the per-leg reference durations."""
    # Slightly conservative profile prioritizing gate completion.
    return np.array([3.95, 2.55, 3.55, 2.30], dtype=np.float64)


def pid_gains() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return PID gain vectors and integral clamp."""
    kp = np.array([0.55, 0.55, 1.55], dtype=np.float64)
    ki = np.array([0.045, 0.045, 0.05], dtype=np.float64)
    kd = np.array([0.50, 0.50, 0.5], dtype=np.float64)
    i_clamp = np.array([2.0, 2.0, 0.4], dtype=np.float64)
    return kp, ki, kd, i_clamp
