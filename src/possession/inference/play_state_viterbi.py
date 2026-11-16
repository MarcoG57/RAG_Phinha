"""Joint H/A/D Viterbi with independent period boundaries and known-dead constraints."""
from __future__ import annotations

import numpy as np

JOINT_LABELS = np.array(["H", "A", "D"])


def _joint_states(control: np.ndarray, dead: np.ndarray, forced_dead: np.ndarray,
                  control_cost: float, dead_cost: float) -> np.ndarray:
    """Viterbi of one period over H=0, A=1, D=2 (see ``decode_joint``)."""
    n = control.size
    emit = np.stack([-dead + control, -dead - control, dead + np.abs(control)], axis=1)
    emit[forced_dead, :2] = -np.inf
    trans = np.array([[0.0, control_cost, dead_cost],
                      [control_cost, 0.0, dead_cost],
                      [dead_cost, dead_cost, 0.0]])  # trans[from, to]
    states = np.arange(3)
    back = np.zeros((n, 3), dtype=np.int8)
    score = emit[0].copy()
    for t in range(1, n):
        candidates = score[:, None] - trans
        best = np.argmax(candidates, axis=0)
        stay = candidates[states, states]
        best = np.where(stay >= candidates[best, states], states, best)  # ties: keep the state
        back[t] = best
        score = candidates[best, states] + emit[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))  # final tie: H before A before D
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_joint(control_evidence: np.ndarray, dead_evidence: np.ndarray, period_id: np.ndarray,
                 control_switch_cost: float, dead_switch_cost: float,
                 known_dead: np.ndarray | None = None) -> np.ndarray:
    """Decode H/A/D jointly for each period using the configured switch costs.

    Per frame, team evidence c is positive towards H and dead evidence d towards D.
    The emissions are H: c - d, A: -c - d and D: d + abs(c). A team change costs
    control_switch_cost; entering or leaving D costs dead_switch_cost, independent
    of the team. Known-dead frames are forced to D; frames outside periods are X.
    """
    for cost in (control_switch_cost, dead_switch_cost):
        if not np.isfinite(cost) or cost < 0:
            raise ValueError("switch costs must be finite and non-negative")
    control = np.asarray(control_evidence, dtype=np.float64)
    dead = np.asarray(dead_evidence, dtype=np.float64)
    period_id = np.asarray(period_id)
    if control.shape != period_id.shape or dead.shape != period_id.shape:
        raise ValueError("evidence and period_id must have the same shape")
    if not (np.isfinite(control).all() and np.isfinite(dead).all()):
        raise ValueError("evidence must be finite")
    forced = (np.zeros(period_id.shape, dtype=bool) if known_dead is None
              else np.asarray(known_dead, dtype=bool))
    if forced.shape != period_id.shape:
        raise ValueError("known_dead must align with period_id")
    labels = np.full(period_id.size, "X", dtype="<U1")
    for period in np.unique(period_id[period_id > 0]):
        idx = np.flatnonzero(period_id == period)
        if not (np.diff(idx) == 1).all():
            raise ValueError(f"period {int(period)} is not contiguous")
        labels[idx] = JOINT_LABELS[_joint_states(control[idx], dead[idx], forced[idx],
                                                 control_switch_cost, dead_switch_cost)]
    return labels
