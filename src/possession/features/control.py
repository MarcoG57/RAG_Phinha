"""Nearest-team distances and validation of recorded control-feature parameters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from possession.data.atd import AWAY, HOME, AtdData

@dataclass(frozen=True)
class ControlConfig:
    contact_scale_m: float = 1.5
    margin_scale_m: float = 1.0
    max_radius_m: float = 5.0

    def __post_init__(self):
        for name in ("contact_scale_m", "margin_scale_m", "max_radius_m"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} debe ser un número positivo finito")

def nearest_distances(data: AtdData) -> tuple[np.ndarray, np.ndarray]:
    """Distancia en metros del jugador más cercano de cada equipo al balón."""
    scale = np.array(data.pitch, dtype=np.float32)
    dist = np.linalg.norm(data.xy * scale - (data.ball_xy * scale)[:, None, :], axis=2)
    dist = np.where(data.observed, dist, np.inf)
    out = []
    for team in (HOME, AWAY):
        cols = data.meta.slot_team == team
        out.append(dist[:, cols].min(axis=1) if cols.any() else np.full(data.n_frames, np.inf))
    dist_h, dist_a = out
    missing_ball = ~data.ball_observed
    dist_h[missing_ball] = np.inf
    dist_a[missing_ball] = np.inf
    return dist_h, dist_a
