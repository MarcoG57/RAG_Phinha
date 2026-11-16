"""Evidencia de balón muerto por frame (E2, C2).

Número con signo por frame: positivo favorece «parado» (``D``), negativo «en juego».
Ninguna señal basta sola (#24), así que se suman evidencias débiles con pesos fijados a
priori con significado físico:

- **balón fuera del campo** (observado fuera de [0,1] con margen): fuerte a favor de D;
- **quietud colectiva**: velocidad media de los jugadores observados, suavizada, por
  debajo de un ritmo de trote ligero favorece D; por encima, juego;
- **balón ausente**: señal débil a favor de D sólo junto con quietud colectiva observada;
- **balón rápido** (desplazamiento entre frames consecutivos observados): en contra de D;
- ``alive_bias``: lo ambiguo se resuelve como juego.

No usa el canal de velocidad del balón del ATD (cobertura 8,7 %, sin relación con el
desplazamiento; #21 §5).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from possession.data.atd import AtdData


@dataclass(frozen=True)
class PlayStateConfig:
    out_margin: float = 0.01          # fracción del campo tolerada fuera de [0,1]
    out_weight: float = 1.0
    still_center_mps: float = 1.8     # velocidad media colectiva neutra (entre caminar y trotar)
    still_scale_mps: float = 0.8
    still_weight: float = 0.5
    smooth_s: float = 1.0             # media móvil centrada de la velocidad colectiva
    missing_ball_weight: float = 0.15
    fast_ball_mps: float = 12.0
    fast_ball_weight: float = 0.5
    alive_bias: float = 0.1

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{f.name} debe ser finito y no negativo")
        for name in ("still_scale_mps", "smooth_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} debe ser positivo")


def rolling_nanmean(x: np.ndarray, window: int,
                    period_id: np.ndarray | None = None) -> np.ndarray:
    """Media móvil centrada; separa bloques contiguos si recibe period_id."""
    x = np.asarray(x, dtype=np.float64)
    if period_id is not None:
        period_id = np.asarray(period_id)
        if period_id.shape != x.shape:
            raise ValueError("x y period_id deben tener la misma forma")
    if window <= 1 or x.size == 0:
        return x.copy()
    kernel = np.ones(window)
    pad = (window // 2, window - 1 - window // 2)
    starts = (np.array([0]) if period_id is None else
              np.r_[0, np.flatnonzero(period_id[1:] != period_id[:-1]) + 1])
    stops = np.r_[starts[1:], x.size]
    out = np.full(x.shape, np.nan)
    for start, stop in zip(starts, stops):
        block = x[start:stop]
        valid = np.isfinite(block)
        total = np.convolve(np.pad(np.where(valid, block, 0.0), pad), kernel, mode="valid")
        count = np.convolve(np.pad(valid.astype(np.float64), pad), kernel, mode="valid")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[start:stop] = np.where(count > 0, total / count, np.nan)
    return out


def ball_speed_mps(data: AtdData) -> np.ndarray:
    """Desplazamiento entre frames observados de la misma parte, en m/s; NaN si no."""
    xy = data.ball_xy.astype(np.float64) * np.array(data.pitch)
    speed = np.full(data.n_frames, np.nan)
    same_period = data.period_id[1:] == data.period_id[:-1]
    delta = np.linalg.norm(np.diff(xy, axis=0), axis=1) * data.fps
    speed[1:] = np.where(same_period, delta, np.nan)
    return speed


def team_speed_mps(data: AtdData) -> np.ndarray:
    """Velocidad media del ATD sobre los jugadores observados de cada frame; NaN sin ninguno."""
    speed = np.where(data.observed, data.speed, np.nan).astype(np.float64)
    count = np.isfinite(speed).sum(axis=1)
    total = np.nansum(speed, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def dead_evidence(data: AtdData, config: PlayStateConfig | None = None) -> np.ndarray:
    config = config or PlayStateConfig()
    m = config.out_margin
    bx, by = data.ball_xy[:, 0], data.ball_xy[:, 1]
    out = data.ball_observed & ((bx < -m) | (bx > 1 + m) | (by < -m) | (by > 1 + m))

    collective = rolling_nanmean(
        team_speed_mps(data), max(1, int(round(config.smooth_s * data.fps))), data.period_id
    )
    still = np.clip((config.still_center_mps - collective) / config.still_scale_mps, -1.0, 1.0)
    # ATD speeds are float32; neutral values can otherwise look a few ulps slow.
    neutral = np.isclose(
        collective, config.still_center_mps, rtol=np.finfo(np.float32).eps, atol=0.0
    )
    still = np.where(np.isfinite(still) & ~neutral, still, 0.0)

    fast = np.nan_to_num(ball_speed_mps(data), nan=0.0) > config.fast_ball_mps
    still_observed = (collective < config.still_center_mps) & ~neutral
    missing_ball = ~data.ball_observed & still_observed & (config.still_weight > 0)

    return (config.out_weight * out
            + config.still_weight * still
            + config.missing_ball_weight * missing_ball
            - config.fast_ball_weight * fast
            - config.alive_bias).astype(np.float64)
