"""Controlled-reference conversion for external training data."""
from __future__ import annotations

import numpy as np

FPS = 25
ALREADY_CONTROLLED = frozenset({"skillcorner"})


def controlled(reference: np.ndarray, period_id: np.ndarray, min_s: float) -> np.ndarray:
    """Absorbe rachas H/A cortas del otro equipo; X y D cierran el tramo vivo."""
    out = np.asarray(reference).copy()
    for period in np.unique(period_id[period_id > 0]):
        idx = np.flatnonzero(period_id == period)
        lab = out[idx]
        if not lab.size:
            continue
        edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        previous = None  # equipo en posesión dentro del tramo vivo actual
        for start, end in zip(np.r_[0, edges], np.r_[edges, lab.size]):
            label = lab[start]
            if label in ("D", "X"):
                previous = None
            elif previous is not None and label != previous and end - start < min_s * FPS:
                lab[start:end] = previous
            else:
                previous = label
        out[idx] = lab
    return out


def possession_lengths(reference: np.ndarray, period_id: np.ndarray) -> list[int]:
    """Longitud de cada racha H/A por parte; D y X separan posesiones."""
    lengths = []
    for period in np.unique(period_id[period_id > 0]):
        lab = reference[period_id == period]
        if not lab.size:
            continue
        edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        lengths += [int(e - s) for s, e in zip(np.r_[0, edges], np.r_[edges, lab.size]) if lab[s] in "HA"]
    return lengths
