"""Selected training preparation and official scoring for external matches."""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from possession.data.atd import AtdData

ROOT = Path(__file__).resolve().parents[3]
SCENARIOS = ("clean", "combined_dead", "ball_long_gaps_40", "dead_ball_missing")
_COMPONENT_SCENARIOS = ("combined", "ball_long_gaps_20", "player_dropout", "coordinate_noise")
COMBINED = ("ball_long_gaps_20", "player_dropout", "coordinate_noise")
DEAD_BALL_MISSING_SHARE = .8
NEEDS_REFERENCE = ("dead_ball_missing", "combined_dead")
LONG_GAP_SHARE = {"ball_long_gaps_20": .2, "ball_long_gaps_40": .4}
LONG_GAP_SECONDS = (2, 30)


def degrade(data: AtdData, scenario: str, seed: int = 0, reference: np.ndarray | None = None) -> AtdData:
    """Apply only the preparation/check scenarios retained for the selected GRU model."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown degradation scenario: {scenario}")
    return _degrade(data, scenario, seed, reference)


def _degrade(data: AtdData, scenario: str, seed: int, reference: np.ndarray | None = None) -> AtdData:
    """Internal closure of combined_dead: long gaps, player dropout and coordinate noise."""
    if scenario not in SCENARIOS + _COMPONENT_SCENARIOS:
        raise ValueError(f"unknown degradation scenario: {scenario}")
    if scenario in NEEDS_REFERENCE:
        if reference is None or np.asarray(reference).shape != (data.n_frames,):
            raise ValueError(f"{scenario} needs the match reference, one label per frame")
        base = _degrade(data, "ball_long_gaps_20" if scenario == "dead_ball_missing" else "combined", seed)
        rng = np.random.default_rng(seed + 10)
        dead = (np.asarray(reference) == "D") & (data.period_id > 0)
        edges = np.flatnonzero(np.diff(np.r_[0, dead.astype(np.int8), 0]))
        mask = np.zeros(data.n_frames, dtype=bool)
        for start, end in zip(edges[::2], edges[1::2]):
            if rng.random() < DEAD_BALL_MISSING_SHARE:
                mask[start:end] = True
        out = replace(base, ball_xy=base.ball_xy.copy(), ball_observed=base.ball_observed.copy(),
                      provenance={**data.provenance, "degradation": {"scenario": scenario, "seed": seed}})
        out.ball_observed[mask] = False
        out.ball_xy[mask] = np.nan
        return out
    if scenario == "combined":
        out = data
        for offset, part in enumerate(COMBINED):
            out = _degrade(out, part, seed + offset)
        return replace(out, provenance={**data.provenance, "degradation": {"scenario": scenario, "seed": seed}})
    rng = np.random.default_rng(seed)
    out = replace(data, xy=data.xy.copy(), speed=data.speed.copy(), observed=data.observed.copy(),
                   ball_xy=data.ball_xy.copy(), ball_observed=data.ball_observed.copy(),
                   provenance={**data.provenance, "degradation": {"scenario": scenario, "seed": seed}})
    if scenario in LONG_GAP_SHARE:
        low, high = (seconds * data.fps for seconds in LONG_GAP_SECONDS)
        mask = np.zeros(data.n_frames, dtype=bool)
        # Gaps stay inside one period, so none spans a period boundary.
        for period in np.unique(data.period_id[data.period_id > 0]):
            rows = np.flatnonzero(data.period_id == period)
            target = LONG_GAP_SHARE[scenario] * rows.size
            while mask[rows].sum() < target:
                # Trim the last gap to what is missing, keeping the 2 s minimum run.
                remaining = int(np.ceil(target - mask[rows].sum()))
                length = min(max(low, min(int(rng.integers(low, high + 1)), remaining)), rows.size)
                start = int(rng.integers(0, rows.size - length + 1))
                mask[rows[start:start + length]] = True
        out.ball_observed[mask] = False
        out.ball_xy[mask] = np.nan
    elif scenario == "player_dropout":
        mask = rng.random(data.observed.shape) < .2
        out.observed[mask] = False
        out.xy[mask] = np.nan
        out.speed[mask] = np.nan
    elif scenario == "coordinate_noise":
        scale = np.array(data.pitch)
        out.xy += (rng.normal(0, .5, size=data.xy.shape) / scale).astype(np.float32)
        out.ball_xy += (rng.normal(0, .5, size=data.ball_xy.shape) / scale).astype(np.float32)
    return out


def _official_metrics():
    sys.path.insert(0, str(ROOT / "scripts"))
    from official_checkout import checked_official
    path = checked_official() / "evaluation/possession_metrics.py"
    spec = importlib.util.spec_from_file_location("external_official_possession_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _aggregate_metrics(value):
    """Retain official metric values while omitting per-frame transition traces."""
    if isinstance(value, dict):
        return {key: _aggregate_metrics(item) for key, item in value.items()
                if key not in {"pairs", "missed", "spurious"}}
    return value


def score_predictions(reference, predictions, period_id, fps) -> dict:
    """Use unchanged official matching; no C4 fixed-length submission validation.

    Preserve every source gap. Insert an ungraded guard longer than the maximum
    official transition tolerance before each following period. This separates
    spells AND transition matching across halves without changing the scorer.
    No reference enters the inference functions above.
    """
    if type(fps) is not int or fps != 25:
        raise ValueError("official tolerances require 25 fps")
    reference, predictions, period_id = map(np.asarray, (reference, predictions, period_id))
    if reference.ndim != 1 or reference.shape != predictions.shape or reference.shape != period_id.shape:
        raise ValueError("reference, predictions and period_id need equal 1D shapes")
    if not np.isin(reference, list("HADX")).all() or not np.isin(predictions, list("HADX")).all():
        raise ValueError("canonical labels must be H/A/D/X")
    if not np.issubdtype(period_id.dtype, np.integer) or not np.isin(period_id, [0, 1, 2, 3, 4]).all():
        raise ValueError("period_id must contain integers 0..4")
    for period in np.unique(period_id[period_id > 0]):
        indices = np.flatnonzero(period_id == period)
        if not (np.diff(indices) == 1).all():
            raise ValueError("periods must be contiguous")
    return evaluate_periods(reference, predictions, period_id, _official_metrics())


def evaluate_periods(reference, predictions, period_id, metrics):
    """Score validated 25-fps arrays with contiguous, positive period identities.

    External loaders and the synthetic comparator validate their own period-ID
    domains before calling this common matching boundary. Zero is ungraded.
    """
    gt = reference.copy()
    gt[period_id == 0] = "X"
    if not (gt != "X").any():
        raise ValueError("no graded reference frames")
    starts = np.flatnonzero((period_id > 0) & np.r_[True, period_id[1:] != period_id[:-1]])
    boundaries = starts[1:]
    guards = np.repeat(boundaries, max(metrics.TAUS) + 1)
    per_period = {}
    for period in np.unique(period_id[period_id > 0]):
        mask = period_id == period
        if (gt[mask] != "X").any():
            per_period[str(int(period))] = metrics.evaluate(metrics.label_codes(gt[mask]),
                                                            metrics.label_codes(predictions[mask]), bootstrap_n=0)
    gt = np.insert(gt, guards, "X")
    pred = np.insert(predictions, guards, "X")
    result = metrics.evaluate(metrics.label_codes(gt), metrics.label_codes(pred), bootstrap_n=0)
    result["source_n_frames"] = int(reference.size)
    result["period_boundary_sentinels"] = int(guards.size)
    result["per_period"] = per_period
    return _aggregate_metrics(result)


def quality_summary(data: AtdData, reference) -> dict:
    return {"n_frames": data.n_frames, "fps": data.fps, "pitch_m": list(data.pitch),
            "period_frames": {str(int(p)): int((data.period_id == p).sum()) for p in np.unique(data.period_id)},
            "record_missing_frames": int((~data.record_present).sum()),
            "ball_coverage": float(data.ball_observed.mean()),
            "player_coverage": float(data.observed.mean()),
            "slots_by_team": {team: int((data.meta.slot_team == team).sum()) for team in ("H", "A")},
            "reference_counts": {label: int((np.asarray(reference) == label).sum()) for label in "HADX"}}
