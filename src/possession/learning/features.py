"""Compact C2 features v1. Each row uses only its frame (ball speed: t-1).

Coordinates stay normalized; supplied player speed and ball displacement use m/s.
No fitted preprocessing, references, identifiers or tags enter these functions.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from possession.data.atd import AtdData
from possession.features.control import nearest_distances

FEATURE_VERSION = "compact-observations-v1"


def _mean(values: np.ndarray) -> np.ndarray:
    count = np.isfinite(values).sum(axis=1)
    return np.divide(np.nansum(values, axis=1), count,
                     out=np.full(values.shape[0], np.nan), where=count > 0)


def player_features(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    """16 player-only features for the dead/alive head, one row per input frame."""
    columns, names = [], []
    for team in ("H", "A"):
        slots = data.meta.slot_team == team
        mask = data.observed[:, slots]
        xy = np.where(mask[:, :, None], data.xy[:, slots], np.nan).astype(np.float64)
        speed = np.where(mask, data.speed[:, slots], np.nan).astype(np.float64)
        values = [mask.mean(axis=1) if slots.any() else np.zeros(data.n_frames)]
        suffixes = ["observed_fraction"]
        for axis, label in enumerate(("x", "y")):
            center = _mean(xy[:, :, axis])
            spread = np.sqrt(_mean((xy[:, :, axis] - center[:, None]) ** 2))
            values.extend((center, spread))
            suffixes.extend((f"{label}_mean_norm", f"{label}_std_norm"))
        valid_speed = np.isfinite(speed)
        speed_count = valid_speed.sum(axis=1)
        fast_fraction = np.divide(((speed > 2.0) & valid_speed).sum(axis=1), speed_count,
                                  out=np.full(data.n_frames, np.nan), where=speed_count > 0)
        values.extend((_mean(speed), fast_fraction,
                       speed_count / max(1, int(slots.sum()))))
        suffixes.extend(("speed_mean_mps", "moving_fraction", "speed_observed_fraction"))
        columns.extend(values)
        names.extend(f"{team}_{name}" for name in suffixes)
    values = np.column_stack(columns)
    values[~np.isfinite(values)] = np.nan
    return values, tuple(names)


def control_features(data: AtdData, ball_speed_frames: int = 1) -> tuple[np.ndarray, tuple[str, ...]]:
    """Player features plus observed ball, availability and nearest distances.

    No interpolation or window crosses a period. Missing coordinates/distances are
    NaN alongside masks; absence alone is not a dead-ball label. ``ball_speed_mps`` is the
    displacement over ``ball_speed_frames`` frames (1: consecutive frames, the original
    feature; more frames average out per-frame position noise).
    """
    k = int(ball_speed_frames)
    if k < 1:
        raise ValueError("ball_speed_frames must be a positive integer")
    players, names = player_features(data)
    ball = np.where(data.ball_observed[:, None], data.ball_xy, np.nan).astype(np.float64)
    dist_h, dist_a = nearest_distances(data)
    ball_speed = np.full(data.n_frames, np.nan)
    valid_pair = (data.ball_observed[k:] & data.ball_observed[:-k]
                  & (data.period_id[k:] == data.period_id[:-k]) & (data.period_id[k:] > 0))
    metres = ball * np.array(data.pitch)
    delta = np.linalg.norm(metres[k:] - metres[:-k], axis=1) * data.fps / k
    ball_speed[k:] = np.where(valid_pair, delta, np.nan)
    extra = np.column_stack((data.ball_observed, ball, dist_h, dist_a, ball_speed))
    extra[~np.isfinite(extra)] = np.nan
    return np.column_stack((players, extra)), names + (
        "ball_observed", "ball_x_norm", "ball_y_norm", "nearest_H_m", "nearest_A_m", "ball_speed_mps",
    )


# Temporal recipe v1: the same per-frame features plus context from neighbouring frames.
# Windows and thresholds are fixed a priori (physical meaning), never selected on scores.
# Every window, carry and gap stops at period boundaries; rows stay one per input frame.
TEMPORAL_FEATURE_VERSION = "temporal-observations-v1"
WINDOWS_S = (1.0, 3.0, 10.0)
CLOSE_M = 2.0  # a player this close to the ball is treated as the last team near it


def _segments(period_id: np.ndarray):
    for period in np.unique(period_id[period_id > 0]):
        yield np.flatnonzero(period_id == period)


def window_mean(values: np.ndarray, period_id: np.ndarray, frames: int) -> np.ndarray:
    """Centred NaN-aware mean over ``frames`` frames, never crossing a period."""
    out = np.full(values.shape[0], np.nan)
    half = frames // 2
    for idx in _segments(period_id):
        segment = values[idx]
        valid = np.isfinite(segment)
        total = np.r_[0.0, np.cumsum(np.where(valid, segment, 0.0))]
        count = np.r_[0, np.cumsum(valid)]
        at = np.arange(idx.size)
        lo, hi = np.clip(at - half, 0, idx.size), np.clip(at + half + 1, 0, idx.size)
        n = count[hi] - count[lo]
        out[idx] = np.divide(total[hi] - total[lo], n, out=np.full(idx.size, np.nan), where=n > 0)
    return out


def nearest_event(mask: np.ndarray, period_id: np.ndarray, fps: float):
    """Index of the last and next frame where ``mask`` holds, and seconds to each, per period.

    Indices are -1 and seconds NaN when there is no such frame in the period.
    """
    n = mask.shape[0]
    last, nxt = np.full(n, -1), np.full(n, -1)
    since, until = np.full(n, np.nan), np.full(n, np.nan)
    for idx in _segments(period_id):
        at = np.arange(idx.size)
        hit = mask[idx]
        before = np.maximum.accumulate(np.where(hit, at, -1))
        after = np.minimum.accumulate(np.where(hit, at, idx.size)[::-1])[::-1]
        has_before, has_after = before >= 0, after < idx.size
        last[idx[has_before]] = idx[before[has_before]]
        nxt[idx[has_after]] = idx[after[has_after]]
        since[idx[has_before]] = (at - before)[has_before] / fps
        until[idx[has_after]] = (after - at)[has_after] / fps
    return last, nxt, since, until


def _take(values: np.ndarray, index: np.ndarray) -> np.ndarray:
    return np.where(index >= 0, values[np.maximum(index, 0)], np.nan)


def _player_context(players: np.ndarray, names: tuple[str, ...], data: AtdData):
    columns, labels = [], []
    for name in ("H_speed_mean_mps", "A_speed_mean_mps", "H_moving_fraction", "A_moving_fraction"):
        for seconds in WINDOWS_S:
            columns.append(window_mean(players[:, names.index(name)], data.period_id, int(seconds * data.fps)))
            labels.append(f"{name}_w{seconds:g}s")
    return columns, labels


def temporal_player_features(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    """Dead/alive head: player features plus windowed collective speed. Still no ball input."""
    players, names = player_features(data)
    columns, labels = _player_context(players, names, data)
    values = np.column_stack((players, *columns)).astype(np.float32)
    return values, names + tuple(labels)


def temporal_control_features(data: AtdData, ball_speed_frames: int = 1) -> tuple[np.ndarray, tuple[str, ...]]:
    """H/A head: control features plus where the ball was last and will next be seen and who was near it."""
    base, names = control_features(data, ball_speed_frames)
    players, player_names = player_features(data)
    columns, labels = _player_context(players, player_names, data)
    margin = base[:, names.index("nearest_A_m")] - base[:, names.index("nearest_H_m")]  # > 0 favours H
    for name, series in (("margin_m", margin), ("ball_observed", base[:, names.index("ball_observed")]),
                         ("ball_x_norm", base[:, names.index("ball_x_norm")])):
        for seconds in WINDOWS_S:
            columns.append(window_mean(series, data.period_id, int(seconds * data.fps)))
            labels.append(f"{name}_w{seconds:g}s")
    last, nxt, since, until = nearest_event(np.isfinite(margin), data.period_id, data.fps)
    columns += [since, until, _take(margin, last), _take(margin, nxt)]
    labels += ["since_ball_s", "until_ball_s", "margin_at_last_ball_m", "margin_at_next_ball_m"]
    near = np.fmin(base[:, names.index("nearest_H_m")], base[:, names.index("nearest_A_m")])
    team = np.sign(margin)
    last, nxt, since, until = nearest_event(np.isfinite(margin) & (near < CLOSE_M), data.period_id, data.fps)
    columns += [_take(team, last), since, _take(team, nxt), until]
    labels += ["last_close_team", "since_close_s", "next_close_team", "until_close_s"]
    values = np.column_stack((base, *columns)).astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    return values, names + tuple(labels)


def swap_teams(data: AtdData) -> AtdData:
    """Training augmentation: home and away exchanged and the pitch rotated 180 degrees."""
    flipped = np.where(data.meta.slot_team == "H", "A", "H")
    return replace(data, meta=replace(data.meta, slot_team=flipped),
                   xy=(1 - data.xy).astype(data.xy.dtype), ball_xy=(1 - data.ball_xy).astype(data.ball_xy.dtype))


# Team-shape recipe: ball-free evidence of which team attacks, from player geometry only.
# The attacking direction of each team comes from its own mean position in the period
# (a team defends the half it stands in on average), so no label or ball enters.
SHAPE_FEATURE_VERSION = "team-shape-v2"
_CHUNK = 20_000


def _attack_sign(x: np.ndarray, period_id: np.ndarray) -> np.ndarray:
    """+1 where the team attacks towards x = 1 in that period, -1 towards x = 0, NaN unknown."""
    sign = np.full(x.shape[0], np.nan)
    for idx in _segments(period_id):
        centre = np.nanmean(x[idx]) if np.isfinite(x[idx]).any() else np.nan
        sign[idx] = np.nan if not np.isfinite(centre) else (1.0 if centre < .5 else -1.0)
    return sign


def _forward_velocity(x: np.ndarray, period_id: np.ndarray, fps: float, length: float) -> np.ndarray:
    """Per-slot x velocity (m/s) by a centred 4-frame difference, never across a period."""
    v = np.full(x.shape, np.nan)
    same = (period_id[4:] == period_id[:-4]) & (period_id[4:] > 0)
    v[2:-2] = np.where(same[:, None], (x[4:] - x[:-4]) * length * fps / 4, np.nan)
    return v


def team_shape_features(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    length = float(data.pitch[0])
    scale = np.array(data.pitch, dtype=np.float32)
    columns, names, advance, forward = [], [], {}, {}
    teams = {team: data.meta.slot_team == team for team in ("H", "A")}
    for team, slots in teams.items():
        mask = data.observed[:, slots]
        x = np.where(mask, data.xy[:, slots, 0], np.nan).astype(np.float64)
        sign = _attack_sign(x, data.period_id)
        ahead = (x - .5) * sign[:, None] * length  # metres beyond the halfway line, towards attack
        advance[team] = _mean(ahead)
        velocity = _forward_velocity(x, data.period_id, data.fps, length) * sign[:, None]
        forward[team] = _mean(velocity)
        count = np.isfinite(ahead).sum(axis=1)
        in_half = np.divide((ahead > 0).sum(axis=1), count, out=np.full(data.n_frames, np.nan), where=count > 0)
        front = np.where(count > 0, np.max(np.where(np.isfinite(ahead), ahead, -np.inf),
                                         axis=1, initial=-np.inf), np.nan)
        back = np.where(count > 0, np.min(np.where(np.isfinite(ahead), ahead, np.inf),
                                        axis=1, initial=np.inf), np.nan)
        columns += [advance[team], in_half, front, back]
        names += [f"{team}_advance_m", f"{team}_attacking_half_fraction", f"{team}_front_m", f"{team}_back_m"]
        for seconds in (1.0, 3.0):
            columns.append(window_mean(forward[team], data.period_id, int(seconds * data.fps)))
            names.append(f"{team}_forward_speed_w{seconds:g}s")
    # Pressure: mean distance from each player to the nearest opponent, per team.
    pressure = {team: np.full(data.n_frames, np.nan) for team in teams}
    for start in range(0, data.n_frames, _CHUNK):
        sl = slice(start, start + _CHUNK)
        points = {team: np.where(data.observed[sl, slots, None], data.xy[sl, slots] * scale, np.nan)
                  for team, slots in teams.items()}
        if points["H"].shape[1] and points["A"].shape[1]:
            d = np.linalg.norm(points["H"][:, :, None] - points["A"][:, None], axis=3)
            d = np.where(np.isfinite(d), d, np.inf)  # unseen pairs never count as nearest
            for team, axis in (("H", 2), ("A", 1)):
                nearest = d.min(axis=axis).astype(np.float64)
                # A missing player or a missing opposing team has no distance.
                # Infinity is only a sentinel for pairwise minimisation, never a mean input.
                nearest[~np.isfinite(nearest)] = np.nan
                pressure[team][sl] = _mean(nearest)
    for team in teams:
        p = pressure[team]
        columns += [p, window_mean(p, data.period_id, int(3 * data.fps))]
        names += [f"{team}_nearest_opponent_m", f"{team}_nearest_opponent_m_w3s"]
    columns += [advance["H"] - advance["A"], forward["H"] - forward["A"],
                window_mean(forward["H"] - forward["A"], data.period_id, int(3 * data.fps))]
    names += ["advance_diff_m", "forward_speed_diff", "forward_speed_diff_w3s"]
    values = np.column_stack(columns).astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    return values, tuple(names)


def shape_player_features(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    a, names_a = temporal_player_features(data)
    b, names_b = team_shape_features(data)
    return np.column_stack((a, b)), names_a + names_b


def shape_control_features(data: AtdData, ball_speed_frames: int = 1) -> tuple[np.ndarray, tuple[str, ...]]:
    a, names_a = temporal_control_features(data, ball_speed_frames)
    b, names_b = team_shape_features(data)
    return np.column_stack((a, b)), names_a + names_b


# On-pitch normalisation: availability features counted against the 11 players a team
# fields, not against the provider's roster slots (20 in DFL, 18 in SkillCorner, 22-24 in
# Metrica ATD). The slot count is a provider artefact; with it a model trained on one
# provider reads another provider's full visibility as missing players.
ONPITCH_FEATURE_VERSION = "onpitch-11-v1"
ONPITCH_PLAYERS = 11
_AVAILABILITY = ("observed_fraction", "speed_observed_fraction")


def on_pitch(values: np.ndarray, names: tuple[str, ...], data: AtdData) -> np.ndarray:
    out = values.copy()
    for team in ("H", "A"):
        slots = int((data.meta.slot_team == team).sum())
        for suffix in _AVAILABILITY:
            column = names.index(f"{team}_{suffix}")
            out[:, column] = values[:, column] * slots / ONPITCH_PLAYERS
    return out


def onpitch_player_features(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    values, names = shape_player_features(data)
    return on_pitch(values, names, data), names


def onpitch_control_features(data: AtdData, ball_speed_frames: int = 1) -> tuple[np.ndarray, tuple[str, ...]]:
    values, names = shape_control_features(data, ball_speed_frames)
    return on_pitch(values, names, data), names


# Ball speed over 5 frames (0.2 s at 25 fps): with independent per-frame position noise the
# one-frame difference is mostly noise (0.5 m noise gives about 18 m/s); 5 frames cut it 5-fold.
# Fixed a priori from the code audit of 8 Nov, not selected on scores.
BALL_SPEED_FEATURE_VERSION = "ball-speed-5-frames-v1"
BALL_SPEED_FRAMES = 5


def onpitch_control_features_speed5(data: AtdData) -> tuple[np.ndarray, tuple[str, ...]]:
    """``onpitch_control_features`` with the ball speed over ``BALL_SPEED_FRAMES`` frames."""
    return onpitch_control_features(data, BALL_SPEED_FRAMES)
