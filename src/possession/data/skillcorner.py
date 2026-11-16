"""SkillCorner open data (broadcast tracking, MIT) as an external match at 25 fps.

Observations come from ``<id>_tracking_extrapolated.jsonl`` and ``<id>_match.json``;
the reference comes only from ``<id>_phases_of_play.csv``. Inside a phase of play the
team in possession owns every frame (consecutive phases of one team form one
possession); outside every phase the ball is dead. That is controlled possession by
construction, close to the official definition, unlike last-touch provider flags.
The declared match periods define the clock. Upstream phases can extend beyond
their declared period: only a tail of a phase starting inside that period is
clipped to its end, with adjustment counts in the reference provenance. A phase
never supplies labels to another period; out-of-span frames remain invalid.
The last period may declare an inclusive endpoint exactly one source frame after
the final tracking sample. That absent endpoint is audited, never synthesised.

The source runs at 10 fps. Resampling to 25 fps is explicit and recorded: positions are
interpolated linearly between the two neighbouring source frames when both have the
object, otherwise the nearest source frame within half a source frame is used; labels
take the nearest source frame. Speeds are central differences of source positions.
Extrapolated players (``is_detected`` false) count as observed only with
``players = "extrapolated"``; the ball counts as observed only when detected.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from possession.data.atd import AtdMeta
from possession.data.external import ExternalMatch, empty_observations, file_fingerprint

SOURCE_FPS = 10
TARGET_FPS = 25


def _read_tracking(path: Path, slot_of: dict, n_slots: int):
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    frames = np.array([row["frame"] for row in rows])
    if (frames.size == 0 or any(type(row["frame"]) is not int or row["frame"] < 0 for row in rows)
            or not np.array_equal(frames, np.arange(frames[0], frames[0] + frames.size))):
        raise ValueError("SkillCorner frames must be contiguous")
    n = frames.size
    xy = np.full((n, n_slots, 2), np.nan)
    detected = np.zeros((n, n_slots), bool)
    ball = np.full((n, 2), np.nan)
    ball_detected = np.zeros(n, bool)
    for i, row in enumerate(rows):
        for player in row["player_data"]:
            slot = slot_of.get(player["player_id"])
            if slot is None or player["x"] is None or player["y"] is None:
                continue
            xy[i, slot] = (player["x"], player["y"])
            detected[i, slot] = bool(player["is_detected"])
        data = row["ball_data"]
        if data["x"] is not None and data["y"] is not None:
            ball[i] = (data["x"], data["y"])
            ball_detected[i] = data["is_detected"] is True
    return int(frames[0]), xy, detected, ball, ball_detected


def _resample(values: np.ndarray, valid: np.ndarray, position: np.ndarray, period: np.ndarray):
    """Interpolate within the target's nearest source period; never borrow across its edge."""
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, values.shape[0] - 1)
    weight = (position - low).reshape((-1,) + (1,) * (values.ndim - 1))
    nearest = np.rint(position).astype(int)
    same = ((period[low] == period[nearest]) & (period[high] == period[nearest])
            & (period[nearest] > 0))
    shape = (-1,) + (1,) * (valid.ndim - 1)
    both = valid[low] & valid[high] & same.reshape(shape)
    out = np.where(both[..., None] if values.ndim > valid.ndim else both,
                   values[low] * (1 - weight) + values[high] * weight, np.nan)
    lone = ~both & valid[nearest] & (period[nearest] > 0).reshape(shape)
    out[lone] = values[nearest][lone]
    return out, both | lone


def load_skillcorner(match_path, tracking_path, phases_path, *, match_id, players="extrapolated"):
    if players not in ("extrapolated", "detected"):
        raise ValueError('players must be "extrapolated" or "detected"')
    match_path, tracking_path, phases_path = Path(match_path), Path(tracking_path), Path(phases_path)
    match = json.loads(match_path.read_text(encoding="utf-8"))
    if f"SC-{match['id']}" != match_id:
        raise ValueError("manifest id must be SC-<SkillCorner match id>")
    home, away = match["home_team"]["id"], match["away_team"]["id"]
    slot_of, slot_team = {}, []
    for player in match["players"]:
        if player["team_id"] not in (home, away) or player["id"] in slot_of:
            raise ValueError("players need a unique id and a team of this match")
        slot_of[player["id"]] = len(slot_team)
        slot_team.append("H" if player["team_id"] == home else "A")
    pitch = (float(match["pitch_length"]), float(match["pitch_width"]))
    meta = AtdMeta(match["home_team"]["name"], match["away_team"]["name"], np.array(slot_team, dtype="U1"))
    first, xy, detected, ball, ball_detected = _read_tracking(tracking_path, slot_of, len(slot_team))
    present = np.isfinite(xy).all(axis=2)
    observed_source = present if players == "extrapolated" else present & detected

    # Source labels: X outside the periods, D outside every phase, the phase team inside.
    n_source = xy.shape[0]
    period_source = np.zeros(n_source, np.int8)
    period_stops, previous_stop = {}, 0
    missing_terminal_source_frames = 0
    if not match["match_periods"]:
        raise ValueError("SkillCorner needs declared periods")
    for period_index, period in enumerate(match["match_periods"]):
        pid, start, end = period["period"], period["start_frame"], period["end_frame"]
        if (any(type(value) is not int for value in (pid, start, end)) or not 1 <= pid <= 4
                or not first <= start < first + n_source or not start <= end <= first + n_source
                or (end == first + n_source and period_index != len(match["match_periods"]) - 1)
                or pid in period_stops
                or start - first < previous_stop):
            raise ValueError("invalid, overlapping or repeated SkillCorner period interval")
        if end == first + n_source:
            missing_terminal_source_frames = 1
        stop = min(end - first + 1, n_source)  # match metadata uses inclusive end_frame
        period_source[start - first:stop] = pid
        period_stops[pid] = stop
        previous_stop = stop
    observed_source &= (period_source > 0)[:, None]
    ball_detected &= (period_source > 0) & np.isfinite(ball).all(axis=1)
    labels_source = np.where(period_source > 0, "D", "X").astype("U1")
    n_phases, clipped_phase_count, clipped_source_frames = 0, 0, 0
    with phases_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["match_id"] != str(match["id"]):
                raise ValueError("phase belongs to another match")
            team = int(row["team_in_possession_id"])
            if team not in (home, away):
                raise ValueError("phase team is not a team of this match")
            start, end = int(row["frame_start"]) - first, int(row["frame_end"]) - first
            pid = int(row["period"])
            if not 0 <= start < end <= n_source or pid not in period_stops or period_source[start] != pid:
                raise ValueError("invalid SkillCorner phase interval or period")
            # Pinned public CSVs sometimes extend a first-half phase into the break
            # or next half. Keep the independently declared period clock authoritative.
            clipped_end = min(end, period_stops[pid])
            if clipped_end != end:
                clipped_phase_count += 1
                clipped_source_frames += end - clipped_end
            end = clipped_end
            label = "H" if team == home else "A"
            if np.isin(labels_source[start:end], ["A" if label == "H" else "H"]).any():
                raise ValueError("conflicting SkillCorner phase teams")
            labels_source[start:end] = label
            n_phases += 1
    labels_source[period_source == 0] = "X"

    # Source speeds (m/s) by central differences, only between observed neighbours.
    speed_source = np.full(present.shape, np.nan)
    same = ((period_source[2:] == period_source[1:-1]) & (period_source[:-2] == period_source[1:-1])
            & (period_source[1:-1] > 0))
    both = observed_source[2:] & observed_source[:-2] & observed_source[1:-1] & same[:, None]
    step = np.linalg.norm(xy[2:] - xy[:-2], axis=2) * SOURCE_FPS / 2
    speed_source[1:-1] = np.where(both, step, np.nan)

    n_target = int((n_source - 1) * TARGET_FPS / SOURCE_FPS) + 1
    position = np.arange(n_target) * SOURCE_FPS / TARGET_FPS
    data = empty_observations(match_id, "skillcorner", TARGET_FPS, pitch, meta, n_target)
    nearest = np.rint(position).astype(int)
    data.period_id[:] = period_source[nearest]
    reference = labels_source[nearest].copy()
    norm = np.array(pitch)
    players_xy, players_ok = _resample(xy / norm + .5, observed_source, position, period_source)
    data.xy[:] = np.where(players_ok[..., None], players_xy, np.nan)
    data.observed[:] = players_ok
    player_speed, speed_ok = _resample(speed_source, np.isfinite(speed_source), position, period_source)
    data.speed[:] = np.where(players_ok & speed_ok, player_speed, np.nan)
    ball_xy, ball_ok = _resample(ball / norm + .5, ball_detected, position, period_source)
    data.ball_xy[:] = np.where(ball_ok[:, None], ball_xy, np.nan)
    data.ball_observed[:] = ball_ok
    data.record_present[:] = data.period_id > 0
    data.provenance = {"kind": "external", "format": "skillcorner", "source_fps": SOURCE_FPS,
                       "resampling": "linear between observed neighbours in same period, else nearest within 0.5 source frames",
                       "players": players, "coordinate_units": "normalized", "speed_units": "m/s",
                       "period_bounds_source": "match.json match_periods; no GT"}
    coverage = {"source_player_samples": int(present.sum()),
                "source_detected_player_samples": int((present & detected).sum()),
                "source_extrapolated_player_samples": int((present & ~detected).sum()),
                "target_observed_player_samples": int(data.observed.sum()),
                "target_player_samples": int(data.observed.size),
                "target_observed_ball_frames": int(data.ball_observed.sum()),
                "target_frames": int(data.n_frames)}
    period_adjustments = {
        "policy": "allow only final period end_frame exactly one source frame beyond tracking; no new samples",
        "clipped_period_count": missing_terminal_source_frames,
        "missing_terminal_source_frames": missing_terminal_source_frames}
    data.provenance.update({"coverage": coverage, "includes_extrapolated_players": players == "extrapolated",
                            "period_boundary_adjustments": period_adjustments})
    return ExternalMatch(match_id, data, reference,
                         {"kind": "external", "format": "skillcorner", "match": file_fingerprint(match_path),
                          "tracking": file_fingerprint(tracking_path), "phases": file_fingerprint(phases_path),
                          "reference_source": "phases_of_play team_in_possession; D outside phases",
                          "phase_count": n_phases, "players": players, "source_match_id": str(match["id"]),
                          "period_boundary_adjustments": period_adjustments,
                          "phase_boundary_adjustments": {
                              "policy": "clip phase tail to declared period end only when start is inside; no cross-period labels",
                              "clipped_phase_count": clipped_phase_count,
                              "clipped_source_frames": clipped_source_frames,
                              "frame_count_basis": "sum of removed phase tails at source_fps",
                              "source_fps": SOURCE_FPS},
                          "coverage": coverage, "includes_extrapolated_players": players == "extrapolated",
                          "unknown_reference_frames": int((reference == "X").sum())})
