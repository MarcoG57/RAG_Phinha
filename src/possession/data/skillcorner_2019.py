"""SkillCorner 2019/20 open data (commit 1ed50b7, 2019/20, MIT) as an external match at 25 fps.

Broadcast tracking at 10 fps from ``structured_data.json`` and ``match_data.json``. Identified
players (``trackable_object`` of a listed player of either team) get a slot; objects with only
``group_name`` (unidentified) and referees are ignored; the ball is the match's ball object.
There is no detected/extrapolated flag: an object present in a frame counts as observed.

Reference: the per-frame automatic ``possession.group`` ("home team" -> H, "away team" -> A).
Every other frame is X (unknown): null possession (no tracking during replays/close-ups, ball in
flight or dead ball) and frames outside the declared periods. The source has NO dead-ball label,
so these matches must never supervise the dead head (``dead_supervision`` false in the provenance,
honoured by ``workflow.head_inputs``); this module never emits D.
Resampling 10 -> 25 fps and speeds follow ``possession.data.skillcorner`` exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from possession.data.atd import AtdMeta
from possession.data.external import ExternalMatch, empty_observations, file_fingerprint
from possession.data.skillcorner import SOURCE_FPS, TARGET_FPS, _resample

GROUP = {"home team": "H", "away team": "A"}


def load_skillcorner_2019(match_path, tracking_path, *, match_id) -> ExternalMatch:
    match_path, tracking_path = Path(match_path), Path(tracking_path)
    match = json.loads(match_path.read_text(encoding="utf-8"))
    if f"SCL-{match['id']}" != match_id:
        raise ValueError("manifest id must be SCL-<SkillCorner match id>")
    home, away = match["home_team"]["id"], match["away_team"]["id"]
    slot_of, slot_team = {}, []
    for player in match["players"]:
        if player["team_id"] not in (home, away) or player["trackable_object"] in slot_of:
            raise ValueError("players need a unique trackable object and a team of this match")
        slot_of[player["trackable_object"]] = len(slot_team)
        slot_team.append("H" if player["team_id"] == home else "A")
    ball_id = match["ball"]["trackable_object"]
    pitch = (float(match["pitch_length"]), float(match["pitch_width"]))
    meta = AtdMeta(match["home_team"]["name"], match["away_team"]["name"], np.array(slot_team, dtype="U1"))

    rows = json.loads(tracking_path.read_text(encoding="utf-8"))
    frames = np.array([row["frame"] for row in rows])
    if frames.size == 0 or not np.array_equal(frames, np.arange(frames[0], frames[0] + frames.size)):
        raise ValueError("legacy SkillCorner frames must be contiguous")
    n_source, n_slots = frames.size, len(slot_team)
    xy = np.full((n_source, n_slots, 2), np.nan)
    ball = np.full((n_source, 2), np.nan)
    period_source = np.array([row["period"] or 0 for row in rows], dtype=np.int8)
    labels_source = np.array([GROUP.get(row["possession"]["group"], "X") for row in rows], dtype="U1")
    for i, row in enumerate(rows):
        for obj in row["data"]:
            key = obj.get("trackable_object")
            if key is None or obj.get("x") is None or obj.get("y") is None:
                continue
            if key == ball_id:
                ball[i] = (obj["x"], obj["y"])
            elif key in slot_of:
                xy[i, slot_of[key]] = (obj["x"], obj["y"])
    for pid in np.unique(period_source[period_source > 0]):
        if not (np.diff(np.flatnonzero(period_source == pid)) == 1).all():
            raise ValueError("legacy SkillCorner periods must be contiguous")
    labels_source[period_source == 0] = "X"
    observed_source = np.isfinite(xy).all(axis=2) & (period_source > 0)[:, None]
    ball_observed = np.isfinite(ball).all(axis=1) & (period_source > 0)

    speed_source = np.full(observed_source.shape, np.nan)
    same = ((period_source[2:] == period_source[1:-1]) & (period_source[:-2] == period_source[1:-1])
            & (period_source[1:-1] > 0))
    both = observed_source[2:] & observed_source[:-2] & observed_source[1:-1] & same[:, None]
    step = np.linalg.norm(xy[2:] - xy[:-2], axis=2) * SOURCE_FPS / 2
    speed_source[1:-1] = np.where(both, step, np.nan)

    n_target = int((n_source - 1) * TARGET_FPS / SOURCE_FPS) + 1
    position = np.arange(n_target) * SOURCE_FPS / TARGET_FPS
    data = empty_observations(match_id, "skillcorner_legacy", TARGET_FPS, pitch, meta, n_target)
    nearest = np.rint(position).astype(int)
    data.period_id[:] = period_source[nearest]
    reference = labels_source[nearest].copy()
    norm = np.array(pitch)
    players_xy, players_ok = _resample(xy / norm + .5, observed_source, position, period_source)
    data.xy[:] = np.where(players_ok[..., None], players_xy, np.nan)
    data.observed[:] = players_ok
    player_speed, speed_ok = _resample(speed_source, np.isfinite(speed_source), position, period_source)
    data.speed[:] = np.where(players_ok & speed_ok, player_speed, np.nan)
    ball_xy, ball_ok = _resample(ball / norm + .5, ball_observed, position, period_source)
    data.ball_xy[:] = np.where(ball_ok[:, None], ball_xy, np.nan)
    data.ball_observed[:] = ball_ok
    data.record_present[:] = data.period_id > 0
    data.provenance = {"kind": "external", "format": "skillcorner_legacy", "source_fps": SOURCE_FPS,
                       "resampling": "as possession.data.skillcorner", "coordinate_units": "normalized",
                       "speed_units": "m/s"}
    return ExternalMatch(match_id, data, reference,
                         {"kind": "external", "format": "skillcorner_legacy", "source": "SkillCorner/opendata@1ed50b7",
                          "dead_supervision": False,
                          "license": "MIT", "match": file_fingerprint(match_path),
                          "tracking": file_fingerprint(tracking_path),
                          "reference_source": "automatic per-frame possession.group; null -> X; never D",
                          "dead_head": "excluded (no dead-ball supervision)",
                          "unknown_reference_frames": int((reference == "X").sum())})
