"""Local Sportec/DFL XML adapter; reference labels never enter observations.

Schema sources and tested limits are recorded in docs/EXTERNAL_VALIDATION.md.
Two streaming XML passes bound parser memory; dense C1 arrays scale with N*S.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from possession.data.atd import AtdMeta
from possession.data.external import ExternalMatch, empty_observations, file_fingerprint

PERIODS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}


def _frames(path, match_id):
    """Stream only attributes; no XML tree or reference channel is retained."""
    current = None
    stack = []
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start":
            stack.append(element)
            if element.tag == "FrameSet":
                current = dict(element.attrib)
                if current.get("GameSection") not in PERIODS:
                    raise ValueError("unknown Sportec period")
                if current.get("MatchId", match_id) != match_id:
                    raise ValueError("FrameSet MatchId disagrees with metadata")
            continue
        if element.tag == "Frame":
            if current is None:
                raise ValueError("Frame outside FrameSet")
            yield current, dict(element.attrib)
        elif element.tag == "PitchSize":
            yield None, dict(element.attrib)
        elif element.tag == "FrameSet":
            current = None
        element.clear()
        stack.pop()
        if stack:
            stack[-1].remove(element)


def _number(value):
    if value is None or value == "":
        return np.nan
    number = float(value)
    if not np.isfinite(number):
        return np.nan
    return number


def load_sportec(metadata_path, positions_path, *, match_id, fps, coordinate_origin,
                  coordinate_units, speed_units):
    try:
        return _load_sportec(metadata_path, positions_path, match_id=match_id, fps=fps,
                             coordinate_origin=coordinate_origin, coordinate_units=coordinate_units,
                             speed_units=speed_units)
    except ET.ParseError as error:
        raise ValueError(f"malformed Sportec XML: {error}") from error


def _load_sportec(metadata_path, positions_path, *, match_id, fps, coordinate_origin,
                   coordinate_units, speed_units):
    if (fps, coordinate_origin, coordinate_units, speed_units) != (25, "center", "m", "km/h"):
        raise ValueError("Sportec requires explicit 25 fps, center origin, m coordinates and km/h speeds")
    metadata_path, positions_path = Path(metadata_path), Path(positions_path)
    root = ET.parse(metadata_path).getroot()
    general = root.find(".//MatchInformation/General")
    if general is None or general.get("MatchId") != match_id:
        raise ValueError("metadata MatchId must equal manifest id")
    environment = root.find(".//MatchInformation/Environment")
    if environment is None:
        raise ValueError("metadata pitch missing")
    pitch = (float(environment.attrib["PitchX"]), float(environment.attrib["PitchY"]))
    teams = root.findall(".//MatchInformation/Teams/Team")
    by_side = {team.get("Role"): team for team in teams}
    if len(teams) != 2 or set(by_side) != {"home", "guest"}:
        raise ValueError("metadata needs exactly one home and one guest team")
    team_map, player_map, slot_team = {}, {}, []
    for role, side in (("home", "H"), ("guest", "A")):
        team = by_side[role]
        team_id = team.get("TeamId")
        if not team_id or team_id in team_map:
            raise ValueError("team ids must be nonempty and unique")
        team_map[team_id] = side
        players = team.findall("./Players/Player")
        if not players:
            raise ValueError("each team needs player metadata")
        for player in players:
            pid = player.get("PersonId")
            if not pid or pid in player_map:
                raise ValueError("player ids must be nonempty and unique")
            player_map[pid] = (len(slot_team), team_id)
            slot_team.append(side)
    referees = {node.get("PersonId") for node in root.findall(".//Referees/Referee")}
    extents = {}
    pitch_seen = False
    with positions_path.open("rb") as handle:
        for group, frame in _frames(handle, match_id):
            if group is None:
                if not np.allclose([float(frame["X"]), float(frame["Y"])], pitch, rtol=0, atol=.01):
                    raise ValueError("positions pitch disagrees with metadata pitch")
                pitch_seen = True
                continue
            n = int(frame["N"])
            if n < 0:
                raise ValueError("negative Sportec frame")
            period = PERIODS[group["GameSection"]]
            low, high = extents.get(period, (n, n))
            extents[period] = (min(low, n), max(high, n))
    if not extents or not pitch_seen:
        raise ValueError("positions need frames and pitch-size metadata")
    ordered = sorted(extents.items())
    if any(right[1][0] <= left[1][1] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("period frame counters overlap or reverse; explicit alignment required")
    offset, maximum = ordered[0][1][0], ordered[-1][1][1]
    meta = AtdMeta(by_side["home"].get("TeamName", "H"), by_side["guest"].get("TeamName", "A"),
                   np.array(slot_team, dtype="U1"))
    data = empty_observations(match_id, "sportec", fps, pitch, meta, maximum - offset + 1)
    for period, (low, high) in ordered:
        data.period_id[low - offset: high - offset + 1] = period
    reference = np.full(data.n_frames, "X", dtype="U1")
    # One duplicate bitmap per slot plus ball; no Python dict per position.
    seen = np.zeros((data.n_frames, meta.n_slots + 1), bool)
    skipped_officials = 0
    with positions_path.open("rb") as handle:
        for group, frame in _frames(handle, match_id):
            if group is None:
                continue
            team_id, pid = group.get("TeamId"), group.get("PersonId")
            if pid in referees:
                skipped_officials += 1
                continue
            ball = team_id == "BALL"
            if ball:
                slot = meta.n_slots
            else:
                if team_id not in team_map or pid not in player_map or player_map[pid][1] != team_id:
                    raise ValueError("unknown or mismatched Sportec player/team")
                slot = player_map[pid][0]
            i = int(frame["N"]) - offset
            if seen[i, slot]:
                raise ValueError("duplicate Sportec frame/object key")
            seen[i, slot] = True
            data.record_present[i] = True
            point = np.array([_number(frame.get("X")), _number(frame.get("Y"))]) / pitch + .5
            if np.isfinite(point).all() and (np.abs(point - .5) > 2).any():
                raise ValueError("coordinates inconsistent with metre pitch units")
            if ball:
                data.ball_xy[i] = point if np.isfinite(point).all() else np.nan
                data.ball_observed[i] = np.isfinite(point).all()
                status, possession = frame.get("BallStatus"), frame.get("BallPossession")
                reference[i] = "D" if status == "0" else ({"1": "H", "2": "A"}.get(possession, "X") if status == "1" else "X")
            else:
                data.xy[i, slot] = point if np.isfinite(point).all() else np.nan
                data.observed[i, slot] = np.isfinite(point).all()
                speed = _number(frame.get("S"))
                if speed < 0:
                    raise ValueError("negative Sportec speed")
                data.speed[i, slot] = speed / 3.6 if data.observed[i, slot] else np.nan
    data.provenance = {"kind": "external", "format": "sportec", "source_frame_offset": offset,
                       "source_period_extents": {str(p): list(bounds) for p, bounds in ordered},
                       "coordinate_units": "normalized", "coordinate_origin": "lower-left; y up",
                       "speed_units": "m/s", "skipped_official_records": skipped_officials,
                       "period_bounds_source": "FrameSet GameSection and observed counter extents; no GT"}
    return ExternalMatch(match_id, data, reference,
                         {"kind": "external", "format": "sportec", "metadata": file_fingerprint(metadata_path),
                          "positions": file_fingerprint(positions_path), "reference_source": "BallStatus/BallPossession",
                          "unknown_reference_frames": int((reference == "X").sum())})
