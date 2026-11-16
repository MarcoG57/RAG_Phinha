"""External corpus boundary: observations and evaluation references stay separate."""
from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from possession.data.atd import AtdData, AtdMeta

MAX_EXTERNAL_SLOTS = 64  # includes substitutes; above the 40 slots in the open DFL matches
MAX_DENSE_INPUT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ExternalMatch:
    match_id: str
    observations: AtdData
    reference: np.ndarray
    provenance: dict

    def __post_init__(self):
        if not self.match_id or self.reference.shape != (self.observations.n_frames,):
            raise ValueError("match id and one reference per frame are required")
        if not np.isin(self.reference, list("HADX")).all():
            raise ValueError("reference labels must be H/A/D/X")


def file_fingerprint(path: Path) -> dict:
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "size_bytes": path.stat().st_size}


def empty_observations(match_id, source, fps, pitch, meta, n_frames):
    """Allocate C1 arrays; absence always remains NaN plus a false mask."""
    if type(fps) is not int or fps != 25:
        raise ValueError("external evaluation requires native 25 fps; no resampling is implicit")
    pitch = tuple(float(v) for v in pitch)
    if len(pitch) != 2 or not (50 <= pitch[0] <= 130 and 30 <= pitch[1] <= 100):
        raise ValueError("pitch dimensions must be plausible metres")
    if type(n_frames) is not int or not 0 < n_frames <= 2_000_000:
        raise ValueError("frame span must be 1..2000000; inspect source counters before allocating")
    n, s = n_frames, meta.n_slots
    if not 1 <= s <= MAX_EXTERNAL_SLOTS:
        raise ValueError(f'external slots must be 1..{MAX_EXTERNAL_SLOTS}; inspect player metadata')
    # C1: 13*S+15 bytes/frame; Sportec adds reference U1 and duplicate bitmap S+1.
    # This bounds input arrays, not inference/training temporaries or total process RAM.
    dense_bytes = n * (14 * s + 20)
    if dense_bytes > MAX_DENSE_INPUT_BYTES:
        raise ValueError(f'external dense input budget exceeded: {dense_bytes} > {MAX_DENSE_INPUT_BYTES} bytes')
    return AtdData(match_id, source, fps, pitch, meta, np.arange(n, dtype=np.int32),
                   np.zeros(n, bool), np.zeros(n, np.int8), np.full((n, 2), np.nan, np.float32),
                   np.zeros(n, bool), np.full((n, s, 2), np.nan, np.float32),
                   np.full((n, s), np.nan, np.float32), np.zeros((n, s), bool), periods_known=True)


def _load_synthetic(path, match_id):
    raw = json.loads(path.read_text(encoding="utf-8"))
    teams = raw["slot_team"]
    if not teams or not set(teams) <= {"H", "A"} or set(teams) != {"H", "A"}:
        raise ValueError("synthetic slots require both H and A")
    data = empty_observations(match_id, "synthetic", raw["fps"], raw["pitch"],
                              AtdMeta("Synthetic home", "Synthetic away", np.array(teams)), raw["n_frames"])
    for period, start, stop in raw["periods"]:
        if type(period) is not int or not 1 <= period <= 4 or not 0 <= start < stop <= data.n_frames:
            raise ValueError("invalid synthetic period interval")
        if (data.period_id[start:stop] != 0).any() or (data.period_id == period).any():
            raise ValueError("overlapping or repeated synthetic period")
        data.period_id[start:stop] = period
    seen = set()
    for row in raw["records"]:
        frame = row["frame"]
        if type(frame) is not int or not 0 <= frame < data.n_frames or frame in seen:
            raise ValueError("invalid/duplicate synthetic frame")
        seen.add(frame)
        data.record_present[frame] = True
        if row.get("ball") is not None:
            data.ball_xy[frame] = row["ball"]
            data.ball_observed[frame] = np.isfinite(data.ball_xy[frame]).all()
        if len(row["players"]) != data.meta.n_slots:
            raise ValueError("one player entry per metadata slot is required")
        for slot, point in enumerate(row["players"]):
            if point is not None:
                data.xy[frame, slot] = point[:2]
                data.speed[frame, slot] = point[2] if len(point) > 2 else np.nan
                data.observed[frame, slot] = np.isfinite(data.xy[frame, slot]).all()
    data.provenance = {"kind": "synthetic", "coordinate_units": "normalized", "speed_units": "m/s"}
    reference = np.array(raw["reference"])
    if reference.shape != (data.n_frames,) or not np.isin(reference, list("HADX")).all():
        raise ValueError("synthetic reference needs one canonical H/A/D/X per frame")
    reference = reference.astype("U1")
    reference[data.period_id == 0] = "X"
    return ExternalMatch(match_id, data, reference,
                         {"kind": "synthetic", "fixture": file_fingerprint(path)})


def _read_manifest(manifest_path: Path):
    manifest_path = Path(manifest_path).resolve()
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    matches = manifest.get("matches", [])
    if not isinstance(matches, list) or not all(isinstance(row, dict) for row in matches):
        raise ValueError("manifest matches must be tables")
    ids = [row.get("id") for row in matches]
    if not matches or any(not isinstance(mid, str) or not mid for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest needs nonempty matches with unique string ids")
    if not isinstance(manifest.get("provenance", {}), dict) or any(
            not isinstance(row.get("provenance", {}), dict) for row in matches):
        raise ValueError("manifest provenance must be tables")
    return manifest_path, manifest, matches


def _local_path(manifest_path: Path, row: dict, key: str) -> Path:
    value = row[key]
    if not isinstance(value, str) or not value or Path(value).is_absolute() or "://" in value:
        raise ValueError(f"{key} must be a local path relative to the manifest")
    return (manifest_path.parent / value).resolve()


def iter_matches(manifest_path: Path) -> Iterator[ExternalMatch]:
    """Yield dense matches; acquisition audit belongs before runner-generated variants.

    Paths are manifest-relative. Preserve corpus and per-match declarations under
    ``manifest_provenance`` without replacing the source adapter's own provenance.
    """
    manifest_path, manifest, matches = _read_manifest(manifest_path)
    for row in matches:
        def local_path(key):
            return _local_path(manifest_path, row, key)
        if row["format"] == "synthetic":
            match = _load_synthetic(local_path("path"), row["id"])
        elif row["format"] == "sportec":
            from possession.data.sportec import load_sportec
            match = load_sportec(local_path("metadata"), local_path("positions"), match_id=row["id"],
                                **{key: row[key] for key in ("fps", "coordinate_origin", "coordinate_units", "speed_units")})
        elif row["format"] == "skillcorner":
            from possession.data.skillcorner import load_skillcorner
            match = load_skillcorner(local_path("match"), local_path("tracking"), local_path("phases"),
                                   match_id=row["id"], players=row.get("players", "extrapolated"))
        elif row["format"] == "skillcorner_legacy":
            from possession.data.skillcorner_2019 import load_skillcorner_2019
            match = load_skillcorner_2019(local_path("match"), local_path("tracking"), match_id=row["id"])
        else:
            raise ValueError("supported formats are synthetic, sportec, skillcorner, skillcorner_legacy")
        yield replace(match, provenance={**match.provenance, "manifest_provenance": {
            "corpus": manifest.get("provenance", {}), "match": row.get("provenance", {})}})
