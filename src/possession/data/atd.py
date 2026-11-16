"""Lector del tracking automático (ATD) alineado al frame de vídeo (contratos C0/C1).

El ATD oficial es un XML de metadatos y un texto con una línea por registro:

    <frame_id>:<x,y,s>;<x,y,s>;...;<x,y,s>:<x,y,...>

con un grupo por slot de tracking y un grupo final para el balón. Las coordenadas
están normalizadas a 0–1 y una observación ausente es ``NaN``.

Decisiones:

- ``frame = frame_id - 1`` (C0). No se aplican los offsets de ``games.py``.
- Arrays densos indexados por frame. Un registro ausente en el fichero deja su fila
  sin observar y ``record_present=False``; nunca se rellena con 0.
- Equipos desde metadatos: el primer ``Team`` del XML es H (convención FIFA/EPTS,
  la misma del baseline oficial). Un slot es un tracklet, no una persona.
- Coordenadas normalizadas en memoria; metros sólo en las tablas C1, escalando cada
  eje por separado con ``pitch``.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import tempfile
import warnings
import xml.etree.ElementTree as ET
from zipfile import BadZipFile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_DIR = REPO_ROOT / ".official"
CACHE_DIR = REPO_ROOT / "artifacts" / "cache"
CACHE_VERSION = 3

FIELD_SEP = ":"
GROUP_SEP = ";"
VALUE_SEP = ","
HOME, AWAY = "H", "A"
_INTEGER = re.compile(r"^[0-9]+$")
_MAX_AUDIT_ERRORS = 20
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class AtdMeta:
    home_name: str
    away_name: str
    slot_team: np.ndarray  # (S,) "H"/"A" por slot, en el orden de los grupos del raw

    @property
    def n_slots(self) -> int:
        return int(self.slot_team.shape[0])


@dataclass
class AtdData:
    """Tracking de una fuente en arrays densos de longitud ``n_frames``."""

    dataset_id: str
    source: str
    fps: int
    pitch: tuple[float, float]  # (largo, ancho) en metros
    meta: AtdMeta
    frame: np.ndarray  # (N,) int32, 0..N-1
    record_present: np.ndarray  # (N,) bool, el raw trae un registro para el frame
    period_id: np.ndarray  # (N,) int8, 1/2 en partes conocidas, 0 fuera o desconocido
    ball_xy: np.ndarray  # (N, 2) float32 normalizado, NaN si ausente
    ball_observed: np.ndarray  # (N,) bool
    xy: np.ndarray  # (N, S, 2) float32 normalizado, NaN si ausente
    speed: np.ndarray  # (N, S) float32 m/s, NaN si ausente
    observed: np.ndarray  # (N, S) bool
    periods_known: bool = False
    provenance: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.frame.shape[0])

    @property
    def time_s(self) -> np.ndarray:
        return self.frame / float(self.fps)

    def to_frames_df(self) -> pd.DataFrame:
        """Tabla C1 ``frames``: una fila por frame, balón en metros."""
        length, width = self.pitch
        return pd.DataFrame({
            "dataset_id": self.dataset_id,
            "source": self.source,
            "frame": self.frame,
            "time_s": self.time_s,
            "period_id": self.period_id,
            "ball_x_m": self.ball_xy[:, 0].astype(np.float64) * length,
            "ball_y_m": self.ball_xy[:, 1].astype(np.float64) * width,
            "ball_observed": self.ball_observed,
        })

    def to_tracks_df(self, observed_only: bool = True) -> pd.DataFrame:
        """Tabla C1 ``tracks``: una fila por (frame, slot), en metros.

        Con ``observed_only`` se omiten las combinaciones sin observación; la ausencia
        sigue siendo explícita porque esas claves no aparecen y ``observed`` no se inventa.
        """
        length, width = self.pitch
        n, s = self.observed.shape
        frames = np.repeat(self.frame, s)
        slots = np.tile(np.arange(1, s + 1, dtype=np.int16), n)
        teams = np.tile(self.meta.slot_team, n)
        observed = self.observed.reshape(-1)
        x = self.xy[:, :, 0].reshape(-1).astype(np.float64) * length
        y = self.xy[:, :, 1].reshape(-1).astype(np.float64) * width
        keep = observed if observed_only else np.ones_like(observed)
        return pd.DataFrame({
            "dataset_id": self.dataset_id,
            "source": self.source,
            "frame": frames[keep],
            "slot_id": slots[keep],
            "team": teams[keep],
            "x_m": x[keep],
            "y_m": y[keep],
            "observed": observed[keep],
        })


def load_metadata(meta_path: str | os.PathLike) -> dict:
    """Return team sides and names in the XML order used by the raw slots."""
    try:
        root = ET.parse(meta_path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"cannot read ATD metadata: {error}") from error

    teams = list(root.iter("Team"))
    team_ids = [team.get("id") for team in teams]
    team_names = [(team.findtext("Name") or "").strip() for team in teams]
    if len(teams) != 2 or any(not value for value in team_ids):
        raise ValueError("ATD metadata must contain exactly two teams (se esperaban 2 equipos) with ids")
    if len(set(team_ids)) != 2:
        raise ValueError("ATD metadata team ids must be unique")

    players = list(root.iter("Player"))
    if not players:
        raise ValueError("ATD metadata contains no Player slots")
    if any(not name for name in team_names) or len({name.casefold() for name in team_names}) != 2:
        raise ValueError("ATD metadata must contain two unique team names")

    known_ids = set(team_ids)
    slot_teams = []
    for player in players:
        team_id = player.get("teamId")
        if team_id not in known_ids:
            raise ValueError("ATD metadata contains an unknown teamId")
        slot_teams.append(HOME if team_id == team_ids[0] else AWAY)
    return {"team_of_slot": slot_teams, "team_names": team_names}


def read_meta(meta_path: str | os.PathLike) -> AtdMeta:
    """Equipos y asignación slot→equipo desde el XML de metadatos."""
    metadata = load_metadata(meta_path)
    return AtdMeta(*metadata["team_names"], np.array(metadata["team_of_slot"], dtype="<U1"))


def _frame_id(line: str) -> int:
    prefix = line.partition(FIELD_SEP)[0].strip()
    if not _INTEGER.fullmatch(prefix):
        raise ValueError("invalid frame_id")
    return int(prefix)


def _number(token: str) -> float | None:
    value = token.strip()
    if value.casefold() == "nan":
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError("invalid numeric field") from error
    if not math.isfinite(number):
        raise ValueError("infinite numeric field")
    if abs(number) > _FLOAT32_MAX:
        raise ValueError("numeric field exceeds float32 range")
    return number


def _group(text: str, name: str) -> tuple[float | None, float | None, float | None]:
    values = text.split(VALUE_SEP)
    if len(values) != 3:
        raise ValueError(f"{name} must have x,y,speed fields")
    return tuple(_number(value) for value in values)


def _parse_record(line: str) -> tuple[int, list[tuple], tuple]:
    fields = line.rstrip("\r\n").split(FIELD_SEP)
    if len(fields) != 3:
        raise ValueError("record must have frame, players and ball fields")
    frame_id = _frame_id(fields[0] + FIELD_SEP)
    if not fields[1]:
        raise ValueError("record has no player groups")
    player_groups = [
        _group(value, f"slot {slot}") for slot, value in enumerate(fields[1].split(GROUP_SEP), 1)
    ]
    ball = _group(fields[2], "ball")
    return frame_id, player_groups, ball


def _parse_line(line: str, n_slots: int, line_no: int) -> tuple[int, np.ndarray, np.ndarray]:
    try:
        frame_id, groups, ball = _parse_record(line)
    except ValueError as error:
        message = str(error)
        if message == "record must have frame, players and ball fields":
            message = "se esperaban 3 campos separados por ':'"
        elif message == "invalid frame_id":
            message = "frame_id no entero"
        elif message == "invalid numeric field":
            message = "valor no numérico"
        elif message == "infinite numeric field":
            message = "valor infinito"
        elif message == "numeric field exceeds float32 range":
            message = "valor fuera del rango float32"
        elif message == "record has no player groups":
            message = "sin grupos de jugador"
        elif message.startswith("slot "):
            message = f"slot {message.split(' ', 2)[1]} debe tener 3 valores (x,y,s)"
        elif message.startswith("ball "):
            message = "el balón necesita 3 valores (x,y,s)"
        raise ValueError(f"línea {line_no}: {message}") from None
    if len(groups) != n_slots:
        raise ValueError(f"línea {line_no}: {len(groups)} grupos de jugador, metadatos indican {n_slots}")
    players = np.array(groups, dtype=np.float64)
    ball_xy = np.array(ball[:2], dtype=np.float64)
    return frame_id, players, ball_xy


def _expected_count(value, name: str, *, allow_zero: bool = False) -> int:
    if type(value) is not int or value < (0 if allow_zero else 1):
        qualifier = "a non-negative" if allow_zero else "a positive"
        raise ValueError(f"{name} must be {qualifier} integer")
    return value


def _audit_stream(stream, expected_frames: int, n_slots: int | None = None) -> dict:
    errors = []
    error_count = 0
    seen = bytearray(expected_frames)
    seen_count = 0
    n_records = 0
    last_frame_id = None
    first_frame_id = None
    slot_count = n_slots
    counts = {
        "invalid_ids": 0, "duplicates": 0, "out_of_order": 0,
        "out_of_range": 0, "structural_errors": 0, "sequence_mismatches": 0,
    }

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < _MAX_AUDIT_ERRORS:
            errors.append(message)

    for line_number, line in enumerate(stream, 1):
        n_records += 1
        if not line.strip():
            counts["structural_errors"] += 1
            add_error(f"line {line_number}: empty record")
            continue
        try:
            frame_id = _frame_id(line)
        except ValueError as error:
            counts["invalid_ids"] += 1
            add_error(f"line {line_number}: {error}")
            continue

        if first_frame_id is None:
            first_frame_id = frame_id
        if frame_id != n_records:
            counts["sequence_mismatches"] += 1
            add_error(f"line {line_number}: expected frame_id {n_records}, found {frame_id}")
        if last_frame_id is not None and frame_id < last_frame_id:
            counts["out_of_order"] += 1
            add_error(f"line {line_number}: frame_id is out of order")
        last_frame_id = frame_id
        if not 1 <= frame_id <= expected_frames:
            counts["out_of_range"] += 1
            add_error(f"line {line_number}: frame_id is outside the expected range")
        elif seen[frame_id - 1]:
            counts["duplicates"] += 1
            add_error(f"line {line_number}: duplicate frame_id {frame_id}")
        else:
            seen[frame_id - 1] = 1
            seen_count += 1

        try:
            _, players, _ = _parse_record(line)
        except ValueError as error:
            counts["structural_errors"] += 1
            add_error(f"line {line_number}: {error}")
            continue
        if slot_count is None:
            slot_count = len(players)
        elif len(players) != slot_count:
            counts["structural_errors"] += 1
            add_error(f"line {line_number}: expected {slot_count} player slots, found {len(players)}")

    missing_count = expected_frames - seen_count
    if n_records != expected_frames:
        add_error(f"record count {n_records} does not match expected {expected_frames}")
    if missing_count:
        add_error(f"missing {missing_count} expected frame ids")
    if error_count > len(errors):
        errors.append(f"{error_count - len(errors)} additional errors omitted")

    return {
        "valid": error_count == 0,
        "expected_frames": expected_frames,
        "n_records": n_records,
        "n_slots": slot_count,
        "first_frame_id": first_frame_id,
        "last_frame_id": last_frame_id,
        "missing_frames": missing_count,
        "counts": counts,
        "error_count": error_count,
        "errors": errors,
    }


def audit_atd(path: str | os.PathLike, expected_frames: int, n_slots: int | None = None) -> dict:
    """Check IDs, row structure and slot count without retaining raw records."""
    _expected_count(expected_frames, "expected_frames")
    if n_slots is not None:
        _expected_count(n_slots, "n_slots")
    with open(path, encoding="utf-8", newline="") as stream:
        return _audit_stream(stream, expected_frames, n_slots)


def _validate_settings(settings: Mapping) -> tuple[dict, int, list[tuple[str | int, int, int]]]:
    required = {
        "dataset_id", "source", "fps", "expected_frames", "pitch_length_m",
        "pitch_width_m", "periods",
    }
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping")
    missing = required - settings.keys()
    if missing:
        raise ValueError("missing settings: " + ", ".join(sorted(missing)))
    if not isinstance(settings["dataset_id"], str) or not settings["dataset_id"].strip():
        raise ValueError("dataset_id must be a non-empty string")
    if settings["source"] != "tactical":
        raise ValueError("only the tactical source is configured")
    expected = _expected_count(settings["expected_frames"], "expected_frames")

    numbers = {}
    for name in ("fps", "pitch_length_m", "pitch_width_m"):
        if isinstance(settings[name], bool):
            raise ValueError(f"{name} must be positive and finite")
        try:
            number = float(settings[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be positive and finite") from error
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be positive and finite")
        numbers[name] = number

    periods = settings["periods"]
    if not isinstance(periods, (list, tuple)) or not periods:
        raise ValueError("periods must be a non-empty list")
    validated = []
    previous_stop = 0
    identifiers = set()
    for period in periods:
        if not isinstance(period, Mapping) or not {"period_id", "start", "stop"} <= period.keys():
            raise ValueError("each period needs period_id, start and stop")
        period_id = period["period_id"]
        if not isinstance(period_id, (str, int)) or isinstance(period_id, bool) or period_id == "":
            raise ValueError("period_id must be a non-empty string or integer")
        if period_id in identifiers:
            raise ValueError("period_id values must be unique")
        start = _expected_count(period["start"], "period start", allow_zero=True)
        stop = _expected_count(period["stop"], "period stop")
        if start >= stop or stop > expected or start < previous_stop:
            raise ValueError("periods must be ordered, non-overlapping and inside the frame range")
        identifiers.add(period_id)
        validated.append((period_id, start, stop))
        previous_stop = stop
    return numbers, expected, validated


def iter_atd(path: str | os.PathLike, metadata: Mapping, settings: Mapping):
    """Yield C1 rows after strict preflight, sharing the dense reader's record parser."""
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("team_of_slot"), list):
        raise ValueError("metadata must include team_of_slot in XML order")
    teams = metadata["team_of_slot"]
    if not teams or any(team not in {HOME, AWAY} for team in teams):
        raise ValueError("metadata team_of_slot must contain H/A sides")
    numbers, expected_frames, periods = _validate_settings(settings)

    with open(path, encoding="utf-8", newline="") as stream:
        report = _audit_stream(stream, expected_frames, len(teams))
        if not report["valid"]:
            raise ValueError("invalid ATD: " + "; ".join(report["errors"]))
        stream.seek(0)
        for line in stream:
            frame_id, players, ball = _parse_record(line)
            frame = frame_id - 1
            period_id = next(
                (identifier for identifier, start, stop in periods if start <= frame < stop), None
            )
            ball_observed = ball[0] is not None and ball[1] is not None
            frame_row = {
                "dataset_id": settings["dataset_id"],
                "source": settings["source"],
                "frame": frame,
                "time_s": frame / numbers["fps"],
                "period_id": period_id,
                "ball_x_m": ball[0] * numbers["pitch_length_m"] if ball_observed else None,
                "ball_y_m": ball[1] * numbers["pitch_width_m"] if ball_observed else None,
                "ball_observed": ball_observed,
            }
            track_rows = []
            for slot_id, (team, position) in enumerate(zip(teams, players)):
                observed = position[0] is not None and position[1] is not None
                track_rows.append({
                    "dataset_id": settings["dataset_id"],
                    "source": settings["source"],
                    "frame": frame,
                    "slot_id": slot_id,
                    "team": team,
                    "x_m": position[0] * numbers["pitch_length_m"] if observed else None,
                    "y_m": position[1] * numbers["pitch_width_m"] if observed else None,
                    "observed": observed,
                })
            yield frame_row, track_rows


def _period_array(n_frames: int, periods: list[tuple[int, int, int]] | None) -> np.ndarray:
    out = np.zeros(n_frames, dtype=np.int8)
    for period, first, last in periods or ():
        out[max(first, 0):min(last + 1, n_frames)] = period
    return out


def read_atd(
    meta_path: str | os.PathLike,
    raw_path: str | os.PathLike,
    *,
    source: str,
    pitch: tuple[float, float],
    fps: int,
    n_frames: int | None = None,
    periods: list[tuple[int, int, int]] | None = None,
    dataset_id: str = "BAR-ATH",
) -> AtdData:
    """Lee el par XML/TXT en arrays densos.

    ``n_frames`` fija la longitud esperada (frames de vídeo de la fuente). ``periods``
    es ``[(period_id, first_frame, last_frame)]`` en frames de vídeo inclusivos; sin
    ella ``period_id`` queda en 0 y ``periods_known=False``.
    """
    meta = read_meta(meta_path)
    s = meta.n_slots
    ids: list[int] = []
    rows_players: list[np.ndarray] = []
    rows_ball: list[np.ndarray] = []
    with open(raw_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            frame_id, players, ball = _parse_line(line, s, line_no)
            ids.append(frame_id)
            rows_players.append(players)
            rows_ball.append(ball)
    if not ids:
        raise ValueError(f"{raw_path}: sin registros")

    frame_ids = np.asarray(ids, dtype=np.int64)
    frames = frame_ids - 1
    if frames.min() < 0:
        raise ValueError(f"{raw_path}: frame_id < 1")
    unique, counts = np.unique(frames, return_counts=True)
    if (counts > 1).any():
        raise ValueError(f"{raw_path}: {int((counts > 1).sum())} frame_id duplicados")
    total = int(n_frames) if n_frames is not None else int(frames.max()) + 1
    if frames.max() >= total:
        raise ValueError(f"{raw_path}: frame {int(frames.max())} fuera de n_frames={total}")

    players = np.stack(rows_players)
    balls = np.stack(rows_ball)
    xy = np.full((total, s, 2), np.nan, dtype=np.float32)
    speed = np.full((total, s), np.nan, dtype=np.float32)
    ball_xy = np.full((total, 2), np.nan, dtype=np.float32)
    present = np.zeros(total, dtype=bool)
    xy[frames] = players[:, :, :2]
    speed[frames] = players[:, :, 2]
    ball_xy[frames] = balls
    present[frames] = True

    observed = np.isfinite(xy).all(axis=2)
    xy[~observed] = np.nan
    ball_observed = np.isfinite(ball_xy).all(axis=1)
    ball_xy[~ball_observed] = np.nan
    speed[~observed] = np.nan

    return AtdData(
        dataset_id=dataset_id,
        source=source,
        fps=int(fps),
        pitch=(float(pitch[0]), float(pitch[1])),
        meta=meta,
        frame=np.arange(total, dtype=np.int32),
        record_present=present,
        period_id=_period_array(total, periods),
        ball_xy=ball_xy,
        ball_observed=ball_observed,
        xy=xy,
        speed=speed,
        observed=observed,
        periods_known=periods is not None,
        provenance={
            "meta_file": Path(meta_path).name,
            "raw_file": Path(raw_path).name,
            "n_records": int(frame_ids.size),
            "frame_id_min": int(frame_ids.min()),
            "frame_id_max": int(frame_ids.max()),
            "frame_ids_contiguous": bool(frame_ids.size == frame_ids.max() - frame_ids.min() + 1),
            "frame_ids_sorted": bool((np.diff(frame_ids) > 0).all()),
        },
    )


def official_games():
    """Importa ``games.py`` del checkout oficial fijado en ``.official/``."""
    if not (OFFICIAL_DIR / "games.py").is_file():
        raise FileNotFoundError(f"falta el checkout oficial en {OFFICIAL_DIR}; ver README.md")
    if str(OFFICIAL_DIR) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_DIR))
    import games  # noqa: PLC0415

    return games


def _cache_key(paths: list[str], config: dict) -> str:
    """Identidad de las entradas (ruta absoluta, tamaño, mtime) más la configuración efectiva."""
    inputs = []
    for path in paths:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        inputs.append([str(resolved), stat.st_size, stat.st_mtime_ns])
    payload = json.dumps({"version": CACHE_VERSION, "inputs": inputs, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _save_cache(path: Path, data: AtdData, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-",
                                         suffix=".npz", delete=False) as handle:
            tmp = Path(handle.name)
            np.savez(
                handle, key=key, dataset_id=data.dataset_id, source=data.source, fps=data.fps,
                pitch=np.array(data.pitch), home_name=data.meta.home_name, away_name=data.meta.away_name,
                slot_team=data.meta.slot_team, record_present=data.record_present,
                period_id=data.period_id, ball_xy=data.ball_xy, xy=data.xy, speed=data.speed,
                periods_known=data.periods_known, provenance=json.dumps(data.provenance),
            )
        try:
            os.replace(tmp, path)
        except PermissionError:
            if os.name != "nt" or not path.is_file():
                raise
            # Windows readers can keep the previous archive open. The freshly
            # validated data remain usable; a later load can refresh the cache.
            warnings.warn("ATD cache could not be replaced; using validated input data without caching",
                          RuntimeWarning, stacklevel=2)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _load_cache(path: Path, key: str) -> AtdData | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle, np.load(handle, allow_pickle=False) as z:
        if str(z["key"]) != key:
            return None
        xy = z["xy"]
        ball_xy = z["ball_xy"]
        return AtdData(
            dataset_id=str(z["dataset_id"]),
            source=str(z["source"]),
            fps=int(z["fps"]),
            pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
            meta=AtdMeta(str(z["home_name"]), str(z["away_name"]), z["slot_team"]),
            frame=np.arange(xy.shape[0], dtype=np.int32),
            record_present=z["record_present"],
            period_id=z["period_id"],
            ball_xy=ball_xy,
            ball_observed=np.isfinite(ball_xy).all(axis=1),
            xy=xy,
            speed=z["speed"],
            observed=np.isfinite(xy).all(axis=2),
            periods_known=bool(z["periods_known"]),
            provenance=json.loads(str(z["provenance"])),
        )


def load_atd(source: str = "tactical", *, use_cache: bool = True, data_dir: str | None = None) -> AtdData:
    """Carga una fuente del partido con rutas, pitch, fps y partes de ``games.py`` oficial.

    Las partes sólo se conocen para ``tactical``; ``broadcast`` no está alineado y su
    ``period_id`` queda en 0 con ``periods_known=False``.
    """
    games = official_games()
    game = games.GAME
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game, data_dir, source)
    periods = None
    if source == games.DEFAULT_SOURCE:
        periods = [tuple(int(v) for v in block) for block in games.half_video_blocks(game)]
    config = {
        "dataset_id": game,
        "source": source,
        "pitch": [float(v) for v in games.GAMES[game]["pitch"]],
        "fps": int(games.GAMES[game]["fps"]),
        "n_frames": int(games.source_frames(source, game)),
        "periods": periods,
    }
    cache_path = CACHE_DIR / f"atd_{game}_{source}.npz"
    key = _cache_key([meta_path, raw_path], config)
    if use_cache:
        try:
            cached = _load_cache(cache_path, key)
        except (ValueError, EOFError, KeyError, BadZipFile, FileNotFoundError):
            # Cache is disposable; authoritative inputs still pass read_atd.
            cached = None
        if cached is not None:
            return cached
    data = read_atd(
        meta_path, raw_path, source=source, pitch=tuple(config["pitch"]), fps=config["fps"],
        n_frames=config["n_frames"], periods=periods, dataset_id=game,
    )
    if use_cache:
        _save_cache(cache_path, data, key)
    return data
