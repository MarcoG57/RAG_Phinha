"""Input validation and reproducibility metadata for CPU inference."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from functools import wraps
from pathlib import Path

import numpy as np

from possession.data.atd import official_games

LIBRARIES = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")


def output_directory(root: Path, name: str, area: str = "final") -> Path:
    parent = (root / "artifacts" / area).resolve()
    if not isinstance(name, str) or not name.strip():
        raise ValueError("--name must identify a new child of artifacts/" + area)
    output = (parent / name).resolve()
    if output == parent or not output.is_relative_to(parent):
        raise ValueError("--name must resolve to a child of artifacts/" + area)
    if output.exists():
        raise ValueError(f"output directory already exists: {output}; choose a new --name")
    return output


def file_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def write_json(path: Path, value: dict) -> None:
    path.write_text(json_text(value), encoding="utf-8")


def json_text(value: dict) -> str:
    """Reject non-JSON TOML types and nonfinite numbers before reserving a run."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError("run metadata requires JSON-compatible values and finite numbers") from error


def cli_errors(function):
    """Expected command failures are concise; interrupts retain their conventional code."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt:
            print("FAIL interrupted", file=sys.stderr)
            return 130
        except Exception as error:
            print(f"FAIL {error}", file=sys.stderr)
            return 1
    return wrapped


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resolve_atd_inputs(data_dir: str | None) -> dict[str, Path]:
    """Resolve the same tactical metadata/raw pair used by ``load_atd``."""
    games = official_games()
    meta, raw = games.require(["atd_meta", "atd_raw"], games.GAME, data_dir, "tactical")
    return {"metadata": Path(meta).resolve(), "raw": Path(raw).resolve()}


def input_file_manifest(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def validate_atd_alignment(data) -> None:
    provenance = data.provenance
    record_present = np.asarray(data.record_present)
    checks = {
        "record_present": record_present.shape == (data.n_frames,) and bool(record_present.all()),
        "frame_ids_contiguous": provenance.get("frame_ids_contiguous") is True,
        "frame_ids_sorted": provenance.get("frame_ids_sorted") is True,
        "frame_id_min": provenance.get("frame_id_min") == 1,
        "n_records": provenance.get("n_records") == data.n_frames,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("ATD frame alignment invalid: " + ", ".join(failed))


def feature_versions(recipe: str) -> list[str]:
    from possession.learning.features import (BALL_SPEED_FEATURE_VERSION, FEATURE_VERSION, ONPITCH_FEATURE_VERSION,
                                              SHAPE_FEATURE_VERSION, TEMPORAL_FEATURE_VERSION)
    return {"gru11": [FEATURE_VERSION, TEMPORAL_FEATURE_VERSION, SHAPE_FEATURE_VERSION, ONPITCH_FEATURE_VERSION],
            "gru11b": [FEATURE_VERSION, TEMPORAL_FEATURE_VERSION, SHAPE_FEATURE_VERSION, ONPITCH_FEATURE_VERSION,
                       BALL_SPEED_FEATURE_VERSION]}[recipe]


def library_versions(learner: str) -> dict:
    names = LIBRARIES + (("torch",) if learner == "gru" else ())
    return {"python": sys.version.split()[0], **{name: importlib.metadata.version(name) for name in names}}
