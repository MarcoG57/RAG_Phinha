"""Shared validation and metadata helpers for training and evaluation."""
from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path

from inference_support import (cli_errors as cli_errors, file_manifest as file_manifest,
                               output_directory as output_directory, write_json as write_json)


def finite_values(values, *, positive=False, label="costs"):
    values = list(values)
    if (not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                          or not math.isfinite(v) or (v <= 0 if positive else v < 0)
                          for v in values) or len(set(values)) != len(values)):
        domain = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be distinct, finite, {domain} numbers")
    return values


def manifest_kind(path: Path, *, minimum=1) -> str:
    rows = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("matches", [])
    if not isinstance(rows, list) or len(rows) < minimum:
        raise ValueError(f"manifest requires at least {minimum} matches")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("manifest matches must be tables with id and format")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(mid, str) or not mid for mid in ids) or len(set(ids)) != len(ids):
        raise ValueError("manifest needs unique nonempty match ids")
    formats = {row.get("format") for row in rows}
    if formats != {"synthetic"} and not formats <= {"sportec", "skillcorner", "skillcorner_legacy"}:
        raise ValueError("one evidence kind per run; do not mix fixtures and external matches")
    return "synthetic" if formats == {"synthetic"} else "external"


def json_text(value: dict) -> str:
    """Reject non-JSON TOML types and nonfinite numbers before reserving a run."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError("run metadata requires JSON-compatible values and finite numbers") from error


def degradation_specs(texts, default_seed=None, *, label="--train-variants"):
    """Distinct (scenario, seed) pairs from ``SCENARIO:SEED`` (``SCENARIO`` alone takes ``default_seed``).

    Clean ignores its seed, so two clean entries are duplicates.
    """
    from possession.evaluation.external import SCENARIOS
    specs, keys = [], set()
    for text in texts:
        scenario, separator, seed = str(text).partition(":")
        if not separator and default_seed is not None:
            seed = default_seed
        try:
            seed = int(seed)
        except ValueError:
            raise ValueError(f"{label} needs SCENARIO:SEED with an integer seed: {text}") from None
        if scenario not in SCENARIOS:
            raise ValueError(f"{label}: unknown scenario {scenario}")
        key = (scenario, None if scenario == "clean" else seed)
        if key in keys:
            raise ValueError(f"{label} must be distinct: {text}")
        keys.add(key)
        specs.append((scenario, seed))
    return specs
